/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_ADD_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_ADD_H_

#include <algorithm>

#include "fixedpoint/fixedpoint.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/compatibility.h"
#include "tflite/kernels/internal/optimized/avx2_quantization_utils.h"
#include "tflite/kernels/internal/optimized/cpu_check.h"
#include "tflite/kernels/internal/optimized/neon_check.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/reference/integer_ops/add.h"
#include "tflite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
inline void AddElementwiseInt8(int size, const ArithmeticParams& params,
                               const int8* input1_data, const int8* input2_data,
                               int8* output_data) {
  ruy::profiler::ScopeLabel label("AddElementwiseInt8/8bit");
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);

#ifdef USE_NEON
  const int8x16_t output_activation_min_vector =
      vdupq_n_s8(params.quantized_activation_min);
  const int8x16_t output_activation_max_vector =
      vdupq_n_s8(params.quantized_activation_max);

  const int input1_left_shift = params.left_shift + params.input1_shift;
  const int input2_left_shift = params.left_shift + params.input2_shift;
  const int32x4_t input1_left_dup = vdupq_n_s32(input1_left_shift);
  const int32x4_t input2_left_dup = vdupq_n_s32(input2_left_shift);

  const int16x8_t input1_offset_dup = vdupq_n_s16(params.input1_offset);
  const int16x8_t input2_offset_dup = vdupq_n_s16(params.input2_offset);

  for (; i <= size - 16; i += 16) {
    const int8x16_t input1_val_original = vld1q_s8(input1_data + i);
    const int8x16_t input2_val_original = vld1q_s8(input2_data + i);

    const int16x8_t input1_val_s16_high =
        vmovl_s8(vget_high_s8(input1_val_original));
    const int16x8_t input1_val_s16_low =
        vmovl_s8(vget_low_s8(input1_val_original));

    const int16x8_t input2_val_s16_high =
        vmovl_s8(vget_high_s8(input2_val_original));
    const int16x8_t input2_val_s16_low =
        vmovl_s8(vget_low_s8(input2_val_original));
    const int16x8_t input1_val_high =
        vaddq_s16(input1_val_s16_high, input1_offset_dup);
    const int16x8_t input2_val_high =
        vaddq_s16(input2_val_s16_high, input2_offset_dup);
    const int16x8_t input1_val_low =
        vaddq_s16(input1_val_s16_low, input1_offset_dup);
    const int16x8_t input2_val_low =
        vaddq_s16(input2_val_s16_low, input2_offset_dup);
    const int16x4_t input1_val_high_high = vget_high_s16(input1_val_high);
    const int16x4_t input1_val_high_low = vget_low_s16(input1_val_high);
    const int16x4_t input1_val_low_high = vget_high_s16(input1_val_low);
    const int16x4_t input1_val_low_low = vget_low_s16(input1_val_low);
    const int16x4_t input2_val_high_high = vget_high_s16(input2_val_high);
    const int16x4_t input2_val_high_low = vget_low_s16(input2_val_high);
    const int16x4_t input2_val_low_high = vget_high_s16(input2_val_low);
    const int16x4_t input2_val_low_low = vget_low_s16(input2_val_low);
    int32x4_t x111 = vmovl_s16(input1_val_low_low);
    int32x4_t x112 = vmovl_s16(input1_val_low_high);
    int32x4_t x121 = vmovl_s16(input1_val_high_low);
    int32x4_t x122 = vmovl_s16(input1_val_high_high);
    int32x4_t x211 = vmovl_s16(input2_val_low_low);
    int32x4_t x212 = vmovl_s16(input2_val_low_high);
    int32x4_t x221 = vmovl_s16(input2_val_high_low);
    int32x4_t x222 = vmovl_s16(input2_val_high_high);

    x111 = vshlq_s32(x111, input1_left_dup);
    x112 = vshlq_s32(x112, input1_left_dup);
    x121 = vshlq_s32(x121, input1_left_dup);
    x122 = vshlq_s32(x122, input1_left_dup);
    x211 = vshlq_s32(x211, input2_left_dup);
    x212 = vshlq_s32(x212, input2_left_dup);
    x221 = vshlq_s32(x221, input2_left_dup);
    x222 = vshlq_s32(x222, input2_left_dup);
    x111 = vqrdmulhq_n_s32(x111, params.input1_multiplier);
    x112 = vqrdmulhq_n_s32(x112, params.input1_multiplier);
    x121 = vqrdmulhq_n_s32(x121, params.input1_multiplier);
    x122 = vqrdmulhq_n_s32(x122, params.input1_multiplier);
    x211 = vqrdmulhq_n_s32(x211, params.input2_multiplier);
    x212 = vqrdmulhq_n_s32(x212, params.input2_multiplier);
    x221 = vqrdmulhq_n_s32(x221, params.input2_multiplier);
    x222 = vqrdmulhq_n_s32(x222, params.input2_multiplier);
    int32x4_t s11 = vaddq_s32(x111, x211);
    int32x4_t s12 = vaddq_s32(x112, x212);
    int32x4_t s21 = vaddq_s32(x121, x221);
    int32x4_t s22 = vaddq_s32(x122, x222);
    s11 = vqrdmulhq_n_s32(s11, params.output_multiplier);
    s12 = vqrdmulhq_n_s32(s12, params.output_multiplier);
    s21 = vqrdmulhq_n_s32(s21, params.output_multiplier);
    s22 = vqrdmulhq_n_s32(s22, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    s11 = RoundingDivideByPOT(s11, -params.output_shift);
    s12 = RoundingDivideByPOT(s12, -params.output_shift);
    s21 = RoundingDivideByPOT(s21, -params.output_shift);
    s22 = RoundingDivideByPOT(s22, -params.output_shift);
    const int16x4_t s11_narrowed = vmovn_s32(s11);
    const int16x4_t s12_narrowed = vmovn_s32(s12);
    const int16x4_t s21_narrowed = vmovn_s32(s21);
    const int16x4_t s22_narrowed = vmovn_s32(s22);
    const int16x8_t s1 = vaddq_s16(vcombine_s16(s11_narrowed, s12_narrowed),
                                   vdupq_n_s16(params.output_offset));
    const int16x8_t s2 = vaddq_s16(vcombine_s16(s21_narrowed, s22_narrowed),
                                   vdupq_n_s16(params.output_offset));
    const int8x16_t s = vcombine_s8(vqmovn_s16(s1), vqmovn_s16(s2));

    const int8x16_t clamped =
        vmaxq_s8(output_activation_min_vector,
                 vminq_s8(output_activation_max_vector, s));
    vst1q_s8(output_data + i, clamped);
  }
#elif defined(USE_RVV)
  // RVV optimized path for int8 element-wise add.
  // Pipeline: i8 load -> i16 widen + offset -> i32 widen + shift -> qrdmulh
  //           -> sum -> output qrdmulh -> RoundingDivideByPOT -> offset
  //           -> clamp -> saturating narrow -> i8 store.
  // qrdmulh matches NEON vqrdmulhq_n_s32: (a * b * 2 + 2^31) >> 32.
  {
    const int32 combined1_shift = params.left_shift + params.input1_shift;
    const int32 combined2_shift = params.left_shift + params.input2_shift;
    size_t vl;
    for (; i < size; i += vl) {
      vl = __riscv_vsetvl_e8m1(size - i);
      // Load int8, widen to i16, add input offsets.
      vint16m2_t v1 = __riscv_vsext_vf2_i16m2(
          __riscv_vle8_v_i8m1(input1_data + i, vl), vl);
      vint16m2_t v2 = __riscv_vsext_vf2_i16m2(
          __riscv_vle8_v_i8m1(input2_data + i, vl), vl);
      v1 = __riscv_vadd_vx_i16m2(v1, static_cast<int16_t>(params.input1_offset), vl);
      v2 = __riscv_vadd_vx_i16m2(v2, static_cast<int16_t>(params.input2_offset), vl);
      // Widen i16 -> i32.
      vint32m4_t x1 = __riscv_vsext_vf2_i32m4(v1, vl);
      vint32m4_t x2 = __riscv_vsext_vf2_i32m4(v2, vl);
      // Combined left shift (left_shift + input_shift).
      x1 = __riscv_vsll_vx_i32m4(x1, combined1_shift, vl);
      x2 = __riscv_vsll_vx_i32m4(x2, combined2_shift, vl);
      // qrdmulh: (a * b * 2 + 2^31) >> 32 — matches NEON vqrdmulhq_n_s32.
      {
        vint64m8_t p = __riscv_vwmul_vx_i64m8(x1, params.input1_multiplier, vl);
        p = __riscv_vsll_vx_i64m8(p, 1, vl);
        p = __riscv_vadd_vx_i64m8(p, static_cast<int64_t>(1) << 31, vl);
        x1 = __riscv_vnsra_wx_i32m4(p, 32, vl);
      }
      {
        vint64m8_t p = __riscv_vwmul_vx_i64m8(x2, params.input2_multiplier, vl);
        p = __riscv_vsll_vx_i64m8(p, 1, vl);
        p = __riscv_vadd_vx_i64m8(p, static_cast<int64_t>(1) << 31, vl);
        x2 = __riscv_vnsra_wx_i32m4(p, 32, vl);
      }
      // Sum the two scaled inputs.
      vint32m4_t s = __riscv_vadd_vv_i32m4(x1, x2, vl);
      // Output qrdmulh.
      {
        vint64m8_t p = __riscv_vwmul_vx_i64m8(s, params.output_multiplier, vl);
        p = __riscv_vsll_vx_i64m8(p, 1, vl);
        p = __riscv_vadd_vx_i64m8(p, static_cast<int64_t>(1) << 31, vl);
        s = __riscv_vnsra_wx_i32m4(p, 32, vl);
      }
      // RoundingDivideByPOT — matches gemmlowp::RoundingDivideByPOT.
      {
        const int pot = -params.output_shift;
        if (pot > 0) {
          vint32m4_t rnd = __riscv_vsra_vx_i32m4(s, pot - 1, vl);
          rnd = __riscv_vand_vx_i32m4(rnd, 1, vl);
          s = __riscv_vsra_vx_i32m4(s, pot, vl);
          s = __riscv_vadd_vv_i32m4(s, rnd, vl);
        }
      }
      // Add output offset.
      s = __riscv_vadd_vx_i32m4(s, params.output_offset, vl);
      // Clamp to activation range, saturating narrow i32 -> i16 -> i8.
      s = __riscv_vmax_vx_i32m4(s, params.quantized_activation_min, vl);
      s = __riscv_vmin_vx_i32m4(s, params.quantized_activation_max, vl);
      vint16m2_t s16 = __riscv_vnclip_wx_i16m2(s, 0, /*rounding=*/0, vl);
      vint8m1_t out = __riscv_vnclip_wx_i8m1(s16, 0, /*rounding=*/0, vl);
      __riscv_vse8_v_i8m1(output_data + i, out, vl);
    }
  }
#endif  // NEON/RVV

  for (; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32 shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32 scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32 scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32 raw_sum = scaled_input1_val + scaled_input2_val;
    const int32 raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<int8>(clamped_output);
  }
}

// Element-wise add is used for the non-broadcast add.
inline void AddElementwiseInt16(int size, const ArithmeticParams& params,
                                const int16* input1_data,
                                const int16* input2_data, int16* output_data) {
  ruy::profiler::ScopeLabel label("AddElementwiseInt16/16bit");
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -32768);
  TFLITE_DCHECK_GT(params.input2_offset, -32768);
  TFLITE_DCHECK_LT(params.input1_offset, 32768);
  TFLITE_DCHECK_LT(params.input2_offset, 32768);

#ifdef __AVX2__
  const int32_t input1_left_shift = params.left_shift + params.input1_shift;
  const int32_t input2_left_shift = params.left_shift + params.input2_shift;
  const __m256i input1_offset = _mm256_set1_epi32(params.input1_offset);
  const __m256i input2_offset = _mm256_set1_epi32(params.input2_offset);
  const __m256i output_offset = _mm256_set1_epi32(params.output_offset);
  const __m256i clamp_max_v =
      _mm256_set1_epi32(params.quantized_activation_max);
  const __m256i clamp_min_v =
      _mm256_set1_epi32(params.quantized_activation_min);

  for (; i <= size - 16; i += 16) {
    const __m256i input1_val_original =
        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(input1_data + i));
    const __m256i input2_val_original =
        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(input2_data + i));

    __m256i s11 =
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(input1_val_original));
    __m256i s12 =
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(input1_val_original, 1));
    __m256i s21 =
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(input2_val_original));
    __m256i s22 =
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(input2_val_original, 1));

    s11 = _mm256_add_epi32(s11, input1_offset);
    s12 = _mm256_add_epi32(s12, input1_offset);
    s21 = _mm256_add_epi32(s21, input2_offset);
    s22 = _mm256_add_epi32(s22, input2_offset);

    s11 = avx2_utils::MultiplyByQuantizedMultiplier(
        s11, params.input1_multiplier, input1_left_shift);
    s12 = avx2_utils::MultiplyByQuantizedMultiplier(
        s12, params.input1_multiplier, input1_left_shift);
    s21 = avx2_utils::MultiplyByQuantizedMultiplier(
        s21, params.input2_multiplier, input2_left_shift);
    s22 = avx2_utils::MultiplyByQuantizedMultiplier(
        s22, params.input2_multiplier, input2_left_shift);

    __m256i s1 = _mm256_add_epi32(s11, s21);
    __m256i s2 = _mm256_add_epi32(s12, s22);

    s1 = avx2_utils::MultiplyByQuantizedMultiplier(s1, params.output_multiplier,
                                                   params.output_shift);
    s2 = avx2_utils::MultiplyByQuantizedMultiplier(s2, params.output_multiplier,
                                                   params.output_shift);

    s1 = _mm256_add_epi32(s1, output_offset);
    s2 = _mm256_add_epi32(s2, output_offset);

    s1 = _mm256_min_epi32(s1, clamp_max_v);
    s1 = _mm256_max_epi32(s1, clamp_min_v);
    s2 = _mm256_min_epi32(s2, clamp_max_v);
    s2 = _mm256_max_epi32(s2, clamp_min_v);

    avx2_utils::CastInt32ToInt16AndStore(output_data + i, s1);
    avx2_utils::CastInt32ToInt16AndStore(output_data + i + 8, s2);
  }

#elif defined(USE_NEON)
  const int32x4_t output_activation_min_vector =
      vdupq_n_s32(params.quantized_activation_min);
  const int32x4_t output_activation_max_vector =
      vdupq_n_s32(params.quantized_activation_max);

  const int input1_left_shift = params.left_shift + params.input1_shift;
  const int input2_left_shift = params.left_shift + params.input2_shift;
  const int32x4_t input1_left_dup = vdupq_n_s32(input1_left_shift);
  const int32x4_t input2_left_dup = vdupq_n_s32(input2_left_shift);

  const int32x4_t input1_offset_dup = vdupq_n_s32(params.input1_offset);
  const int32x4_t input2_offset_dup = vdupq_n_s32(params.input2_offset);
  const int32x4_t output_offset_dup = vdupq_n_s32(params.output_offset);

  // Use the size 16 batch as it is effective on pixel 3/4.
  for (; i <= size - 16; i += 16) {
    const int16x8_t input11_val_original = vld1q_s16(input1_data + i);
    const int16x8_t input12_val_original = vld1q_s16(input2_data + i);
    const int16x8_t input21_val_original = vld1q_s16(input1_data + 8 + i);
    const int16x8_t input22_val_original = vld1q_s16(input2_data + 8 + i);

    int32x4_t x111 = vmovl_s16(vget_low_s16(input11_val_original));
    int32x4_t x112 = vmovl_s16(vget_high_s16(input11_val_original));
    int32x4_t x121 = vmovl_s16(vget_low_s16(input12_val_original));
    int32x4_t x122 = vmovl_s16(vget_high_s16(input12_val_original));
    int32x4_t x211 = vmovl_s16(vget_low_s16(input21_val_original));
    int32x4_t x212 = vmovl_s16(vget_high_s16(input21_val_original));
    int32x4_t x221 = vmovl_s16(vget_low_s16(input22_val_original));
    int32x4_t x222 = vmovl_s16(vget_high_s16(input22_val_original));

    x111 = vaddq_s32(x111, input1_offset_dup);
    x112 = vaddq_s32(x112, input1_offset_dup);
    x121 = vaddq_s32(x121, input2_offset_dup);
    x122 = vaddq_s32(x122, input2_offset_dup);
    x211 = vaddq_s32(x211, input1_offset_dup);
    x212 = vaddq_s32(x212, input1_offset_dup);
    x221 = vaddq_s32(x221, input2_offset_dup);
    x222 = vaddq_s32(x222, input2_offset_dup);

    x111 = vshlq_s32(x111, input1_left_dup);
    x112 = vshlq_s32(x112, input1_left_dup);
    x121 = vshlq_s32(x121, input2_left_dup);
    x122 = vshlq_s32(x122, input2_left_dup);
    x211 = vshlq_s32(x211, input1_left_dup);
    x212 = vshlq_s32(x212, input1_left_dup);
    x221 = vshlq_s32(x221, input2_left_dup);
    x222 = vshlq_s32(x222, input2_left_dup);

    x111 = vqrdmulhq_n_s32(x111, params.input1_multiplier);
    x112 = vqrdmulhq_n_s32(x112, params.input1_multiplier);
    x121 = vqrdmulhq_n_s32(x121, params.input2_multiplier);
    x122 = vqrdmulhq_n_s32(x122, params.input2_multiplier);
    x211 = vqrdmulhq_n_s32(x211, params.input1_multiplier);
    x212 = vqrdmulhq_n_s32(x212, params.input1_multiplier);
    x221 = vqrdmulhq_n_s32(x221, params.input2_multiplier);
    x222 = vqrdmulhq_n_s32(x222, params.input2_multiplier);

    int32x4_t s11 = vaddq_s32(x111, x121);
    int32x4_t s12 = vaddq_s32(x112, x122);
    int32x4_t s21 = vaddq_s32(x211, x221);
    int32x4_t s22 = vaddq_s32(x212, x222);

    s11 = vqrdmulhq_n_s32(s11, params.output_multiplier);
    s12 = vqrdmulhq_n_s32(s12, params.output_multiplier);
    s21 = vqrdmulhq_n_s32(s21, params.output_multiplier);
    s22 = vqrdmulhq_n_s32(s22, params.output_multiplier);

    using gemmlowp::RoundingDivideByPOT;
    s11 = RoundingDivideByPOT(s11, -params.output_shift);
    s12 = RoundingDivideByPOT(s12, -params.output_shift);
    s21 = RoundingDivideByPOT(s21, -params.output_shift);
    s22 = RoundingDivideByPOT(s22, -params.output_shift);

    s11 = vaddq_s32(s11, output_offset_dup);
    s12 = vaddq_s32(s12, output_offset_dup);
    s21 = vaddq_s32(s21, output_offset_dup);
    s22 = vaddq_s32(s22, output_offset_dup);

    s11 = vmaxq_s32(output_activation_min_vector,
                    vminq_s32(output_activation_max_vector, s11));
    s12 = vmaxq_s32(output_activation_min_vector,
                    vminq_s32(output_activation_max_vector, s12));
    s21 = vmaxq_s32(output_activation_min_vector,
                    vminq_s32(output_activation_max_vector, s21));
    s22 = vmaxq_s32(output_activation_min_vector,
                    vminq_s32(output_activation_max_vector, s22));

    const int16x8_t s1 = vcombine_s16(vqmovn_s32(s11), vqmovn_s32(s12));
    const int16x8_t s2 = vcombine_s16(vqmovn_s32(s21), vqmovn_s32(s22));

    vst1q_s16(output_data + i, s1);
    vst1q_s16(output_data + 8 + i, s2);
  }
#elif defined(USE_RVV)
  // RVV optimized path for int16 element-wise add.
  // Pipeline: i16 load -> i32 widen + offset + shift -> qrdmulh -> sum
  //           -> output qrdmulh -> RoundingDivideByPOT -> offset -> clamp
  //           -> saturating narrow i32 -> i16 store.
  // Uses e16m2/e32m4 LMUL chain.
  {
    const int32 combined1_shift = params.left_shift + params.input1_shift;
    const int32 combined2_shift = params.left_shift + params.input2_shift;
    size_t vl;
    for (; i < size; i += vl) {
      vl = __riscv_vsetvl_e16m2(size - i);
      // Load int16, widen to i32, add input offsets.
      vint32m4_t x1 = __riscv_vsext_vf2_i32m4(
          __riscv_vle16_v_i16m2(input1_data + i, vl), vl);
      vint32m4_t x2 = __riscv_vsext_vf2_i32m4(
          __riscv_vle16_v_i16m2(input2_data + i, vl), vl);
      x1 = __riscv_vadd_vx_i32m4(x1, params.input1_offset, vl);
      x2 = __riscv_vadd_vx_i32m4(x2, params.input2_offset, vl);
      // Combined left shift (left_shift + input_shift).
      x1 = __riscv_vsll_vx_i32m4(x1, combined1_shift, vl);
      x2 = __riscv_vsll_vx_i32m4(x2, combined2_shift, vl);
      // qrdmulh: (a * b * 2 + 2^31) >> 32 — matches NEON vqrdmulhq_n_s32.
      {
        vint64m8_t p = __riscv_vwmul_vx_i64m8(x1, params.input1_multiplier, vl);
        p = __riscv_vsll_vx_i64m8(p, 1, vl);
        p = __riscv_vadd_vx_i64m8(p, static_cast<int64_t>(1) << 31, vl);
        x1 = __riscv_vnsra_wx_i32m4(p, 32, vl);
      }
      {
        vint64m8_t p = __riscv_vwmul_vx_i64m8(x2, params.input2_multiplier, vl);
        p = __riscv_vsll_vx_i64m8(p, 1, vl);
        p = __riscv_vadd_vx_i64m8(p, static_cast<int64_t>(1) << 31, vl);
        x2 = __riscv_vnsra_wx_i32m4(p, 32, vl);
      }
      // Sum the two scaled inputs.
      vint32m4_t s = __riscv_vadd_vv_i32m4(x1, x2, vl);
      // Output qrdmulh.
      {
        vint64m8_t p = __riscv_vwmul_vx_i64m8(s, params.output_multiplier, vl);
        p = __riscv_vsll_vx_i64m8(p, 1, vl);
        p = __riscv_vadd_vx_i64m8(p, static_cast<int64_t>(1) << 31, vl);
        s = __riscv_vnsra_wx_i32m4(p, 32, vl);
      }
      // RoundingDivideByPOT — matches gemmlowp::RoundingDivideByPOT.
      {
        const int pot = -params.output_shift;
        if (pot > 0) {
          vint32m4_t rnd = __riscv_vsra_vx_i32m4(s, pot - 1, vl);
          rnd = __riscv_vand_vx_i32m4(rnd, 1, vl);
          s = __riscv_vsra_vx_i32m4(s, pot, vl);
          s = __riscv_vadd_vv_i32m4(s, rnd, vl);
        }
      }
      // Add output offset and clamp.
      s = __riscv_vadd_vx_i32m4(s, params.output_offset, vl);
      s = __riscv_vmax_vx_i32m4(s, params.quantized_activation_min, vl);
      s = __riscv_vmin_vx_i32m4(s, params.quantized_activation_max, vl);
      // Saturating narrow i32 -> i16 (matches NEON vqmovn_s32).
      vint16m2_t out = __riscv_vnclip_wx_i16m2(s, 0, /*rounding=*/0, vl);
      __riscv_vse16_v_i16m2(output_data + i, out, vl);
    }
  }
#endif  // AVX2/NEON/RVV

  for (; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32 shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32 scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32 scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32 raw_sum = scaled_input1_val + scaled_input2_val;
    const int32 raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<int16>(clamped_output);
  }
}

// Scalar-broadcast add that can be used for inner loop of more general
// broadcast add, so that, for example, scalar-broadcast with batch will still
// be fast.
inline void AddScalarBroadcast(int size, const ArithmeticParams& params,
                               int8 input1_data, const int8* input2_data,
                               int8* output_data) {
  using gemmlowp::RoundingDivideByPOT;

  ruy::profiler::ScopeLabel label("AddScalarBroadcastInt8/8bit");
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);

  int i = 0;

#ifdef USE_NEON
  const int32x4_t left_shift_dup = vdupq_n_s32(params.left_shift);
  const int8x8_t output_activation_min_vector =
      vdup_n_s8(params.quantized_activation_min);
  const int8x8_t output_activation_max_vector =
      vdup_n_s8(params.quantized_activation_max);

  // Process broadcast scalar.
  const int8x8_t input1_val_original = vdup_n_s8(input1_data);
  const int16x8_t input1_val_s16 = vmovl_s8(input1_val_original);
  const int16x8_t input1_val =
      vaddq_s16(input1_val_s16, vdupq_n_s16(params.input1_offset));
  const int16x4_t input1_val_high = vget_high_s16(input1_val);
  const int16x4_t input1_val_low = vget_low_s16(input1_val);
  int32x4_t x11 = vmovl_s16(input1_val_low);
  int32x4_t x12 = vmovl_s16(input1_val_high);
  x11 = vshlq_s32(x11, left_shift_dup);
  x12 = vshlq_s32(x12, left_shift_dup);
  x11 = vqrdmulhq_n_s32(x11, params.input1_multiplier);
  x12 = vqrdmulhq_n_s32(x12, params.input1_multiplier);
  const int32x4_t input1_shift_dup = vdupq_n_s32(params.input1_shift);
  x11 = vshlq_s32(x11, input1_shift_dup);
  x12 = vshlq_s32(x12, input1_shift_dup);

  for (; i <= size - 8; i += 8) {
    const int8x8_t input2_val_original = vld1_s8(input2_data + i);
    const int16x8_t input2_val_s16 = vmovl_s8(input2_val_original);
    const int16x8_t input2_val =
        vaddq_s16(input2_val_s16, vdupq_n_s16(params.input2_offset));
    const int16x4_t input2_val_high = vget_high_s16(input2_val);
    const int16x4_t input2_val_low = vget_low_s16(input2_val);
    int32x4_t x21 = vmovl_s16(input2_val_low);
    int32x4_t x22 = vmovl_s16(input2_val_high);
    x21 = vshlq_s32(x21, left_shift_dup);
    x22 = vshlq_s32(x22, left_shift_dup);
    x21 = vqrdmulhq_n_s32(x21, params.input2_multiplier);
    x22 = vqrdmulhq_n_s32(x22, params.input2_multiplier);
    const int32x4_t input2_shift_dup = vdupq_n_s32(params.input2_shift);
    x21 = vshlq_s32(x21, input2_shift_dup);
    x22 = vshlq_s32(x22, input2_shift_dup);
    int32x4_t s1 = vaddq_s32(x11, x21);
    int32x4_t s2 = vaddq_s32(x12, x22);
    s1 = vqrdmulhq_n_s32(s1, params.output_multiplier);
    s2 = vqrdmulhq_n_s32(s2, params.output_multiplier);
    s1 = RoundingDivideByPOT(s1, -params.output_shift);
    s2 = RoundingDivideByPOT(s2, -params.output_shift);
    const int16x4_t s1_narrowed = vmovn_s32(s1);
    const int16x4_t s2_narrowed = vmovn_s32(s2);
    const int16x8_t s = vaddq_s16(vcombine_s16(s1_narrowed, s2_narrowed),
                                  vdupq_n_s16(params.output_offset));
    const int8x8_t clamped =
        vmax_s8(output_activation_min_vector,
                vmin_s8(output_activation_max_vector, vqmovn_s16(s)));
    vst1_s8(output_data + i, clamped);
  }
#elif defined(USE_RVV)
  // RVV optimized path for int8 scalar-broadcast add.
  // Precompute scalar input1 outside the loop, then process input2 vectors.
  // qrdmulh matches NEON vqrdmulhq_n_s32: (a * b * 2 + 2^31) >> 32.
  {
    // Precompute scalar input1: offset -> left_shift -> qrdmulh -> input1_shift.
    int32_t scalar_x1 = static_cast<int32_t>(input1_data) + params.input1_offset;
    scalar_x1 = scalar_x1 * (1 << params.left_shift);
    {
      int64_t p = static_cast<int64_t>(scalar_x1) * params.input1_multiplier;
      p = (p << 1) + (static_cast<int64_t>(1) << 31);
      scalar_x1 = static_cast<int32_t>(p >> 32);
    }
    if (params.input1_shift < 0) {
      scalar_x1 >>= -params.input1_shift;
    } else if (params.input1_shift > 0) {
      scalar_x1 <<= params.input1_shift;
    }

    size_t vl;
    for (; i < size; i += vl) {
      vl = __riscv_vsetvl_e8m1(size - i);
      // Load int8 input2, widen to i16, add offset.
      vint16m2_t v2 = __riscv_vsext_vf2_i16m2(
          __riscv_vle8_v_i8m1(input2_data + i, vl), vl);
      v2 = __riscv_vadd_vx_i16m2(v2, static_cast<int16_t>(params.input2_offset), vl);
      // Widen i16 -> i32.
      vint32m4_t x2 = __riscv_vsext_vf2_i32m4(v2, vl);
      // Left shift.
      x2 = __riscv_vsll_vx_i32m4(x2, params.left_shift, vl);
      // qrdmulh with input2_multiplier.
      {
        vint64m8_t p = __riscv_vwmul_vx_i64m8(x2, params.input2_multiplier, vl);
        p = __riscv_vsll_vx_i64m8(p, 1, vl);
        p = __riscv_vadd_vx_i64m8(p, static_cast<int64_t>(1) << 31, vl);
        x2 = __riscv_vnsra_wx_i32m4(p, 32, vl);
      }
      // Apply input2_shift.
      if (params.input2_shift < 0) {
        x2 = __riscv_vsra_vx_i32m4(x2, -params.input2_shift, vl);
      } else if (params.input2_shift > 0) {
        x2 = __riscv_vsll_vx_i32m4(x2, params.input2_shift, vl);
      }
      // Add precomputed scalar input1.
      vint32m4_t s = __riscv_vadd_vx_i32m4(x2, scalar_x1, vl);
      // Output qrdmulh.
      {
        vint64m8_t p = __riscv_vwmul_vx_i64m8(s, params.output_multiplier, vl);
        p = __riscv_vsll_vx_i64m8(p, 1, vl);
        p = __riscv_vadd_vx_i64m8(p, static_cast<int64_t>(1) << 31, vl);
        s = __riscv_vnsra_wx_i32m4(p, 32, vl);
      }
      // RoundingDivideByPOT.
      {
        const int pot = -params.output_shift;
        if (pot > 0) {
          vint32m4_t rnd = __riscv_vsra_vx_i32m4(s, pot - 1, vl);
          rnd = __riscv_vand_vx_i32m4(rnd, 1, vl);
          s = __riscv_vsra_vx_i32m4(s, pot, vl);
          s = __riscv_vadd_vv_i32m4(s, rnd, vl);
        }
      }
      // Add output offset.
      s = __riscv_vadd_vx_i32m4(s, params.output_offset, vl);
      // Clamp and saturating narrow i32 -> i16 -> i8.
      s = __riscv_vmax_vx_i32m4(s, params.quantized_activation_min, vl);
      s = __riscv_vmin_vx_i32m4(s, params.quantized_activation_max, vl);
      vint16m2_t s16 = __riscv_vnclip_wx_i16m2(s, 0, /*rounding=*/0, vl);
      vint8m1_t out = __riscv_vnclip_wx_i8m1(s16, 0, /*rounding=*/0, vl);
      __riscv_vse8_v_i8m1(output_data + i, out, vl);
    }
  }
#endif  // NEON/RVV

  if (i < size) {
    // Process broadcast scalar.
    const int32 input1_val = params.input1_offset + input1_data;
    const int32 shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32 scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);

    for (; i < size; ++i) {
      const int32 input2_val = params.input2_offset + input2_data[i];
      const int32 shifted_input2_val = input2_val * (1 << params.left_shift);
      const int32 scaled_input2_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_input2_val, params.input2_multiplier,
              params.input2_shift);
      const int32 raw_sum = scaled_input1_val + scaled_input2_val;
      const int32 raw_output =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              raw_sum, params.output_multiplier, params.output_shift) +
          params.output_offset;
      const int32 clamped_output =
          std::min(params.quantized_activation_max,
                   std::max(params.quantized_activation_min, raw_output));
      output_data[i] = static_cast<int8>(clamped_output);
    }
  }
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8* input1_data,
                const RuntimeShape& input2_shape, const int8* input2_data,
                const RuntimeShape& output_shape, int8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  ruy::profiler::ScopeLabel label("AddInt8/8bit");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  AddElementwiseInt8(flat_size, params, input1_data, input2_data, output_data);
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, int16* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  ruy::profiler::ScopeLabel label("AddInt16/16bit");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_GT(params.input1_offset, -32768);
  TFLITE_DCHECK_GT(params.input2_offset, -32768);
  TFLITE_DCHECK_LT(params.input1_offset, 32768);
  TFLITE_DCHECK_LT(params.input2_offset, 32768);
  AddElementwiseInt16(flat_size, params, input1_data, input2_data, output_data);
}

inline void BroadcastAddDispatch(const ArithmeticParams& params,
                                 const RuntimeShape& input1_shape,
                                 const int8* input1_data,
                                 const RuntimeShape& input2_shape,
                                 const int8* input2_data,
                                 const RuntimeShape& output_shape,
                                 int8* output_data) {
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast) {
    return reference_integer_ops::BroadcastAdd6DSlow(
        params, input1_shape, input1_data, input2_shape, input2_data,
        output_shape, output_data);
  }

  optimized_ops::BinaryBroadcastFiveFold(
      params, input1_shape, input1_data, input2_shape, input2_data,
      output_shape, output_data, AddElementwiseInt8, AddScalarBroadcast);
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_ADD_H_
