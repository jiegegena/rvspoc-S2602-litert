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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MUL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MUL_H_

#include <algorithm>

#include "fixedpoint/fixedpoint.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/compatibility.h"
#include "tflite/kernels/internal/optimized/cpu_check.h"
#include "tflite/kernels/internal/optimized/neon_check.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"
#include "tflite/kernels/internal/optimized/rvv_utils.h"
#include "tflite/kernels/internal/reference/integer_ops/mul.h"
#include "tflite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

// Element-wise mul that can often be used for inner loop of broadcast Mul as
// well as the non-broadcast Mul.
inline void MulElementwise(int size, const ArithmeticParams& params,
                           const int8* input1_data, const int8* input2_data,
                           int8* output_data) {
  ruy::profiler::ScopeLabel label("MulElementwiseInt8/8bit");
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);
#ifdef USE_NEON
  const int16x8_t input1_offset_vector = vdupq_n_s16(params.input1_offset);
  const int16x8_t input2_offset_vector = vdupq_n_s16(params.input2_offset);
  const int16x8_t output_offset_vector = vdupq_n_s16(params.output_offset);
  const auto output_activation_min_vector =
      vdupq_n_s8(params.quantized_activation_min);
  const auto output_activation_max_vector =
      vdupq_n_s8(params.quantized_activation_max);
  const int left_shift = std::max(0, params.output_shift);
  const int right_shift = std::max(0, -params.output_shift);
  const int32x4_t left_shift_vec = vdupq_n_s32(left_shift);
  for (; i <= size - 16; i += 16) {
    // We load / store 16 at a time, multiplying as four sets of 4 int32s.
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
        vaddq_s16(input1_val_s16_high, input1_offset_vector);
    const int16x8_t input2_val_high =
        vaddq_s16(input2_val_s16_high, input2_offset_vector);
    const int16x8_t input1_val_low =
        vaddq_s16(input1_val_s16_low, input1_offset_vector);
    const int16x8_t input2_val_low =
        vaddq_s16(input2_val_s16_low, input2_offset_vector);
    const int16x4_t input1_val_high_high = vget_high_s16(input1_val_high);
    const int16x4_t input1_val_high_low = vget_low_s16(input1_val_high);
    const int16x4_t input1_val_low_high = vget_high_s16(input1_val_low);
    const int16x4_t input1_val_low_low = vget_low_s16(input1_val_low);
    const int16x4_t input2_val_high_high = vget_high_s16(input2_val_high);
    const int16x4_t input2_val_high_low = vget_low_s16(input2_val_high);
    const int16x4_t input2_val_low_high = vget_high_s16(input2_val_low);
    const int16x4_t input2_val_low_low = vget_low_s16(input2_val_low);

    auto p1 = vmull_s16(input2_val_high_high, input1_val_high_high);
    auto p2 = vmull_s16(input2_val_high_low, input1_val_high_low);
    auto p3 = vmull_s16(input2_val_low_high, input1_val_low_high);
    auto p4 = vmull_s16(input2_val_low_low, input1_val_low_low);

    p1 = vshlq_s32(p1, left_shift_vec);
    p2 = vshlq_s32(p2, left_shift_vec);
    p3 = vshlq_s32(p3, left_shift_vec);
    p4 = vshlq_s32(p4, left_shift_vec);

    p1 = vqrdmulhq_n_s32(p1, params.output_multiplier);
    p2 = vqrdmulhq_n_s32(p2, params.output_multiplier);
    p3 = vqrdmulhq_n_s32(p3, params.output_multiplier);
    p4 = vqrdmulhq_n_s32(p4, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    p1 = RoundingDivideByPOT(p1, right_shift);
    p2 = RoundingDivideByPOT(p2, right_shift);
    p3 = RoundingDivideByPOT(p3, right_shift);
    p4 = RoundingDivideByPOT(p4, right_shift);

    const auto p1_narrowed = vqmovn_s32(p1);
    const auto p2_narrowed = vqmovn_s32(p2);
    const auto p3_narrowed = vqmovn_s32(p3);
    const auto p4_narrowed = vqmovn_s32(p4);

    const int16x8_t p_part1 =
        vaddq_s16(vcombine_s16(p2_narrowed, p1_narrowed), output_offset_vector);
    const int16x8_t p_part2 =
        vaddq_s16(vcombine_s16(p4_narrowed, p3_narrowed), output_offset_vector);
    const int8x16_t p = vcombine_s8(vqmovn_s16(p_part2), vqmovn_s16(p_part1));

    const auto clamped = vmaxq_s8(output_activation_min_vector,
                                  vminq_s8(output_activation_max_vector, p));
    vst1q_s8(output_data + i, clamped);
  }
#endif  // NEON

#ifdef USE_RVV
  // RVV optimized path: LMUL=1 with vsetvl auto-adapts to VLEN.
  // VLEN=128: 16 elements/iter; VLEN=256: 32 elements/iter.
  {
    const int8_t output_activation_min = params.quantized_activation_min;
    const int8_t output_activation_max = params.quantized_activation_max;
    const int left_shift = std::max(0, params.output_shift);
    const int right_shift = std::max(0, -params.output_shift);

    for (; i < size;) {
      const size_t vl = __riscv_vsetvl_e8m1(size - i);

      // Load int8 inputs
      vint8m1_t v_in1_8 = __riscv_vle8_v_i8m1(input1_data + i, vl);
      vint8m1_t v_in2_8 = __riscv_vle8_v_i8m1(input2_data + i, vl);

      // Widen int8 → int16
      vint16m2_t v_in1_16 = __riscv_vsext_vf2_i16m2(v_in1_8, vl);
      vint16m2_t v_in2_16 = __riscv_vsext_vf2_i16m2(v_in2_8, vl);

      // Add input offsets
      v_in1_16 = __riscv_vadd_vx_i16m2(
          v_in1_16, static_cast<int16_t>(params.input1_offset), vl);
      v_in2_16 = __riscv_vadd_vx_i16m2(
          v_in2_16, static_cast<int16_t>(params.input2_offset), vl);

      // Widening multiply: int16 * int16 → int32
      vint32m4_t v_prod = __riscv_vwmul_vv_i32m4(v_in1_16, v_in2_16, vl);

      // Left shift
      v_prod = __riscv_vsll_vx_i32m4(v_prod, left_shift, vl);

      // Multiply by output multiplier: qrdmulh(prod, multiplier)
      vint32m4_t v_out = optimized_rvv::RvvVqrdmulhScalar_i32m4(
          v_prod, params.output_multiplier, vl);

      // Rounding divide by power of 2
      v_out = optimized_rvv::RvvRoundingDivideByPOT_i32m4(
          v_out, right_shift, vl);

      // Add output offset
      v_out = __riscv_vadd_vx_i32m4(v_out, params.output_offset, vl);

      // Clamp to int16 range, narrow int32 → int16
      v_out = __riscv_vmin_vx_i32m4(v_out, 32767, vl);
      v_out = __riscv_vmax_vx_i32m4(v_out, -32768, vl);
      vuint32m4_t v_out_u32 = __riscv_vreinterpret_v_i32m4_u32m4(v_out);
      vint16m2_t v_out_16 = __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vnsrl_wx_u16m2(v_out_u32, 0, vl));

      // Clamp to int8 range
      v_out_16 = __riscv_vmin_vx_i16m2(
          v_out_16, static_cast<int16_t>(output_activation_max), vl);
      v_out_16 = __riscv_vmax_vx_i16m2(
          v_out_16, static_cast<int16_t>(output_activation_min), vl);

      // Narrow int16 → int8 via unsigned clip with truncation rounding
      vuint16m2_t v_out_u16 = __riscv_vreinterpret_v_i16m2_u16m2(v_out_16);
      vint8m1_t v_out_8 = __riscv_vreinterpret_v_u8m1_i8m1(
          __riscv_vnclipu_wx_u8m1(v_out_u16, 0, __RISCV_VXRM_RDN, vl));
      __riscv_vse8_v_i8m1(output_data + i, v_out_8, vl);

      i += vl;
    }
  }
#endif  // USE_RVV

  for (; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<int8>(clamped_output);
  }
}

// Broadcast mul that can often be used for inner loop of broadcast Mul.
inline void MulSimpleBroadcast(int size, const ArithmeticParams& params,
                               const int8 broadcast_value,
                               const int8* input2_data, int8* output_data) {
  ruy::profiler::ScopeLabel label("BroadMulSimpleBroadcastInt8/8bit");
  const int16 input1_val = params.input1_offset + broadcast_value;

  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);
#ifdef USE_NEON
  const auto input2_offset_vector = vdupq_n_s16(params.input2_offset);
  const auto output_offset_vector = vdupq_n_s16(params.output_offset);
  const auto output_activation_min_vector =
      vdupq_n_s8(params.quantized_activation_min);
  const auto output_activation_max_vector =
      vdupq_n_s8(params.quantized_activation_max);
  const int left_shift = std::max(0, params.output_shift);
  const int right_shift = std::max(0, -params.output_shift);
  const int32x4_t left_shift_vec = vdupq_n_s32(left_shift);
  for (; i <= size - 16; i += 16) {
    // We load / store 16 at a time, multiplying as four sets of 4 int32s.
    const auto input2_val_original = vld1q_s8(input2_data + i);
    const auto input2_val_s16_high =
        vmovl_s8(vget_high_s8(input2_val_original));
    const auto input2_val_s16_low = vmovl_s8(vget_low_s8(input2_val_original));

    const auto input2_val_high =
        vaddq_s16(input2_val_s16_high, input2_offset_vector);
    const auto input2_val_low =
        vaddq_s16(input2_val_s16_low, input2_offset_vector);

    const auto input2_val_low_low = vget_low_s16(input2_val_low);
    const auto input2_val_low_high = vget_high_s16(input2_val_low);
    const auto input2_val_high_low = vget_low_s16(input2_val_high);
    const auto input2_val_high_high = vget_high_s16(input2_val_high);

    auto p1 = vmull_n_s16(input2_val_high_high, input1_val);
    auto p2 = vmull_n_s16(input2_val_high_low, input1_val);
    auto p3 = vmull_n_s16(input2_val_low_high, input1_val);
    auto p4 = vmull_n_s16(input2_val_low_low, input1_val);

    p1 = vshlq_s32(p1, left_shift_vec);
    p2 = vshlq_s32(p2, left_shift_vec);
    p3 = vshlq_s32(p3, left_shift_vec);
    p4 = vshlq_s32(p4, left_shift_vec);

    p1 = vqrdmulhq_n_s32(p1, params.output_multiplier);
    p2 = vqrdmulhq_n_s32(p2, params.output_multiplier);
    p3 = vqrdmulhq_n_s32(p3, params.output_multiplier);
    p4 = vqrdmulhq_n_s32(p4, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    p1 = RoundingDivideByPOT(p1, right_shift);
    p2 = RoundingDivideByPOT(p2, right_shift);
    p3 = RoundingDivideByPOT(p3, right_shift);
    p4 = RoundingDivideByPOT(p4, right_shift);

    const auto p1_narrowed = vqmovn_s32(p1);
    const auto p2_narrowed = vqmovn_s32(p2);
    const auto p3_narrowed = vqmovn_s32(p3);
    const auto p4_narrowed = vqmovn_s32(p4);

    const int16x8_t p_part1 =
        vaddq_s16(vcombine_s16(p2_narrowed, p1_narrowed), output_offset_vector);
    const int16x8_t p_part2 =
        vaddq_s16(vcombine_s16(p4_narrowed, p3_narrowed), output_offset_vector);
    const int8x16_t p = vcombine_s8(vqmovn_s16(p_part2), vqmovn_s16(p_part1));

    const auto clamped = vmaxq_s8(output_activation_min_vector,
                                  vminq_s8(output_activation_max_vector, p));
    vst1q_s8(output_data + i, clamped);
  }
#endif  // NEON

#ifdef USE_RVV
  // RVV optimized path: LMUL=1 with vsetvl auto-adapts to VLEN.
  // Pre-computes scalar input1 contribution, then vectorizes input2.
  {
    const int8_t output_activation_min = params.quantized_activation_min;
    const int8_t output_activation_max = params.quantized_activation_max;
    const int left_shift = std::max(0, params.output_shift);
    const int right_shift = std::max(0, -params.output_shift);

    for (; i < size;) {
      const size_t vl = __riscv_vsetvl_e8m1(size - i);

      // Load int8 input2
      vint8m1_t v_in2_8 = __riscv_vle8_v_i8m1(input2_data + i, vl);

      // Widen int8 → int16
      vint16m2_t v_in2_16 = __riscv_vsext_vf2_i16m2(v_in2_8, vl);

      // Add input2 offset
      v_in2_16 = __riscv_vadd_vx_i16m2(
          v_in2_16, static_cast<int16_t>(params.input2_offset), vl);

      // Widening multiply: scalar input1_val * int16 → int32
      vint32m4_t v_prod = __riscv_vwmul_vx_i32m4(v_in2_16, input1_val, vl);

      // Left shift
      v_prod = __riscv_vsll_vx_i32m4(v_prod, left_shift, vl);

      // Multiply by output multiplier: qrdmulh(prod, multiplier)
      vint32m4_t v_out = optimized_rvv::RvvVqrdmulhScalar_i32m4(
          v_prod, params.output_multiplier, vl);

      // Rounding divide by power of 2
      v_out = optimized_rvv::RvvRoundingDivideByPOT_i32m4(
          v_out, right_shift, vl);

      // Add output offset
      v_out = __riscv_vadd_vx_i32m4(v_out, params.output_offset, vl);

      // Clamp to int16 range, narrow int32 → int16
      v_out = __riscv_vmin_vx_i32m4(v_out, 32767, vl);
      v_out = __riscv_vmax_vx_i32m4(v_out, -32768, vl);
      vuint32m4_t v_out_u32 = __riscv_vreinterpret_v_i32m4_u32m4(v_out);
      vint16m2_t v_out_16 = __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vnsrl_wx_u16m2(v_out_u32, 0, vl));

      // Clamp to int8 range
      v_out_16 = __riscv_vmin_vx_i16m2(
          v_out_16, static_cast<int16_t>(output_activation_max), vl);
      v_out_16 = __riscv_vmax_vx_i16m2(
          v_out_16, static_cast<int16_t>(output_activation_min), vl);

      // Narrow int16 → int8 via unsigned clip with truncation rounding
      vuint16m2_t v_out_u16 = __riscv_vreinterpret_v_i16m2_u16m2(v_out_16);
      vint8m1_t v_out_8 = __riscv_vreinterpret_v_u8m1_i8m1(
          __riscv_vnclipu_wx_u8m1(v_out_u16, 0, __RISCV_VXRM_RDN, vl));
      __riscv_vse8_v_i8m1(output_data + i, v_out_8, vl);

      i += vl;
    }
  }
#endif  // USE_RVV

  for (; i < size; ++i) {
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<int8>(clamped_output);
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8* input1_data,
                const RuntimeShape& input2_shape, const int8* input2_data,
                const RuntimeShape& output_shape, int8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  ruy::profiler::ScopeLabel label("MulInt8/8bit");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  MulElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void BroadcastMulDispatch(const ArithmeticParams& params,
                                 const RuntimeShape& input1_shape,
                                 const int8* input1_data,
                                 const RuntimeShape& input2_shape,
                                 const int8* input2_data,
                                 const RuntimeShape& output_shape,
                                 int8* output_data) {
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast) {
    return reference_integer_ops::BroadcastMul6DSlow(
        params, input1_shape, input1_data, input2_shape, input2_data,
        output_shape, output_data);
  }

  optimized_ops::BinaryBroadcastFiveFold(
      params, input1_shape, input1_data, input2_shape, input2_data,
      output_shape, output_data, MulElementwise, MulSimpleBroadcast);
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MUL_H_
