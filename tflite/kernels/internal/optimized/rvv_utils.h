/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_UTILS_H_

#include "tflite/kernels/internal/optimized/rvv_check.h"

#ifdef USE_RVV

#include <algorithm>
#include <cstdint>

namespace tflite {
namespace optimized_rvv { 

// Equivalent to NEON vqrdmulhq_n_s32 for int32m4:
// vqrdmulhq_n_s32(a, b) = (a * b * 2 + 2^31) >> 32 = (a * b + 2^30) >> 31
// This also matches gemmlowp::SaturatingRoundingDoublingHighMul for positive products.
inline vint32m4_t RvvVqrdmulhScalar_i32m4(vint32m4_t a, int32_t b,
                                           size_t vl) {
  vint64m8_t prod = __riscv_vwmul_vx_i64m8(a, b, vl);
  prod = __riscv_vadd_vx_i64m8(prod, static_cast<int64_t>(1) << 30, vl);
  return __riscv_vnsra_wx_i32m4(prod, 31, vl);
}

// Equivalent to gemmlowp::RoundingDivideByPOT(x, exponent) for int32m4.
// Matches: result = (x >> exponent) + ((x >> (exponent-1)) & 1)
inline vint32m4_t RvvRoundingDivideByPOT_i32m4(vint32m4_t x, int exponent,
                                                size_t vl) {
  if (exponent > 0) {
    vint32m4_t rounding = __riscv_vsra_vx_i32m4(x, exponent - 1, vl);
    rounding = __riscv_vand_vx_i32m4(rounding, 1, vl);
    vint32m4_t result = __riscv_vsra_vx_i32m4(x, exponent, vl);
    return __riscv_vadd_vv_i32m4(result, rounding, vl);
  }
  return x;
}

// Equivalent to NEON MultiplyByQuantizedMultiplier4Rows for int32m4.
// Applies: left_shift → qrdmulh(multiplier) → rounding_right_shift
inline vint32m4_t RvvMultiplyByQuantizedMultiplier_i32m4(
    vint32m4_t x, int32_t multiplier, int shift, size_t vl) {
  const int left_shift = std::max(shift, 0);
  const int right_shift = std::min(shift, 0);
  x = __riscv_vsll_vx_i32m4(x, left_shift, vl);
  x = RvvVqrdmulhScalar_i32m4(x, multiplier, vl);
  x = RvvRoundingDivideByPOT_i32m4(x, -right_shift, vl);
  return x;
}

// Per-channel version: vector multiplier and vector shift.
// shift[i] = left_shift or negative right_shift.
// left_shift = max(shift, 0), right_shift = min(shift, 0).
inline vint32m4_t RvvMultiplyByQuantizedMultiplierPerChannel_i32m4(
    vint32m4_t x, vint32m4_t multiplier, vint32m4_t shift, size_t vl) {
  // left_shift = max(shift, 0)
  vint32m4_t vzero = __riscv_vmv_v_x_i32m4(0, vl);
  vint32m4_t left_shift = __riscv_vmax_vv_i32m4(shift, vzero, vl);
  // right_shift = min(shift, 0)
  vint32m4_t right_shift = __riscv_vmin_vv_i32m4(shift, vzero, vl);
  // x << left_shift
  x = __riscv_vsll_vv_i32m4(x, __riscv_vreinterpret_v_i32m4_u32m4(left_shift), vl);
  // qrdmulh(x, multiplier): (x * multiplier + 2^30) >> 31
  vint64m8_t prod = __riscv_vwmul_vv_i64m8(x, multiplier, vl);
  prod = __riscv_vadd_vx_i64m8(prod, static_cast<int64_t>(1) << 30, vl);
  x = __riscv_vnsra_wx_i32m4(prod, 31, vl);
  // RoundingDivideByPOT(x, -right_shift)
  // -right_shift is non-negative
  vint32m4_t neg_right_shift = __riscv_vneg_v_i32m4(right_shift, vl);
  // rounding = (x >> (exp-1)) & 1
  vint32m4_t exp_minus1 = __riscv_vsub_vx_i32m4(neg_right_shift, 1, vl);
  vint32m4_t rounding = __riscv_vsra_vv_i32m4(x, __riscv_vreinterpret_v_i32m4_u32m4(exp_minus1), vl);
  rounding = __riscv_vand_vx_i32m4(rounding, 1, vl);
  // result = (x >> exp) + rounding
  vint32m4_t result = __riscv_vsra_vv_i32m4(x, __riscv_vreinterpret_v_i32m4_u32m4(neg_right_shift), vl);
  result = __riscv_vadd_vv_i32m4(result, rounding, vl);
  return result;
}

}  // namespace optimized_rvv
}  // namespace tflite

#endif  // USE_RVV

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_UTILS_H_
