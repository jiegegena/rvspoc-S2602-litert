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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_UINT8_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_UINT8_H_

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tflite {
namespace optimized_rvv {

// UINT8 per-tensor GEMM: dst = requantize(lhs * rhs + bias)
//
// Layout (matching optimized_ops::Conv UINT8):
//   lhs: n x k row-major (uint8)
//   rhs: k x m col-major (uint8)
//   dst: n x m col-major (uint8)
//   bias: optional, length n (int32)
//
// Uses widening multiply-accumulate with zero-point handling.
inline void RvvGemmUint8Uniform(
    int m, int n, int k, const uint8_t* lhs_data, int lhs_zp,
    const uint8_t* rhs_data, int rhs_zp, int dst_zp,
    const int32_t* bias_data, int32_t multiplier, int shift,
    int32_t clamp_min, int32_t clamp_max, uint8_t* dst_data) {
  for (int m_idx = 0; m_idx < m; ++m_idx) {
    const uint8_t* rhs_col = rhs_data + static_cast<size_t>(m_idx) * k;
    uint8_t* dst_col = dst_data + static_cast<size_t>(m_idx) * n;

    for (int n_idx = 0; n_idx < n; ++n_idx) {
      const uint8_t* lhs_row = lhs_data + static_cast<size_t>(n_idx) * k;

      // Vectorized dot-product over k dimension
      size_t remaining = static_cast<size_t>(k);
      const size_t vl0 = __riscv_vsetvl_e32m4(remaining);
      vint32m4_t v_acc = __riscv_vmv_v_x_i32m4(0, vl0);

      const uint8_t* p_lhs = lhs_row;
      const uint8_t* p_rhs = rhs_col;
      while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e8m4(remaining);
        vuint8m4_t v_lhs_u8 = __riscv_vle8_v_u8m4(p_lhs, vl);
        vuint8m4_t v_rhs_u8 = __riscv_vle8_v_u8m4(p_rhs, vl);

        // Widen u8 to u16, then reinterpret as s16 for signed arithmetic
        vuint16m8_t v_lhs_u16 = __riscv_vzext_vf2_u16m8(v_lhs_u8, vl);
        vuint16m8_t v_rhs_u16 = __riscv_vzext_vf2_u16m8(v_rhs_u8, vl);
        vint16m8_t v_lhs_s16 = __riscv_vreinterpret_v_u16m8_i16m8(v_lhs_u16);
        vint16m8_t v_rhs_s16 = __riscv_vreinterpret_v_u16m8_i16m8(v_rhs_u16);

        // Add zero points
        v_lhs_s16 = __riscv_vadd_vx_i16m8(v_lhs_s16, lhs_zp, vl);
        v_rhs_s16 = __riscv_vadd_vx_i16m8(v_rhs_s16, rhs_zp, vl);

        // Widening MAC: s16 * s16 -> s32
        vint16m2_t lhs_lo = __riscv_vlmul_trunc_v_i16m8_i16m2(v_lhs_s16);
        vint16m2_t rhs_lo = __riscv_vlmul_trunc_v_i16m8_i16m2(v_rhs_s16);
        v_acc = __riscv_vwmacc_vv_i32m4(v_acc, lhs_lo, rhs_lo, vl);

        p_lhs += vl;
        p_rhs += vl;
        remaining -= vl;
      }

      // Horizontal reduction
      vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvl_e32m1(1));
      vint32m1_t v_red =
          __riscv_vredsum_vs_i32m4_i32m1(v_acc, v_zero, vl0);
      int32_t acc = __riscv_vmv_x_s_i32m1_i32(v_red);

      if (bias_data != nullptr) {
        acc += bias_data[n_idx];
      }

      // Uniform quantization
      acc = (acc * multiplier) >> shift;

      // Add destination zero point and clamp
      acc += dst_zp;
      acc = std::max(acc, clamp_min);
      acc = std::min(acc, clamp_max);
      dst_col[n_idx] = static_cast<uint8_t>(acc);
    }
  }
}

}  // namespace optimized_rvv
}  // namespace tflite

#endif  // __riscv_vector

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_UINT8_H_
