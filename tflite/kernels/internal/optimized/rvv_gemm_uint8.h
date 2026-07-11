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
// All vector operations use m1 grouping to ensure consistent element
// counts across load, accumulate, and reduce steps.
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

      // Vectorized dot-product over k dimension.
      // Use m1 grouping for all operations so vl is consistent:
      //   e8m1 -> e16m2 -> e32m4 widening chain, all with the same vl.
      size_t remaining = static_cast<size_t>(k);
      const size_t vl0 = __riscv_vsetvl_e32m4(remaining);
      vint32m4_t v_acc = __riscv_vmv_v_x_i32m4(0, vl0);

      const uint8_t* p_lhs = lhs_row;
      const uint8_t* p_rhs = rhs_col;
      while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t v_lhs_u8 = __riscv_vle8_v_u8m1(p_lhs, vl);
        vuint8m1_t v_rhs_u8 = __riscv_vle8_v_u8m1(p_rhs, vl);

        // Widen u8 to u16, then reinterpret as s16 for signed arithmetic
        vuint16m2_t v_lhs_u16 = __riscv_vzext_vf2_u16m2(v_lhs_u8, vl);
        vuint16m2_t v_rhs_u16 = __riscv_vzext_vf2_u16m2(v_rhs_u8, vl);
        vint16m2_t v_lhs_s16 = __riscv_vreinterpret_v_u16m2_i16m2(v_lhs_u16);
        vint16m2_t v_rhs_s16 = __riscv_vreinterpret_v_u16m2_i16m2(v_rhs_u16);

        // Subtract zero points: effective value = val - zp
        v_lhs_s16 = __riscv_vsub_vx_i16m2(v_lhs_s16, lhs_zp, vl);
        v_rhs_s16 = __riscv_vsub_vx_i16m2(v_rhs_s16, rhs_zp, vl);

        // Widening MAC: s16 * s16 -> s32, accumulating into v_acc (m4)
        // Source m2 (16 elements) -> dest m4 (16 elements), vl matches.
        v_acc = __riscv_vwmacc_vv_i32m4(v_acc, v_lhs_s16, v_rhs_s16, vl);

        p_lhs += vl;
        p_rhs += vl;
        remaining -= vl;
      }

      // Horizontal reduction using vl0 (matches accumulator element count)
      vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvl_e32m1(1));
      vint32m1_t v_red =
          __riscv_vredsum_vs_i32m4_i32m1(v_acc, v_zero, vl0);
      int32_t acc = __riscv_vmv_x_s_i32m1_i32(v_red);

      if (bias_data != nullptr) {
        acc += bias_data[n_idx];
      }

      // Requantization using single-rounding formula (matching ruy's
      // ApplyMultiplier exactly). Uses 64-bit arithmetic to avoid
      // int32 overflow.
      {
        int total_shift = 31 - shift;
        int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
        int64_t result =
            static_cast<int64_t>(acc) * static_cast<int64_t>(multiplier) + round;
        acc = static_cast<int32_t>(result >> total_shift);
      }

      // Add destination zero point and clamp
      acc += dst_zp;
      acc = std::max(acc, clamp_min);
      acc = std::min(acc, clamp_max);
      dst_col[n_idx] = static_cast<uint8_t>(acc);
    }
  }
}

// UINT8 per-tensor GEMM with N-dimension tiling and loop reordering.
// N_TILE=4: 4*4(acc) + 1(rhs_m1) + 1(lhs_m1) = 18 regs.
inline void RvvGemmUint8UniformTiled(
    int m, int n, int k, const uint8_t* lhs_data, int lhs_zp,
    const uint8_t* rhs_data, int rhs_zp, int dst_zp,
    const int32_t* bias_data, int32_t multiplier, int shift,
    int32_t clamp_min, int32_t clamp_max, uint8_t* dst_data) {
  constexpr int N_TILE = 4;
  const bool has_bias = (bias_data != nullptr);

  // Process full tiles of N_TILE output channels
  int n_block;
  for (n_block = 0; n_block <= n - N_TILE; n_block += N_TILE) {
    for (int m_idx = 0; m_idx < m; ++m_idx) {
      const uint8_t* rhs_col =
          rhs_data + static_cast<size_t>(m_idx) * k;
      uint8_t* dst_col =
          dst_data + static_cast<size_t>(m_idx) * n;

      // Initialize N_TILE accumulators (N_TILE=4, unrolled)
      const size_t k_size = static_cast<size_t>(k);
      size_t vl0 = __riscv_vsetvl_e32m4(k_size);
      vint32m4_t v_zero = __riscv_vmv_v_x_i32m4(0, vl0);
      vint32m4_t acc0 = v_zero;
      vint32m4_t acc1 = v_zero;
      vint32m4_t acc2 = v_zero;
      vint32m4_t acc3 = v_zero;

      size_t k_off = 0;
      while (k_off < k_size) {
        size_t vl = __riscv_vsetvl_e8m1(k_size - k_off);
        vuint8m1_t v_rhs_u8 =
            __riscv_vle8_v_u8m1(rhs_col + k_off, vl);
        vuint16m2_t v_rhs_u16 =
            __riscv_vzext_vf2_u16m2(v_rhs_u8, vl);
        vint16m2_t v_rhs_s16 =
            __riscv_vreinterpret_v_u16m2_i16m2(v_rhs_u16);
        v_rhs_s16 = __riscv_vsub_vx_i16m2(v_rhs_s16, rhs_zp, vl);

        // Channel 0
        {
          const uint8_t* lhs_row = lhs_data + static_cast<size_t>(n_block) * k;
          vuint8m1_t v_lhs = __riscv_vle8_v_u8m1(lhs_row + k_off, vl);
          vuint16m2_t v_lhs_u16 = __riscv_vzext_vf2_u16m2(v_lhs, vl);
          vint16m2_t v_lhs_s16 =
              __riscv_vreinterpret_v_u16m2_i16m2(v_lhs_u16);
          v_lhs_s16 = __riscv_vsub_vx_i16m2(v_lhs_s16, lhs_zp, vl);
          acc0 = __riscv_vwmacc_vv_i32m4(acc0, v_lhs_s16, v_rhs_s16, vl);
        }
        // Channel 1
        {
          const uint8_t* lhs_row = lhs_data + static_cast<size_t>(n_block + 1) * k;
          vuint8m1_t v_lhs = __riscv_vle8_v_u8m1(lhs_row + k_off, vl);
          vuint16m2_t v_lhs_u16 = __riscv_vzext_vf2_u16m2(v_lhs, vl);
          vint16m2_t v_lhs_s16 =
              __riscv_vreinterpret_v_u16m2_i16m2(v_lhs_u16);
          v_lhs_s16 = __riscv_vsub_vx_i16m2(v_lhs_s16, lhs_zp, vl);
          acc1 = __riscv_vwmacc_vv_i32m4(acc1, v_lhs_s16, v_rhs_s16, vl);
        }
        // Channel 2
        {
          const uint8_t* lhs_row = lhs_data + static_cast<size_t>(n_block + 2) * k;
          vuint8m1_t v_lhs = __riscv_vle8_v_u8m1(lhs_row + k_off, vl);
          vuint16m2_t v_lhs_u16 = __riscv_vzext_vf2_u16m2(v_lhs, vl);
          vint16m2_t v_lhs_s16 =
              __riscv_vreinterpret_v_u16m2_i16m2(v_lhs_u16);
          v_lhs_s16 = __riscv_vsub_vx_i16m2(v_lhs_s16, lhs_zp, vl);
          acc2 = __riscv_vwmacc_vv_i32m4(acc2, v_lhs_s16, v_rhs_s16, vl);
        }
        // Channel 3
        {
          const uint8_t* lhs_row = lhs_data + static_cast<size_t>(n_block + 3) * k;
          vuint8m1_t v_lhs = __riscv_vle8_v_u8m1(lhs_row + k_off, vl);
          vuint16m2_t v_lhs_u16 = __riscv_vzext_vf2_u16m2(v_lhs, vl);
          vint16m2_t v_lhs_s16 =
              __riscv_vreinterpret_v_u16m2_i16m2(v_lhs_u16);
          v_lhs_s16 = __riscv_vsub_vx_i16m2(v_lhs_s16, lhs_zp, vl);
          acc3 = __riscv_vwmacc_vv_i32m4(acc3, v_lhs_s16, v_rhs_s16, vl);
        }
        k_off += vl;
      }

      // Reduction, requantize, clamp, store for each tile element
      vint32m1_t v_szero =
          __riscv_vmv_v_x_i32m1(0, __riscv_vsetvl_e32m1(1));
      {
        vint32m1_t v_red =
            __riscv_vredsum_vs_i32m4_i32m1(acc0, v_szero, vl0);
        int32_t val = __riscv_vmv_x_s_i32m1_i32(v_red);
        if (has_bias) val += bias_data[n_block];
        int total_shift = 31 - shift;
        int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
        int64_t result = static_cast<int64_t>(val) * multiplier + round;
        val = static_cast<int32_t>(result >> total_shift);
        val += dst_zp;
        val = std::max(val, clamp_min);
        val = std::min(val, clamp_max);
        dst_col[n_block] = static_cast<uint8_t>(val);
      }
      {
        vint32m1_t v_red =
            __riscv_vredsum_vs_i32m4_i32m1(acc1, v_szero, vl0);
        int32_t val = __riscv_vmv_x_s_i32m1_i32(v_red);
        if (has_bias) val += bias_data[n_block + 1];
        int total_shift = 31 - shift;
        int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
        int64_t result = static_cast<int64_t>(val) * multiplier + round;
        val = static_cast<int32_t>(result >> total_shift);
        val += dst_zp;
        val = std::max(val, clamp_min);
        val = std::min(val, clamp_max);
        dst_col[n_block + 1] = static_cast<uint8_t>(val);
      }
      {
        vint32m1_t v_red =
            __riscv_vredsum_vs_i32m4_i32m1(acc2, v_szero, vl0);
        int32_t val = __riscv_vmv_x_s_i32m1_i32(v_red);
        if (has_bias) val += bias_data[n_block + 2];
        int total_shift = 31 - shift;
        int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
        int64_t result = static_cast<int64_t>(val) * multiplier + round;
        val = static_cast<int32_t>(result >> total_shift);
        val += dst_zp;
        val = std::max(val, clamp_min);
        val = std::min(val, clamp_max);
        dst_col[n_block + 2] = static_cast<uint8_t>(val);
      }
      {
        vint32m1_t v_red =
            __riscv_vredsum_vs_i32m4_i32m1(acc3, v_szero, vl0);
        int32_t val = __riscv_vmv_x_s_i32m1_i32(v_red);
        if (has_bias) val += bias_data[n_block + 3];
        int total_shift = 31 - shift;
        int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
        int64_t result = static_cast<int64_t>(val) * multiplier + round;
        val = static_cast<int32_t>(result >> total_shift);
        val += dst_zp;
        val = std::max(val, clamp_min);
        val = std::min(val, clamp_max);
        dst_col[n_block + 3] = static_cast<uint8_t>(val);
      }
    }
  }

  // Handle remaining output channels (< N_TILE)
  for (; n_block < n; ++n_block) {
    const uint8_t* lhs_row =
        lhs_data + static_cast<size_t>(n_block) * k;
    for (int m_idx = 0; m_idx < m; ++m_idx) {
      const uint8_t* rhs_col =
          rhs_data + static_cast<size_t>(m_idx) * k;
      uint8_t* dst_col =
          dst_data + static_cast<size_t>(m_idx) * n;

      size_t remaining = static_cast<size_t>(k);
      size_t vl0 = __riscv_vsetvl_e32m4(remaining);
      vint32m4_t v_acc = __riscv_vmv_v_x_i32m4(0, vl0);

      size_t k_off = 0;
      while (k_off < static_cast<size_t>(k)) {
        size_t vl = __riscv_vsetvl_e8m1(
            static_cast<size_t>(k) - k_off);
        vuint8m1_t v_lhs_u8 =
            __riscv_vle8_v_u8m1(lhs_row + k_off, vl);
        vuint8m1_t v_rhs_u8 =
            __riscv_vle8_v_u8m1(rhs_col + k_off, vl);

        vuint16m2_t v_lhs_u16 =
            __riscv_vzext_vf2_u16m2(v_lhs_u8, vl);
        vuint16m2_t v_rhs_u16 =
            __riscv_vzext_vf2_u16m2(v_rhs_u8, vl);
        vint16m2_t v_lhs_s16 =
            __riscv_vreinterpret_v_u16m2_i16m2(v_lhs_u16);
        vint16m2_t v_rhs_s16 =
            __riscv_vreinterpret_v_u16m2_i16m2(v_rhs_u16);

        v_lhs_s16 = __riscv_vsub_vx_i16m2(v_lhs_s16, lhs_zp, vl);
        v_rhs_s16 = __riscv_vsub_vx_i16m2(v_rhs_s16, rhs_zp, vl);

        v_acc = __riscv_vwmacc_vv_i32m4(v_acc, v_lhs_s16,
                                        v_rhs_s16, vl);
        k_off += vl;
      }

      vint32m1_t v_szero =
          __riscv_vmv_v_x_i32m1(0, __riscv_vsetvl_e32m1(1));
      vint32m1_t v_red =
          __riscv_vredsum_vs_i32m4_i32m1(v_acc, v_szero, vl0);
      int32_t val = __riscv_vmv_x_s_i32m1_i32(v_red);

      if (has_bias) val += bias_data[n_block];

      int total_shift = 31 - shift;
      int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
      int64_t result =
          static_cast<int64_t>(val) *
              static_cast<int64_t>(multiplier) +
          round;
      val = static_cast<int32_t>(result >> total_shift);

      val += dst_zp;
      val = std::max(val, clamp_min);
      val = std::min(val, clamp_max);
      dst_col[n_block] = static_cast<uint8_t>(val);
    }
  }
}

}  // namespace optimized_rvv
}  // namespace tflite

#endif  // __riscv_vector

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_UINT8_H_
