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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_INT8_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_INT8_H_

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tflite {
namespace optimized_rvv {

// INT8 per-channel GEMM: dst = requantize(lhs * rhs + bias)
//
// Layout (matching optimized_integer_ops::ConvPerChannel):
//   lhs: n x k row-major (int8, zero_point=0)
//   rhs: k x m col-major (int8)
//   dst: n x m col-major (int8)
//   bias: optional, length n (int32)
//   multiplier_per_channel: per-row quantization multiplier (int32)
//   shift_per_channel: per-row quantization shift (int)
//
// Uses widening multiply-accumulate: s8 * s8 -> s32.
// All vector operations use m1 grouping to ensure consistent element
// counts across load, accumulate, and reduce steps.
inline void RvvGemmInt8PerChannel(
    int m, int n, int k, const int8_t* lhs_data, const int8_t* rhs_data,
    int rhs_zp, int dst_zp, const int32_t* bias_data,
    const int32_t* multiplier_per_channel, const int* shift_per_channel,
    int32_t clamp_min, int32_t clamp_max, int8_t* dst_data) {
  for (int m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t* rhs_col = rhs_data + static_cast<size_t>(m_idx) * k;
    int8_t* dst_col = dst_data + static_cast<size_t>(m_idx) * n;

    for (int n_idx = 0; n_idx < n; ++n_idx) {
      const int8_t* lhs_row = lhs_data + static_cast<size_t>(n_idx) * k;

      // Vectorized dot-product over k dimension using widening MAC.
      // Use m1 grouping for all operations so vl is consistent:
      //   e8m1 -> e16m2 -> e32m4 widening chain, all with the same vl.
      size_t remaining = static_cast<size_t>(k);
      const size_t vl0 = __riscv_vsetvl_e32m4(remaining);
      vint32m4_t v_acc = __riscv_vmv_v_x_i32m4(0, vl0);

      const int8_t* p_lhs = lhs_row;
      const int8_t* p_rhs = rhs_col;
      while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e8m1(remaining);
        vint8m1_t v_lhs = __riscv_vle8_v_i8m1(p_lhs, vl);
        vint8m1_t v_rhs = __riscv_vle8_v_i8m1(p_rhs, vl);

        // Widen s8 to s16
        vint16m2_t v_lhs16 = __riscv_vsext_vf2_i16m2(v_lhs, vl);
        vint16m2_t v_rhs16 = __riscv_vsext_vf2_i16m2(v_rhs, vl);

        // Subtract rhs zero point from rhs: effective = val - zp
        v_rhs16 = __riscv_vsub_vx_i16m2(v_rhs16, rhs_zp, vl);

        // Widening MAC: s16 * s16 -> s32, accumulating into v_acc (m4)
        v_acc = __riscv_vwmacc_vv_i32m4(v_acc, v_lhs16, v_rhs16, vl);

        p_lhs += vl;
        p_rhs += vl;
        remaining -= vl;
      }

      // Horizontal reduction using vl0 (matches accumulator element count)
      vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvl_e32m1(1));
      vint32m1_t v_red =
          __riscv_vredsum_vs_i32m4_i32m1(v_acc, v_zero, vl0);
      int32_t acc = __riscv_vmv_x_s_i32m1_i32(v_red);

      // Add bias
      if (bias_data != nullptr) {
        acc += bias_data[n_idx];
      }

      // Per-channel requantization (single-rounding, matching ruy)
      {
        int total_shift = 31 - shift_per_channel[n_idx];
        int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
        int64_t result = static_cast<int64_t>(acc) *
                             static_cast<int64_t>(multiplier_per_channel[n_idx]) +
                         round;
        acc = static_cast<int32_t>(result >> total_shift);
      }

      // Add destination zero point and clamp
      acc += dst_zp;
      acc = std::max(acc, clamp_min);
      acc = std::min(acc, clamp_max);
      dst_col[n_idx] = static_cast<int8_t>(acc);
    }
  }
}

// INT8 per-tensor GEMM with uniform multiplier.
inline void RvvGemmInt8Uniform(
    int m, int n, int k, const int8_t* lhs_data, const int8_t* rhs_data,
    int lhs_zp, int rhs_zp, int dst_zp, const int32_t* bias_data,
    int32_t multiplier, int shift, int32_t clamp_min, int32_t clamp_max,
    int8_t* dst_data) {
  for (int m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t* rhs_col = rhs_data + static_cast<size_t>(m_idx) * k;
    int8_t* dst_col = dst_data + static_cast<size_t>(m_idx) * n;

    for (int n_idx = 0; n_idx < n; ++n_idx) {
      const int8_t* lhs_row = lhs_data + static_cast<size_t>(n_idx) * k;

      size_t remaining = static_cast<size_t>(k);
      const size_t vl0 = __riscv_vsetvl_e32m4(remaining);
      vint32m4_t v_acc = __riscv_vmv_v_x_i32m4(0, vl0);

      const int8_t* p_lhs = lhs_row;
      const int8_t* p_rhs = rhs_col;
      while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e8m1(remaining);
        vint8m1_t v_lhs = __riscv_vle8_v_i8m1(p_lhs, vl);
        vint8m1_t v_rhs = __riscv_vle8_v_i8m1(p_rhs, vl);

        // Subtract zero points: effective value = val - zp
        vint16m2_t v_lhs16 = __riscv_vsext_vf2_i16m2(v_lhs, vl);
        vint16m2_t v_rhs16 = __riscv_vsext_vf2_i16m2(v_rhs, vl);
        v_lhs16 = __riscv_vsub_vx_i16m2(v_lhs16, lhs_zp, vl);
        v_rhs16 = __riscv_vsub_vx_i16m2(v_rhs16, rhs_zp, vl);

        // Widening MAC: s16 * s16 -> s32
        v_acc = __riscv_vwmacc_vv_i32m4(v_acc, v_lhs16, v_rhs16, vl);

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

      // Uniform requantization (single-rounding, matching ruy)
      {
        int total_shift = 31 - shift;
        int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
        int64_t result =
            static_cast<int64_t>(acc) * static_cast<int64_t>(multiplier) + round;
        acc = static_cast<int32_t>(result >> total_shift);
      }

      acc += dst_zp;
      acc = std::max(acc, clamp_min);
      acc = std::min(acc, clamp_max);
      dst_col[n_idx] = static_cast<int8_t>(acc);
    }
  }
}

// INT8 per-channel GEMM with N-dimension tiling and loop reordering.
// Key optimizations vs the simple RvvGemmInt8PerChannel:
//   1. N-tiling: process N_TILE output channels together, sharing rhs loads
//   2. Loop reorder: N outer, M inner — filter rows loaded once
//   3. LMUL=1 loads, m4 accumulator: e8m1->e16m2->e32m4 chain
//   4. N_TILE=4: 4*4(acc) + 1(rhs_m1) + 1(lhs_m1) = 18 regs
inline void RvvGemmInt8PerChannelTiled(
    int m, int n, int k, const int8_t* lhs_data, const int8_t* rhs_data,
    int rhs_zp, int dst_zp, const int32_t* bias_data,
    const int32_t* multiplier_per_channel, const int* shift_per_channel,
    int32_t clamp_min, int32_t clamp_max, int8_t* dst_data) {
  constexpr int N_TILE = 4;
  const bool has_bias = (bias_data != nullptr);

  // Process full tiles of N_TILE output channels
  int n_block;
  for (n_block = 0; n_block <= n - N_TILE; n_block += N_TILE) {
    for (int m_idx = 0; m_idx < m; ++m_idx) {
      const int8_t* rhs_col =
          rhs_data + static_cast<size_t>(m_idx) * k;
      int8_t* dst_col =
          dst_data + static_cast<size_t>(m_idx) * n;

      // Initialize N_TILE=4 accumulators (explicitly unrolled)
      const size_t k_size = static_cast<size_t>(k);
      size_t vl0 = __riscv_vsetvl_e32m4(k_size);
      vint32m4_t v_zero = __riscv_vmv_v_x_i32m4(0, vl0);
      vint32m4_t acc0 = v_zero, acc1 = v_zero, acc2 = v_zero, acc3 = v_zero;

      size_t k_off = 0;
      while (k_off < k_size) {
        size_t vl = __riscv_vsetvl_e8m1(k_size - k_off);
        vint8m1_t v_rhs = __riscv_vle8_v_i8m1(rhs_col + k_off, vl);
        vint16m2_t v_rhs16 = __riscv_vsext_vf2_i16m2(v_rhs, vl);
        v_rhs16 = __riscv_vsub_vx_i16m2(v_rhs16, rhs_zp, vl);

        // Channel 0
        { const int8_t* r = lhs_data + static_cast<size_t>(n_block) * k;
          vint8m1_t vl8 = __riscv_vle8_v_i8m1(r + k_off, vl);
          vint16m2_t vl16 = __riscv_vsext_vf2_i16m2(vl8, vl);
          acc0 = __riscv_vwmacc_vv_i32m4(acc0, vl16, v_rhs16, vl); }
        // Channel 1
        { const int8_t* r = lhs_data + static_cast<size_t>(n_block + 1) * k;
          vint8m1_t vl8 = __riscv_vle8_v_i8m1(r + k_off, vl);
          vint16m2_t vl16 = __riscv_vsext_vf2_i16m2(vl8, vl);
          acc1 = __riscv_vwmacc_vv_i32m4(acc1, vl16, v_rhs16, vl); }
        // Channel 2
        { const int8_t* r = lhs_data + static_cast<size_t>(n_block + 2) * k;
          vint8m1_t vl8 = __riscv_vle8_v_i8m1(r + k_off, vl);
          vint16m2_t vl16 = __riscv_vsext_vf2_i16m2(vl8, vl);
          acc2 = __riscv_vwmacc_vv_i32m4(acc2, vl16, v_rhs16, vl); }
        // Channel 3
        { const int8_t* r = lhs_data + static_cast<size_t>(n_block + 3) * k;
          vint8m1_t vl8 = __riscv_vle8_v_i8m1(r + k_off, vl);
          vint16m2_t vl16 = __riscv_vsext_vf2_i16m2(vl8, vl);
          acc3 = __riscv_vwmacc_vv_i32m4(acc3, vl16, v_rhs16, vl); }
        k_off += vl;
      }

      // Reduction, requantize, clamp, store (unrolled)
      vint32m1_t v_szero =
          __riscv_vmv_v_x_i32m1(0, __riscv_vsetvl_e32m1(1));
      auto reduce_and_store = [&](vint32m4_t acc, int idx) {
        vint32m1_t r = __riscv_vredsum_vs_i32m4_i32m1(acc, v_szero, vl0);
        int32_t v = __riscv_vmv_x_s_i32m1_i32(r);
        if (has_bias) v += bias_data[idx];
        int ts = 31 - shift_per_channel[idx];
        int64_t rnd = static_cast<int64_t>(1) << (ts - 1);
        int64_t res = static_cast<int64_t>(v) *
                          static_cast<int64_t>(multiplier_per_channel[idx]) +
                      rnd;
        v = static_cast<int32_t>(res >> ts);
        v += dst_zp;
        v = std::max(v, clamp_min);
        v = std::min(v, clamp_max);
        dst_col[idx] = static_cast<int8_t>(v);
      };
      reduce_and_store(acc0, n_block);
      reduce_and_store(acc1, n_block + 1);
      reduce_and_store(acc2, n_block + 2);
      reduce_and_store(acc3, n_block + 3);
    }
  }

  // Handle remaining output channels (< N_TILE)
  for (; n_block < n; ++n_block) {
    const int8_t* lhs_row =
        lhs_data + static_cast<size_t>(n_block) * k;
    for (int m_idx = 0; m_idx < m; ++m_idx) {
      const int8_t* rhs_col =
          rhs_data + static_cast<size_t>(m_idx) * k;
      int8_t* dst_col =
          dst_data + static_cast<size_t>(m_idx) * n;

      size_t remaining = static_cast<size_t>(k);
      size_t vl0 = __riscv_vsetvl_e32m4(remaining);
      vint32m4_t v_acc = __riscv_vmv_v_x_i32m4(0, vl0);

      size_t k_off = 0;
      while (k_off < static_cast<size_t>(k)) {
        size_t vl = __riscv_vsetvl_e8m1(
            static_cast<size_t>(k) - k_off);
        vint8m1_t v_lhs =
            __riscv_vle8_v_i8m1(lhs_row + k_off, vl);
        vint8m1_t v_rhs =
            __riscv_vle8_v_i8m1(rhs_col + k_off, vl);

        vint16m2_t v_lhs16 =
            __riscv_vsext_vf2_i16m2(v_lhs, vl);
        vint16m2_t v_rhs16 =
            __riscv_vsext_vf2_i16m2(v_rhs, vl);
        v_rhs16 = __riscv_vsub_vx_i16m2(v_rhs16, rhs_zp, vl);

        v_acc = __riscv_vwmacc_vv_i32m4(v_acc, v_lhs16,
                                        v_rhs16, vl);
        k_off += vl;
      }

      vint32m1_t v_szero =
          __riscv_vmv_v_x_i32m1(0, __riscv_vsetvl_e32m1(1));
      vint32m1_t v_red =
          __riscv_vredsum_vs_i32m4_i32m1(v_acc, v_szero, vl0);
      int32_t val = __riscv_vmv_x_s_i32m1_i32(v_red);

      if (has_bias) val += bias_data[n_block];

      int total_shift = 31 - shift_per_channel[n_block];
      int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
      int64_t result =
          static_cast<int64_t>(val) *
              static_cast<int64_t>(
                  multiplier_per_channel[n_block]) +
          round;
      val = static_cast<int32_t>(result >> total_shift);

      val += dst_zp;
      val = std::max(val, clamp_min);
      val = std::min(val, clamp_max);
      dst_col[n_block] = static_cast<int8_t>(val);
    }
  }
}

}  // namespace optimized_rvv
}  // namespace tflite

#endif  // __riscv_vector

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_INT8_H_
