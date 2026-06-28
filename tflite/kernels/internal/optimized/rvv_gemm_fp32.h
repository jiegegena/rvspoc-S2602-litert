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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_FP32_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_FP32_H_

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <algorithm>
#include <cstddef>

namespace tflite {
namespace optimized_rvv {

// FP32 GEMM: dst = clamp(lhs * rhs + bias, clamp_min, clamp_max)
//
// Layout (matching optimized_ops::Conv FP32):
//   lhs: n x k row-major    -> lhs[n_idx * k + k_idx]
//   rhs: k x m col-major    -> rhs[m_idx * k + k_idx]
//   dst: n x m col-major    -> dst[m_idx * n + n_idx]
//   bias: optional, length n
//
// The inner loop over k performs a vectorized dot-product using RVV FMA.
// LMUL=4 is used for good throughput while keeping register pressure low.
inline void RvvGemmFp32(int m, int n, int k, const float* lhs_data,
                        const float* rhs_data, const float* bias_data,
                        float clamp_min, float clamp_max, float* dst_data) {
  for (int m_idx = 0; m_idx < m; ++m_idx) {
    const float* rhs_col = rhs_data + static_cast<size_t>(m_idx) * k;
    float* dst_col = dst_data + static_cast<size_t>(m_idx) * n;

    for (int n_idx = 0; n_idx < n; ++n_idx) {
      const float* lhs_row = lhs_data + static_cast<size_t>(n_idx) * k;

      // Vectorized dot-product over k dimension
      size_t remaining = static_cast<size_t>(k);
      const size_t vl0 = __riscv_vsetvl_e32m4(remaining);
      vfloat32m4_t v_acc = __riscv_vfmv_v_f_f32m4(0.0f, vl0);

      const float* p_lhs = lhs_row;
      const float* p_rhs = rhs_col;
      while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e32m4(remaining);
        vfloat32m4_t v_lhs = __riscv_vle32_v_f32m4(p_lhs, vl);
        vfloat32m4_t v_rhs = __riscv_vle32_v_f32m4(p_rhs, vl);
        v_acc = __riscv_vfmacc_vv_f32m4(v_acc, v_lhs, v_rhs, vl);
        p_lhs += vl;
        p_rhs += vl;
        remaining -= vl;
      }

      // Horizontal reduction
      vfloat32m1_t v_zero =
          __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
      vfloat32m1_t v_red =
          __riscv_vfredusum_vs_f32m4_f32m1(v_acc, v_zero, vl0);
      float acc = __riscv_vfmv_f_s_f32m1_f32(v_red);

      // Add bias and clamp
      if (bias_data != nullptr) {
        acc += bias_data[n_idx];
      }
      acc = std::max(acc, clamp_min);
      acc = std::min(acc, clamp_max);
      dst_col[n_idx] = acc;
    }
  }
}

// Optimized FP32 GEMM for small K (common in depthwise conv pointwise).
// Processes K in chunks of 4 using explicit unrolling.
inline void RvvGemmFp32SmallK(int m, int n, int k, const float* lhs_data,
                               const float* rhs_data, const float* bias_data,
                               float clamp_min, float clamp_max,
                               float* dst_data) {
  const bool has_bias = (bias_data != nullptr);

  for (int m_idx = 0; m_idx < m; ++m_idx) {
    const float* rhs_col = rhs_data + static_cast<size_t>(m_idx) * k;
    float* dst_col = dst_data + static_cast<size_t>(m_idx) * n;

    for (int n_idx = 0; n_idx < n; ++n_idx) {
      const float* lhs_row = lhs_data + static_cast<size_t>(n_idx) * k;

      float acc = 0.0f;
      int ki = 0;

      // Process 4 elements at a time
      for (; ki <= k - 4; ki += 4) {
        acc += lhs_row[ki] * rhs_col[ki];
        acc += lhs_row[ki + 1] * rhs_col[ki + 1];
        acc += lhs_row[ki + 2] * rhs_col[ki + 2];
        acc += lhs_row[ki + 3] * rhs_col[ki + 3];
      }
      // Handle remaining
      for (; ki < k; ++ki) {
        acc += lhs_row[ki] * rhs_col[ki];
      }

      if (has_bias) {
        acc += bias_data[n_idx];
      }
      acc = std::max(acc, clamp_min);
      acc = std::min(acc, clamp_max);
      dst_col[n_idx] = acc;
    }
  }
}

}  // namespace optimized_rvv
}  // namespace tflite

#endif  // __riscv_vector

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_FP32_H_
