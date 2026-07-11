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
// RVV GEMM backend for RISC-V Vector Extension.
//
// This file provides GemmImplUsingRvv, a template specialization that
// dispatches FP32/INT8/UINT8 GEMM to RVV-optimized kernels, bypassing
// ruy which has no RISC-V support.
//
// For matrix layouts not matching the Conv emit pattern, it falls back
// to GemmImplUsingRuy (which will use scalar code on RISC-V).

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RVV_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RVV_H_

#if defined(__riscv_vector)

#include <algorithm>
#include <cstdint>
#include <vector>

#include "tflite/kernels/cpu_backend_context.h"
#include "tflite/kernels/cpu_backend_gemm_params.h"
#include "tflite/kernels/cpu_backend_gemm_ruy.h"
#include "tflite/kernels/cpu_backend_threadpool.h"
#include "tflite/kernels/internal/optimized/rvv_gemm_fp32.h"
#include "tflite/kernels/internal/optimized/rvv_gemm_int8.h"
#include "tflite/kernels/internal/optimized/rvv_gemm_uint8.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

using cpu_backend_gemm::QuantizationFlavor;

// Threadpool tasks for parallel GEMM execution.
// Each task processes a chunk of the M dimension.
struct RvvGemmFp32Task : cpu_backend_threadpool::Task {
  int m_chunk, n, k;
  const float* lhs_data;
  const float* rhs_chunk;
  const float* bias_data;
  float clamp_min, clamp_max;
  float* dst_chunk;
  void Run() override {
    if (k >= 16) {
      optimized_rvv::RvvGemmFp32Tiled(m_chunk, n, k, lhs_data, rhs_chunk,
                                       bias_data, clamp_min, clamp_max,
                                       dst_chunk);
    } else {
      optimized_rvv::RvvGemmFp32SmallK(m_chunk, n, k, lhs_data, rhs_chunk,
                                        bias_data, clamp_min, clamp_max,
                                        dst_chunk);
    }
  }
};

struct RvvGemmInt8PerChannelTask : cpu_backend_threadpool::Task {
  int m_chunk, n, k;
  const int8_t* lhs_data;
  const int8_t* rhs_chunk;
  int rhs_zp, dst_zp;
  const int32_t* bias_data;
  const int32_t* mul_pc;
  const int* shift_pc;
  int32_t clamp_min, clamp_max;
  int8_t* dst_chunk;
  void Run() override {
    if (k >= 16) {
      optimized_rvv::RvvGemmInt8PerChannelTiled(
          m_chunk, n, k, lhs_data, rhs_chunk, rhs_zp, dst_zp, bias_data,
          mul_pc, shift_pc, clamp_min, clamp_max, dst_chunk);
    } else {
      optimized_rvv::RvvGemmInt8PerChannel(
          m_chunk, n, k, lhs_data, rhs_chunk, rhs_zp, dst_zp, bias_data,
          mul_pc, shift_pc, clamp_min, clamp_max, dst_chunk);
    }
  }
};

struct RvvGemmUint8Task : cpu_backend_threadpool::Task {
  int m_chunk, n, k;
  const uint8_t* lhs_data;
  int lhs_zp;
  const uint8_t* rhs_chunk;
  int rhs_zp, dst_zp;
  const int32_t* bias_data;
  int32_t multiplier;
  int shift;
  int32_t clamp_min, clamp_max;
  uint8_t* dst_chunk;
  void Run() override {
    if (k >= 16) {
      optimized_rvv::RvvGemmUint8UniformTiled(
          m_chunk, n, k, lhs_data, lhs_zp, rhs_chunk, rhs_zp, dst_zp,
          bias_data, multiplier, shift, clamp_min, clamp_max, dst_chunk);
    } else {
      optimized_rvv::RvvGemmUint8Uniform(
          m_chunk, n, k, lhs_data, lhs_zp, rhs_chunk, rhs_zp, dst_zp,
          bias_data, multiplier, shift, clamp_min, clamp_max, dst_chunk);
    }
  }
};

// Default: fall back to ruy for unsupported type combinations.
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImplUsingRvv : GemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar,
                                            DstScalar, quantization_flavor> {};

// FP32 GEMM specialization.
template <>
struct GemmImplUsingRvv<float, float, float, float,
                         QuantizationFlavor::kFloatingPoint> {
  static void Run(
      const MatrixParams<float>& lhs_params, const float* lhs_data,
      const MatrixParams<float>& rhs_params, const float* rhs_data,
      const MatrixParams<float>& dst_params, float* dst_data,
      const GemmParams<float, float, QuantizationFlavor::kFloatingPoint>&
          params,
      CpuBackendContext* context) {
    // Check layout matches Conv FP32 emit pattern:
    //   lhs (filter): n x k row-major
    //   rhs (im2col): k x m col-major
    //   dst (output): n x m col-major
    const bool layout_ok = lhs_params.order == Order::kRowMajor &&
                           rhs_params.order == Order::kColMajor &&
                           dst_params.order == Order::kColMajor &&
                           lhs_params.rows == dst_params.rows &&
                           lhs_params.cols == rhs_params.rows &&
                           rhs_params.cols == dst_params.cols;
    if (!layout_ok) {
      // Fall back to ruy for unexpected layouts
      GemmImplUsingRuy<float, float, float, float,
                       QuantizationFlavor::kFloatingPoint>::Run(
          lhs_params, lhs_data, rhs_params, rhs_data, dst_params, dst_data,
          params, context);
      return;
    }

    const int m = rhs_params.cols;
    const int n = lhs_params.rows;
    const int k = lhs_params.cols;
    const int max_threads =
        (context != nullptr) ? context->max_num_threads() : 1;
    const int n_threads = std::min(max_threads, m);

    if (n_threads <= 1) {
      if (k >= 16) {
        optimized_rvv::RvvGemmFp32Tiled(m, n, k, lhs_data, rhs_data,
                                         params.bias, params.clamp_min,
                                         params.clamp_max, dst_data);
      } else {
        optimized_rvv::RvvGemmFp32SmallK(m, n, k, lhs_data, rhs_data,
                                          params.bias, params.clamp_min,
                                          params.clamp_max, dst_data);
      }
      return;
    }

    // Multi-threaded: split M dimension across threads
    std::vector<RvvGemmFp32Task> tasks(n_threads);
    for (int i = 0; i < n_threads; ++i) {
      const int s = i * m / n_threads;
      const int e = (i + 1) * m / n_threads;
      auto& t = tasks[i];
      t.m_chunk = e - s;
      t.n = n;
      t.k = k;
      t.lhs_data = lhs_data;
      t.rhs_chunk = rhs_data + static_cast<size_t>(s) * k;
      t.bias_data = params.bias;
      t.clamp_min = params.clamp_min;
      t.clamp_max = params.clamp_max;
      t.dst_chunk = dst_data + static_cast<size_t>(s) * n;
    }
    cpu_backend_threadpool::Execute(n_threads, tasks.data(), context);
  }
};

// INT8 per-channel GEMM specialization (for EfficientDet INT8).
template <>
struct GemmImplUsingRvv<int8_t, int8_t, int32_t, int8_t,
                         QuantizationFlavor::kIntegerWithPerRowMultiplier> {
  static void Run(
      const MatrixParams<int8_t>& lhs_params, const int8_t* lhs_data,
      const MatrixParams<int8_t>& rhs_params, const int8_t* rhs_data,
      const MatrixParams<int8_t>& dst_params, int8_t* dst_data,
      const GemmParams<int32_t, int8_t,
                       QuantizationFlavor::kIntegerWithPerRowMultiplier>&
          params,
      CpuBackendContext* context) {
    const bool layout_ok = lhs_params.order == Order::kRowMajor &&
                           rhs_params.order == Order::kColMajor &&
                           dst_params.order == Order::kColMajor &&
                           lhs_params.rows == dst_params.rows &&
                           lhs_params.cols == rhs_params.rows &&
                           rhs_params.cols == dst_params.cols &&
                           lhs_params.zero_point == 0 &&
                           params.multiplier_fixedpoint_perchannel != nullptr &&
                           params.multiplier_exponent_perchannel != nullptr;
    if (!layout_ok) {
      GemmImplUsingRuy<int8_t, int8_t, int32_t, int8_t,
                       QuantizationFlavor::kIntegerWithPerRowMultiplier>::Run(
          lhs_params, lhs_data, rhs_params, rhs_data, dst_params, dst_data,
          params, context);
      return;
    }

    const int m = rhs_params.cols;
    const int n = lhs_params.rows;
    const int k = lhs_params.cols;
    const int max_threads =
        (context != nullptr) ? context->max_num_threads() : 1;
    const int n_threads = std::min(max_threads, m);
    const int rhs_zp = rhs_params.zero_point;
    const int dst_zp = dst_params.zero_point;
    const int32_t cmin = static_cast<int32_t>(params.clamp_min);
    const int32_t cmax = static_cast<int32_t>(params.clamp_max);

    if (n_threads <= 1) {
      if (k >= 16) {
        optimized_rvv::RvvGemmInt8PerChannelTiled(
            m, n, k, lhs_data, rhs_data, rhs_zp, dst_zp, params.bias,
            params.multiplier_fixedpoint_perchannel,
            params.multiplier_exponent_perchannel, cmin, cmax, dst_data);
      } else {
        optimized_rvv::RvvGemmInt8PerChannel(
            m, n, k, lhs_data, rhs_data, rhs_zp, dst_zp, params.bias,
            params.multiplier_fixedpoint_perchannel,
            params.multiplier_exponent_perchannel, cmin, cmax, dst_data);
      }
      return;
    }

    std::vector<RvvGemmInt8PerChannelTask> tasks(n_threads);
    for (int i = 0; i < n_threads; ++i) {
      const int s = i * m / n_threads;
      const int e = (i + 1) * m / n_threads;
      auto& t = tasks[i];
      t.m_chunk = e - s;
      t.n = n;
      t.k = k;
      t.lhs_data = lhs_data;
      t.rhs_chunk = rhs_data + static_cast<size_t>(s) * k;
      t.rhs_zp = rhs_zp;
      t.dst_zp = dst_zp;
      t.bias_data = params.bias;
      t.mul_pc = params.multiplier_fixedpoint_perchannel;
      t.shift_pc = params.multiplier_exponent_perchannel;
      t.clamp_min = cmin;
      t.clamp_max = cmax;
      t.dst_chunk = dst_data + static_cast<size_t>(s) * n;
    }
    cpu_backend_threadpool::Execute(n_threads, tasks.data(), context);
  }
};

// UINT8 per-tensor GEMM specialization (for older MobileNet INT8).
template <>
struct GemmImplUsingRvv<uint8_t, uint8_t, int32_t, uint8_t,
                         QuantizationFlavor::kIntegerWithUniformMultiplier> {
  static void Run(
      const MatrixParams<uint8_t>& lhs_params, const uint8_t* lhs_data,
      const MatrixParams<uint8_t>& rhs_params, const uint8_t* rhs_data,
      const MatrixParams<uint8_t>& dst_params, uint8_t* dst_data,
      const GemmParams<int32_t, uint8_t,
                       QuantizationFlavor::kIntegerWithUniformMultiplier>&
          params,
      CpuBackendContext* context) {
    const bool layout_ok = lhs_params.order == Order::kRowMajor &&
                           rhs_params.order == Order::kColMajor &&
                           dst_params.order == Order::kColMajor &&
                           lhs_params.rows == dst_params.rows &&
                           lhs_params.cols == rhs_params.rows &&
                           rhs_params.cols == dst_params.cols;
    if (!layout_ok) {
      GemmImplUsingRuy<uint8_t, uint8_t, int32_t, uint8_t,
                       QuantizationFlavor::kIntegerWithUniformMultiplier>::Run(
          lhs_params, lhs_data, rhs_params, rhs_data, dst_params, dst_data,
          params, context);
      return;
    }

    const int m = rhs_params.cols;
    const int n = lhs_params.rows;
    const int k = lhs_params.cols;
    const int max_threads =
        (context != nullptr) ? context->max_num_threads() : 1;
    const int n_threads = std::min(max_threads, m);
    const int lhs_zp = lhs_params.zero_point;
    const int rhs_zp = rhs_params.zero_point;
    const int dst_zp = dst_params.zero_point;
    const int32_t cmin = static_cast<int32_t>(params.clamp_min);
    const int32_t cmax = static_cast<int32_t>(params.clamp_max);

    if (n_threads <= 1) {
      if (k >= 16) {
        optimized_rvv::RvvGemmUint8UniformTiled(
            m, n, k, lhs_data, lhs_zp, rhs_data, rhs_zp, dst_zp, params.bias,
            params.multiplier_fixedpoint, params.multiplier_exponent, cmin,
            cmax, dst_data);
      } else {
        optimized_rvv::RvvGemmUint8Uniform(
            m, n, k, lhs_data, lhs_zp, rhs_data, rhs_zp, dst_zp, params.bias,
            params.multiplier_fixedpoint, params.multiplier_exponent, cmin,
            cmax, dst_data);
      }
      return;
    }

    std::vector<RvvGemmUint8Task> tasks(n_threads);
    for (int i = 0; i < n_threads; ++i) {
      const int s = i * m / n_threads;
      const int e = (i + 1) * m / n_threads;
      auto& t = tasks[i];
      t.m_chunk = e - s;
      t.n = n;
      t.k = k;
      t.lhs_data = lhs_data;
      t.lhs_zp = lhs_zp;
      t.rhs_chunk = rhs_data + static_cast<size_t>(s) * k;
      t.rhs_zp = rhs_zp;
      t.dst_zp = dst_zp;
      t.bias_data = params.bias;
      t.multiplier = params.multiplier_fixedpoint;
      t.shift = params.multiplier_exponent;
      t.clamp_min = cmin;
      t.clamp_max = cmax;
      t.dst_chunk = dst_data + static_cast<size_t>(s) * n;
    }
    cpu_backend_threadpool::Execute(n_threads, tasks.data(), context);
  }
};

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // __riscv_vector

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RVV_H_
