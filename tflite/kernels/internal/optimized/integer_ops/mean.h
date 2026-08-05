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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MEAN_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MEAN_H_

#include <algorithm>

#include "tflite/kernels/cpu_backend_context.h"
#include "tflite/kernels/cpu_backend_threadpool.h"
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"

#ifdef USE_RVV
namespace tflite {
namespace optimized_rvv {

// Equivalent to NEON vqrdmulhq_n_s32 for int32m4:
// Computes high32(a * b * 2 + 2^30) >> 31 using 64-bit intermediate.
inline vint32m4_t RvvVqrdmulhScalar_i32m4(vint32m4_t a, int32_t b,
                                           size_t vl) {
  vint64m8_t prod = __riscv_vwmul_vx_i64m8(a, b, vl);
  prod = __riscv_vsll_vx_i64m8(prod, 1, vl);
  prod = __riscv_vadd_vx_i64m8(prod, static_cast<int64_t>(1) << 30, vl);
  return __riscv_vnsra_wx_i32m4(prod, 31, vl);
}

// Equivalent to gemmlowp::RoundingDivideByPOT(x, exponent) for int32m4.
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

}  // namespace optimized_rvv
}  // namespace tflite
#endif  // USE_RVV

namespace tflite {
namespace optimized_integer_ops {

inline void MeanImpl(const tflite::MeanParams& op_params,
                     const RuntimeShape& input_shape, const int8_t* input_data,
                     int32 multiplier, int32 shift, int32 bias,
                     const RuntimeShape& output_shape, int8_t* output_data,
                     int start_depth, int end_depth) {
  ruy::profiler::ScopeLabel label("Mean4D/Int8/MeanImpl");

  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  const int output_batch = output_shape.Dims(0);
  const int output_height = output_shape.Dims(2);
  const int output_width = output_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  constexpr static int32_t kMinValue = std::numeric_limits<int8_t>::min();
  constexpr static int32_t kMaxValue = std::numeric_limits<int8_t>::max();

#ifdef USE_NEON
  const int32x4_t bias_dup = vdupq_n_s32(bias);
  const int32x4_t min_dup = vdupq_n_s32(kMinValue);
  const int32x4_t max_dup = vdupq_n_s32(kMaxValue);
#endif  // USE_NEON
  for (int out_b = 0; out_b < output_batch; ++out_b) {
    int out_d = start_depth;
#ifdef USE_NEON

    for (; out_d <= end_depth - 16; out_d += 16) {
      int32x4x4_t temp_sum;
      temp_sum.val[0] = vdupq_n_s32(0);
      temp_sum.val[1] = vdupq_n_s32(0);
      temp_sum.val[2] = vdupq_n_s32(0);
      temp_sum.val[3] = vdupq_n_s32(0);
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          const int8_t* input_data_ptr =
              input_data + Offset(input_shape, out_b, in_h, in_w, out_d);
          int8x16_t input_data_val = vld1q_s8(input_data_ptr);

          int16x8_t input_data_low_shift =
              vmovl_s8(vget_low_s8(input_data_val));
          int16x8_t input_data_high_shift =
              vmovl_s8(vget_high_s8(input_data_val));

          int32x4_t input_low_low =
              vmovl_s16(vget_low_s16(input_data_low_shift));
          int32x4_t input_high_low =
              vmovl_s16(vget_high_s16(input_data_low_shift));
          int32x4_t input_low_high =
              vmovl_s16(vget_low_s16(input_data_high_shift));
          int32x4_t input_high_high =
              vmovl_s16(vget_high_s16(input_data_high_shift));

          temp_sum.val[0] = vaddq_s32(temp_sum.val[0], input_low_low);
          temp_sum.val[1] = vaddq_s32(temp_sum.val[1], input_high_low);
          temp_sum.val[2] = vaddq_s32(temp_sum.val[2], input_low_high);
          temp_sum.val[3] = vaddq_s32(temp_sum.val[3], input_high_high);
        }
      }

      temp_sum =
          MultiplyByQuantizedMultiplier4Rows(temp_sum, multiplier, shift);

      temp_sum.val[0] = vaddq_s32(temp_sum.val[0], bias_dup);
      temp_sum.val[1] = vaddq_s32(temp_sum.val[1], bias_dup);
      temp_sum.val[2] = vaddq_s32(temp_sum.val[2], bias_dup);
      temp_sum.val[3] = vaddq_s32(temp_sum.val[3], bias_dup);

      temp_sum.val[0] = vminq_s32(vmaxq_s32(temp_sum.val[0], min_dup), max_dup);
      temp_sum.val[1] = vminq_s32(vmaxq_s32(temp_sum.val[1], min_dup), max_dup);
      temp_sum.val[2] = vminq_s32(vmaxq_s32(temp_sum.val[2], min_dup), max_dup);
      temp_sum.val[3] = vminq_s32(vmaxq_s32(temp_sum.val[3], min_dup), max_dup);

      int16x4_t narrowed_low_low = vmovn_s32(temp_sum.val[0]);
      int16x4_t narrowed_high_low = vmovn_s32(temp_sum.val[1]);
      int16x4_t narrowed_low_high = vmovn_s32(temp_sum.val[2]);
      int16x4_t narrowed_high_high = vmovn_s32(temp_sum.val[3]);

      int16x8_t combined_low =
          vcombine_s16(narrowed_low_low, narrowed_high_low);
      int16x8_t combined_high =
          vcombine_s16(narrowed_low_high, narrowed_high_high);

      int8x8_t narrowed_low = vmovn_s16(combined_low);
      int8x8_t narrowed_high = vmovn_s16(combined_high);

      int8x16_t combined_output = vcombine_s8(narrowed_low, narrowed_high);

      int8_t* output_data_ptr =
          output_data + Offset(output_shape, out_b, 0, 0, out_d);
      vst1q_s8(output_data_ptr, combined_output);
    }
#endif  // USE_NEON

#ifdef USE_RVV
    // RVV optimized path: LMUL=1 with vsetvl auto-adapts to VLEN.
    // VLEN=128: 16 elements/iter; VLEN=256: 32 elements/iter.
    for (; out_d < end_depth;) {
      const size_t vl = __riscv_vsetvl_e8m1(end_depth - out_d);

      // Initialize int32 accumulators to zero
      vint32m4_t v_acc = __riscv_vmv_v_x_i32m4(0, vl);

      // Accumulate over input_height × input_width
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          const int8_t* input_data_ptr =
              input_data + Offset(input_shape, out_b, in_h, in_w, out_d);
          vint8m1_t v_in = __riscv_vle8_v_i8m1(input_data_ptr, vl);
          // Widen int8 → int16 → int32 and accumulate
          vint16m2_t v_in_16 = __riscv_vsext_vf2_i16m2(v_in, vl);
          vint32m4_t v_in_32 = __riscv_vsext_vf2_i32m4(v_in_16, vl);
          v_acc = __riscv_vadd_vv_i32m4(v_acc, v_in_32, vl);
        }
      }

      // Apply quantized multiplier: left_shift → qrdmulh → rounding_divide
      v_acc = optimized_rvv::RvvMultiplyByQuantizedMultiplier_i32m4(
          v_acc, multiplier, shift, vl);

      // Add bias
      v_acc = __riscv_vadd_vx_i32m4(v_acc, bias, vl);

      // Clamp to int8 range
      v_acc = __riscv_vmin_vx_i32m4(v_acc, kMaxValue, vl);
      v_acc = __riscv_vmax_vx_i32m4(v_acc, kMinValue, vl);

      // Narrow int32 → int16 via unsigned shift
      vuint32m4_t v_acc_u32 = __riscv_vreinterpret_v_i32m4_u32m4(v_acc);
      vint16m2_t v_out_16 = __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vnsrl_wx_u16m2(v_acc_u32, 0, vl));

      // Narrow int16 → int8 via unsigned clip with truncation rounding
      vuint16m2_t v_out_u16 = __riscv_vreinterpret_v_i16m2_u16m2(v_out_16);
      vint8m1_t v_out_8 = __riscv_vreinterpret_v_u8m1_i8m1(
          __riscv_vnclipu_wx_u8m1(v_out_u16, 0, __RISCV_VXRM_RDN, vl));

      int8_t* output_data_ptr =
          output_data + Offset(output_shape, out_b, 0, 0, out_d);
      __riscv_vse8_v_i8m1(output_data_ptr, v_out_8, vl);

      out_d += vl;
    }
#endif  // USE_RVV

    for (; out_d < end_depth; ++out_d) {
      int acc = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          acc += input_data[Offset(input_shape, out_b, in_h, in_w, out_d)];
        }
      }

      acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
      acc += bias;
      acc = std::min(std::max(acc, kMinValue), kMaxValue);
      output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
          static_cast<int8_t>(acc);
    }
  }
}

struct MeanWorkerTask : cpu_backend_threadpool::Task {
  MeanWorkerTask(const tflite::MeanParams& op_params,
                 const RuntimeShape& input_shape, const int8_t* input_data,
                 int32 multiplier, int32 shift, int32 bias,
                 const RuntimeShape& output_shape, int8_t* output_data,
                 int start_height, int end_height)
      : op_params(op_params),
        input_shape(input_shape),
        input_data(input_data),
        multiplier(multiplier),
        shift(shift),
        bias(bias),
        output_shape(output_shape),
        output_data(output_data),
        start_height(start_height),
        end_height(end_height) {}

  void Run() override {
    MeanImpl(op_params, input_shape, input_data, multiplier, shift, bias,
             output_shape, output_data, start_height, end_height);
  }

 private:
  const tflite::MeanParams& op_params;
  const RuntimeShape& input_shape;
  const int8_t* input_data;
  int32 multiplier;
  int32 shift;
  int32 bias;
  const RuntimeShape& output_shape;
  int8_t* output_data;
  int start_height;
  int end_height;
};

inline void Mean(const tflite::MeanParams& op_params,
                 const RuntimeShape& unextended_input_shape,
                 const int8_t* input_data, int32 input_zero_point,
                 float input_scale, const RuntimeShape& unextended_output_shape,
                 int8_t* output_data, int32 output_zero_point,
                 float output_scale, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("Mean4D/Int8");
  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  TFLITE_CHECK_EQ(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_CHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const float num_elements_in_axis = input_width * input_height;

  float temp = input_zero_point * input_scale / output_scale;
  temp = temp > 0 ? temp + 0.5f : temp - 0.5f;
  int32_t bias = output_zero_point - static_cast<int32_t>(temp);
  float real_scale = input_scale / (num_elements_in_axis * output_scale);

  int32 multiplier, shift;
  QuantizeMultiplier(real_scale, &multiplier, &shift);

  constexpr int kMinDepthPerThread = 8;
  int thread_count = output_depth / kMinDepthPerThread;
  thread_count = thread_count > 0 ? thread_count : 1;
  const int capped_thread_count =
      std::min(thread_count, cpu_backend_context->max_num_threads());

  if (capped_thread_count == 1) {
    MeanImpl(op_params, input_shape, input_data, multiplier, shift, bias,
             output_shape, output_data, 0, output_depth);
  } else {
    // Instead parallel for batch, we loop for the output_depth since batch
    // is typical 1.
    std::vector<MeanWorkerTask> tasks;
    // TODO(b/131746020) don't create new heap allocations every time.
    // At least we make it a single heap allocation by using reserve().
    tasks.reserve(capped_thread_count);
    int depth_start = 0;
    for (int i = 0; i < capped_thread_count; ++i) {
      // Try to distribute the tasks as even as possible.
      int depth_end = depth_start +
                      (output_depth - depth_start) / (capped_thread_count - i);
      tasks.emplace_back(op_params, input_shape, input_data, multiplier, shift,
                         bias, output_shape, output_data, depth_start,
                         depth_end);
      depth_start = depth_end;
    }
    cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                    cpu_backend_context);
  }
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MEAN_H_
