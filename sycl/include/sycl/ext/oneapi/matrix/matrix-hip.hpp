//===-------- matrix-hip.hpp - matrix ext impl ---*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#pragma once

#include "matrix-unified-utils.hpp"

#include <sycl/access/access.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/marray.hpp>
#include <sycl/multi_ptr.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <type_traits>

#define __HIP_PLATFORM_AMD_MFMA__

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace detail {

constexpr int WAVEFRONT_SIZE = 64;

template <typename T, sycl::ext::oneapi::experimental::matrix::use Use,
          size_t Rows, size_t Cols,
          sycl::ext::oneapi::experimental::matrix::layout Layout =
              sycl::ext::oneapi::experimental::matrix::layout::dynamic,
          typename Cond = void>
struct joint_matrix_hip;

using bfloat16x4 = __attribute__((__vector_size__(4 * sizeof(__bf16)))) __fp16;
using float16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using int32x4 = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int;
using int32x16 = __attribute__((__vector_size__(16 * sizeof(int32_t)))) int;
using doublex4 = __attribute__((__vector_size__(4 * sizeof(double)))) double;
using uint32x4 = __attribute__((__vector_size__(16))) uint32_t;

template <typename T> struct to_hip_type {
  using type = T;
};

template <> struct to_hip_type<bfloat16> {
  using type = __bf16;
};

template <> struct to_hip_type<half> {
  using type = __fp16;
};

template <> struct to_hip_type<int8_t> {
  using type = int32_t;
};

template <typename T> struct default_mfma_vec_size {
  static constexpr int value = 4;
};

template <> struct default_mfma_vec_size<float> {
  static constexpr int value = 1;
};

template <> struct default_mfma_vec_size<double> {
  static constexpr int value = 1;
};

template <int element_size> struct default_swizzle_vec_size {
  static constexpr int value = 4;
};

template <> struct default_swizzle_vec_size<4> {
  static constexpr int value = 1;
};

template <> struct default_swizzle_vec_size<8> {
  static constexpr int value = 1;
};

template <sycl::ext::oneapi::experimental::matrix::use Use, typename S>
struct operand_tile_policy {
  static constexpr bool input_prepacked = false;
  static constexpr bool apply_swizzle_on_store = true;
  static constexpr bool apply_swizzle_on_load = true;
};

template <>
struct operand_tile_policy<
    sycl::ext::oneapi::experimental::matrix::use::b, sycl::half> {
  // For fp16 B, the caller must provide a pre-packed tile that already
  // incorporates the local reorder required by the thesis design.
  static constexpr bool input_prepacked = true;
  static constexpr bool apply_swizzle_on_store = true;
  static constexpr bool apply_swizzle_on_load = true;
};

template <>
struct operand_tile_policy<
    sycl::ext::oneapi::experimental::matrix::use::b, float> {
  static constexpr bool input_prepacked = false;
  static constexpr bool apply_swizzle_on_store = false;
  static constexpr bool apply_swizzle_on_load = false;
};

template <>
struct operand_tile_policy<
    sycl::ext::oneapi::experimental::matrix::use::b, double> {
  static constexpr bool input_prepacked = false;
  static constexpr bool apply_swizzle_on_store = false;
  static constexpr bool apply_swizzle_on_load = false;
};

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR(TYPE, USE, M, N, SIZE)                \
  template <sycl::ext::oneapi::experimental::matrix::layout Layout>            \
  struct joint_matrix_hip<                                                     \
      TYPE, sycl::ext::oneapi::experimental::matrix::use::USE, M, N, Layout,   \
      typename std::enable_if_t<                                               \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::row_major ||    \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::col_major>> {   \
    sycl::marray<TYPE, SIZE> wi_marray;                                        \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 8, 32, 4)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 8, 32, 4)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, a, 16, 4, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, b, 4, 16, 1)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 8, 32, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 16, 4)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, a, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, b, 16, 16, 4)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, a, 32, 2, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, b, 2, 32, 1)

// Compound logical tiles exposed at the SYCL layer:
//   FP32 m32n32k32 -> 16 x mfma_f32_32x32x2f32
//   FP64 m16n16k16 ->  4 x mfma_f64_16x16x4f64
// Global->LDS and LDS->VGPR helpers below are intentionally organized around
// these compound tiles rather than the native K=2/K=4 MFMA fragments.
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, a, 32, 32, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, b, 32, 32, 16)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(TYPE, M, N)                       \
  template <>                                                                  \
  struct joint_matrix_hip<                                                     \
      TYPE, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,   \
      sycl::ext::oneapi::experimental::matrix::layout::dynamic> {              \
    sycl::marray<TYPE, (M * N) / WAVEFRONT_SIZE> wi_marray;                    \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 32, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(double, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 32, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 16, 16)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC

template <sycl::ext::oneapi::experimental::matrix::layout Layout, typename S,
          typename T, size_t M, size_t N, access::address_space Space,
          access::decorated IsDecorated, typename Group>
void load_accumulator_layoutT(
    joint_matrix_hip<
        S, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride, Group &sg) {
  const auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
                   sg.get_local_linear_id();

  if constexpr (std::is_same_v<S, double>) {
    const auto thread_x = idx % N;
    const auto thread_y = idx / N;

    if constexpr (Layout ==
                  sycl::ext::oneapi::experimental::matrix::layout::row_major) {
      for (int i = 0; i < 4; ++i) {
        const int s_idx = thread_x + i * 4 * stride + thread_y * stride;
        res.wi_marray[i] = src[s_idx];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        const int s_idx = i * 4 + thread_x * stride + thread_y;
        res.wi_marray[i] = src[s_idx];
      }
    }
  } else if constexpr (std::is_same_v<S, float> || std::is_same_v<S, int32_t>) {
    if constexpr (M == 16 && N == 16) {
      const auto thread_x = idx % N;
      const auto thread_y = idx / N;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::layout::
                                  row_major) {
        for (int i = 0; i < 4; ++i) {
          const int s_idx = thread_x + i * stride + thread_y * 4 * stride;
          res.wi_marray[i] = src[s_idx];
        }
      } else {
        for (int i = 0; i < 4; ++i) {
          const int s_idx = i + thread_x * stride + thread_y * 4;
          res.wi_marray[i] = src[s_idx];
        }
      }
    } else if constexpr (M == 32 && N == 32) {
      const auto thread_x = idx % N;
      const auto thread_y = idx / N;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::layout::
                                  row_major) {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int s_idx =
                thread_x + i * stride + thread_y * 4 * stride + j * 8 * N;
            res.wi_marray[i + 4 * j] = src[s_idx];
          }
        }
      } else {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int s_idx = i + thread_x * stride + thread_y * 4 + j * 8;
            res.wi_marray[i + 4 * j] = src[s_idx];
          }
        }
      }
    }
  }
}

template <
    typename Group, typename S, typename T, size_t M, size_t N,
    access::address_space Space, access::decorated IsDecorated,
    typename = std::enable_if_t<std::is_same_v<S, std::remove_const_t<T>>>>
void load_accumulator_hip(
    joint_matrix_hip<
        S, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout layout, Group &sg) {
  if (layout == sycl::ext::oneapi::experimental::matrix::layout::row_major)
    load_accumulator_layoutT<
        sycl::ext::oneapi::experimental::matrix::layout::row_major>(res, src,
                                                                    stride, sg);
  else
    load_accumulator_layoutT<
        sycl::ext::oneapi::experimental::matrix::layout::col_major>(res, src,
                                                                    stride, sg);
}

template <
    typename Group, typename S, typename T, size_t M, size_t N,
    sycl::ext::oneapi::experimental::matrix::use Use,
    sycl::ext::oneapi::experimental::matrix::layout Layout,
    access::address_space Space, access::decorated IsDecorated,
    typename = typename std::enable_if_t<
        (Layout == sycl::ext::oneapi::experimental::matrix::layout::row_major ||
         Layout ==
             sycl::ext::oneapi::experimental::matrix::layout::col_major) &&
        std::is_same_v<S, std::remove_const_t<T>>>>
void load_multiplicand_hip(joint_matrix_hip<S, Use, M, N, Layout> &res,
                           multi_ptr<T, Space, IsDecorated> src, size_t stride,
                           Group &sg) {
  const auto lane = sg.get_local_linear_id();

  if constexpr (std::is_same_v<S, float>) {
    const int tidx = lane % 32;
    const int ty = lane / 32;
    if constexpr (Use ==
                  sycl::ext::oneapi::experimental::matrix::use::a) {
      if constexpr (Layout ==
                    sycl::ext::oneapi::experimental::matrix::layout::row_major)
        res.wi_marray[0] = src[tidx * stride + ty];
      else
        res.wi_marray[0] = src[ty * stride + tidx];
    } else {
      if constexpr (Layout ==
                    sycl::ext::oneapi::experimental::matrix::layout::row_major)
        res.wi_marray[0] = src[ty * stride + tidx];
      else
        res.wi_marray[0] = src[tidx * stride + ty];
    }
  } else if constexpr (std::is_same_v<S, double>) {
    const int tidx = lane % 16;
    const int ty = (lane / 16) % 4;
    if constexpr (M == 16 && N == 16) {
      if constexpr (Use ==
                    sycl::ext::oneapi::experimental::matrix::use::a) {
        for (int i = 0; i < 4; ++i) {
          if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                      layout::row_major)
            res.wi_marray[i] = src[tidx * stride + (4 * i + ty)];
          else
            res.wi_marray[i] = src[(4 * i + ty) * stride + tidx];
        }
      } else {
        for (int i = 0; i < 4; ++i) {
          if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                      layout::row_major)
            res.wi_marray[i] = src[(4 * i + ty) * stride + tidx];
          else
            res.wi_marray[i] = src[tidx * stride + (4 * i + ty)];
        }
      }
    } else {
      if constexpr (Use ==
                    sycl::ext::oneapi::experimental::matrix::use::a) {
        if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                    layout::row_major)
          res.wi_marray[0] = src[tidx * stride + ty];
        else
          res.wi_marray[0] = src[ty * stride + tidx];
      } else {
        if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                    layout::row_major)
          res.wi_marray[0] = src[ty * stride + tidx];
        else
          res.wi_marray[0] = src[tidx * stride + ty];
      }
    }
  } else {
    if constexpr (Use ==
                  sycl::ext::oneapi::experimental::matrix::use::a) {
      const int tidx = lane % M;
      const int ty = lane / M;
      for (int i = 0; i < 4; ++i) {
        if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                    layout::row_major)
          res.wi_marray[i] = src[tidx * stride + ty * 4 + i];
        else
          res.wi_marray[i] = src[(ty * 4 + i) * stride + tidx];
      }
    } else {
      const int tidx = lane % N;
      const int ty = lane / N;
      for (int i = 0; i < 4; ++i) {
        if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                    layout::row_major)
          res.wi_marray[i] = src[(ty * 4 + i) * stride + tidx];
        else
          res.wi_marray[i] = src[tidx * stride + ty * 4 + i];
      }
    }
  }
}

template <typename Group,
          sycl::ext::oneapi::experimental::matrix::layout Layout, typename T,
          size_t M, size_t N, access::address_space Space,
          access::decorated IsDecorated>
void store_layoutT(
    const joint_matrix_hip<
        T, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride, Group &sg) {
  const auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
                   sg.get_local_linear_id();

  if constexpr (std::is_same_v<T, double>) {
    const auto thread_x = idx % N;
    const auto thread_y = idx / N;

    if constexpr (Layout ==
                  sycl::ext::oneapi::experimental::matrix::layout::row_major) {
      for (int i = 0; i < 4; ++i) {
        const int d_idx = thread_x + i * 4 * stride + thread_y * stride;
        dst[d_idx] = src.wi_marray[i];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        const int d_idx = i * 4 + thread_x * stride + thread_y;
        dst[d_idx] = src.wi_marray[i];
      }
    }
  } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t>) {
    if constexpr (M == 16 && N == 16) {
      const auto thread_x = idx % N;
      const auto thread_y = idx / N;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::layout::
                                  row_major) {
        for (int i = 0; i < 4; ++i) {
          const int d_idx = thread_x + i * stride + thread_y * 4 * stride;
          dst[d_idx] = src.wi_marray[i];
        }
      } else {
        for (int i = 0; i < 4; ++i) {
          const int d_idx = i + thread_x * stride + thread_y * 4;
          dst[d_idx] = src.wi_marray[i];
        }
      }
    } else if constexpr (M == 32 && N == 32) {
      const auto thread_x = idx % N;
      const auto thread_y = idx / N;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::layout::
                                  row_major) {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int d_idx =
                thread_x + i * stride + thread_y * 4 * stride + j * 8 * stride;
            dst[d_idx] = src.wi_marray[i + 4 * j];
          }
        }
      } else {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int d_idx = i + thread_x * stride + thread_y * 4 + j * 8;
            dst[d_idx] = src.wi_marray[i + 4 * j];
          }
        }
      }
    }
  }
}

template <typename Group, typename T, size_t M, size_t N,
          access::address_space Space, access::decorated IsDecorated>
void joint_matrix_store_hip(
    const joint_matrix_hip<
        T, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout layout, Group &sg) {
  if (sycl::ext::oneapi::experimental::matrix::layout::row_major == layout) {
    store_layoutT<Group,
                  sycl::ext::oneapi::experimental::matrix::layout::row_major>(
        src, dst, stride, sg);
  } else {
    store_layoutT<Group,
                  sycl::ext::oneapi::experimental::matrix::layout::col_major>(
        src, dst, stride, sg);
  }
}

template <typename Tm, typename Tc, std::size_t M, std::size_t K, std::size_t N,
          sycl::ext::oneapi::experimental::matrix::layout LayoutA,
          sycl::ext::oneapi::experimental::matrix::layout LayoutB>
void joint_matrix_mad_hip(
    joint_matrix_hip<
        Tc, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &D,
    const joint_matrix_hip<Tm, sycl::ext::oneapi::experimental::matrix::use::a,
                           M, K, LayoutA> &A,
    const joint_matrix_hip<Tm, sycl::ext::oneapi::experimental::matrix::use::b,
                           K, N, LayoutB> &B,
    const joint_matrix_hip<
        Tc, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &C) {
#ifdef __gfx90a__
  if constexpr (std::is_same_v<Tm, sycl::half>) {
    if constexpr (M == 16 && N == 16) {
      auto result = __builtin_amdgcn_mfma_f32_16x16x16f16(
          *reinterpret_cast<const float16x4 *>(&A.wi_marray),
          *reinterpret_cast<const float16x4 *>(&B.wi_marray),
          *reinterpret_cast<const floatx4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(float));
    } else if constexpr (M == 32 && N == 32) {
      auto result = __builtin_amdgcn_mfma_f32_32x32x8f16(
          *reinterpret_cast<const float16x4 *>(&A.wi_marray),
          *reinterpret_cast<const float16x4 *>(&B.wi_marray),
          *reinterpret_cast<const floatx16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(float));
    }
  } else if constexpr (std::is_same_v<Tm, bfloat16>) {
    if constexpr (M == 16 && N == 16) {
      auto result = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          *reinterpret_cast<const bfloat16x4 *>(&A.wi_marray),
          *reinterpret_cast<const bfloat16x4 *>(&B.wi_marray),
          *reinterpret_cast<const floatx4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(float));
    } else if constexpr (M == 32 && N == 32) {
      auto result = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
          *reinterpret_cast<const bfloat16x4 *>(&A.wi_marray),
          *reinterpret_cast<const bfloat16x4 *>(&B.wi_marray),
          *reinterpret_cast<const floatx16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(float));
    }
  } else if constexpr (std::is_same_v<Tm, double>) {
    static_assert(M == 16 && N == 16 && (K == 4 || K == 16),
                  "FP64 HIP joint_matrix supports native m16n16k4 or "
                  "compound m16n16k16 only");
    if constexpr (M == 16 && N == 16 && K == 16) {
      auto accum = *reinterpret_cast<const doublex4 *>(&C.wi_marray);
      for (int i = 0; i < 4; ++i) {
        accum = __builtin_amdgcn_mfma_f64_16x16x4f64(
            A.wi_marray[i], B.wi_marray[i], accum, 0, 0, 0);
      }
      std::memcpy(&D.wi_marray, &accum, 4 * sizeof(double));
    } else if constexpr (M == 16 && N == 16 && K == 4) {
      auto result = __builtin_amdgcn_mfma_f64_16x16x4f64(
          A.wi_marray[0], B.wi_marray[0],
          *reinterpret_cast<const doublex4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(double));
    }
  } else if constexpr (std::is_same_v<Tm, float>) {
    static_assert(M == 32 && N == 32 && (K == 2 || K == 32),
                  "FP32 HIP joint_matrix supports native m32n32k2 or "
                  "compound m32n32k32 only");
    if constexpr (M == 32 && N == 32 && K == 32) {
      auto accum = *reinterpret_cast<const floatx16 *>(&C.wi_marray);
      for (int i = 0; i < 16; ++i) {
        accum = __builtin_amdgcn_mfma_f32_32x32x2f32(
            A.wi_marray[i], B.wi_marray[i], accum, 0, 0, 0);
      }
      std::memcpy(&D.wi_marray, &accum, 16 * sizeof(float));
    } else if constexpr (M == 32 && N == 32 && K == 2) {
      auto result = __builtin_amdgcn_mfma_f32_32x32x2f32(
          A.wi_marray[0], B.wi_marray[0],
          *reinterpret_cast<const floatx16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(float));
    }
  } else if constexpr (std::is_same_v<Tm, int8_t>) {
    if constexpr (M == 16 && N == 16) {
      auto result = __builtin_amdgcn_mfma_i32_16x16x16i8(
          *reinterpret_cast<const Tc *>(&A.wi_marray),
          *reinterpret_cast<const Tc *>(&B.wi_marray),
          *reinterpret_cast<const int32x4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(int32_t));
    } else if constexpr (M == 32 && N == 32) {
      auto result = __builtin_amdgcn_mfma_i32_32x32x8i8(
          *reinterpret_cast<const Tc *>(&A.wi_marray),
          *reinterpret_cast<const Tc *>(&B.wi_marray),
          *reinterpret_cast<const int32x16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(int32_t));
    }
  }
#endif // __gfx90a__
}

std::pair<int, int> reverse_index_map_transposed(int tid, int local_id) {
  return std::make_pair((tid / 16) * 4 + local_id, tid % 16);
}

std::pair<int, int> reverse_index_map(int tid, int local_id) {
  return std::make_pair(tid % 16, (tid / 16) * 4 + local_id);
}

template <int element_size, int VecSize>
inline int swizzle_col(int row, int col, int stride) {
  const int dtype_bits = element_size * 8;
  const int numBanks = 32;
  const int bankBitWidth = 32;
  const int SIMDWidth = 16;

  const int elemsPerOneBanksRow = (numBanks * bankBitWidth) / dtype_bits;
  const int perPhase = std::max(1, elemsPerOneBanksRow / stride);
  const int maxPhase = std::min(SIMDWidth / perPhase, stride / VecSize);
  const int phase = (row / perPhase) % maxPhase;

  return (((col / VecSize) ^ phase) * VecSize) + (col % VecSize);
}

template <int element_size = 2,
          int VecSize = default_swizzle_vec_size<element_size>::value>
std::pair<int, int> make_mfma_swizzle_layout(const int row, const int col,
                                             int stride) {
  return std::make_pair(row, swizzle_col<element_size, VecSize>(row, col, stride));
}

template <typename PtrT>
inline bool is_byte_aligned(const PtrT *ptr, std::size_t alignment) {
  return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

template <typename S, int VecSize>
inline void store_swizzled_chunk(S *dst, const S *src_chunk) {
  constexpr std::size_t kChunkBytes = VecSize * sizeof(S);

  if (!is_byte_aligned(dst, kChunkBytes)) {
    for (int i = 0; i < VecSize; ++i)
      dst[i] = src_chunk[i];
    return;
  }

  if constexpr (kChunkBytes == 16) {
    uint32x4 tmp;
    std::memcpy(&tmp, src_chunk, kChunkBytes);
    *reinterpret_cast<uint32x4 *>(dst) = tmp;
  } else if constexpr (kChunkBytes == 8) {
    uint64_t tmp;
    std::memcpy(&tmp, src_chunk, kChunkBytes);
    *reinterpret_cast<uint64_t *>(dst) = tmp;
  } else if constexpr (kChunkBytes == 4) {
    uint32_t tmp;
    std::memcpy(&tmp, src_chunk, kChunkBytes);
    *reinterpret_cast<uint32_t *>(dst) = tmp;
  } else {
    for (int i = 0; i < VecSize; ++i)
      dst[i] = src_chunk[i];
  }
}

template <typename S, size_t TileRows, size_t TileCols,
          int VecSize = default_mfma_vec_size<S>::value>
void store_mfma_tile_direct(S *lds_dst, const S *src, size_t src_stride,
                            size_t lds_stride, int lane) {
  constexpr int kTransferBytes = 16;
  constexpr int kTransferElems = kTransferBytes / static_cast<int>(sizeof(S));
  constexpr int kSubChunks = kTransferElems / VecSize;
  constexpr int kTotalElems = static_cast<int>(TileRows * TileCols);
  constexpr bool kUsePacketPath = (kTransferElems % VecSize == 0) &&
                                  (TileCols % kTransferElems == 0);

  static_assert(kTransferElems % VecSize == 0,
                "16B packet must be divisible by VecSize");

  if constexpr (kUsePacketPath) {
    constexpr int kTotalPackets = kTotalElems / kTransferElems;

    for (int packet = lane; packet < kTotalPackets; packet += WAVEFRONT_SIZE) {
      const int base = packet * kTransferElems;
      const int row = base / static_cast<int>(TileCols);
      const int col0 = base % static_cast<int>(TileCols);

      const S *src_packet = src + row * static_cast<int>(src_stride) + col0;

      if (!is_byte_aligned(src_packet, kTransferBytes)) {
        for (int e = 0; e < kTransferElems; ++e)
          lds_dst[row * static_cast<int>(lds_stride) + col0 + e] =
              src_packet[e];
        continue;
      }

      const auto *packet_ptr = reinterpret_cast<const uint32x4 *>(src_packet);
      const uint32x4 packet_data = *packet_ptr;
      const auto *packet_elems = reinterpret_cast<const S *>(&packet_data);

      for (int sub = 0; sub < kSubChunks; ++sub) {
        const int col = col0 + sub * VecSize;
        store_swizzled_chunk<S, VecSize>(
            lds_dst + row * static_cast<int>(lds_stride) + col,
            packet_elems + sub * VecSize);
      }
    }
    return;
  }

  for (int idx = lane; idx < kTotalElems; idx += WAVEFRONT_SIZE) {
    const int row = idx / static_cast<int>(TileCols);
    const int col = idx % static_cast<int>(TileCols);

    lds_dst[row * static_cast<int>(lds_stride) + col] =
        src[row * static_cast<int>(src_stride) + col];
  }
}

// ============================================================
// store_mfma_tile_swizzled: Cooperatively write a plain
// row-major tile into LDS with swizzle layout.
//
// Final version:
//   - preserves the original interface/signature;
//   - prefers a 16B vectorized GMEM->VGPR path when the tile shape
//     and source address alignment allow it;
//   - writes LDS back in VecSize-sized swizzled chunks;
//   - falls back to scalar stores when vectorized transfer is unsafe.
// ============================================================
template <typename S, size_t TileRows, size_t TileCols,
          int VecSize = default_mfma_vec_size<S>::value>
void store_mfma_tile_swizzled(S *lds_dst, const S *src,
                              size_t src_stride, size_t lds_stride,
                              int lane) {
  constexpr int kTransferBytes = 16;
  constexpr int kTransferElems = kTransferBytes / static_cast<int>(sizeof(S));
  constexpr int kSubChunks = kTransferElems / VecSize;
  constexpr int kTotalElems = static_cast<int>(TileRows * TileCols);
  constexpr bool kUsePacketPath = (kTransferElems % VecSize == 0) &&
                                  (TileCols % kTransferElems == 0);

  static_assert(kTransferElems % VecSize == 0,
                "16B packet must be divisible by VecSize");

  if constexpr (kUsePacketPath) {
    constexpr int kTotalPackets = kTotalElems / kTransferElems;

    for (int packet = lane; packet < kTotalPackets; packet += WAVEFRONT_SIZE) {
      const int base = packet * kTransferElems;
      const int row = base / static_cast<int>(TileCols);
      const int col0 = base % static_cast<int>(TileCols);

      const S *src_packet =
          src + row * static_cast<int>(src_stride) + col0;

      if (!is_byte_aligned(src_packet, kTransferBytes)) {
        for (int e = 0; e < kTransferElems; ++e) {
          const int col = col0 + e;
          auto [sw_row, sw_col] =
              make_mfma_swizzle_layout<sizeof(S), VecSize>(row, col,
                                                           static_cast<int>(lds_stride));
          lds_dst[sw_row * static_cast<int>(lds_stride) + sw_col] =
              src_packet[e];
        }
        continue;
      }

      const auto *packet_ptr = reinterpret_cast<const uint32x4 *>(src_packet);
      const uint32x4 packet_data = *packet_ptr;
      const auto *packet_elems = reinterpret_cast<const S *>(&packet_data);

      for (int sub = 0; sub < kSubChunks; ++sub) {
        const int col = col0 + sub * VecSize;
        const int sw_col = swizzle_col<sizeof(S), VecSize>(
            row, col, static_cast<int>(lds_stride));

        store_swizzled_chunk<S, VecSize>(
            lds_dst + row * static_cast<int>(lds_stride) + sw_col,
            packet_elems + sub * VecSize);
      }
    }
    return;
  }

  for (int idx = lane; idx < kTotalElems; idx += WAVEFRONT_SIZE) {
    const int row = idx / static_cast<int>(TileCols);
    const int col = idx % static_cast<int>(TileCols);

    auto [sw_row, sw_col] =
        make_mfma_swizzle_layout<sizeof(S), VecSize>(row, col,
                                                     static_cast<int>(lds_stride));

    lds_dst[sw_row * static_cast<int>(lds_stride) + sw_col] =
        src[row * static_cast<int>(src_stride) + col];
  }
}

template <sycl::ext::oneapi::experimental::matrix::use Use, typename S,
          size_t TileRows, size_t TileCols,
          int VecSize = default_mfma_vec_size<S>::value>
void store_operand_tile_to_lds(S *lds_dst, const S *src, size_t src_stride,
                               size_t lds_stride, int lane) {
  if constexpr (std::is_same_v<S, float>) {
    static_assert(TileRows == 32 && TileCols == 32,
                  "FP32 operand LDS transfers use compound m32n32k32 tiles");
  } else if constexpr (std::is_same_v<S, double>) {
    static_assert(TileRows == 16 && TileCols == 16,
                  "FP64 operand LDS transfers use compound m16n16k16 tiles");
  }

  if constexpr (operand_tile_policy<Use, S>::apply_swizzle_on_store) {
    store_mfma_tile_swizzled<S, TileRows, TileCols, VecSize>(
        lds_dst, src, src_stride, lds_stride, lane);
  } else {
    store_mfma_tile_direct<S, TileRows, TileCols, VecSize>(
        lds_dst, src, src_stride, lds_stride, lane);
  }
}

template <typename S, sycl::ext::oneapi::experimental::matrix::use Use,
          size_t M, size_t N,
          sycl::ext::oneapi::experimental::matrix::layout Layout>
void load_mfma_tile_direct(joint_matrix_hip<S, Use, M, N, Layout> &res,
                           const S *lds_ptr, size_t stride, int lane) {
  if constexpr (std::is_same_v<S, sycl::half> ||
                std::is_same_v<S, bfloat16>) {
    if constexpr (Use ==
                  sycl::ext::oneapi::experimental::matrix::use::a) {
      for (int i = 0; i < 4; ++i) {
        const auto [row, col] = reverse_index_map(lane, i);
        res.wi_marray[i] = lds_ptr[row * stride + col];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        const auto [row, col] = reverse_index_map_transposed(lane, i);
        res.wi_marray[i] = lds_ptr[row * stride + col];
      }
    }
  } else if constexpr (std::is_same_v<S, double>) {
    static_assert(M == 16 && N == 16,
                  "FP64 LDS->VGPR load uses compound m16n16k16 tiles");
    const int tidx = lane % 16;
    const int ty = (lane / 16) % 4;
    if constexpr (Use ==
                  sycl::ext::oneapi::experimental::matrix::use::a) {
      for (int i = 0; i < 4; ++i) {
        const int row = tidx;
        const int col = 4 * i + ty;
        res.wi_marray[i] = lds_ptr[row * stride + col];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        const int row = 4 * i + ty;
        const int col = tidx;
        res.wi_marray[i] = lds_ptr[row * stride + col];
      }
    }
  } else if constexpr (std::is_same_v<S, float>) {
    static_assert(M == 32 && N == 32,
                  "FP32 LDS->VGPR load uses compound m32n32k32 tiles");
    const int tidx = lane % 32;
    const int ty = lane / 32;
    if constexpr (Use ==
                  sycl::ext::oneapi::experimental::matrix::use::a) {
      for (int i = 0; i < 16; ++i) {
        const int row = tidx;
        const int col = 2 * i + ty;
        res.wi_marray[i] = lds_ptr[row * stride + col];
      }
    } else {
      for (int i = 0; i < 16; ++i) {
        const int row = 2 * i + ty;
        const int col = tidx;
        res.wi_marray[i] = lds_ptr[row * stride + col];
      }
    }
  }
}

// ============================================================
// load_mfma_tile_swizzled: Load a compound MFMA tile from
// swizzled LDS. Supports:
//   - fp16/bf16: m16n16k16 (1x mfma_f32_16x16x16f16)
//   - fp64:      m16n16k16 (4x mfma_f64_16x16x4f64)
//   - fp32:      m32n32k32 (16x mfma_f32_32x32x2f32)
//   - B(fp32/fp64): direct LDS load without swizzle, per thesis 3.5.4
//
// Parameters:
//   res     - joint_matrix_hip to fill
//   lds_ptr - pointer to the start of the swizzled LDS tile
//   stride  - LDS row stride (in elements)
//   lane    - lane ID within wavefront (0..63)
// ============================================================
template <typename S, sycl::ext::oneapi::experimental::matrix::use Use,
          size_t M, size_t N,
          sycl::ext::oneapi::experimental::matrix::layout Layout>
void load_mfma_tile_swizzled(joint_matrix_hip<S, Use, M, N, Layout> &res,
                             const S *lds_ptr, size_t stride, int lane) {
  if constexpr (!operand_tile_policy<Use, S>::apply_swizzle_on_load) {
    load_mfma_tile_direct(res, lds_ptr, stride, lane);
    return;
  }

  if constexpr (std::is_same_v<S, sycl::half> ||
                std::is_same_v<S, bfloat16>) {
    if constexpr (Use ==
                  sycl::ext::oneapi::experimental::matrix::use::a) {
      for (int i = 0; i < 4; ++i) {
        auto [row, col] = reverse_index_map(lane, i);
        std::tie(row, col) =
            make_mfma_swizzle_layout<sizeof(S)>(row, col, stride);
        res.wi_marray[i] = lds_ptr[row * stride + col];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        auto [row, col] = reverse_index_map_transposed(lane, i);
        std::tie(row, col) =
            make_mfma_swizzle_layout<sizeof(S)>(row, col, stride);
        res.wi_marray[i] = lds_ptr[row * stride + col];
      }
    }
  } else if constexpr (std::is_same_v<S, double>) {
    static_assert(M == 16 && N == 16,
                  "FP64 swizzled LDS->VGPR load uses compound m16n16k16 tiles");
    const int tidx = lane % 16;
    const int ty = (lane / 16) % 4;
    if constexpr (Use ==
                  sycl::ext::oneapi::experimental::matrix::use::a) {
      for (int i = 0; i < 4; ++i) {
        int row = tidx;
        int col = 4 * i + ty;
        std::tie(row, col) =
            make_mfma_swizzle_layout<sizeof(double)>(row, col, stride);
        res.wi_marray[i] = lds_ptr[row * stride + col];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        int row = 4 * i + ty;
        int col = tidx;
        std::tie(row, col) =
            make_mfma_swizzle_layout<sizeof(double)>(row, col, stride);
        res.wi_marray[i] = lds_ptr[row * stride + col];
      }
    }
  } else if constexpr (std::is_same_v<S, float>) {
    static_assert(M == 32 && N == 32,
                  "FP32 swizzled LDS->VGPR load uses compound m32n32k32 tiles");
    const int tidx = lane % 32;
    const int ty = lane / 32;
    if constexpr (Use ==
                  sycl::ext::oneapi::experimental::matrix::use::a) {
      for (int i = 0; i < 16; ++i) {
        int row = tidx;
        int col = 2 * i + ty;
        std::tie(row, col) =
            make_mfma_swizzle_layout<sizeof(float)>(row, col, stride);
        res.wi_marray[i] = lds_ptr[row * stride + col];
      }
    } else {
      for (int i = 0; i < 16; ++i) {
        int row = 2 * i + ty;
        int col = tidx;
        std::tie(row, col) =
            make_mfma_swizzle_layout<sizeof(float)>(row, col, stride);
        res.wi_marray[i] = lds_ptr[row * stride + col];
      }
    }
  }
}

template <typename S, sycl::ext::oneapi::experimental::matrix::use Use,
          size_t M, size_t N,
          sycl::ext::oneapi::experimental::matrix::layout Layout>
void load_operand_tile_from_lds(joint_matrix_hip<S, Use, M, N, Layout> &res,
                                const S *lds_ptr, size_t stride, int lane) {
  if constexpr (operand_tile_policy<Use, S>::apply_swizzle_on_load) {
    load_mfma_tile_swizzled(res, lds_ptr, stride, lane);
  } else {
    load_mfma_tile_direct(res, lds_ptr, stride, lane);
  }
}

} // namespace detail
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
