// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Dtype.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
template <typename T, size_t N>
class Block {
public:
    bool OPEN3D_HOST_DEVICE operator==(const Block<T, N>& other) const {
        bool is_eq = true;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (size_t i = 0; i < N; ++i) {
            is_eq = is_eq && (data_[i] == other.data_[i]);
        }
        return is_eq;
    }

    void OPEN3D_HOST_DEVICE operator=(const Block<T, N>& other) {
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (size_t i = 0; i < N; ++i) {
            data_[i] = other.data_[i];
        }
    }

    const T& OPEN3D_HOST_DEVICE operator()(size_t i) const { return data_[i]; }
    T& OPEN3D_HOST_DEVICE operator()(size_t i) { return data_[i]; }

private:
    T data_[N];
};

template <typename T, size_t N>
struct BlockHash {
public:
    uint64_t OPEN3D_HOST_DEVICE operator()(const Block<T, N>& key) const {
        uint64_t hash = UINT64_C(14695981039346656037);
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (size_t i = 0; i < N; ++i) {
            hash ^= static_cast<uint64_t>(key(i));
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }
};

}  // namespace core
}  // namespace open3d

// TODO: dispatch more combinations.
#define DISPATCH_DTYPE_AND_DIM_TO_TEMPLATE(DTYPE, DIM, ...)                  \
    [&] {                                                                    \
        if (DTYPE == open3d::core::Dtype::Int32) {                           \
            if (DIM == 1) {                                                  \
                using key_t = Block<int, 1>;                                 \
                using hash_t = BlockHash<int, 1>;                            \
                return __VA_ARGS__();                                        \
            } else if (DIM == 2) {                                           \
                using key_t = Block<int, 2>;                                 \
                using hash_t = BlockHash<int, 2>;                            \
                return __VA_ARGS__();                                        \
            } else if (DIM == 3) {                                           \
                using key_t = Block<int, 3>;                                 \
                using hash_t = BlockHash<int, 3>;                            \
                return __VA_ARGS__();                                        \
            }                                                                \
        } else if (DTYPE == open3d::core::Dtype::Int64) {                    \
            if (DIM == 1) {                                                  \
                using key_t = Block<int64_t, 1>;                             \
                using hash_t = BlockHash<int64_t, 1>;                        \
                return __VA_ARGS__();                                        \
            } else if (DIM == 2) {                                           \
                using key_t = Block<int64_t, 2>;                             \
                using hash_t = BlockHash<int64_t, 2>;                        \
                return __VA_ARGS__();                                        \
            } else if (DIM == 3) {                                           \
                using key_t = Block<int64_t, 3>;                             \
                using hash_t = BlockHash<int64_t, 3>;                        \
                return __VA_ARGS__();                                        \
            }                                                                \
        } else {                                                             \
            utility::LogError("Unsupported dtype {} and dim {} combination", \
                              DTYPE.ToString(), DIM);                        \
        }                                                                    \
    }()
