#pragma once

#include<device/runtime.cuh>
#include<device/constants.h>
#include<device/variables.cuh>

namespace device::kernel {
    using constants::device_arithmetic;

    template<device_arithmetic T>
    __global__ void add(const tensor::variables<T> A, const tensor::variables<T> B, tensor::variables<T> R) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= R.n)
            return;

        R[i] = A[i % A.n] + B[i % B.n];
    }

    template<device_arithmetic T>
    __global__ void sub(const tensor::variables<T> A, const tensor::variables<T> B, tensor::variables<T> R) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= R.n)
            return;

        R[i] = A[i % A.n] - B[i % B.n];
    }

    template<device_arithmetic T>
    __global__ void mul(const tensor::variables<T> A, const tensor::variables<T> B, tensor::variables<T> R) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= A.n)
            return;

        R[i] = A[i % A.n] * B[i % B.n];
    }

    template<device_arithmetic T>
    __global__ void div(const tensor::variables<T> A, const tensor::variables<T> B, tensor::variables<T> R) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= A.n)
            return;

        T denominator = B[i % B.n];
        if (denominator == 0) {
            atomicExch(&runtime::d_error_code, constants::error::DIVISION_BY_ZERO);
            return;
        }

        R[i] = A[i % A.n] / B[i % B.n];
    }

    template<device_arithmetic T>
    using elew_kernel_ptr_t = void (*)(const tensor::variables<T>, const tensor::variables<T>, tensor::variables<T>);

    template<device_arithmetic T>
    __global__ void add_multiple(const tensor::variables<T> *A, tensor::variables<T> R, size_t count) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= R.n)
            return;

        T sum = 0;
        for (size_t j = 0; j < count; j++)
            sum += A[j][i];

        R.data[i] = sum;
    }

    template<device_arithmetic T>
    __global__ void matmul(
        const tensor::variables<T> A, const tensor::variables<T> B, tensor::variables<T> R,
        size_t I, size_t J, size_t K
        ) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i >= I || j >= J) return;

        T sum = 0.f;
        for(size_t k = 0; k < K; k++)
            sum += A[(i * K) + k] * B[(k * J) + j];

        R[(i * J) + j] = sum;
    }

    template<device_arithmetic T>
    __global__ void dot(const tensor::variables<T> A, const tensor::variables<T> B, tensor::variables<T> R) {
        __shared__ T partial[constants::BLOCK_SIZE];

        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        partial[threadIdx.x] = (i < A.n)? A[i] * B[i]: 0;

        __syncthreads();
        if (threadIdx.x == 0) {
            T sum = 0;
            for(size_t j = 0; j < constants::BLOCK_SIZE; j++)
                sum += partial[j];
            atomicAdd(R.data, sum);
        }
    }
}