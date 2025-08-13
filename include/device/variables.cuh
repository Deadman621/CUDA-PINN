#pragma once

#include<device/constants.h>

namespace device::tensor {
    template<constants::device_arithmetic T>
    struct variables {
        T* data = nullptr;
        size_t* shape = nullptr;
        size_t* stride = nullptr;
        size_t dim = 0;
        size_t n;
        bool transposed = false;

        __device__ T& operator[](size_t index) {
            if (transposed) {
                size_t real_index = 0;
                auto i = static_cast<long long int>(this->dim) - 1;
                while(i >= 0) {
                    real_index += ((index % this->shape[i]) * this->stride[i]);
                    index /= this->shape[i--];
                }
                return data[real_index];
            }

            return data[index];
        }

        __device__ const T& operator[](size_t index) const {
            if (transposed) {
                size_t real_index = 0;
                auto i = static_cast<long long int>(this->dim) - 1;
                while(i >= 0) {
                    real_index += ((index % this->shape[i]) * this->stride[i]);
                    index /= this->shape[i--];
                }
                return data[real_index];
            }

            return data[index];
        }

        __device__ T& operator*() { return *this->data; }
        __device__ const T& operator*() const { return *this->data; }

        __device__ size_t calculate_stride(size_t* weights) {
            size_t index = 0;
            for(size_t i = 0; i < this->n; i++)
                index += weights[i] * this->stride[i];
            return index;
        }
    };
}