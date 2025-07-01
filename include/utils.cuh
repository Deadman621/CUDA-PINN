#pragma once

#include<initializer_list>
#include<iostream>
#include<cstring>
#include<array>
#include<type_traits>

namespace kernel {
    template<typename T>
    __global__ void add(T* A, T* out, size_t count, size_t N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        T sum = 0;
        for(size_t j = 0; j < count; j++)  
            sum += *((A + j * N) + i);

        out[i] = sum;
    }
}

// type unsafe function.
template<typename First, typename... Rest>
static size_t deduceSize(const First& __ax, const Rest&... __bx) { return __ax.size(); }

template<typename T>
class tensor {
    private:    
        T* x = nullptr;
        size_t n;
    
    public:
        tensor(const std::initializer_list<T>& nums = {0}) {
            this->n = nums.size();
            this->x = new T[this->n];        
            std::copy(nums.begin(), nums.end(), this->x);
        }

        tensor(const size_t& n) {
            this->n = n;
            this->x = new T[this->n];
            std::memset(this->x, 0, this->n * sizeof(T));
        }

        tensor(const tensor& obj) {
            this->n = obj.n;
            this->x = new T[this->n];
            std::copy(obj.x, obj.x + obj.n, this->x);
        }

        tensor& operator=(const tensor& obj) {
            if (this == &obj) return *this;

            delete[] this->x;
            this->n = obj.n;
            this->x = new T[n];
            if (obj.x != nullptr)
                std::copy(obj.x, obj.x + n, x);
            return *this;
        }

        tensor<T> operator+(tensor<T>& obj) { return tensor<T>::add(*this, obj); }

        size_t size(void) const { return this->n; }
        T* raw(void) const { return this->x; }

        template<typename... Tensors>
        static tensor<T> add(const Tensors&... tensors) {
            constexpr size_t count = sizeof...(tensors); 
            static_assert((std::is_same_v<tensor<T>, Tensors> && ...), "All arguments must be tensor<T>");
            static_assert(std::is_arithmetic_v<T>, "Only arithmetic types supported");
            
            if constexpr (count == 0) 
            throw std::invalid_argument("add() needs at least one tensor");
            
            size_t tensor_size = deduceSize(tensors...);
            
            if (!((tensors.size() == tensor_size) && ...)) 
                throw std::invalid_argument("mismatch in tensor sizes");   
            
            std::array<T*, count> ts = { tensors.raw()... };
            T* h_raw_ptr = new T[count * tensor_size];
            for(size_t i = 0; i < count; i++) {
                for(size_t j = 0; j < tensor_size; j++) 
                    h_raw_ptr[i * tensor_size + j] = ts.at(i)[j];
            }

            // Debugging ONLY
            /* std::cout << std::endl << "Linearized = [";
            for(size_t i = 0; i < count; i++) {
                for(size_t j = 0; j < tensor_size; j++)
                    std::cout << h_raw_ptr[i * tensor_size + j] << ", ";
            }
            std::cout << "\b\b]" << std::endl; */    

            // device variables
            T* d_raw_ptr, *d_outptr;

            cudaMalloc(&d_raw_ptr, sizeof(T) * count * tensor_size);
            cudaMalloc(&d_outptr, sizeof(T) * tensor_size);
            cudaMemcpy(d_raw_ptr, h_raw_ptr, sizeof(T) * count * tensor_size, cudaMemcpyHostToDevice);

            tensor<T> result(tensor_size);

            int threadsperblock(256);
            int blocks = (tensor_size + threadsperblock - 1) / threadsperblock;

            kernel::add<T><<<blocks, threadsperblock>>>(d_raw_ptr, d_outptr, count, tensor_size);

            cudaMemcpy(result.raw(), d_outptr, sizeof(T) * tensor_size, cudaMemcpyDeviceToHost);

            cudaFree(d_raw_ptr);
            cudaFree(d_outptr);
            
            cudaDeviceSynchronize();
            
            return result;
        }

        ~tensor(void) {
            delete[] x;
            this->x = nullptr;
        }
};