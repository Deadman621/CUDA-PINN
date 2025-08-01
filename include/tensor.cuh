#pragma once

#include<iostream>
#include<thread>
#include<type_traits>
#include<optional>
#include<cstdio>
#include<sstream>
#include<unordered_set>
#include<init_tensor.h>
#include<kernel.cuh>

static const size_t OFFSET_TO_GPU = 10000; 

// type unsafe function.
template <typename First, typename... Rest>
static auto &GetFirstTensor(const First &__ax, const Rest &...__bx) { return __ax; }

template <typename T>
class tensor: public init_tensor<T> {
    private:
        kernel::device<T> device;
        bool transposed = false;

        bool mem_avail_h(void) const noexcept { return this->x; }
        bool mem_avail_d(void) const noexcept { return this->device.x; }
        virtual bool mem_avail(void) const noexcept override { return this->x && this->device.x; }

        void setup_device_memory(bool shape) {
            this->device.allocate(this->n, this->dim());
            if (shape) this->device.copy_from(this->x, this->shape.data(), this->stride.data(), cudaMemcpyHostToDevice);
            else this->device.copy_from(this->x, cudaMemcpyHostToDevice);
        }

        void realloc_device_memory(bool shape) {
            this->device.reallocate(this->n, this->dim());
            if (shape) this->device.copy_from(this->x, this->shape.data(), this->stride.data(), cudaMemcpyHostToDevice);
            else this->device.copy_from(this->x, cudaMemcpyHostToDevice);            
        }

    public:
        tensor(void) = default;
        tensor(const std::initializer_list<T>& list): init_tensor<T>(list) { this->setup_device_memory(true); }
        tensor(const std::initializer_list<init_tensor<T>>& list): init_tensor<T>(list) { this->setup_device_memory(true); }
        tensor(const T &scalar): init_tensor<T>(scalar) { this->setup_device_memory(false); }
        tensor(as_shape_t, const std::vector<size_t>& shape): init_tensor<T>(as_shape, shape) { this->setup_device_memory(true); }
        tensor(const tensor &obj): init_tensor<T>(obj), device{obj.device} {}
        tensor(tensor&& obj) noexcept: init_tensor<T>(obj), device{std::move(device)}, transposed{false} {}
        
        tensor &operator=(const tensor &obj) {
            if (this == &obj)
            return *this;
            
            init_tensor<T>::operator=(obj);
            this->realloc_device_memory(true);

            return *this;
        }
        
        tensor& operator=(tensor&& obj) noexcept {
            if (this == &obj) return *this;

            init_tensor<T>::operator=(obj);
            this->device = std::move(obj.device);
            
            return *this;
        }

        tensor& operator=(const std::initializer_list<T> &list) {
            init_tensor<T>::operator=(list);
            this->realloc_device_memory(true);

            return *this;     
        }

        tensor& operator=(const std::initializer_list<init_tensor<T>> &list) {
            init_tensor<T>::operator=(list);
            this->realloc_device_memory(true);

            return *this;     
        }

        tensor& operator=(const T& scalar) {
            init_tensor<T>::operator=(scalar);
            this->realloc_device_memory(false);

            return *this;
        }

        using init_tensor<T>::operator();

        /**
         * @brief Compare two tensors
         * @param obj a tensor object
         * @returns boolean true or false
         *
         * This overloaded operator only compares the shapes and strides of a tensor, it doens't compare it
         * element by element.
        */
        bool operator==(const tensor &obj) const noexcept { return this->shape == obj.shape && this->n == obj.n; }
        tensor<T> operator+(tensor<T> &obj) { return tensor<T>::add(*this, obj); }
        tensor<T> operator*(const tensor<T> &obj) { 
            switch(obj.dim()) {
                case 0: return tensor<T>::operator*(*obj.x);
                case 1: return tensor<T>::dot(*this, obj);
                case 2: return tensor<T>::matmul(*this, obj); 
                default: throw std::invalid_argument{"tensor::operator*: multiplication on unsupported dimension"};
            } 
        }

        tensor<T> operator*(const T& scalar) {
            if (!this->mem_avail())
                throw std::runtime_error{"tensor::operator*: cannot do arithmetic with uninitialized tensor"};

            tensor<T> result = *this;

            if (result.n > OFFSET_TO_GPU) {
                dim3 blockSize(256);
                dim3 gridSize((result.n * blockSize.x - 1) / blockSize.x);
                kernel::scalar_dist<<<gridSize, blockSize>>>(result.device.data(), scalar, result.n);
                result.device.copy_to(result.x, cudaMemcpyDeviceToHost);
            }

            else {
                for(size_t i = 0; i < result.n; i++)
                    result.x[i] *= scalar;
                result.device.copy_from(result.x, cudaMemcpyHostToDevice);
            }

            return result;
        }

        template <typename... Tensors>
        static tensor<T> add(const Tensors &...tensors) {
            constexpr size_t count = sizeof...(tensors);
            static_assert((std::is_same_v<tensor<T>, Tensors> && ...), "add: all arguments must be tensor<T>");
            static_assert(std::is_arithmetic_v<T>, "add: only arithmetic types supported");
            if constexpr (count == 0)
                throw std::invalid_argument("tensor::add: need at least one tensor");

            auto &first = GetFirstTensor(tensors...);
            auto tensor_size = first.size();
            auto tensor_shape = first.get_shape();

            if (!((tensors.mem_avail()) && ...))
                throw std::invalid_argument{"tensor::add: cannot do arithmetic with uninitialized tensor(s)"};

            if (!((tensors == first) && ...))
                throw std::invalid_argument("tensor::add: mismatch in tensor shape/size");

            tensor<T> result(as_shape, tensor_shape);
                
            std::vector<kernel::d_variables<T>> devices = {tensors.device.data()...};

            kernel::d_variables<T> *d_ptr;
            size_t mem_size = sizeof(kernel::d_variables<T>) * devices.size();

            cudaMalloc(&d_ptr, mem_size);
            cudaMemcpy(d_ptr, devices.data(), mem_size, cudaMemcpyHostToDevice);
            
            int block_size(256);
            int grid_size = (tensor_size + block_size - 1) / block_size;
            
            kernel::add<T><<<grid_size, block_size>>>(d_ptr, result.device, count, tensor_size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) 
                printf("add - kernel launch failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_ptr);
            
            result.device.copy_to(result.x, cudaMemcpyDeviceToHost);

            return result;
        }

        static tensor<T> matmul(const tensor<T>& a, const tensor<T>& b) {
            if (!a.mem_avail() || !b.mem_avail()) throw std::invalid_argument{"tensor::matmul: cannot multiply uninitialized tensors"};
            if (a.dim() != 2 || b.dim() != 2) throw std::invalid_argument{"tensor::matmul: given tensor(s) are not matrices"};
            if (a.shape[1] != b.shape[0]) throw std::invalid_argument{"tensor::matmul: invalid shapes"};
            
            size_t i = a.shape[0], j = b.shape[1], k = a.shape[1]; 
            tensor<T> result(as_shape, {i, j});
            
            dim3 block(16, 16);
            dim3 grid_size((i + block.x - 1) / block.x, (j + block.y - 1) / block.y);
            kernel::matmul<<<grid_size, block>>>(a.device.data(), b.device.data(), result.device.data(), i, j, k);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) 
                printf("matmul - kernel launch failed: %s\n", cudaGetErrorString(err));
            
            result.device.copy_to(result.x, cudaMemcpyDeviceToHost);

            return result;
        }

        static tensor<T> dot(const tensor<T>& a, const tensor<T>& b) {
            if (!a.mem_avail() || !b.mem_avail()) throw std::invalid_argument{"tensor::dot: cannot multiply uninitialized tensors"};
            if (a.dim() != 1 || b.dim() != 1) throw std::invalid_argument{"tensor::dot: given tensor(s) are not vectors"};
            if (a.n != b.n) throw std::invalid_argument{"tensor::dot: cannot perform dot operation with unmatched sizes"};
            
            tensor<T> result(0);
            size_t N = a.n;
            dim3 block(256);
            dim3 grid_size((N + block.x - 1) / block.x);
            kernel::dot<<<grid_size, block>>>(a.device.data(), b.device.data(), result.device.data(), N);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) 
            printf("dot - kernel launch failed: %s\n", cudaGetErrorString(err)); 
            
            result.device.copy_to(result.x, cudaMemcpyDeviceToHost);

            return result;
        }

        static tensor<T> einsum(const tensor<T>& a, const tensor<T>& b) {
            // Will do later...
        }

        /**
         * @brief returns a transposed tensor
         */
        static tensor<T> transpose(const tensor<T>& a, const std::initializer_list<size_t> order = {}) {
            
            tensor<T> result = a;
            size_t size = order.size();
            std::vector<size_t>& shape = result.shape;
            std::vector<size_t>& stride = result.stride;
            kernel::device<T>& device = result.device;

            if (size == 0) {
                size_t start = 0, end = shape.size() - 1;
                while(start < end) {
                    std::swap(shape[start], shape[end]);
                    std::swap(stride[start], stride[end]);
                    start++, end--;
                }
            }

            else {
                if (size != a.shape.size()) 
                    throw std::invalid_argument{"tensor::transpose: axis order must match dimensions"};

                std::unordered_set<size_t> set;

                size_t k = 0;
                for(const auto& i: order) {
                    if (i >= a.dim()) throw std::invalid_argument{"tensor::transpose: invalid order"};
                    if (!set.insert(i).second) throw std::invalid_argument{"tensor::transpose: duplicates not allowed"};
                    
                    shape[k] = a.shape[i];
                    stride[k] = a.stride[i];

                    k++;
                }
            }

            if (a.dim() > 1) result.transposed = device.transposed = true;
            device.copy_from(shape.data(), stride.data(), cudaMemcpyHostToDevice);

            return result;
        }
        
        /**
         * @brief transposes the tensor inplace
         */
        tensor<T>& transpose(const std::initializer_list<size_t> order = {}) {

            size_t size = order.size();
            
            if (size == 0) {
                size_t start = 0, end = this->shape.size() - 1;
                while(start < end) {
                    std::swap(this->shape[start], this->shape[end]);
                    std::swap(this->stride[start], this->stride[end]);
                    start++, end--;
                }
            }
            
            else {
                std::vector<size_t> new_shape; new_shape.resize(this->shape.size());
                std::vector<size_t> new_stride; new_stride.resize(this->stride.size());

                if (size != this->shape.size()) 
                    throw std::invalid_argument{"tensor::transpose: axis order must match dimensions"};

                std::unordered_set<size_t> set;

                size_t k = 0;
                for(const auto& i: order) {
                    if (i >= this->dim()) throw std::invalid_argument{"tensor::transpose: invalid order"};
                    if (!set.insert(i).second) throw std::invalid_argument{"tensor::transpose: duplicates not allowed"};
                    
                    new_shape[k] = this->shape[i];
                    new_stride[k] = this->stride[i];

                    k++;
                }

                this->shape = new_shape;
                this->stride = new_stride;
            }           

            if (this->dim() > 1) this->transposed = this->device.transposed = true;
            this->device.copy_from(this->shape.data(), this->stride.data(), cudaMemcpyHostToDevice);
            
            return *this;
        }

        ~tensor(void) { this->transposed = false; }
};