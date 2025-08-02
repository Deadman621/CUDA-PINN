#pragma once

#include<iostream>
#include<thread>
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

    template<typename S, typename U>
    friend tensor<U> operator*(const S& scalar, const tensor<U>& obj);

    private:
        kernel::device<T> device;
        bool transposed = false;
        using s_size_t = typename init_tensor<T>::s_size_t;

        bool mem_avail_h(void) const noexcept { return this->x; }
        bool mem_avail_d(void) const noexcept { return this->device.x; }
        virtual bool mem_avail(void) const noexcept override { return this->x && this->device.x; }

        template<typename... Tensors>
        static bool memory_check(const Tensors &...tensors) noexcept {
            return (
                (tensors.mem_avail()) && ...
            );
        }
        
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
        tensor(const std::initializer_list<T> list): init_tensor<T>(list) { this->setup_device_memory(true); }
        tensor(const std::initializer_list<init_tensor<T>> list): init_tensor<T>(list) { this->setup_device_memory(true); }
        tensor(const T &scalar): init_tensor<T>(scalar) { this->setup_device_memory(false); }
        tensor(as_shape_t, const std::vector<size_t>& shape): init_tensor<T>(as_shape, shape) { this->setup_device_memory(true); }
        tensor(const tensor &obj): init_tensor<T>(obj), device{obj.device}, transposed{obj.transposed} {}
        tensor(tensor&& obj) noexcept: init_tensor<T>(obj), device{std::move(device)}, transposed{obj.transposed} {}
        
        tensor &operator=(const tensor &obj) {
            if (this == &obj)
            return *this;
            
            init_tensor<T>::operator=(obj);
            this->realloc_device_memory(true);
            this->transposed = obj.transposed;

            return *this;
        }
        
        tensor& operator=(tensor&& obj) noexcept {
            if (this == &obj) return *this;

            init_tensor<T>::operator=(obj);
            this->device = std::move(obj.device);
            this->transposed = obj.transposed;
            
            return *this;
        }

        tensor& operator=(const std::initializer_list<T> &list) {
            init_tensor<T>::operator=(list);
            this->realloc_device_memory(true);
            this->transposed = false;

            return *this;     
        }

        tensor& operator=(const std::initializer_list<init_tensor<T>> &list) {
            init_tensor<T>::operator=(list);
            this->realloc_device_memory(true);
            this->transposed = false;

            return *this;     
        }

        tensor& operator=(const T& scalar) {
            init_tensor<T>::operator=(scalar);
            this->realloc_device_memory(false);
            this->transposed = false;

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
        bool operator!=(const tensor &obj) const noexcept { return this->shape != obj.shape || this->n != obj.n; }
        tensor<T> operator+(tensor<T> &obj) const { return tensor<T>::add(*this, obj); }
        tensor<T> operator*(const tensor<T> &obj) const { 
            switch(obj.dim()) {
                case 0: return tensor<T>::operator*(*obj.x);
                case 1: return tensor<T>::dot(*this, obj);
                case 2: return tensor<T>::matmul(*this, obj); 
                default: throw std::invalid_argument{"tensor::operator*: multiplication on unsupported dimension"};
            } 
        }

        tensor<T> operator*(const T& scalar) const {
            if (!this->mem_avail())
                throw std::runtime_error{"tensor::operator*: cannot do arithmetic with uninitialized tensor"};

            tensor<T> result = *this;

            if (result.n > OFFSET_TO_GPU) {
                dim3 blockSize(256);
                dim3 gridSize((result.n * blockSize.x - 1) / blockSize.x);
                kernel::scalar_dist<<<gridSize, blockSize>>>(result.device.data(), scalar);
                result.device.copy_to(result.x, cudaMemcpyDeviceToHost);
            }

            else {
                for(size_t i = 0; i < result.n; i++)
                    result.x[i] *= scalar;

                result.device.copy_from(result.x, cudaMemcpyHostToDevice);
            }

            return result;
        }

        static bool broadcast_possible(const std::vector<size_t>& a, const std::vector<size_t>& b) {
            s_size_t i = static_cast<s_size_t>(a.size()) - 1;
            s_size_t j = static_cast<s_size_t>(b.size()) - 1;
            
            while(i >= 0 || j >= 0) {
                size_t x = i >= 0? a[i--]: 1;
                size_t y = j >= 0? b[j--]: 1;
                if ((x != y) && (x != 1 && y != 1)) 
                    return false;
            }

            return true;
        }

        static tensor<T> add(const tensor<T>& a, const tensor<T>& b) {
            if (!tensor<T>::memory_check(a, b)) 
                throw std::invalid_argument{"tensor::add: cannot do arithmetic with uninitialized tensor(s)"};

            if (a != b) throw std::invalid_argument("tensor::add: incompaitable shape or size");

            tensor<T> result(as_shape, a.shape);
            
            int block_size = 256;
            int grid_size = (result.n + block_size - 1) / block_size;
            kernel::add<<<grid_size, block_size>>>(result.device.data(), a.device.data(), b.device.data());

            result.device.copy_to(result.x, cudaMemcpyDeviceToHost);

            return result;
        }

        template <typename... Tensors>
        static tensor<T> add(const Tensors &...tensors) {
            constexpr size_t count = sizeof...(tensors);
            static_assert((std::is_same_v<tensor<T>, Tensors> && ...), "tensor::add: all arguments must be tensor<T>");
            static_assert(std::is_arithmetic_v<T>, "tensor::add: only arithmetic types supported");
            
            if constexpr (count == 0)
                throw std::invalid_argument("tensor::add: need at least one tensor");

            const tensor<T> &first = GetFirstTensor(tensors...);
            const std::vector<size_t> &tensor_shape = first.shape;
            size_t tensor_size = first.n;

            if (!tensor<T>::memory_check(tensors...))
                throw std::invalid_argument{"tensor::add: cannot do arithmetic with uninitialized tensor(s)"};

            if (!((tensors == first) && ...))
                throw std::invalid_argument("tensor::add: incompaitable shape or size");

            tensor<T> result(as_shape, tensor_shape);
                
            std::vector<kernel::d_variables<T>> devices = {tensors.device.data()...};

            kernel::d_variables<T> *d_ptr;
            size_t mem_size = sizeof(kernel::d_variables<T>) * devices.size();

            cudaMalloc(&d_ptr, mem_size);
            cudaMemcpy(d_ptr, devices.data(), mem_size, cudaMemcpyHostToDevice);
            
            int block_size(256);
            int grid_size = (tensor_size + block_size - 1) / block_size;
            
            kernel::add_multiple<T><<<grid_size, block_size>>>(d_ptr, result.device, count);
            cudaFree(d_ptr);
            
            result.device.copy_to(result.x, cudaMemcpyDeviceToHost);

            return result;
        }

        static tensor<T> matmul(const tensor<T>& a, const tensor<T>& b) {
            if (!tensor<T>::memory_check(a, b)) 
                throw std::invalid_argument{"tensor::matmul: cannot multiply uninitialized tensors"};
            
            if (a.dim() != 2 || b.dim() != 2) 
                throw std::invalid_argument{"tensor::matmul: given tensor(s) are not matrices"};
            
            if (a.shape[1] != b.shape[0]) throw std::invalid_argument{"tensor::matmul: invalid shapes"};
            
            size_t i = a.shape[0], j = b.shape[1], k = a.shape[1]; 
            tensor<T> result(as_shape, {i, j});
            
            dim3 block(16, 16);
            dim3 grid_size((i + block.x - 1) / block.x, (j + block.y - 1) / block.y);
            kernel::matmul<<<grid_size, block>>>(a.device.data(), b.device.data(), result.device.data(), i, j, k);
            
            result.device.copy_to(result.x, cudaMemcpyDeviceToHost);

            return result;
        }

        static tensor<T> dot(const tensor<T>& a, const tensor<T>& b) {
            if (!tensor<T>::memory_check(a, b)) 
                throw std::invalid_argument{"tensor::dot: cannot multiply uninitialized tensors"};

            if (a.dim() != 1 || b.dim() != 1) 
                throw std::invalid_argument{"tensor::dot: given tensor(s) are not vectors"};

            if (a.n != b.n) 
                throw std::invalid_argument{"tensor::dot: cannot perform dot operation with unmatched sizes"};
            
            tensor<T> result(0);
            size_t N = a.n;

            dim3 block(256);
            dim3 grid_size((N + block.x - 1) / block.x);
            kernel::dot<<<grid_size, block>>>(a.device.data(), b.device.data(), result.device.data());
            
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
                s_size_t start = 0, end = static_cast<s_size_t>(shape.size()) - 1;
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
                s_size_t start = 0, end = this->shape.size() - 1;
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

template<typename S, typename T>
tensor<T> operator*(const S& scalar, const tensor<T>& obj) {
    static_assert(std::is_arithmetic<S>::value, "tensor::operator*: non-arithmetic types not supported");
    return obj.operator*(static_cast<T>(scalar));
}