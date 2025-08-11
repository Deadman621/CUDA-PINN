#pragma once

#include<stdexcept>
#include<device_math.h>

namespace kernel {

    using device_math::device_arithmetic;
    template<device_arithmetic T>
    struct d_variables {
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

    template<device_arithmetic T>
    class device: public d_variables<T> {
        
        private:
            bool data_allocated = false;
            bool shape_allocated = false;

            void copy_to_x(T* x, cudaMemcpyKind kind) const {
                if (this->n > 0) {
                    if (!x) throw std::invalid_argument{"device::copy_to: passing nullptr not allowed"};
                    if (!data_allocated) throw std::invalid_argument{"device::copy_to: copying to unallocated raw data (x)"};
                
                    size_t mem_size_x = this->n * sizeof(T); 
                    const T *src = this->data; T *dst = x;

                    switch(kind) {                            
                        case cudaMemcpyDeviceToHost:
                        case cudaMemcpyDeviceToDevice:
                            break;

                        default:
                            throw std::invalid_argument{"device::copy_to: invalid cudaMemcpyKind"};
                            break;                        
                    }
                    
                    cudaMemcpy(dst, src, mem_size_x, kind);
                }
            }

            void copy_to_s(size_t* shape, size_t* stride, cudaMemcpyKind kind) {
                if (this->dim > 0) {
                    if (!(shape || stride))
                        throw std::invalid_argument{"device::copy_to: passing nullptrs not allowed"};

                    if (!shape_allocated)
                        throw std::invalid_argument{"device::copy_to: copying to unallocated shape and stride"};

                    size_t mem_size_s = this->dim * sizeof(size_t);
                    const size_t *shape_src = this->shape, *stride_src = this->stride;
                    size_t *shape_dst = shape, *stride_dst = stride;

                    switch(kind) {
                        case cudaMemcpyDeviceToHost:
                        case cudaMemcpyDeviceToDevice:

                        default:
                            throw std::invalid_argument{"device::copy_to: invalid cudaMemcpyKind"};
                            break;
                    }

                    cudaMemcpy(shape_dst, shape_src, mem_size_s, kind);
                    cudaMemcpy(stride_dst, stride_src, mem_size_s, kind);
                }
            }

        public:
            device(void) = default;
            
            device(const device& obj) {
                this->n = obj.n;
                this->dim = obj.dim;
                this->allocate(this->n, this->dim).copy_from(obj.data, obj.shape, obj.stride, cudaMemcpyDeviceToDevice);
            } 

            device(device&& obj) noexcept: data_allocated{obj.data_allocated}, shape_allocated{obj.shape_allocated} {
                this->data = obj.data;
                this->n = obj.n;
                this->dim = obj.dim;
                this->shape = obj.shape;
                this->stride = obj.stride;
                this->transposed = obj.transposed;

                obj.data = nullptr;
                obj.n = 0;
                obj.dim = 0;
                obj.shape = nullptr;
                obj.stride = nullptr;
                obj.transposed = false;
            }

            device& allocate(size_t n, size_t dim) {
                if (data_allocated || shape_allocated)
                    throw std::invalid_argument{
                        "device::allocate: allocating already allocated objects - use reallocate instead"};

                this->n = n;
                this->dim = dim;

                size_t mem_size_x = this->n * sizeof(T); 
                size_t mem_size_s = this->dim * sizeof(size_t);

                if (this->n > 0) { 
                    cudaMalloc(&this->data, mem_size_x);
                    cudaMemset(this->data, 0, mem_size_x);
                    data_allocated = true;
                }

                if (this->dim > 0) {
                    cudaMalloc(&this->shape, mem_size_s); 
                    cudaMalloc(&this->stride, mem_size_s);
                    shape_allocated = true;
                }

                return *this;
            }
            
            device& reshape(size_t dim) noexcept {
                if (dim == 0) {                   
                    if (this->shape) cudaFree(this->shape);
                    if (this->stride) cudaFree(this->stride);

                    shape_allocated = false;
                    
                    this->dim = 0;               
                    this->transposed = false;
                    this->shape = this->stride = nullptr;
                }

                else if (this->dim != dim) {
                    if (this->shape) cudaFree(this->shape);
                    if (this->stride) cudaFree(this->stride);

                    this->dim = dim;
                    size_t mem_size_s = this->dim * sizeof(size_t);
                    cudaMalloc(&this->shape, mem_size_s);
                    cudaMalloc(&this->stride, mem_size_s);
                    shape_allocated = true;
                }

                return *this;
            }

            device& resize(size_t n) noexcept {      
                if (n == 0) {
                    if (this->data) cudaFree(this->data);                    
                    data_allocated = false;
                    this->data = nullptr;
                    this->n = 0;
                }

                else if (this->n != n) {
                    if (this->data) cudaFree(this->data);

                    this->n = n;
                    size_t mem_size_x = this->n * sizeof(T);
                    cudaMalloc(&this->data, mem_size_x);  
                    cudaMemset(this->data, 0, mem_size_x);
                    data_allocated = true;
                }

                return *this;
            }

            device& reallocate(size_t n, size_t dim) noexcept { return this->resize(n).reshape(dim); }
            
            device& copy_from(const T* x, cudaMemcpyKind kind) {
                if (this->n > 0) {
                    if (!x) throw std::invalid_argument{"device::copy_from: passing nullptr not allowed"};
                    if (!data_allocated) throw std::invalid_argument{"device::copy_from: copying to unallocated raw data (x)"};
                
                    switch(kind) {
                        case cudaMemcpyHostToDevice:                          
                        case cudaMemcpyDeviceToDevice:
                            break;

                        default:
                            throw std::invalid_argument{"device::copy_from: invalid cudaMemcpyKind"};
                            break;                        
                    }
                    
                    size_t mem_size_x = this->n * sizeof(T); 
                    const T *src = x; T *dst = this->data;

                    cudaMemcpy(dst, src, mem_size_x, kind);
                }

                return *this;
            } 

            const device& copy_to(T* x, cudaMemcpyKind kind) const {
                this->copy_to_x(x, kind);
                return *this;                
            }

            device& copy_from(const size_t* shape, const size_t* stride, cudaMemcpyKind kind) {                
                if (this->dim > 0) {
                    if (!(shape || stride))
                        throw std::invalid_argument{"device::copy_from: passing nullptrs not allowed"};

                    if (!shape_allocated)
                        throw std::invalid_argument{"device::copy_from: copying to unallocated shape and stride"};

                    size_t mem_size_s = this->dim * sizeof(size_t);
                    const size_t *shape_src = shape, *stride_src = stride;
                    size_t *shape_dst = this->shape, *stride_dst = this->stride;

                    switch(kind) {
                        case cudaMemcpyHostToDevice: 
                        case cudaMemcpyDeviceToDevice:
                            break;

                        default:
                            throw std::invalid_argument{"device::copy_from: invalid cudaMemcpyKind"};
                            break;
                    }

                    cudaMemcpy(shape_dst, shape_src, mem_size_s, kind);
                    cudaMemcpy(stride_dst, stride_src, mem_size_s, kind);
                }

                return *this;     
            }

            const device& copy_to(size_t* shape, size_t* stride, cudaMemcpyKind kind) const {                
                this->copy_to_s(shape, stride, kind);
                return *this;     
            }
        
            /**
             * @brief copy will always be made to the internal object except if Host is destination
             */
            device& copy_from(const T* x, const size_t* shape, const size_t* stride, cudaMemcpyKind kind) {
                return this->copy_from(x, kind).copy_from(shape, stride, kind);
            }

            device& copy_to(T* x, size_t* shape, size_t* stride, cudaMemcpyKind kind) {
                return this->copy_to(x, kind).copy_to(shape, stride, kind);
            }

            const device& copy_to(T* x, size_t* shape, size_t* stride, cudaMemcpyKind kind) const {
                return this->copy_to(x, kind).copy_to(shape, stride, kind);
            }

            d_variables<T>& d_var(void) { return *this; }
            const d_variables<T>& d_var(void) const { return *this; }

            device& relinquish(void) noexcept {
                if (this->data) cudaFree(this->data);
                if (this->shape) cudaFree(this->shape);
                if (this->stride) cudaFree(this->stride);

                data_allocated = shape_allocated = false;
                
                this->data = nullptr;
                this->shape = this->stride = nullptr;
                this->n = 0;
                this->dim = 0;               
                this->transposed = false;

                return *this;
            }

            device& operator=(const device& obj) noexcept {
                if (this == &obj) return *this;
                if (this->data) cudaFree(this->data);
                if (this->shape) cudaFree(this->shape);
                if (this->stride) cudaFree(this->stride);

                this->n = obj.n;
                this->dim = obj.dim;
                this->transposed = obj.transposed;
                this->allocate(this->n, this->dim).copy_from(obj.data, obj.shape, obj.stride, cudaMemcpyDeviceToDevice);

                return *this;
            }

            device& operator=(device&& obj) noexcept {
                if (this == &obj) return *this;

                this->data = obj.data;
                this->n = obj.n;
                this->dim = obj.dim;
                this->shape = obj.shape;
                this->stride = obj.stride;
                this->transposed = obj.transposed;

                obj.data = nullptr;
                obj.n = 0;
                obj.dim = 0;
                obj.shape = nullptr;
                obj.stride = nullptr;
                obj.transposed = false;

                return *this;
            }

            ~device(void) {
                if (this->data) cudaFree(this->data);
                if (this->shape) cudaFree(this->shape);
                if (this->stride) cudaFree(this->stride);

                data_allocated = shape_allocated = false;
                
                this->data = nullptr;
                this->shape = this->stride = nullptr;
                this->n = 0;
                this->dim = 0;
                this->transposed = false;
            }
    };

    // Use obj.x[index] if you're sure that obj is not transposed

    template<device_arithmetic T>
    __global__ void add(const d_variables<T> a, const d_variables<T> b, d_variables<T> out) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= out.n) 
            return;

        out[i] = a[i % a.n] + b[i % b.n];
    }

    template<device_arithmetic T>
    __global__ void add_multiple(const d_variables<T> *A, d_variables<T> out, size_t count) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= out.n)
            return;

        T sum = 0;
        for (size_t j = 0; j < count; j++) 
            sum += A[j][i];

        out.data[i] = sum;
    }

    template<device_arithmetic T>
    __global__ void sub(const d_variables<T> a, const d_variables<T> b, d_variables<T> out) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= out.n) 
            return;

        out[i] = a[i % a.n] - b[i % b.n];
    }

    template<device_arithmetic T>
    __global__ void mul(const d_variables<T> A, const d_variables<T> B, d_variables<T> R) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= A.n)
            return;
        
        R[i] = A[i % A.n] * B[i % B.n];
    }

    template<device_arithmetic T>
    __global__ void div(const d_variables<T> A, const d_variables<T> B, d_variables<T> R) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= A.n)
            return;
        
        R[i] = A[i % A.n] / B[i % B.n];
    }

    template<device_arithmetic T>
    __global__ void matmul(const d_variables<T> A, const d_variables<T> B, d_variables<T> R, size_t I, size_t J, size_t K) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i >= I || j >= J) return;
                                                                  
        T sum = 0.f;
        for(size_t k = 0; k < K; k++)
            sum += A[(i * K) + k] * B[(k * J) + j];

        R[(i * J) + j] = sum;
    }

    template<device_arithmetic T>
    __global__ void dot(const d_variables<T> A, const d_variables<T> B, d_variables<T> R) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= A.n)
            return;
        
        T result = A[i] * B[i];
        atomicAdd(R.data, result);
    }
}