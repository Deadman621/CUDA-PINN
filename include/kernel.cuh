#pragma once

namespace kernel {
    
    template<typename T>
    struct d_variables {
        T* x = nullptr;
        size_t* shape = nullptr;
        size_t* stride = nullptr;
        size_t dim = 0;
        size_t n;
        bool transposed = false;

        __device__ T& operator[](size_t index) {
            if (transposed) {
                size_t real_index = 0;
                for(int i = this->dim - 1; i >= 0; i--) {
                    real_index += ((index % this->shape[i]) * this->stride[i]);
                    index /= this->shape[i];
                }  

                return x[real_index];                
            }

            return x[index];
        }

        __device__ const T& operator[](size_t index) const {
            if (transposed) {
                size_t real_index = 0;
                for(int i = this->dim - 1; i >= 0; i--) {
                    real_index += ((index % this->shape[i]) * this->stride[i]);
                    index /= this->shape[i];
                }  

                return x[real_index];                
            }

            return x[index];
        }

        __device__ T& operator*() { return *this->x; }
        __device__ const T& operator*() const { return *this->x; }

        __device__ size_t calculate_stride(size_t* weights) {
            size_t index = 0;
            for(size_t i = 0; i < this->n; i++)
                index += weights[i] * this->stride[i];
            return index;
        } 
    };

    template<typename T>
    class device: public d_variables<T> {
        
        private:
            bool x_allocated = false;
            bool shape_allocated = false;

            void copy_to_x(T* x, cudaMemcpyKind kind) const {
                if (this->n > 0) {
                    if (!x) throw std::invalid_argument{"device::copy_to: passing nullptr not allowed"};
                    if (!x_allocated) throw std::invalid_argument{"device::copy_to: copying to unallocated raw data (x)"};
                
                    size_t mem_size_x = this->n * sizeof(T); 
                    const T *src = this->x; T *dst = x;

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
                this->allocate(this->n, this->dim).copy(obj.x, obj.shape, obj.stride, cudaMemcpyDeviceToDevice);
            } 

            device& allocate(size_t n, size_t dim) {
                if (x_allocated || shape_allocated)
                    throw std::invalid_argument{
                        "device::allocate: reallocating already allocated objects - use reallocate instead"};

                this->n = n;
                this->dim = dim;

                size_t mem_size_x = this->n * sizeof(T); 
                size_t mem_size_s = this->dim * sizeof(size_t);

                if (this->n > 0) { 
                    cudaMalloc(&this->x, mem_size_x);
                    cudaMemset(this->x, 0, mem_size_x);
                    x_allocated = true;
                }

                if (this->dim > 0) {
                    cudaMalloc(&this->shape, mem_size_s); 
                    cudaMalloc(&this->stride, mem_size_s);
                    shape_allocated = true;
                }

                return *this;
            }
            
            device& reshape(size_t dim) {
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

            device& resize(size_t n) {      
                if (n == 0) {
                    if (this->x) cudaFree(this->x);                    
                    x_allocated = false;
                    this->x = nullptr;
                    this->n = 0;
                }

                else if (this->n != n) {
                    if (this->x) cudaFree(this->x);

                    this->n = n;
                    size_t mem_size_x = this->n * sizeof(T);
                    cudaMalloc(&this->x, mem_size_x);  
                    cudaMemset(this->x, 0, mem_size_x);
                    x_allocated = true;
                }

                return *this;
            }

            device& reallocate(size_t n, size_t dim) {
                return this->resize(n).reshape(dim);
            }
            
            device& copy_from(const T* x, cudaMemcpyKind kind) {
                if (this->n > 0) {
                    if (!x) throw std::invalid_argument{"device::copy_from: passing nullptr not allowed"};
                    if (!x_allocated) throw std::invalid_argument{"device::copy_from: copying to unallocated raw data (x)"};
                
                    switch(kind) {
                        case cudaMemcpyHostToDevice:                          
                        case cudaMemcpyDeviceToDevice:
                            break;

                        default:
                            throw std::invalid_argument{"device::copy_from: invalid cudaMemcpyKind"};
                            break;                        
                    }
                    
                    size_t mem_size_x = this->n * sizeof(T); 
                    const T *src = x; T *dst = this->x;

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

            d_variables<T>& data(void) { return *this; }
            const d_variables<T>& data(void) const { return *this; }

            device& relinquish(void) {
                if (this->x) cudaFree(this->x);
                if (this->shape) cudaFree(this->shape);
                if (this->stride) cudaFree(this->stride);

                x_allocated = shape_allocated = false;
                
                this->x = nullptr;
                this->shape = this->stride = nullptr;
                this->n = 0;
                this->dim = 0;               
                this->transposed = false;

                return *this;
            }

            device& operator=(const device& obj) {

                if (this->x) cudaFree(this->x);
                if (this->shape) cudaFree(this->shape);
                if (this->stride) cudaFree(this->stride);

                this->n = obj.n;
                this->dim = obj.dim;
                this->transposed = obj.transposed;
                this->allocate(this->n, this->dim).copy(obj.x, obj.shape, obj.stride, cudaMemcpyDeviceToDevice);
            }

            ~device(void) {
                if (this->x) cudaFree(this->x);
                if (this->shape) cudaFree(this->shape);
                if (this->stride) cudaFree(this->stride);

                x_allocated = shape_allocated = false;
                
                this->x = nullptr;
                this->shape = this->stride = nullptr;
                this->n = 0;
                this->dim = 0;
                this->transposed = false;
            }
    };

    template <typename T>
    __global__ void add(const d_variables<T> *A, d_variables<T> out, size_t count, size_t N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N)
            return;

        T sum = 0;
        for (size_t j = 0; j < count; j++) 
            sum += A[j].operator[](i);

        out.x[i] = sum;
    }

    template<typename T>
    __global__ void matmul(const d_variables<T> A, const d_variables<T> B, d_variables<T> R, size_t I, size_t J, size_t K) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i >= I || j >= J) return;
                                                                  
        T sum = 0.f;
        for(size_t k = 0; k < K; k++)
            sum += A[(i * K) + k] * B[(k * J) + j];

        R.x[(i * J) + j] = sum;
    }

    template<typename T>
    __global__ void dot(const d_variables<T> A, const d_variables<T> B, d_variables<T> R, size_t N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N)
            return;
        
        T result = A[i] * B[i];
        atomicAdd(R.x, result);
    }

    template<typename T>
    __global__ void scalar_dist(d_variables<T> t, T s, size_t N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        t.x[i] *= s;
    }
}