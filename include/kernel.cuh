#pragma once

namespace kernel {
    
    template<typename T>
    struct d_variables {
        T* x = nullptr;
        size_t* shape = nullptr;
        size_t* stride = nullptr;
        size_t dim = 0;
        size_t n;

        __device__ T& operator[](size_t index) {
            size_t real_index = 0;
            for(size_t i = 0; i < this->n; i++) {
                real_index += (index % this->shape[i]) * this->stride[i];
                index /= this->shape[i];
            }  

            return x[real_index];
        }

        __device__ const T& operator[](size_t index) const {
            size_t real_index = 0;
            for(size_t i = 0; i < this->n; i++) {
                real_index += (index % this->shape[i]) * this->stride[i];
                index /= this->shape[i];
            }  

            return x[real_index];
        }
    };

    template<typename T>
    class device: public d_variables<T> {
        
        private:
            bool x_allocated = false;
            bool shape_allocated = false;

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
                        "tensor::device::allocate: reallocating already allocated objects - use reallocate instead"};

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
                if (this->dim != dim && dim != 0) {
                    this->dim = dim;

                    if (this->shape) cudaFree(this->shape);
                    if (this->stride) cudaFree(this->stride);

                    size_t mem_size_s = this->dim * sizeof(size_t);
                    cudaMalloc(&this->shape, mem_size_s);
                    cudaMalloc(&this->stride, mem_size_s);
                    shape_allocated = true;
                }

                return *this;
            }

            device& resize(size_t n) {      
                if (this->n != n && n != 0) {
                    this->n = n;
                    if (this->x) cudaFree(this->x);
                    
                    size_t mem_size_x = this->n * sizeof(T);
                    cudaMalloc(&this->x, mem_size_x);  
                    cudaMemset(this->x, 0, mem_size_x);
                    x_allocated = true;
                }

                return *this;
            }

            device& reallocate(size_t n, size_t dim) {
                if (!x_allocated || !shape_allocated)
                    throw std::invalid_argument {
                        "tensor::device::reallocate: calling reallocate on unallocated variable(s) - use allocate instead"
                    };

                return this->resize(n).reshape(dim);
            }
            
            device& copy(const size_t* const x, cudaMemcpyKind kind) {
                if (!x)
                    throw std::invalid_argument{"tensor::device::copy: passing nullptr not allowed"};

                if (!x_allocated)
                    throw std::invalid_argument{"tensor::device::copy: copying to unallocated raw data (x)"};
            
                size_t mem_size_x = this->n * sizeof(T); 
                
                if (this->n > 0) cudaMemcpy(this->x, x, mem_size_x, kind);

                return *this;
            }

            device& copy(const size_t* const shape, const size_t* const stride, cudaMemcpyKind kind) {
                if (!(shape || stride))
                    throw std::invalid_argument{"tensor::device::copy: passing nullptrs not allowed"};

                if (!shape_allocated)
                    throw std::invalid_argument{"tensor::device::copy: copying to unallocated shape and stride"};

                size_t mem_size_s = this->dim * sizeof(size_t);
                
                if (this->dim > 0) {
                    cudaMemcpy(this->shape, shape, mem_size_s, kind);
                    cudaMemcpy(this->stride, stride, mem_size_s, kind);
                }

                return *this;     
            }

            device& copy(const T* const x, const size_t* const shape, const size_t* const stride, cudaMemcpyKind kind) {
                return this->copy(x, kind).copy(shape, stride, kind);
            }

            d_variables<T>& data(void) { return *this; }
            const d_variables<T>& data(void) const { return *this; }

            device& relinquish(void) const {
                if (this->x) cudaFree(this->x);
                if (this->shape) cudaFree(this->shape);
                if (this->stride) cudaFree(this->stride);

                x_allocated = shape_allocated = false;
                
                this->x = nullptr;
                this->shape = this->stride = nullptr;
                this->n = 0;
                this->dim = 0;               
            }

            device& operator=(const device& obj) {

                if (this->x) cudaFree(this->x);
                if (this->shape) cudaFree(this->shape);
                if (this->stride) cudaFree(this->stride);

                this->n = obj.n;
                this->dim = obj.dim;
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
            }
    };

    template <typename T>
    __global__ void add(const d_variables<T> *A, T* out, size_t count, size_t N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N)
            return;

        T sum = 0;
        for (size_t j = 0; j < count; j++) 
            sum += A[j][i];

        out[i] = sum;
    }

    template<typename T>
    __global__ void matmul(T* A, T* B, T* R, size_t I, size_t J, size_t K) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i >= I || j >= J) return;
                                                                  
        T sum = 0.f;
        for(size_t k = 0; k < K; k++)
            sum += A[(i * K) + k] * B[(k * J) + j];

        R[(i * J) + j] = sum;
    }

    template<typename T>
    __global__ void dot(T* A, T* B, T* R, size_t n) {
        int i = blockIdx.x + blockDim.x + threadIdx.x;
        if (i > n)
            return;
        
        *R += *A * *B; 
    }

    template<typename T>
    __global__ void scalar_dist(T* t, T s, size_t N) {
        int i = blockIdx.x + blockDim.x + threadIdx.x;
        if (i >= N) return;

        t[i] *= s;
    }

    __device__ size_t calculate_stride(size_t* stride, size_t* weights, size_t N) {
        size_t index = 0;
        for(size_t i = 0; i < N; i++)
            index += weights[i] * stride[i];
        return index;
    } 
}