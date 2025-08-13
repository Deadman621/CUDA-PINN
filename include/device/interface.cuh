#pragma once

#include<stdexcept>
#include<device/variables.cuh>

namespace device {
    namespace tensor {
        template<constants::device_arithmetic T>
        class interface: public variables<T> {
            
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
                interface(void) = default;
                
                interface(const interface& obj) {
                    this->n = obj.n;
                    this->dim = obj.dim;
                    this->allocate(this->n, this->dim).copy_from(obj.data, obj.shape, obj.stride, cudaMemcpyDeviceToDevice);
                } 

                interface(interface&& obj) noexcept: data_allocated{obj.data_allocated}, shape_allocated{obj.shape_allocated} {
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

                interface& allocate(size_t n, size_t dim) {
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
                
                interface& reshape(size_t dim) noexcept {
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

                interface& resize(size_t n) noexcept {      
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

                interface& reallocate(size_t n, size_t dim) noexcept { return this->resize(n).reshape(dim); }
                
                interface& copy_from(const T* x, cudaMemcpyKind kind) {
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

                const interface& copy_to(T* x, cudaMemcpyKind kind) const {
                    this->copy_to_x(x, kind);
                    return *this;                
                }

                interface& copy_from(const size_t* shape, const size_t* stride, cudaMemcpyKind kind) {                
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

                const interface& copy_to(size_t* shape, size_t* stride, cudaMemcpyKind kind) const {                
                    this->copy_to_s(shape, stride, kind);
                    return *this;     
                }
            
                /**
                * @brief copy will always be made to the internal object except if Host is destination
                */
                interface& copy_from(const T* x, const size_t* shape, const size_t* stride, cudaMemcpyKind kind) {
                    return this->copy_from(x, kind).copy_from(shape, stride, kind);
                }

                interface& copy_to(T* x, size_t* shape, size_t* stride, cudaMemcpyKind kind) {
                    return this->copy_to(x, kind).copy_to(shape, stride, kind);
                }

                const interface& copy_to(T* x, size_t* shape, size_t* stride, cudaMemcpyKind kind) const {
                    return this->copy_to(x, kind).copy_to(shape, stride, kind);
                }

                variables<T>& d_var(void) { return *this; }
                const variables<T>& d_var(void) const { return *this; }

                interface& relinquish(void) noexcept {
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

                interface& operator=(const interface& obj) noexcept {
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

                interface& operator=(interface&& obj) noexcept {
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

                ~interface(void) {
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
    }
}