#pragma once

#include<compare>
#include<cstddef>
#include<cstdio>
#include<stdexcept>
#include<unordered_set>
#include<init_tensor.h>
#include<kernel.cuh>
#include<optional>
#include<utility>

/* using std::cout; 
using std::endl;  */

static constexpr size_t OFFSET_TO_GPU = 10000;

// type unsafe function.
template <typename First, typename... Rest>
static auto &GetFirstTensor(const First &__ax, const Rest &...__bx) { return __ax; }

template<arithmetic T>
class tensor: public init_tensor<T> {

    template<arithmetic S, arithmetic U>
    friend tensor<U> operator*(const S& scalar, const tensor<U>& obj);

    private:
        kernel::device<T> device;
        bool transposed = false;

        using typename init_tensor<T>::s_size_t;
        using typename init_tensor<T>::init_tensor_0D;
        using typename init_tensor<T>::init_tensor_1D;
        using typename init_tensor<T>::init_tensor_ND;

        [[nodiscard]] bool mem_avail_h(void) const noexcept { return this->x; }
        [[nodiscard]] bool mem_avail_d(void) const noexcept { return this->device.x; }
        [[nodiscard]] bool mem_avail(void) const noexcept override { return this->x && this->device.x; }

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
        
        tensor(const init_tensor_0D &scalar): init_tensor<T>(scalar) { this->setup_device_memory(false); }
        tensor(const init_tensor_1D list): init_tensor<T>(list) { this->setup_device_memory(true); }
        tensor(const init_tensor_ND list): init_tensor<T>(list) { this->setup_device_memory(true); }
        
        tensor(as_shape_t, const std::vector<size_t>& shape): init_tensor<T>(as_shape, shape) { 
            this->setup_device_memory(true); 
        }
        
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

        tensor& operator=(const init_tensor_1D &list) {
            init_tensor<T>::operator=(list);
            this->realloc_device_memory(true);
            this->transposed = false;

            return *this;     
        }

        tensor& operator=(const init_tensor_ND &list) {
            init_tensor<T>::operator=(list);
            this->realloc_device_memory(true);
            this->transposed = false;

            return *this;     
        }

        tensor& operator=(const init_tensor_0D& scalar) {
            init_tensor<T>::operator=(scalar);
            this->realloc_device_memory(false);
            this->transposed = false;

            return *this;
        }

        using init_tensor<T>::operator();

        tensor<T>& resize(const std::vector<size_t>& shape) {
            init_tensor<T>::resize(shape);
            this->realloc_device_memory(true);
            this->transposed = false;
            return *this;
        }

        /**
         * @brief Compare two tensors
         * @param obj a tensor object
         * @returns boolean true or false
         *
         * This overloaded operator only compares the shapes and strides of a tensor, it doens't compare it
         * element by element.
        */
        inline bool operator==(const tensor &obj) const noexcept { return this->shape == obj.shape && this->n == obj.n; }
        inline bool operator!=(const tensor &obj) const noexcept { return this->shape != obj.shape || this->n != obj.n; }
        tensor<T> operator+(tensor<T> &obj) const { return tensor<T>::add(*this, obj); }
        tensor<T>& operator+=(const tensor<T> &obj) { return this->add(obj); }
        
        tensor<T> operator*(const tensor<T> &obj) const { 
            if (!tensor<T>::memory_check(*this, obj))
                throw std::runtime_error{"tensor::operator*: cannot do arithmetic with uninitialized tensor(s)"};
    
            switch(obj.dim()) {
                case 0: return tensor<T>::operator*(*obj.x);
                case 1: return tensor<T>::dot(*this, obj);
                case 2: return tensor<T>::matmul(*this, obj); 
                default: throw std::invalid_argument{"tensor::operator*: multiplication on unsupported dimension"};
            } 
        }
        
        tensor<T> operator*(const T& scalar) const {
            if (!tensor<T>::memory_check(*this))
                throw std::runtime_error{"tensor::operator*: cannot do arithmetic with uninitialized tensor"};

            tensor<T> result = *this;

            if (result.n > OFFSET_TO_GPU) {
                dim3 blockSize(256);
                dim3 grid_size((result.n * blockSize.x - 1) / blockSize.x);
                kernel::scalar_dist<<<grid_size, blockSize>>>(result.device.data(), scalar);
                result.device.copy_to(result.x, cudaMemcpyDeviceToHost);
            }

            else {
                for(size_t i = 0; i < result.n; i++)
                    result.x[i] *= scalar;

                result.device.copy_from(result.x, cudaMemcpyHostToDevice);
            }

            return result;
        }

        tensor<T>& operator*=(const tensor<T> &obj) { 
            if (!tensor<T>::memory_check(*this, obj))
                throw std::runtime_error{"tensor::operator*=: cannot do arithmetic with uninitialized tensor(s)"};

            switch(obj.dim()) {
                case 0: return tensor<T>::operator*=(*obj.x);
                case 1: return tensor<T>::dot(obj);
                case 2: return tensor<T>::matmul(obj); 
                default: throw std::invalid_argument{"tensor::operator*: multiplication on unsupported dimension"};
            }             
        }

        tensor<T>& operator*=(const T& scalar) {
            if (!tensor<T>::memory_check(*this))
                throw std::runtime_error{"tensor::operator*=: cannot do arithmetic with uninitialized tensor"};
            
            if (this->n > OFFSET_TO_GPU) {
                dim3 block_size(256);
                dim3 grid_size((this->n * block_size.x - 1) / block_size.x);
                kernel::scalar_dist<<<grid_size, block_size>>>(this->device.data(), scalar);
                this->device.copy_to(this->x, cudaMemcpyDeviceToHost);
            }     
            
            else {
                for(size_t i = 0; i < this->n; i++)
                    this->x[i] *= scalar;

                this->device.copy_from(this->x, cudaMemcpyHostToDevice);
            }

            return *this;
        }

        tensor& assign(const init_tensor_0D& scalar) {
            init_tensor<T>::assign(scalar);
            this->device.copy_from(this->x, cudaMemcpyHostToDevice);
            return *this;
        }

        tensor& assign(const init_tensor_1D list) {
            init_tensor<T>::assign(list);
            this->device.copy_from(this->x, cudaMemcpyHostToDevice);
            return *this;
        }  

        tensor& assign(const init_tensor_ND list) {
            init_tensor<T>::assign(list);
            this->device.copy_from(this->x, cudaMemcpyHostToDevice);
            return *this;
        }

        using const_ptr = const tensor<T>*;
        using op_tensor = std::optional<std::pair<const_ptr, const_ptr>>;

        static op_tensor broadcast_order(const tensor<T>&& a, const tensor<T>&& b) = delete;
        static op_tensor broadcast_order(const tensor<T>&& a, const tensor<T>&  b) = delete;
        static op_tensor broadcast_order(const tensor<T>&  a, const tensor<T>&& b) = delete;

        static op_tensor broadcast_order(const tensor<T>&  a, const tensor<T>&  b) {
            if (!memory_check(a, b)) 
                throw std::invalid_argument{"tensor::broadcast_order"};

            const std::vector<size_t>& s_a = a.shape;
            const std::vector<size_t>& s_b = b.shape;
        
            auto i = static_cast<s_size_t>(s_a.size()) - 1;
            auto j = static_cast<s_size_t>(s_b.size()) - 1;
            size_t m = 0, n = 0;

            std::strong_ordering cmp = j <=> i;
            while(i >= 0 || j >= 0) {
                size_t x = i >= 0? s_a[i--]: 1;
                size_t y = j >= 0? s_b[j--]: 1;
                if (cmp == 0) cmp = s_b[n++] <=> s_a[m++]; 
                if ((x != y) && (x != 1 && y != 1)) 
                    return std::nullopt;
            }   

            if (cmp == std::strong_ordering::greater) return std::make_pair(&b, &a); 
            if (cmp == std::strong_ordering::less   ) return std::make_pair(&a, &b);
            else return std::make_pair(&a, &b);
        }

        tensor<T>& add(const tensor<T>& t) {
            if (!tensor<T>::memory_check(*this, t)) 
                throw std::invalid_argument{"tensor::add: cannot do arithmetic with uninitialized tensor(s)"};

            auto broadcast_check = tensor<T>::broadcast_order(*this, t);
            if (*this != t && !broadcast_check.has_value())
                throw std::invalid_argument{"tensor::add: incompaitable shape or size"};
        
            auto x = broadcast_check.value().first;
            if (x != this) throw std::invalid_argument{"tensor::add: incompaitable shape or size"};

            int block_size = 256;
            int grid_size = (x->n + block_size - 1) / block_size;

            kernel::add<<<grid_size, block_size>>>(this->device.data(), t.device.data(), this->device.data());
            this->device.copy_to(this->x, cudaMemcpyDeviceToHost);

            return *this;
        }   

        static tensor<T> add(const tensor<T>& a, const tensor<T>& b) {
            if (!tensor<T>::memory_check(a, b)) 
                throw std::invalid_argument{"tensor::add: cannot do arithmetic with uninitialized tensor(s)"};

            auto broadcast_check = tensor<T>::broadcast_order(a, b);
            if (a != b && !broadcast_check.has_value())
                throw std::invalid_argument("tensor::add: incompaitable shape or size"); 

            tensor<T> result(as_shape, broadcast_check.value().first->shape);
            
            int block_size = 256;
            int grid_size = (result.n + block_size - 1) / block_size;
            kernel::add<<<grid_size, block_size>>>(a.device.data(), b.device.data(), result.device.data());

            result.device.copy_to(result.x, cudaMemcpyDeviceToHost);

            return result;
        }

        //broadcasting not available here
        template<typename... Tensors>
        static tensor<T> add(const Tensors &...tensors) {
            constexpr size_t count = sizeof...(tensors);
            static_assert((std::is_same_v<tensor<T>, Tensors> && ...), "tensor::add: all arguments must be tensor<T>");            
            if constexpr (count == 0)
                throw std::invalid_argument("tensor::add: need at least one tensor");

            const tensor<T> &first = GetFirstTensor(tensors...);
            const std::vector<size_t> &tensor_shape = first.shape;
            const auto tensor_size = static_cast<s_size_t>(first.n);

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

        tensor<T>& matmul(const tensor<T>& t) {
            if (!tensor<T>::memory_check(*this, t)) 
                throw std::invalid_argument{"tensor::matmul: cannot multiply uninitialized tensors"};        
                
            if (this->dim() != 2 || t.dim() != 2) 
                throw std::invalid_argument{"tensor::matmul: given tensor(s) are not matrices"};

            if (this->shape[1] != t.shape[0]) throw std::invalid_argument{"tensor::matmul: invalid shapes"};  
            
            size_t i = this->shape[0], j = t.shape[1], k = this->shape[1]; 
            tensor<T> temp = *this;

            this->resize({i, j});
            
            dim3 block_size(16, 16);
            dim3 grid_size((i + block_size.x - 1) / block_size.x, (j + block_size.y - 1) / block_size.y);
            kernel::matmul<<<grid_size, block_size>>>(temp.device.data(), t.device.data(), this->device.data(), i, j, k);
            
            this->device.copy_to(this->x, cudaMemcpyDeviceToHost);

            return *this;            
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

        tensor& dot(const tensor<T>& t) {
            if (!tensor<T>::memory_check(*this, t))
                throw std::invalid_argument{"tensor::dot: cannot multiply uninitialized tensors"};

            if (this->dim() != 1 || t.dim() != 1)
                throw std::invalid_argument{"tensor::dot: given tensor(s) are not vectors"};

            if (this->n != t.n)
                throw std::invalid_argument{"tensor::dot: cannot perform dot operation with unmatched sizes"};

            tensor<T> temp = *this;
            this->resize({});

            dim3 block_size(256);
            dim3 grid_size((temp.n + block_size.x - 1) / block_size.x);
            kernel::dot<<<grid_size, block_size>>>(temp.device.data(), t.device.data(), this->device.data());
            
            this->device.copy_to(this->x, cudaMemcpyDeviceToHost);

            return *this;
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
                    ++start, --end;
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

        ~tensor(void) override { this->transposed = false; }
};

template<arithmetic S, arithmetic T>
tensor<T> operator*(const S& scalar, const tensor<T>& obj) {
    static_assert(std::is_arithmetic_v<S>, "tensor::operator*: non-arithmetic types not supported");
    return obj.operator*(static_cast<T>(scalar));
}

template<arithmetic T>
tensor<T> make_tensor(const typename tensor<T>::init_tensor_0D& scalar) { return tensor<T>(scalar); }

template<arithmetic T>
tensor<T> make_tensor(const typename tensor<T>::init_tensor_1D list) { return tensor<T>(list); }

template<arithmetic T>
tensor<T> make_tensor(const typename tensor<T>::init_tensor_ND list) { return tensor<T>(list); }