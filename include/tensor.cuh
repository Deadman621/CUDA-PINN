#pragma once

#include<cstdio>
#include<utility>
#include<compare>
#include<cstddef>
#include<optional>
#include<unordered_set>
#include<host/runtime.h>
#include<device/kernel.cuh>
#include<host/init_tensor.h>

/* using std::cout; 
using std::endl;  */

using device_constants::BLOCK_SIZE;
//using device_constants::OFFSET_TO_GPU;

template<typename First, typename... Rest>
static auto &GetFirstTensor(const First &__ax, const Rest &...__bx) { return __ax; }

template<arithmetic T>
class tensor: public init_tensor<T> {
    private:
        mutable kernel::device<T> device;
        mutable bool host_dirty = false;
        bool transposed = false;

        using typename init_tensor<T>::data_t;
        using typename init_tensor<T>::shape_t;
        using typename init_tensor<T>::s_size_t;
        using typename init_tensor<T>::init_tensor_0D;
        using typename init_tensor<T>::init_tensor_1D;
        using typename init_tensor<T>::init_tensor_ND;

        [[nodiscard]] bool mem_avail_h(void) const noexcept { return this->data.get(); }
        [[nodiscard]] bool mem_avail_d(void) const noexcept { return this->device.x; }
        [[nodiscard]] bool mem_avail(void) const noexcept override { return this->data.get() && this->device.data; }

        template<typename... Tensors>
        static bool memory_check(const Tensors &...tensors) noexcept {
            return (
                (tensors.mem_avail()) && ...
            );
        }
        
        void setup_device_memory(bool shape) {
            this->device.allocate(this->n, this->dim());
            if (shape) this->device.copy_from(this->data.get(), this->shape.data(), this->stride.data(), cudaMemcpyHostToDevice);
            else this->device.copy_from(this->data.get(), cudaMemcpyHostToDevice);
            this->host_dirty = false;    
        }

        void realloc_device_memory(bool shape) {
            this->device.reallocate(this->n, this->dim());
            if (shape) this->device.copy_from(this->data.get(), this->shape.data(), this->stride.data(), cudaMemcpyHostToDevice);
            else this->device.copy_from(this->data.get(), cudaMemcpyHostToDevice);         
            this->host_dirty = false;
        }

        bool sync_device(void) const { 
            if (!this->host_dirty) return false;
            this->device.copy_from(this->data.get(), cudaMemcpyHostToDevice); 
            this->host_dirty = false;
            return true;
        }
        
        template<typename... Tensors>
        static size_t sync_device(const Tensors &...tensors) { return (static_cast<size_t>(tensors.sync_device()) + ...); } 
        
    public:
        tensor(void) = default;
        
        tensor(const init_tensor_0D sclr): init_tensor<T>(sclr) { this->setup_device_memory(false); }
        tensor(const init_tensor_1D list): init_tensor<T>(list) { this->setup_device_memory(true);  }
        tensor(const init_tensor_ND list): init_tensor<T>(list) { this->setup_device_memory(true);  }
        
        tensor(as_shape_t, const shape_t& shape): init_tensor<T>(as_shape, shape) { 
            this->setup_device_memory(true); 
        }
        
        tensor(const tensor &obj)
        :init_tensor<T>(obj), transposed{obj.transposed}, host_dirty{obj.host_dirty} {
            if (this->host_dirty) return;
            this->device = obj.device;
        }
        
        tensor(tensor&& obj) noexcept
        :init_tensor<T>(obj), device{std::move(device)}, transposed{obj.transposed}, host_dirty{obj.host_dirty} {}
        
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
            this->host_dirty = obj.host_dirty;
            
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
        using init_tensor<T>::operator[];

        template<typename... Indices>
        T& operator()(Indices... indices) {
            this->host_dirty = true;
            return init_tensor<T>::operator()(indices...);
        }

        T& operator[](size_t index) noexcept {
            this->host_dirty = true;
            return init_tensor<T>::operator[](index);
        }
        
        inline bool operator!=(const tensor &obj) const noexcept { return this->shape != obj.shape || this->n != obj.n; }
        inline bool operator==(const tensor &obj) const noexcept { return this->shape == obj.shape && this->n == obj.n; }
        inline bool operator<=(const tensor &obj) const noexcept { return this->shape <= obj.shape && this->n <= obj.n; }
        inline bool operator>=(const tensor &obj) const noexcept { return this->shape >= obj.shape && this->n >= obj.n; }
        inline bool operator< (const tensor &obj) const noexcept { return this->shape <  obj.shape && this->n <  obj.n; }
        inline bool operator> (const tensor &obj) const noexcept { return this->shape >  obj.shape && this->n >  obj.n; }
        
        inline bool operator!(void) const noexcept { return !this->mem_avail(); }
        
        inline tensor<T> operator+(const tensor<T> &obj) const { return tensor<T>::add(*this, obj); }
        inline tensor<T> operator-(const tensor<T> &obj) const { return tensor<T>::sub(*this, obj); }
        inline tensor<T> operator*(const tensor<T> &obj) const { return tensor<T>::mul(*this, obj); }
        inline tensor<T> operator/(const tensor<T> &obj) const { return tensor<T>::div(*this, obj); }

        inline tensor<T>& operator+=(const tensor<T> &obj) { return this->add(obj); }
        inline tensor<T>& operator-=(const tensor<T> &obj) { return this->sub(obj); }
        inline tensor<T>& operator*=(const tensor<T> &obj) { return this->mul(obj); }
        inline tensor<T>& operator/=(const tensor<T> &obj) { return this->div(obj); }

        inline tensor<T>& operator++(void) { return this->add(tensor<T>(1)); }
        inline tensor<T>& operator--(void) { return this->sub(tensor<T>(1)); }

        inline tensor<T> operator++(int) noexcept { 
            tensor<T> temp = *this;
            this->add(tensor<T>(1));
            return temp;
        }

        inline tensor<T> operator--(int) noexcept {
            tensor<T> temp = *this;
            this->sub(tensor<T>(1));
            return temp;            
        }
        
        using const_ptr = const tensor<T>*;
        using op_tensor = std::optional<std::pair<const_ptr, const_ptr>>;

        static op_tensor broadcast_order(const tensor<T>&& a, const tensor<T>&& b) = delete;
        static op_tensor broadcast_order(const tensor<T>&& a, const tensor<T>&  b) = delete;
        static op_tensor broadcast_order(const tensor<T>&  a, const tensor<T>&& b) = delete;

        using init_tensor<T>::at;

        template<typename... Indices>
        T& at(Indices... indices) {
            this->host_dirty = true;
            return init_tensor<T>::at(indices...);
        }

        tensor<T>& resize(const shape_t& shape) {
            init_tensor<T>::resize(shape);
            this->realloc_device_memory(true);
            this->transposed = false;

            return *this;
        }

        tensor& assign(const init_tensor_0D& scalar) {
            init_tensor<T>::assign(scalar);
            this->device.copy_from(this->data.get(), cudaMemcpyHostToDevice);
            this->host_dirty = false;
            return *this;
        }

        tensor& assign(const init_tensor_1D list) {
            init_tensor<T>::assign(list);
            this->device.copy_from(this->data.get(), cudaMemcpyHostToDevice);
            this->host_dirty = false;
            return *this;
        }  

        tensor& assign(const init_tensor_ND list) {
            init_tensor<T>::assign(list);
            this->device.copy_from(this->data.get(), cudaMemcpyHostToDevice);
            this->host_dirty = false;
            return *this;
        }

        static op_tensor broadcast_order(const tensor<T>&  a, const tensor<T>&  b) {
            if (a == b) return std::make_pair(&a, &b);
            const shape_t& s_a = a.shape;
            const shape_t& s_b = b.shape;
        
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

            if (!broadcast_check.has_value())
                throw std::invalid_argument{"tensor::add: incompaitable shape or size"};
        
            else if (broadcast_check.value().first != this) \
                throw std::invalid_argument{"tensor::add: cannot broadcast this (object)"};
        
            tensor<T>::sync_device(*this, t);

            dim3 block_size(BLOCK_SIZE);
            dim3 grid_size((this->n + block_size.x - 1) / block_size.x);

            kernel::add<<<grid_size, block_size>>>(
                this->device.d_var(), 
                t.device.d_var(), 
                this->device.d_var()
            );

            this->device.copy_to(this->data.get(), cudaMemcpyDeviceToHost);

            return *this;
        }   

        static tensor<T> add(const tensor<T>& a, const tensor<T>& b) {
            if (!tensor<T>::memory_check(a, b)) 
                throw std::invalid_argument{"tensor::add: cannot do arithmetic with uninitialized tensor(s)"};
            
            auto broadcast_check = tensor<T>::broadcast_order(a, b);
            if (!broadcast_check.has_value())
                throw std::invalid_argument("tensor::add: incompaitable shape or size"); 
        
            tensor<T>::sync_device(a, b);
            tensor<T> result(as_shape, broadcast_check.value().first->shape);
            
            dim3 block_size(BLOCK_SIZE);
            dim3 grid_size((result.n + block_size.x - 1) / block_size.x);

            kernel::add<<<grid_size, block_size>>>(
                a.device.d_var(), 
                b.device.d_var(), 
                result.device.d_var()
            );

            result.device.copy_to(result.data.get(), cudaMemcpyDeviceToHost);

            return result;
        }

        tensor<T>& sub(const tensor<T>& t) {
            if (!tensor<T>::memory_check(*this, t)) 
                throw std::invalid_argument{"tensor::sub: cannot do arithmetic with uninitialized tensor(s)"};
            
            auto broadcast_check = tensor<T>::broadcast_order(*this, t);

            if (!broadcast_check.has_value())
                throw std::invalid_argument{"tensor::sub: incompaitable shape or size"};
    
            else if (broadcast_check.value().first != this) 
                throw std::invalid_argument{"tensor::sub: cannot broadcast this (object)"};
        
            tensor<T>::sync_device(*this, t);
            
            dim3 block_size(BLOCK_SIZE);
            dim3 grid_size((this->n + block_size.x - 1) / block_size.x);

            kernel::sub<<<grid_size, block_size>>>(
                this->device.d_var(), 
                t.device.d_var(), 
                this->device.d_var()
            );

            this->device.copy_to(this->data.get(), cudaMemcpyDeviceToHost);

            return *this;
        }   

        static tensor<T> sub(const tensor<T>& a, const tensor<T>& b) {
            if (!tensor<T>::memory_check(a, b)) 
                throw std::invalid_argument{"tensor::sub: cannot do arithmetic with uninitialized tensor(s)"};
            
            auto broadcast_check = tensor<T>::broadcast_order(a, b);
            if (!broadcast_check.has_value())
                throw std::invalid_argument("tensor::sub: incompaitable shape or size"); 
        
            tensor<T>::sync_device(a, b);
            tensor<T> result(as_shape, broadcast_check.value().first->shape);
            
            dim3 block_size(BLOCK_SIZE);
            dim3 grid_size((result.n + block_size.x - 1) / block_size.x);

            kernel::sub<<<grid_size, block_size>>>(
                a.device.d_var(), 
                b.device.d_var(), 
                result.device.d_var()
            );

            result.device.copy_to(result.data.get(), cudaMemcpyDeviceToHost);

            return result;
        }        

        tensor<T>& mul(const tensor<T>& t) {
            if (!tensor<T>::memory_check(*this, t))
                throw std::runtime_error{"tensor::mul: cannot do arithmetic with uninitialized tensor(s)"};

            auto broadcast_check = tensor<T>::broadcast_order(*this, t);

            if (!broadcast_check.has_value())
                throw std::invalid_argument("tensor::mul: shape not suitable for elementwise multiplication"); 

            else if (broadcast_check.value().first != this)
                throw std::invalid_argument("tensor::mul: cannot broadcast this (object)"); 
        
            tensor<T>::sync_device(*this, t);

            dim3 block_size(BLOCK_SIZE);
            dim3 grid_size((this->n * block_size.x - 1) / block_size.x);

            kernel::mul<<<grid_size, block_size>>>(
                this->device.d_var(), 
                t.device.d_var(), 
                this->device.d_var()
            );

            this->device.copy_to(this->data.get(), cudaMemcpyDeviceToHost);

            return *this;   
        }

        static tensor<T> mul(const tensor<T>& a, const tensor<T>& b) {
            if (!tensor<T>::memory_check(a, b))
                throw std::runtime_error{"tensor::mul: cannot do arithmetic with uninitialized tensor(s)"};

            auto broadcast_check = tensor<T>::broadcast_order(a, b);
            if (!broadcast_check.has_value())
                throw std::invalid_argument("tensor::mul: shape not suitable for elementwise multiplication"); 
        
            tensor<T>::sync_device(a, b);
            tensor<T> result(as_shape, broadcast_check.value().first->shape);

            dim3 block_size(BLOCK_SIZE);
            dim3 grid_size((result.n * block_size.x - 1) / block_size.x);

            kernel::mul<<<grid_size, block_size>>>(
                a.device.d_var(), 
                b.device.d_var(), 
                result.device.d_var()
            );

            result.device.copy_to(result.data.get(), cudaMemcpyDeviceToHost);

            return result;
        }

        tensor<T>& div(const tensor<T>& t) {
            if (!tensor<T>::memory_check(*this, t))
                throw std::runtime_error{"tensor::div: cannot do arithmetic with uninitialized tensor(s)"};

            auto broadcast_check = tensor<T>::broadcast_order(*this, t);

            if (!broadcast_check.has_value())
                throw std::invalid_argument("tensor::div: shape not suitable for elementwise divison"); 

            else if (broadcast_check.value().first != this)
                throw std::invalid_argument("tensor::div: cannot broadcast this (object)"); 
        
            tensor<T>::sync_device(*this, t);

            dim3 block_size(BLOCK_SIZE);
            dim3 grid_size((this->n * block_size.x - 1) / block_size.x);

            kernel::div<<<grid_size, block_size>>>(
                this->device.d_var(), 
                t.device.d_var(), 
                this->device.d_var()
            );

            host_runtime::find_error(__PRETTY_FUNCTION__);
            this->device.copy_to(this->data.get(), cudaMemcpyDeviceToHost);

            return *this;   
        }

        static tensor<T> div(const tensor<T>& a, const tensor<T>& b) {
            if (!tensor<T>::memory_check(a, b))
                throw std::runtime_error{"tensor::div: cannot do arithmetic with uninitialized tensor(s)"};

            auto broadcast_check = tensor<T>::broadcast_order(a, b);
            if (!broadcast_check.has_value())
                throw std::invalid_argument("tensor::div: shape not suitable for elementwise division"); 
        
            tensor<T>::sync_device(a, b);
            tensor<T> result(as_shape, broadcast_check.value().first->shape);

            dim3 block_size(BLOCK_SIZE);
            dim3 grid_size((result.n * block_size.x - 1) / block_size.x);

            kernel::div<<<grid_size, block_size>>>(
                a.device.d_var(), 
                b.device.d_var(), 
                result.device.d_var()
            );

            host_runtime::find_error(__func__);
            result.device.copy_to(result.data.get(), cudaMemcpyDeviceToHost);

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
            tensor<T>::sync_device(temp, t);

            this->resize({i, j});
            
            dim3 block_size(16, 16);
            dim3 grid_size(
                (i + block_size.x - 1) / block_size.x, 
                (j + block_size.y - 1) / block_size.y
            );

            kernel::matmul<<<grid_size, block_size>>>(
                temp.device.d_var(), 
                t.device.d_var(), 
                this->device.d_var(), 
                i, j, k
            );
            
            this->device.copy_to(this->data.get(), cudaMemcpyDeviceToHost);

            return *this;            
        }

        static tensor<T> matmul(const tensor<T>& a, const tensor<T>& b) {
            if (!tensor<T>::memory_check(a, b)) 
                throw std::invalid_argument{"tensor::matmul: cannot multiply uninitialized tensors"};
            
            if (a.dim() != 2 || b.dim() != 2) 
                throw std::invalid_argument{"tensor::matmul: given tensor(s) are not matrices"};
            
            if (a.shape[1] != b.shape[0]) throw std::invalid_argument{"tensor::matmul: invalid shapes"};
            
            tensor<T>::sync_device(a, b);
            size_t i = a.shape[0], j = b.shape[1], k = a.shape[1]; 
            tensor<T> result(as_shape, {i, j});
            
            dim3 block(16, 16);
            dim3 grid_size(
                (i + block.x - 1) / block.x, 
                (j + block.y - 1) / block.y
            );

            kernel::matmul<<<grid_size, block>>>(
                a.device.d_var(), 
                b.device.d_var(), 
                result.device.d_var(), 
                i, j, k
            );
            
            result.device.copy_to(result.data.get(), cudaMemcpyDeviceToHost);

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
            tensor<T>::sync_device(temp, t);
            this->resize({});

            dim3 block_size(BLOCK_SIZE);
            dim3 grid_size((temp.n + block_size.x - 1) / block_size.x);

            kernel::dot<<<grid_size, block_size>>>(
                temp.device.d_var(), 
                t.device.d_var(), 
                this->device.d_var()
            );
            
            this->device.copy_to(this->data.get(), cudaMemcpyDeviceToHost);

            return *this;
        }

        static tensor<T> dot(const tensor<T>& a, const tensor<T>& b) {
            if (!tensor<T>::memory_check(a, b)) 
                throw std::invalid_argument{"tensor::dot: cannot multiply uninitialized tensors"};

            if (a.dim() != 1 || b.dim() != 1) 
                throw std::invalid_argument{"tensor::dot: given tensor(s) are not vectors"};

            if (a.n != b.n) 
                throw std::invalid_argument{"tensor::dot: cannot perform dot operation with unmatched sizes"};
            
            tensor<T>::sync_device(a, b);
            tensor<T> result(0);
            size_t N = a.n;

            dim3 block(BLOCK_SIZE);
            dim3 grid_size((N + block.x - 1) / block.x);
            
            kernel::dot<<<grid_size, block>>>(
                a.device.d_var(), 
                b.device.d_var(), 
                result.device.d_var()
            );
            
            result.device.copy_to(result.data.get(), cudaMemcpyDeviceToHost);

            return result;
        }

        static tensor<T> einsum(const tensor<T>& a, const tensor<T>& b) {
            // Will do later...
        }

        template<typename... Tensors>
        static tensor<T> add(const Tensors &...tensors) {
            constexpr size_t count = sizeof...(tensors);
            static_assert((std::is_same_v<tensor<T>, Tensors> && ...), "tensor::add: all arguments must be tensor<T>");            
            if constexpr (count == 0)
                throw std::invalid_argument("tensor::add: need at least one tensor");

            const tensor<T> &first = GetFirstTensor(tensors...);
            const shape_t &tensor_shape = first.shape;
            const auto tensor_size = static_cast<s_size_t>(first.n);

            if (!tensor<T>::memory_check(tensors...))
                throw std::invalid_argument{"tensor::add: cannot do arithmetic with uninitialized tensor(s)"};

            if (!((tensors == first) && ...))
                throw std::invalid_argument("tensor::add: incompaitable shape or size");

            tensor<T>::sync_device(tensors...);
            tensor<T> result(as_shape, tensor_shape);
                
            std::vector<kernel::d_variables<T>> devices = {tensors.device.d_var()...};

            kernel::d_variables<T> *d_ptr;
            size_t mem_size = sizeof(kernel::d_variables<T>) * devices.size();

            cudaMalloc(&d_ptr, mem_size);
            cudaMemcpy(d_ptr, devices.data(), mem_size, cudaMemcpyHostToDevice);
            
            int block_size(BLOCK_SIZE);
            int grid_size = (tensor_size + block_size - 1) / block_size;
            
            kernel::add_multiple<T><<<grid_size, block_size>>>(d_ptr, result.device, count);
            cudaFree(d_ptr);
            
            result.device.copy_to(result.data.get(), cudaMemcpyDeviceToHost);

            return result;
        }

        static tensor<T> transpose(const tensor<T>& a, const std::initializer_list<size_t> order = {}) {
            tensor<T> result = a;
            size_t size = order.size();
            shape_t& shape = result.shape;
            shape_t& stride = result.stride;
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
                shape_t new_shape; new_shape.resize(this->shape.size());
                shape_t new_stride; new_stride.resize(this->stride.size());

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

        ~tensor(void) override { this->transposed = this->host_dirty = false; }
};

template<arithmetic T>
tensor<T> make_tensor(const typename tensor<T>::init_tensor_0D& scalar) { return tensor<T>(scalar); }

template<arithmetic T>
tensor<T> make_tensor(const typename tensor<T>::init_tensor_1D list) { return tensor<T>(list); }

template<arithmetic T>
tensor<T> make_tensor(const typename tensor<T>::init_tensor_ND list) { return tensor<T>(list); }