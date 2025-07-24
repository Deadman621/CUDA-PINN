#pragma once

#include<initializer_list>
#include<iostream>
#include<cstring>
#include<vector>
#include<thread>
#include<type_traits>
#include<optional>
#include<cstdio>
#include<sstream>
#include<unordered_set>
#include<kernel.cuh>

static const size_t OFFSET_TO_GPU = 10000; 

struct as_shape_t {};
inline constexpr as_shape_t as_shape{};

template<typename T>
static std::string vec_to_str(const std::vector<T>& v) {
    std::ostringstream oss;

    oss << "(";
    for(const auto& i: v) {
        oss << i;
        if (i != v.back())
            oss << ", ";
    }
    oss << ")";

    return oss.str();
}

// type unsafe function.
template <typename First, typename... Rest>
static auto &GetFirstTensor(const First &__ax, const Rest &...__bx) { return __ax; }

template <typename T>
class tensor {

    template<typename U>
    friend std::ostream& operator<<(std::ostream&, const tensor<U>&);

    template<typename U>
    friend void opHelper(std::ostream&, const tensor<U>&, size_t, size_t);

    private:
        T *x = nullptr;
        size_t n;

        std::vector<size_t> shape;
        std::vector<size_t> stride;

        kernel::device<T> device;

        void typeCheck(void) const noexcept { static_assert(std::is_arithmetic_v<T>, "Non-arithmetic types not supported"); }
        bool mem_avail_h(void) const noexcept { return this->x; }
        bool mem_avail_d(void) const noexcept { return this->d_x; }
        bool mem_avail(void) const noexcept { return this->x && this->device.x; }

        size_t calculate_stride(std::initializer_list<size_t> weights) {
            if (weights.size() != stride.size())
                throw std::invalid_argument{"calculate_stride: invalid indices provided"};

            size_t i = 0, index = 0;
            for (const auto &w : weights) {
                if (w > this->shape[i] || static_cast<ptrdiff_t>(w) < 0)
                    throw std::out_of_range{"calculate_stride: out of bounds access"};
                index += this->stride[i] * w;
                i++;
            }

            return index;
        }

        template <typename U>
        void infer_shape(const std::initializer_list<U> &nums) {
            if constexpr (std::is_same_v<T, U>) {
                this->shape.push_back(nums.size());
                return;
            }

            else {
                this->shape.push_back(nums.size());
                infer_shape(*nums.begin());
            }
        }

        /**
         * @brief this function assumes shape is already known
         */

        void initialize_strides(void) {
            const size_t shape_size = this->shape.size();
            if (shape_size == 0) throw std::runtime_error{"initialize_strides: shape is not initialized"};
            if (shape_size != this->stride.size()) this->stride.resize(shape_size);
            
            this->stride[this->stride.size() - 1] = 1;
            if (shape_size > 1) {
                for(int i = this->stride.size() - 2; i >= 0; i--) 
                    this->stride[i] = this->shape[i + 1] * this->stride[i + 1];
            }
        }

        template<typename R>
        void reshape_backend(const R& shape) {
            size_t product = 1;
            for(const auto& i: shape)
                product *= i;
            
            if (product != this->n)
                throw std::invalid_argument{"reshape_backend: cannot reshape, element size doesn't match"};

            this->shape.resize(shape.size());
            std::copy(shape.begin(), shape.end(), this->shape.begin());
            this->initialize_strides();

            device.reshape(shape.size());
            device.copy(this->shape.data(), this->stride.data(), cudaMemcpyHostToDevice);
        }

        /* REQUIRES:
            - this->x allocated
            - shape and stride vectors allocated

            Unsafe if these are not initialized.

            This function is an attempt to reduce time complexity from 3N to N and reduce call
            stack operations if the tensor dimensions are large

            WARNING: Potentionally dangerous function, use with high caution
        */
        template <typename U>
        bool validate_shape__initialize_strides__flatten_tensor(const std::initializer_list<U> &nums, T *&raw_ptr, int depth = 0) {
            if constexpr (std::is_same_v<T, U>) {
                if (depth > 0)
                    this->stride[depth - 1] = nums.size();
                this->stride[depth] = 1;

                bool result = nums.size() == this->shape[depth];
                if (result) {
                    std::copy(nums.begin(), nums.end(), raw_ptr);
                    raw_ptr += nums.size();
                }

                return result;
            }

            else {
                if (nums.size() != this->shape[depth])
                    return false;

                for (const auto &i : nums) {
                    if (!validate_shape__initialize_strides__flatten_tensor(i, raw_ptr, depth + 1))
                        return false;
                }

                if (depth > 0)
                    this->stride[depth - 1] = this->shape[depth] * this->stride[depth];

                return true;
            }
        }

        template <typename U>
        bool initialize_tensor(const std::initializer_list<U> &nums) {
            infer_shape(nums);

            this->n = this->shape.size() > 0? 1: 0;
            for (auto const &i: shape)
                n *= i;
            this->stride.resize(this->dim());

            if (n != 0) {
                this->x = new T[this->n];
                std::memset(this->x, 0, this->n * sizeof(T));
            }

            auto x_alias = this->x;
            if (!validate_shape__initialize_strides__flatten_tensor(nums, x_alias))
                return false;

/*             {
                using std::cout;
                using std::endl;

                                cout << "shape = [";
                                for(int i = 0; i < shape.size(); i++)
                                    cout << shape[i] << ", ";
                                cout << "\b\b]" << endl;

                                cout << "Strides = [";
                                for(int i = 0; i < stride.size(); i++)
                                    cout << stride[i] << ", ";
                                cout << "\b\b]" << endl;
            } */

            return true;
        }

        /**
         * @brief Initialize tensors with given shape
         * @param as_shape_t pass as_shape here to distinguish from other overloads
         * @param shape std::initializer_list<size_t> or std::vector<size_t> - shape of tensor
         */
        template<typename R>
        void initialize_tensor(as_shape_t, const R& shape) {
            this->shape.resize(shape.size());
            std::copy(shape.begin(), shape.end(), this->shape.begin());

            this->n = this->shape.size() > 0? 1: 0;
            for (auto const &i: this->shape)
                n *= i;
    
            if (this->n != 0) {
                size_t mem_size = this->n * sizeof(T);
                this->x = new T[this->n];
                std::memset(this->x, 0, mem_size);
                
                size_t dim = this->shape.size();
                if (dim > 0) this->initialize_strides();
            }

            device.allocate(n, dim());
            device.copy(this->shape.data(), this->stride.data(), cudaMemcpyHostToDevice);
        }

        tensor(void) { typeCheck(); }

    public:
        template <typename U>
        tensor(const std::initializer_list<U> &nums): tensor() {
            bool isValid = initialize_tensor(nums);
            if (!isValid)
                throw std::invalid_argument{"tensor: tensor is inconsistent"};

            device.allocate(this->n, this->dim());
            device.copy(this->x, this->shape, this->stride, cudaMemcpyHostToDevice);
        }

        // GPU memory will not be used with scalars
        tensor(const T &scalar): tensor{} { 
            std::initializer_list<size_t> temp = {}; 
            initialize_tensor(as_shape, temp); 
            device.copy(this->x, cudaMemcpyHostToDevice);
        }

        tensor(const std::vector<size_t>& shape): tensor{} { initialize_tensor(as_shape, shape); }
        tensor(as_shape_t, const std::initializer_list<size_t> shape): tensor{} { initialize_tensor(as_shape, shape); }

        tensor(const tensor<T> &obj): tensor{} {
            this->n = obj.n;

            this->shape = obj.shape;
            this->stride = obj.stride;
        
            if (this->n != 0) {
                this->x = new T[this->n];
                std::copy(obj.x, obj.x + obj.n, this->x);  
            }

            device.allocate(this->n, this->dim());
            device.copy(this->x, this->shape, this->stride, cudaMemcpyHostToDevice);
        }

        tensor &operator=(const tensor &obj) {
            if (this == &obj)
                return *this;

            if (x) delete[] x;
            this->x = nullptr;

            this->n = obj.n;
            this->shape = obj.shape;
            this->stride = obj.stride;

            if (this->n != 0) {
                this->x = new T[this->n];
                std::copy(obj.x, obj.x + n, x);
            }

            device.reallocate(this->n, this->dim());
            device.copy(obj.x, obj.shape.data(), obj.stride.data(), cudaMemcpyDeviceToDevice);

            return *this;
        }

        template<typename U>
        tensor& operator=(const std::initializer_list<U> &nums) {
            if (x) delete[] x;
            this->x = nullptr;
            this->n = 0;
            this->shape.resize(0);
            this->stride.resize(0);

            bool isValid = initialize_tensor(nums);
            if (!isValid)
                throw std::invalid_argument{"tensor: tensor is inconsistent"};

            device.reallocate(this->n, this->dim());
            device.copy(this->x, this->shape.data(), this->stride.data(), cudaMemcpyHostToDevice);

            return *this;     
        }

        tensor& operator=(const T& scalar) {
            if (x) delete[] x;
            this->x = this->d_x = nullptr;
            this->n = 0;
            this->shape.resize(0);
            this->stride.resize(0);
            
            std::initializer_list<size_t> temp = {};
            initialize_tensor(as_shape, temp);
            *(this->x) = scalar;
            device.copy(this->x, cudaMemcpyHostToDevice);

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
        bool operator==(const tensor &obj) const noexcept { return this->shape == obj.shape && this->n == obj.n; }
        tensor<T> operator+(tensor<T> &obj) { return tensor<T>::add(*this, obj); }
        tensor<T> operator*(const tensor<T> &obj) { 
            switch(obj.dim()) {
                case 0: return tensor<T>::operator*(*obj.x);
                case 1: return tensor<T>::dot(*this, obj);
                case 2: return tensor<T>::matmul(*this, obj); 
                default: throw std::invalid_argument{"operator*: multiplication on unsupported dimension"};
            } 
        }

        tensor<T> operator*(const T& scalar) {
            if (!this->mem_avail())
                throw std::runtime_error{"operator*: cannot do arithmetic with uninitialized tensor"};

            tensor<T> result(this->shape);

            if (result.n > OFFSET_TO_GPU) {
                dim3 blockSize(256);
                dim3 gridSize((result.n * blockSize.x - 1) / blockSize.x);
                kernel::scalar_dist<<<gridSize, blockSize>>>(result.x, scalar, result.n);
                cudaMemcpy(result.x, result.d_x, result.n, cudaMemcpyDeviceToHost);
            }

            else {
                for(size_t i = 0; i < result.n; i++)
                    result.x[i] *= scalar;
                cudaMemcpy(result.d_x, result.x, result.n, cudaMemcpyHostToDevice);
            }

            return result;
        }

        template<typename... Indices>
        T& operator()(Indices... indices) {
            constexpr size_t count = sizeof...(indices);
            static_assert((std::is_integral_v<Indices> && ...), "operator(): all arguments must be unsigned numbers");
            if (count != this->dim()) throw std::out_of_range{"operator(): invalid access"};
            
            std::initializer_list<size_t> I = { static_cast<size_t>(indices)... };
            size_t linear_index = this->calculate_stride(I);

            return this->x[linear_index];
        }

        inline size_t size(void) const noexcept { return this->n; }
        inline std::vector<size_t> get_shape(void) const noexcept { return this->shape; }
        inline std::vector<size_t> get_stride(void) const noexcept { return this->stride; }
        inline size_t dim(void) const noexcept { return this->shape.size(); }
        inline auto& reshape(std::vector<size_t>& shape) { reshape_backend(shape); return *this; }
        inline auto& reshape(std::initializer_list<size_t> shape) { reshape_backend(shape); return *this; }
        
        template<typename U>
        auto& assign(std::initializer_list<U> nums) {
            if (!mem_avail() || !this->shape.size() || !this->stride.size())
                throw std::runtime_error{"assign: cannot initialize tensor, shape unknown"};

            auto x_alias = this->x;
            if (!validate_shape__initialize_strides__flatten_tensor(nums, x_alias)) 
                throw std::invalid_argument{"assign: cannot initialize tensor of shape " 
                    + vec_to_str(this->shape) + " - inconsistent shape"};

            return *this;
        }

        auto& assign(const T& scalar) {
            if (!mem_avail() || this->shape.size())
                throw std::runtime_error{"assign: cannot initialize tensor, shape unkown or inconsistent"};
            
            *(this->x) = scalar;
            return *this;
        }

        // Temporary - remove it later
        inline T *raw(void) const noexcept { return this->x; }

        template <typename... Tensors>
        static tensor<T> add(const Tensors &...tensors) {
            constexpr size_t count = sizeof...(tensors);
            static_assert((std::is_same_v<tensor<T>, Tensors> && ...), "add: all arguments must be tensor<T>");
            static_assert(std::is_arithmetic_v<T>, "add: only arithmetic types supported");
            if constexpr (count == 0)
                throw std::invalid_argument("add: need at least one tensor");

            auto &first = GetFirstTensor(tensors...);
            auto tensor_size = first.size();
            auto tensor_shape = first.get_shape();

            if (!((tensors.mem_avail()) && ...))
                throw std::invalid_argument{"add: cannot do arithmetic with uninitialized tensor(s)"};

            if (!((tensors == first) && ...))
                throw std::invalid_argument("add: mismatch in tensor shape/size");

            tensor<T> result(tensor_shape);
                
            std::vector<const kernel::d_variables<T>> devices = {tensors.device.data()...};

            kernel::d_variables<T> *d_ptr;
            size_t mem_size = sizeof(kernel::d_variables<T>) * sizeof(devices);

            cudaMalloc(&d_ptr, mem_size);
            cudaMemcpy(d_ptr, devices.data(), mem_size, cudaMemcpyHostToDevice);
            
            int threadsperblock(256);
            int blocks = (tensor_size + threadsperblock - 1) / threadsperblock;
            
            kernel::add<T><<<blocks, threadsperblock>>>(d_ptr, result.device.x, count, tensor_size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) 
                printf("add - kernel launch failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_ptr);
            
            mem_size = sizeof(T) * tensor_size;

            // FIX THIS
            cudaMemcpy(result.x, result.device.x, mem_size, cudaMemcpyDeviceToHost);

            return result;
        }

        static tensor<T> matmul(const tensor<T>& a, const tensor<T>& b) {
            if (!a.mem_avail() || !b.mem_avail()) throw std::invalid_argument{"matmul: cannot multiply uninitialized tensors"};
            if (a.dim() != 2 || b.dim() != 2) throw std::invalid_argument{"matmul: given tensor(s) are not matrices"};
            if (a.shape[1] != b.shape[0]) throw std::invalid_argument{"matmul: invalid shapes"};
            
            size_t i = a.shape[0], j = b.shape[1], k = a.shape[1]; 
            tensor<T> result(as_shape, {i, j});
            
            dim3 block(16, 16);
            dim3 grid_size((i + block.x - 1) / block.x, (j + block.y - 1) / block.y);
            kernel::matmul<<<grid_size, block>>>(a.d_x, b.d_x, result.d_x, i, j, k);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) 
                printf("matmul - kernel launch failed: %s\n", cudaGetErrorString(err));
            
            cudaMemcpy(result.x, result.d_x, sizeof(T) * result.n, cudaMemcpyDeviceToHost);  

            return result;
        }

        static tensor<T> dot(const tensor<T>& a, const tensor<T>& b) {
            if (!a.mem_avail() || !b.mem_avail()) throw std::invalid_argument{"dot: cannot multiply uninitialized tensors"};
            if (a.dim() != 1 || b.dim() != 1) throw std::invalid_argument{"dot: given tensor(s) are not vectors"};
            if (a.n != b.n) throw std::invalid_argument{"dot: cannot perform dot operation with unmatched sizes"};

            size_t N = a.n;
            tensor<T> result(0);
            dim3 block(256);
            dim3 grid_size((N + block.x - 1) / block.x);
            kernel::dot<<<block, grid_size>>>(a.d_x, b.d_x, result.d_x, N);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) 
                printf("dot - kernel launch failed: %s\n", cudaGetErrorString(err));            

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
                    throw std::invalid_argument{"transpose: axis order must match dimensions"};

                std::unordered_set<size_t> set;

                size_t k = 0;
                for(const auto& i: order) {
                    if (i >= a.dim()) throw std::invalid_argument{"transpose: invalid order"};
                    if (!set.insert(i).second) throw std::invalid_argument{"transpose: duplicates not allowed"};
                    
                    shape[k] = a.shape[i];
                    stride[k] = a.stride[i];

                    k++;
                }
            }

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
                    throw std::invalid_argument{"transpose: axis order must match dimensions"};

                std::unordered_set<size_t> set;

                size_t k = 0;
                for(const auto& i: order) {
                    if (i >= this->dim()) throw std::invalid_argument{"transpose: invalid order"};
                    if (!set.insert(i).second) throw std::invalid_argument{"transpose: duplicates not allowed"};
                    
                    new_shape[k] = this->shape[i];
                    new_stride[k] = this->stride[i];

                    k++;
                }

                this->shape = new_shape;
                this->stride = new_stride;

                if (this->d_stride) cudaFree(this->d_stride);
                if (this->dim() > 0) {
                    cudaMalloc(&this->d_stride, this->dim());
                    cudaMemcpy(this->d_stride, this->stride.data(), this->stride.size() * sizeof(size_t), cudaMemcpyHostToDevice);
                }
            }           

            return *this;
        }

        ~tensor(void) {
            if (x) delete[] x;
            this->x = nullptr;
            this->n = 0;
            this->shape.resize(0);
            this->stride.resize(0);
        }
};

template<typename T>
void opHelper(std::ostream& output, const tensor<T>& obj, size_t index, size_t dim = 1) {
    
    size_t offset = obj.stride[dim - 1];
    size_t start = index;
    size_t end = start + obj.shape[dim - 1] * offset;
    
    if (dim == obj.dim()) {
        output << '[';
        for(size_t i = start; i < end; i += offset) {
            output << obj.x[i];
            if (i != end - offset) 
                output << ", ";
        }
        output << ']';

        return;
    }
    
    output << '[';
    for(size_t i = start; i < end; i += offset) {
        opHelper(output, obj, i, dim + 1);
        if (i != end - offset) 
            output << ", ";
    }
    output << ']';

    return;
}

template<typename T>
std::ostream& operator<<(std::ostream& output, const tensor<T>& obj) {
    if (obj.dim() == 0) {
        output << *obj.x;
        return output;
    }

    opHelper(output, obj, 0);

    return output;
}