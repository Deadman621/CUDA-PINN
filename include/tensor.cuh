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

static const size_t OFFSET_TO_GPU = 10000; 

struct as_shape_t {};
inline constexpr as_shape_t as_shape{};

namespace kernel {
    template <typename T>
    __global__ void add(T **A, T *out, size_t count, size_t N) {
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
}

template<typename T>
static std::string vec_to_str(const std::vector<T>& v) {
    std::ostringstream oss;

    oss << "(";
    for(const auto& i: v)
        oss << i << ", ";
    oss << "\b\b)";

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
        T *h_x = nullptr; // host raw pointer
        T *d_x = nullptr; // device raw pointer
        std::vector<size_t> shape;
        std::vector<size_t> stride;
        size_t n;

        struct with_shape_t {};
        static constexpr with_shape_t with_shape{};

        void typeCheck(void) const noexcept { static_assert(std::is_arithmetic_v<T>, "Non-arithmetic types not supported"); }
        bool mem_avail_h(void) const noexcept { return this->h_x; }
        bool mem_avail_d(void) const noexcept { return this->d_x; }
        bool mem_avail(void) const noexcept { return this->h_x && this->d_x; }

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
                this->h_x = new T[this->n];
                std::memset(this->h_x, 0, this->n * sizeof(T));
            }

            auto x_alias = this->h_x;
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
         * @param with_shape_t pass as_shape here to distinguish from other overloads
         * @param shape std::initializer_list<size_t> or std::vector<size_t> - shape of tensor
         */
        template<typename R>
        void initialize_tensor(with_shape_t, const R& shape) {
            this->shape.resize(shape.size());
            std::copy(shape.begin(), shape.end(), this->shape.begin());

            this->n = this->shape.size() > 0? 1: 0;
            for (auto const &i: this->shape)
                n *= i;
    
            if (this->n != 0) {
                size_t mem_size = this->n * sizeof(T);
                this->h_x = new T[this->n];
                std::memset(this->h_x, 0, mem_size);
                size_t dim = this->shape.size();

                if (dim > 0) this->initialize_strides();
                cudaMalloc(&this->d_x, mem_size);
                cudaMemset(this->d_x, 0, mem_size);
            }
        }

        tensor(void) { typeCheck(); }

    public:
        template <typename U>
        tensor(const std::initializer_list<U> &nums): tensor() {
            bool isValid = initialize_tensor(nums);
            if (!isValid)
                throw std::invalid_argument{"tensor: tensor is inconsistent"};

            if (this->n != 0) {
                size_t bytes_to_allocate = sizeof(T) * this->n;
                cudaMalloc(&this->d_x, bytes_to_allocate);
                cudaMemcpy(this->d_x, this->h_x, bytes_to_allocate, cudaMemcpyHostToDevice);
            }
        }

        // GPU memory will not be used with scalars
        tensor(const T &scalar): tensor{} { 
            std::initializer_list<size_t> temp = {}; 
            initialize_tensor(with_shape, temp); 
            cudaMemcpy(this->d_x, &scalar, sizeof(T), cudaMemcpyHostToDevice); 
        }

        tensor(const std::vector<size_t>& shape): tensor{} { initialize_tensor(with_shape, shape); }
        tensor(as_shape_t, const std::initializer_list<size_t> shape): tensor{} { initialize_tensor(with_shape, shape); }

        tensor(const tensor &obj): tensor{} {
            this->n = obj.n;

            this->shape = obj.shape;
            this->stride = obj.stride;
        
            if (this->n != 0) {
                this->h_x = new T[this->n];
                std::copy(obj.h_x, obj.h_x + obj.n, this->h_x);
                const size_t mem_size = sizeof(T) * this->n;
                cudaMalloc(&this->d_x, mem_size);
                cudaMemcpy(this->d_x, obj.d_x, mem_size, cudaMemcpyDeviceToDevice);          
            }  
        }

        tensor &operator=(const tensor &obj) {
            if (this == &obj)
                return *this;

            if (h_x) delete[] h_x;
            if (d_x) cudaFree(this->d_x);
            this->h_x = this->d_x = nullptr;

            this->n = obj.n;
            this->shape = obj.shape;
            this->stride = obj.stride;

            if (this->n != 0) {
                this->h_x = new T[this->n];
                std::copy(obj.h_x, obj.h_x + n, h_x);
                const size_t mem_size = sizeof(T) * this->n;
                cudaMalloc(&this->d_x, mem_size);
                cudaMemcpy(this->d_x, obj.d_x, mem_size, cudaMemcpyDeviceToDevice);
            }

            return *this;
        }

        template<typename U>
        tensor& operator=(const std::initializer_list<U> &nums) {
            if (h_x) delete[] h_x;
            if (d_x) cudaFree(this->d_x);
            this->h_x = this->d_x = nullptr;
            this->n = 0;
            this->shape.resize(0);
            this->stride.resize(0);

            bool isValid = initialize_tensor(nums);
            if (!isValid)
                throw std::invalid_argument{"tensor: tensor is inconsistent"};

            if (this->n != 0) {
                size_t bytes_to_allocate = sizeof(T) * this->n;
                cudaMalloc(&this->d_x, bytes_to_allocate);
                cudaMemcpy(this->d_x, this->h_x, bytes_to_allocate, cudaMemcpyHostToDevice);       
            }

            return *this;     
        }

        tensor& operator=(const T& scalar) {
            if (h_x) delete[] h_x;
            if (d_x) cudaFree(this->d_x);
            this->h_x = this->d_x = nullptr;
            this->n = 0;
            this->shape.resize(0);
            this->stride.resize(0);
            
            std::initializer_list<size_t> temp = {};
            initialize_tensor(as_shape, temp);
            *(this->h_x) = scalar;
            cudaMemcpy(this->d_x, this->h_x, sizeof(T), cudaMemcpyHostToDevice);

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
        tensor<T> operator*(tensor<T> &obj) { 
            switch(obj.dim()) {
                case 0: return tensor<T>::operator*(*obj.h_x);
                case 1: return tensor<T>::dot(*this, obj);
                case 2: return tensor<T>::matmul(*this, obj); 
                default: throw std::invalid_argument{"operator*: multiplication on unsupported dimension"};
            } 
        }

        tensor<T> operator*(const T& scalar) {
            if (!this->mem_avail())
                throw std::runtime_error{"operator*: cannot do arithmetic with uninitialized tensor"};

            tensor<T> result(as_shape, this->shape);

            if (result.n > OFFSET_TO_GPU) {
                dim3 blockSize(256);
                dim3 gridSize((result.n * blockSize.x - 1) / blockSize.x);
                kernel::scalar_dist<<<gridSize, blockSize>>>(result.h_x, scalar, result.n);
                cudaMemcpy(result.h_x, result.d_x, result.n, cudaMemcpyDeviceToHost);
            }

            else {
                for(size_t i = 0; i < result.n; i++)
                    result.h_x[i] *= scalar;
                cudaMemcpy(result.d_x, result.h_x, result.n, cudaMemcpyHostToDevice);
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

            return this->h_x[linear_index];
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

            auto x_alias = this->h_x;
            if (!validate_shape__initialize_strides__flatten_tensor(nums, x_alias)) 
                throw std::invalid_argument{"assign: cannot initialize tensor of shape " 
                    + vec_to_str(this->shape) + " - inconsistent shape"};

            return *this;
        }

        auto& assign(const T& scalar) {
            if (!mem_avail() || this->shape.size())
                throw std::runtime_error{"assign: cannot initialize tensor, shape unkown or inconsistent"};
            
            *(this->h_x) = scalar;
            return *this;
        }

        // Temporary - remove it later
        inline T *raw(void) const noexcept { return this->h_x; }

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
                
            std::vector<T*> d_ptrs = {tensors.d_x...};
            // device variables
            T **d_ptr_2D;
            size_t mem_size = sizeof(T*) * sizeof(d_ptrs);

            cudaMalloc(&d_ptr_2D, mem_size);
            cudaMemcpy(d_ptr_2D, d_ptrs.data(), mem_size, cudaMemcpyHostToDevice);
            
            int threadsperblock(256);
            int blocks = (tensor_size + threadsperblock - 1) / threadsperblock;
            
            kernel::add<T><<<blocks, threadsperblock>>>(d_ptr_2D, result.d_x, count, tensor_size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) 
                printf("add - kernel launch failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_ptr_2D);
            
            mem_size = sizeof(T) * tensor_size;
            cudaMemcpy(result.h_x, result.d_x, mem_size, cudaMemcpyDeviceToHost);

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
            
            cudaMemcpy(result.h_x, result.d_x, sizeof(T) * result.n, cudaMemcpyDeviceToHost);  

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

        ~tensor(void) {
            if (h_x) delete[] h_x;
            if (d_x) cudaFree(this->d_x);
            this->h_x = this->d_x = nullptr;
            this->n = 0;
            this->shape.resize(0);
            this->stride.resize(0);
        }
};

template<typename T>
void opHelper(std::ostream& output, const tensor<T>& obj, size_t index, size_t dim = 1) {
    
    size_t start = index * obj.shape[dim - 1];
    size_t end = start + obj.shape[dim - 1];
    
    if (dim == obj.dim()) {
        output << '[';
        for(size_t i = start; i < end; i++) {
            output << obj.h_x[i];
            if (i != end - 1) 
                output << ", ";
        }
        output << ']';

        return;
    }
    
    output << '[';
    for(size_t i = start; i < end; i++) {
        opHelper(output, obj, i, dim + 1);
        if (i != end - 1) 
            output << ", ";
    }
    output << ']';

    return;
}

template<typename T>
std::ostream& operator<<(std::ostream& output, const tensor<T>& obj) {
    if (obj.dim() == 0) {
        output << *obj.h_x;
        return output;
    }

    opHelper(output, obj, 0);

    return output;
}