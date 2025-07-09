#pragma once

#include <initializer_list>
#include <iostream>
#include <cstring>
#include <vector>
#include <thread>
#include <type_traits>
#include <optional>
#include <cstdio>

static const size_t OFFSET_TO_GPU = 10000; 

namespace kernel
{
    template <typename T>
    __global__ void add(T **A, T *out, size_t count, size_t N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N)
            return;

        T sum = 0;
        for (size_t j = 0; j < count; j++) 
            sum += A[j][i];

        out[i] = sum;
    }
}

// type unsafe function.
template <typename First, typename... Rest>
static auto &GetFirstTensor(const First &__ax, const Rest &...__bx) { return __ax; }

template <typename T>
class tensor {
    private:
        T *h_x = nullptr; // host raw pointer
        T *d_x = nullptr; // device raw pointer
        std::vector<size_t> shape;
        std::vector<size_t> stride;
        size_t n;

        void typeCheck(void) { static_assert(std::is_arithmetic_v<T>, "Non-arithmetic types not supported"); }

        size_t calculate_stride(std::initializer_list<size_t> weights)
        {

            if (weights.size() != stride.size())
                throw std::invalid_argument{"calculate_stride: invalid indices provided"};

            size_t i = 0, index = 0;
            for (const auto &w : weights)
            {
                index += this->stride[i] * w;
                i++;
            }

            return index;
        }

        template <typename U>
        size_t infer_shape(const std::initializer_list<U> &nums)
        {
            if constexpr (std::is_same_v<T, U>)
            {
                this->shape.push_back(nums.size());
                return 0;
            }

            else
            {
                this->shape.push_back(nums.size());
                return infer_shape(*nums.begin());
            }
        }

        /**
         * @brief this function assumes shape is already known
         */

        // TODO: Correct Stride calculation logic.
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

        /* REQUIRES:
            - this->x allocated
            - shape and stride vectors allocated

            Unsafe if these are not initialized.

            This function is an attempt to reduce time complexity from 3N to N and reduce call
            stack operations if the tensor dimensions are large

            WARNING: Potentionally dangerous function, use with high caution
        */
        template <typename U>
        bool validate_shape__initialize_strides__flatten_tensor(const std::initializer_list<U> &nums, T *&raw_ptr, int depth = 0)
        {
            if constexpr (std::is_same_v<T, U>)
            {
                if (depth > 0)
                    this->stride[depth - 1] = nums.size();
                this->stride[depth] = 1;

                bool result = nums.size() == this->shape[depth];
                if (result)
                {
                    std::copy(nums.begin(), nums.end(), raw_ptr);
                    raw_ptr += nums.size();
                }

                return result;
            }

            else
            {
                if (nums.size() != this->shape[depth])
                    return false;

                for (const auto &i : nums)
                {
                    if (!validate_shape__initialize_strides__flatten_tensor(i, raw_ptr, depth + 1))
                        return false;
                }

                if (depth > 0)
                    this->stride[depth - 1] = this->shape[depth] * this->stride[depth];

                return true;
            }
        }

        template <typename U>
        bool initialize_tensor(const std::initializer_list<U> &nums)
        {
            infer_shape(nums);

            this->n = shape.empty() ? 0 : 1;
            for (auto const &i : shape)
                n *= i;
            this->stride.resize(this->dim());
            this->h_x = new T[this->n];
            std::memset(this->h_x, 0, this->n * sizeof(T));

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

        tensor(const size_t &n, const size_t &dim) : n{n} {
            typeCheck();
            size_t mem_size = this->n * sizeof(T);
            this->h_x = new T[this->n];
            std::memset(this->h_x, 0, mem_size);

            if (dim > 0) {
                cudaMalloc(&this->d_x, mem_size);
                cudaMemset(this->d_x, 0, mem_size);

                this->shape.resize(dim);
                this->stride.resize(dim);
                
            }
        }

    public:
        template <typename U>
        tensor(const std::initializer_list<U> &nums)
        {
            typeCheck();
            bool isValid = initialize_tensor(nums);
            if (!isValid)
                throw std::invalid_argument{"tensor: tensor is inconsistent"};

            size_t bytes_to_allocate = sizeof(T) * this->n;
            cudaMalloc(&this->d_x, bytes_to_allocate);
            cudaMemcpy(this->d_x, this->h_x, bytes_to_allocate, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }

        // GPU memory will not be used with scalars
        tensor(const T &scalar) : tensor(1, 0) { *(this->h_x) = scalar; }
        tensor(const size_t &n, const std::vector<size_t> &shape): tensor(n, 0) { this->shape = shape; }
        tensor(const size_t &n, const std::initializer_list<size_t> &shape): tensor(n, shape.size())
        {
            std::copy(shape.begin(), shape.end(), this->shape.begin());
        }

        tensor(const tensor &obj)
        {
            typeCheck();

            this->n = obj.n;
            this->h_x = new T[this->n];
            this->shape = obj.shape;
            this->stride = obj.stride;
            std::copy(obj.h_x, obj.h_x + obj.n, this->h_x);
        }

        tensor &operator=(const tensor &obj) {
            typeCheck();

            if (this == &obj)
                return *this;

            this->~tensor();

            this->n = obj.n;
            this->h_x = new T[n];
            this->shape = obj.shape;
            this->stride = obj.stride;
            if (obj.h_x != nullptr)
                std::copy(obj.h_x, obj.h_x + n, h_x);

            const size_t mem_size = sizeof(T) * this->n;
            cudaMalloc(&this->d_x, mem_size);

            if (obj.d_x != nullptr) cudaMemcpy(this->d_x, obj.d_x, mem_size, cudaMemcpyDeviceToDevice);
            else cudaMemcpy(this->d_x, this->h_x, mem_size, cudaMemcpyHostToDevice);

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
        
        // Not worth running GPU
        tensor<T> operator*(const T& scalar) {
            tensor<T> result = *this;
            for(size_t i = 0; i < this->n; i++)
                this->result[i] *= scalar; 

            return result;
        }

        inline size_t size(void) const noexcept { return this->n; }
        inline std::vector<size_t> get_shape(void) const noexcept { return this->shape; }
        inline std::vector<size_t> get_stride(void) const noexcept { return this->stride; }
        inline size_t dim(void) const noexcept { return this->shape.size(); }
        inline auto reshape(std::vector<size_t>& shape) { 
            size_t product = 1;
            for(const auto& i: shape)
                product *= i;
            
            if (product != this->n)
                throw std::invalid_argument{"reshape: cannot reshape, elements size doesn't match"};

            this->shape = shape; 
            this->initialize_strides();
            return *this;
        }

        inline auto reshape(std::initializer_list<size_t> shape) {
            size_t product = 1;
            for(const auto& i: shape)
                product *= i;
            
            if (product != this->n)
                throw std::invalid_argument{"reshape: cannot reshape, elements size doesn't match"};

            this->shape.resize(shape.size());
            std::copy(shape.begin(), shape.end(), this->shape.begin());
            this->initialize_strides();
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

            if (!((tensors == first) && ...))
                throw std::invalid_argument("add: mismatch in tensor shape/size");

            tensor<T> result(tensor_size, tensor_shape);
            
            if (tensor_shape.size() > 0) {
                
                std::vector<T*> d_ptrs = {tensors.d_x...};
                // device variables
                T **d_ptr_2D, *d_outptr;
                size_t mem_size = sizeof(T*) * sizeof(d_ptrs);

                cudaMalloc(&d_outptr, sizeof(T) * tensor_size);
                cudaMalloc(&d_ptr_2D, mem_size);
                cudaMemcpy(d_ptr_2D, d_ptrs.data(), mem_size, cudaMemcpyHostToDevice);
                
                int threadsperblock(256);
                int blocks = (tensor_size + threadsperblock - 1) / threadsperblock;
                
                kernel::add<T><<<blocks, threadsperblock>>>(d_ptr_2D, d_outptr, count, tensor_size);
                cudaDeviceSynchronize();
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
                }
                
                mem_size = sizeof(T) * tensor_size;
                cudaMemcpy(result.h_x, d_outptr, mem_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(result.d_x, d_outptr, mem_size, cudaMemcpyDeviceToDevice);

                cudaFree(d_ptr_2D);
                cudaFree(d_outptr);
            }

            else *result.h_x = (*tensors.h_x + ...);

            return result;
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