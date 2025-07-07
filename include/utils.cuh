#pragma once

#include<initializer_list>
#include<iostream>
#include<cstring>
#include<vector>
#include<thread>
#include<type_traits> 
#include<optional>

namespace kernel {
    template<typename T>
    __global__ void add(T* A, T* out, size_t count, size_t N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        T sum = 0;
        for(size_t j = 0; j < count; j++)  
            sum += *((A + j * N) + i);

        out[i] = sum;
    }
}

// type unsafe function.
template<typename First, typename... Rest>
static size_t deduceSize(const First& __ax, const Rest&... __bx) { return __ax.size(); }

template<typename First, typename... Rest>
static size_t deduceShape(const First& __ax, const Rest&... __bx) { return __ax.dim(); }

template<typename T>
class tensor {
    private:    
        T* x = nullptr;
        std::vector<size_t> shape;
        std::vector<size_t> stride;
        size_t n;

        void typeCheck(void) { static_assert(std::is_arithmetic_v<T>, "Non-arithmetic types not supported"); }
        
        size_t calculate_stride(std::initializer_list<size_t> weights) {
            
            if (weights.size() != stride.size())
                throw std::invalid_argument{"calculate_stride: invalid indices provided"};

            size_t i = 0, index = 0;
            for(const auto& w: weights) {
                index += this->stride[i] * w;
                i++;
            }

            return index;
        }

        template<typename U>
        size_t infer_shape(const std::initializer_list<U>& nums) {
            if constexpr (std::is_same_v<T, U>) {
                this->shape.push_back(nums.size());
                return 0;
            }

            else {
                this->shape.push_back(nums.size());
                return infer_shape(*nums.begin());
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
        template<typename U>
        bool validate_shape__initialize_strides__flatten_tensor(const std::initializer_list<U>& nums, T*& raw_ptr, int depth = 0) {
            if constexpr (std::is_same_v<T, U>) {
                if (depth > 0) this->stride[depth - 1] = nums.size();
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

                for(const auto& i: nums) {
                    if (!validate_shape__initialize_strides__flatten_tensor(i, raw_ptr, depth + 1))
                        return false;    
                }
                
                if (depth > 0) this->stride[depth - 1] = this->shape[depth] * this->stride[depth];

                return true;
            }   
        }

        template<typename U>
        bool initialize_tensor(const std::initializer_list<U>& nums) {
            infer_shape(nums);

            this->n = shape.empty()? 0: 1;
            for(auto const& i: shape)
                n *= i;
            this->stride.resize(this->dim());
            this->x = new T[this->n];
            std::memset(this->x, 0, this->n * sizeof(T));

            auto x_alias = this->x;
            if (!validate_shape__initialize_strides__flatten_tensor(nums, x_alias))
                return false;

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

            return true;
        }

    public:
        template<typename U>
        tensor(const std::initializer_list<U>& nums) {
            typeCheck();
            bool isValid = initialize_tensor(nums);
            if (!isValid)
                throw std::invalid_argument{"tensor: tensor is inconsistent"};           
        }

        tensor(const size_t& n, const size_t dim) {
            typeCheck();

            this->n = n;
            this->shape.resize(dim);
            this->x = new T[this->n];
            std::memset(this->x, 0, this->n * sizeof(T));
        }

        tensor(const tensor& obj) {
            typeCheck();

            this->n = obj.n;
            this->x = new T[this->n];
            std::copy(obj.x, obj.x + obj.n, this->x);
        }

        tensor& operator=(const tensor& obj) {
            typeCheck();

            if (this == &obj) return *this;

            delete[] this->x;
            this->n = obj.n;
            this->x = new T[n];
            if (obj.x != nullptr)
                std::copy(obj.x, obj.x + n, x);
            return *this;
        }

        bool operator==(const tensor& obj) {
            
        }

        tensor<T> operator+(tensor<T>& obj) { return tensor<T>::add(*this, obj); }

        inline size_t size(void) const noexcept { return this->n; }
        inline size_t dim(void) const noexcept { return this->shape.size(); }
        inline T* raw(void) const noexcept { return this->x; }

        template<typename... Tensors>
        static tensor<T> add(const Tensors&... tensors) {
            constexpr size_t count = sizeof...(tensors); 
            static_assert((std::is_same_v<tensor<T>, Tensors> && ...), "add: all arguments must be tensor<T>");
            static_assert(std::is_arithmetic_v<T>, "add: only arithmetic types supported");
            
            if constexpr (count == 0) 
            throw std::invalid_argument("add: need at least one tensor");
            
            size_t tensor_size = deduceSize(tensors...);
            size_t tensor_shape = deduceShape(tensors...);
            
            if (!((tensors.size() == tensor_size) && ...)) 
                throw std::invalid_argument("add: mismatch in tensor sizes");   
            
            std::vector<T*> ts = { tensors.raw()... };
            T* h_raw_ptr = new T[count * tensor_size];
            for(size_t i = 0; i < count; i++) {
                for(size_t j = 0; j < tensor_size; j++) 
                    h_raw_ptr[i * tensor_size + j] = ts.at(i)[j];
            }

            // Debugging ONLY
            /* std::cout << std::endl << "Linearized = [";
            for(size_t i = 0; i < count; i++) {
                for(size_t j = 0; j < tensor_size; j++)
                    std::cout << h_raw_ptr[i * tensor_size + j] << ", ";
            }
            std::cout << "\b\b]" << std::endl; */    

            // device variables
            T* d_raw_ptr, *d_outptr;

            cudaMalloc(&d_raw_ptr, sizeof(T) * count * tensor_size);
            cudaMalloc(&d_outptr, sizeof(T) * tensor_size);
            cudaMemcpy(d_raw_ptr, h_raw_ptr, sizeof(T) * count * tensor_size, cudaMemcpyHostToDevice);

            tensor<T> result(tensor_size, tensor_shape);

            int threadsperblock(256);
            int blocks = (tensor_size + threadsperblock - 1) / threadsperblock;

            kernel::add<T><<<blocks, threadsperblock>>>(d_raw_ptr, d_outptr, count, tensor_size);

            cudaMemcpy(result.raw(), d_outptr, sizeof(T) * tensor_size, cudaMemcpyDeviceToHost);

            cudaFree(d_raw_ptr);
            cudaFree(d_outptr);
            
            cudaDeviceSynchronize();
            
            return result;
        }

        ~tensor(void) {
            delete[] x;
            this->x = nullptr;
        }
};