#pragma once

#include<initializer_list>
#include<type_traits>
#include<iostream>
#include<algorithm>
#include<stdexcept>
#include<sstream>
#include<concepts>
#include<cstddef>
#include<cstring>
#include<vector>

struct as_shape_t {};
inline constexpr as_shape_t as_shape{}; 

template<typename T>
static std::string vec_to_str(const std::vector<T>& v) {
    std::ostringstream oss;

    if (v.empty()) oss << "(scalar)";
    
    else {
        oss << "(";
        size_t count = 0;
        for(const auto& i: v) {
            oss << i;
            if (++count != v.size())
                oss << ", ";
        }
        oss << ")";
    }

    return oss.str();
}

template<typename T>
concept arithmetic = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
    { a * b } -> std::same_as<T>;
    { a / b } -> std::same_as<T>;
};

template<arithmetic T>
class init_tensor {

    template<arithmetic U>
    friend std::ostream& operator<<(std::ostream&, const init_tensor<U>&);

    template<arithmetic U>
    friend void opHelper(std::ostream&, const init_tensor<U>&, size_t, size_t);

    protected:
        T *x = nullptr;
        size_t n = 0;
        std::vector<size_t> shape;
        std::vector<size_t> stride;

        using s_size_t = std::make_signed_t<size_t>;

        using init_tensor_0D = T;
        using init_tensor_1D = std::initializer_list<T>;
        using init_tensor_ND = std::initializer_list<init_tensor>;

        virtual bool mem_avail(void) const noexcept { return this->x; }

        [[nodiscard]] size_t calculate_stride(std::initializer_list<size_t> weights) const {
            if (weights.size() != this->stride.size())
                throw std::invalid_argument{"tensor::calculate_stride: invalid indices provided"};

            size_t i = 0, index = 0;
            for (const auto &w: weights) {
                if (w >= this->shape[i] || static_cast<ptrdiff_t>(w) < 0)
                    throw std::out_of_range{"tensor::calculate_stride: out of bounds access"};
                index += this->stride[i] * w;
                i++;
            }

            return index;
        }

    public:
        init_tensor(void): shape(0), stride(0) {}

        explicit init_tensor(const init_tensor_0D& scalar): n{1} {
            this->shape.resize(0);
            this->stride.resize(0);
            this->x = new T[this->n];
            *this->x = scalar;
        }

        init_tensor(as_shape_t, const std::vector<size_t>& shape): shape{shape} {
            this->n = 1;
            for(auto const& i: this->shape) 
                this->n *= i;         

            if (this->n != 0) {
                this->x = new T[this->n];
                std::memset(this->x, 0, this->n * sizeof(T));
                this->stride = init_tensor<T>::deduce_stride(this->shape);
            }
        }

        init_tensor(const init_tensor_1D list): n{list.size()} {
            shape.push_back(this->n);
            stride.push_back(1);

            if (this->n == 0) return;
            this->x = new T[this->n];
            std::copy(list.begin(), list.end(), this->x);
        }

        init_tensor(const init_tensor_ND list) {
            const size_t list_size = list.size();
            if (list_size == 0) return;

            const init_tensor<T> &first = *list.begin();
            
            this->shape.push_back(list_size);
            this->stride.push_back(first.n);
            this->shape.insert(this->shape.end(), first.shape.begin(), first.shape.end());
            this->stride.insert(this->stride.end(), first.stride.begin(), first.stride.end());
            this->n = list_size * first.n;
            
            if (this->n > 0) {
                this->x = new T[this->n];
                std::memset(this->x, 0, this->n * sizeof(T));

                T* alias = this->x;
                for(const init_tensor<T>& i: list) {
                    if (i.shape != first.shape) throw std::invalid_argument{"init_tensor::init_tensor: inconistent shape"};                
                    alias = std::copy(i.x, i.x + i.n, alias);
                } 
            }
        }

        init_tensor(const init_tensor& obj): n{obj.n}, shape{obj.shape}, stride{obj.stride} {
            if (this->n == 0) return;
            this->x = new T[this->n];
            std::copy(obj.x, obj.x + obj.n, this->x);
        }

        init_tensor(init_tensor&& obj) noexcept: x{obj.x}, n{obj.n}, shape{std::move(obj.shape)}, stride{std::move(obj.stride)} {
            obj.x = nullptr;
            obj.n = 0;
        }
        
        init_tensor& operator=(const init_tensor& obj) {
            if (this == &obj) return *this;
            
            this->n = obj.n;
            this->shape = obj.shape;
            this->stride = obj.stride;
            
            if (this->n == 0) return *this;
            this->x = new T[this->n];
            std::copy(obj.x, obj.x + obj.n, this->x);

            
            return *this;
        }

        init_tensor& operator=(init_tensor&& obj) noexcept {
            if (this == &obj) return *this;

            this->x = obj.x;
            this->n = obj.n;
            this->shape = std::move(obj.shape);
            this->stride = std::move(obj.stride);
            
            obj.x = nullptr;
            obj.n = 0;


            return *this;
        }

        init_tensor& operator=(const init_tensor_1D list) {
            if (this->x) delete[] this->x;
            this->shape.resize(0);
            this->stride.resize(0);
            
            this->n = list.size();            
            shape.push_back(this->n);
            stride.push_back(1);

            if (this->n == 0) return *this;
            
            this->x = new T[this->n];
            std::copy(list.begin(), list.end(), this->x);   
            
            return *this;
        }

        init_tensor& operator=(const init_tensor_ND list) {  
            if (this->x) delete[] this->x;
            this->shape.resize(0);
            this->stride.resize(0);
            this->n = 0;
            
            const size_t list_size = list.size();
            if (list_size == 0) return *this;

            const init_tensor<T> &first = *list.begin();
            
            this->shape.push_back(list_size);
            this->stride.push_back(first.n);
            this->shape.insert(this->shape.end(), first.shape.begin(), first.shape.end());
            this->stride.insert(this->stride.end(), first.stride.begin(), first.stride.end());
            this->n = list_size * first.n;

            if (this->n == 0) return *this;

            this->x = new T[this->n];
            std::memset(this->x, 0, this->n * sizeof(T));

            T* alias = this->x;
            for(const init_tensor<T>& i: list) {
                if (i.shape != first.shape) throw std::invalid_argument{"init_tensor::operator=: inconistent shape"};                
                alias = std::copy(i.x, i.x + i.n, alias);
            }             

            return *this;
        }

        init_tensor& operator=(const init_tensor_0D& scalar) {
            if (x) delete[] x;
            this->x = nullptr;
            this->n = 1;
            this->shape.resize(0);
            this->stride.resize(0);

            this->x = new T[this->n];
            *(this->x) = scalar;

            return *this;
        }

        template<typename... Indices>
        [[nodiscard]] T& operator()(Indices... indices) {
            constexpr size_t count = sizeof...(indices);
            static_assert((std::is_integral_v<Indices> && ...), "init_tensor::operator(): all arguments must be unsigned numbers");
            if (!this->mem_avail() || count != this->dim()) throw std::out_of_range{"init_tensor::operator(): invalid access"};

            size_t linear_index = this->calculate_stride({ static_cast<size_t>(indices)... });

            return this->x[linear_index];
        }

        template<typename... Indices>
        [[nodiscard]] const T& operator()(Indices... indices) const {
            constexpr size_t count = sizeof...(indices);
            static_assert((std::is_integral_v<Indices> && ...), "init_tensor::operator(): all arguments must be unsigned numbers");
            if (!this->mem_avail() || count != this->dim()) throw std::out_of_range{"init_tensor::operator(): invalid access"};

            size_t linear_index = this->calculate_stride({ static_cast<size_t>(indices)... });

            return this->x[linear_index];
        }

        [[nodiscard]] inline size_t size(void) const noexcept { return this->n; }
        [[nodiscard]] inline std::vector<size_t> get_shape(void) const noexcept { return this->shape; }
        [[nodiscard]] inline std::vector<size_t> get_stride(void) const noexcept { return this->stride; }
        [[nodiscard]] inline size_t dim(void) const noexcept { return this->shape.size(); }
        [[nodiscard]] inline const T *raw(void) const noexcept { return this->x; }

        init_tensor<T>& reshape(const std::vector<size_t>& shape) { 
            size_t product = 1;
            for(const auto& i: shape)
                product *= i;
            
            if (product != this->n)
                throw std::invalid_argument{"init_tensor::reshape: cannot reshape, total size doesn't match"};

            this->shape.resize(shape.size());
            std::copy(shape.begin(), shape.end(), this->shape.begin());
            this->stride = init_tensor<T>::deduce_stride(this->shape);

            return *this; 
        }

        init_tensor<T>& resize(const std::vector<size_t>& shape) {
            if (this->x) delete[] this->x;

            this->shape = shape;
            this->stride = init_tensor<T>::deduce_stride(this->shape);
            this->n = 1;

            if (this->shape.size() > 0) {
                for(const auto& i: this->shape)
                    n *= i;
            }

            if (this->n != 0) this->x = new T[this->n];
            std::memset(this->x, 0, this->n);

            return *this;        
        }

        init_tensor& assign(const init_tensor_1D list) {
            if (!mem_avail())
                throw std::runtime_error{"init_tensor::assign: shape unavailable or not set"};
            
            std::vector<size_t> shape = init_tensor<T>::deduce_shape(list);
            if (this->shape != shape) {
                throw std::invalid_argument{
                    "init_tensor::assign: shape mismatch - " + vec_to_str(this->shape) + " != " + vec_to_str(shape)
                };
            }
            
            if (this->n == 0) return *this;
            std::copy(list.begin(), list.end(), this->x);

            return *this;
        }
        
        init_tensor& assign(const init_tensor_ND list) {
            if (!mem_avail())
                throw std::runtime_error{"init_tensor::assign: shape unavailable or not set"};
                
            std::vector<size_t> shape = init_tensor<T>::deduce_shape(list);
            if (this->shape != shape) {
                throw std::invalid_argument{
                    "init_tensor::assign: shape mismatch - " + vec_to_str(this->shape) + " != " + vec_to_str(shape)
                };
            }

            if (this->n == 0) return *this;
            
            T* alias = this->x;
            for(const init_tensor<T>& i: list)          
                alias = std::copy(i.x, i.x + i.n, alias);
            
            return *this;
        }
        
        init_tensor& assign(const T& scalar) {
            if (!this->mem_avail() || this->shape.empty())
                throw std::runtime_error{"init_tensor::assign: shape unavailable or inconsistent"};

            *(this->x) = scalar;

            return *this;
        }

        init_tensor& flat_assign(const std::vector<T> list) {
            if (!this->mem_avail())
                throw std::runtime_error{"init_tensor::flat_assign: memory unavailable"};

            if (this->n != list.size()) {
                throw std::invalid_argument{
                    "init_tensor::flat_assign: list size doesn't match tensor size - " + list.size() + " != " + this->n
                };
            }

            if (this->n == 0) return *this;
            std::copy(list.begin(), list.end(), this->x);

            return *this;
        }

        static std::vector<size_t> deduce_shape(const init_tensor_1D list) { 
            return {list.size()}; 
        }

        static std::vector<size_t> deduce_shape(const init_tensor_ND list) {
            std::vector<size_t> shape;

            const size_t list_size = list.size();
            if (list_size == 0) return shape;

            const init_tensor<T> &first = *list.begin();

            shape.push_back(list_size);
            shape.insert(shape.end(), first.shape.begin(), first.shape.end());

            for(const init_tensor<T>& i: list) {
                if (i.shape != first.shape)
                    throw std::invalid_argument{"init_tensor::deduce_shape: inconistent shape"};
            }

            return shape;
        }

        static std::vector<size_t> deduce_stride(const std::vector<size_t>& shape) noexcept {
            const auto shape_size = static_cast<s_size_t>(shape.size());
            if (shape_size == 0) return {};

            std::vector<size_t> stride(shape_size);
            stride[shape_size - 1] = 1;
            if (shape_size > 1) {
                for(s_size_t i = shape_size - 2; i >= 0; i--)
                    stride[i] = shape[i + 1] * stride[i + 1];
            }

            return stride;
        }

        static std::vector<T> flatten(const init_tensor_ND list) {
            std::vector<size_t> shape;
            std::vector<T> flat;

            const size_t list_size = list.size();
            if (list_size == 0) return flat;
            
            const init_tensor<T> &first = *list.begin();
            
            shape.push_back(list_size);
            shape.insert(shape.end(), first.shape.begin(), first.shape.end());
            size_t n = list_size * first.n;
            
            flat.resize(n);
            auto alias = flat.begin();
            for(const init_tensor<T>& i: list) {
                if (i.shape != first.shape) throw std::invalid_argument{"init_tensor::flatten: inconsistent shape"}; 
                alias = std::copy(i.x, i.x + i.n, alias);
            }

            return flat;
        }

        virtual ~init_tensor(void) {
            if (this->x) delete[] x;

            this->n = 0;
            this->shape.resize(0);
            this->stride.resize(0);
        }
};

template<arithmetic T>
void opHelper(std::ostream& output, const init_tensor<T>& obj, size_t index, size_t dim = 1) {
    
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
}

template<arithmetic T>
std::ostream& operator<<(std::ostream& output, const init_tensor<T>& obj) {
    if (!obj.mem_avail())
        return output;

    if (obj.dim() == 0) {
        output << *obj.x;
        return output;
    }

    opHelper(output, obj, 0);

    return output;
}