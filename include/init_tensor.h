#pragma once

#include<initializer_list>
#include<algorithm>
#include<stdexcept>
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

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
class init_tensor {

    template<typename U>
    friend std::ostream& operator<<(std::ostream&, const init_tensor<U>&);

    template<typename U>
    friend void opHelper(std::ostream&, const init_tensor<U>&, size_t, size_t);

    protected:
        T *x = nullptr;
        size_t n = 0;
        std::vector<size_t> shape;
        std::vector<size_t> stride;

        virtual bool mem_avail(void) const noexcept { return this->x; }

        size_t calculate_stride(std::initializer_list<size_t> weights) const {
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

        init_tensor(const T& scalar): n{1} {
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

        init_tensor(const std::initializer_list<T> list): n{list.size()} {
            shape.push_back(this->n);
            stride.push_back(1);

            if (this->n == 0) return;
            this->x = new T[this->n];
            std::copy(list.begin(), list.end(), this->x);
        }

        init_tensor(const std::initializer_list<init_tensor<T>> list) {
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

        init_tensor& operator=(const std::initializer_list<T> list) {
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

        init_tensor& operator=(const std::initializer_list<init_tensor<T>> list) {  
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

        init_tensor& operator=(const T& scalar) {
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
        T& operator()(Indices... indices) {
            constexpr size_t count = sizeof...(indices);
            static_assert((std::is_integral_v<Indices> && ...), "init_tensor::operator(): all arguments must be unsigned numbers");
            if (!this->mem_avail() || count != this->dim()) throw std::out_of_range{"init_tensor::operator(): invalid access"};

            size_t linear_index = this->calculate_stride({ static_cast<size_t>(indices)... });

            return this->x[linear_index];
        }

        inline size_t size(void) const noexcept { return this->n; }
        inline std::vector<size_t> get_shape(void) const noexcept { return this->shape; }
        inline std::vector<size_t> get_stride(void) const noexcept { return this->stride; }
        inline size_t dim(void) const noexcept { return this->shape.size(); }
        inline const T *raw(void) const noexcept { return this->x; }

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
            for(const auto& i: this->shape)
                n *= i;

            if (this->n != 0) this->x = new T[this->n];

            return *this;        
        }
        
        init_tensor& assign(const std::initializer_list<init_tensor<T>> list) {
            if (!mem_avail())
                throw std::runtime_error{"init_tensor::assign: shape unavailable or not set"};
                
            std::vector<size_t> __shape__ = init_tensor<T>::deduce_shape(list);
            if (this->shape != __shape__) {
                throw std::invalid_argument{
                    "init_tensor::assign: shape mismatch - " + vec_to_str(this->shape) + " != " + vec_to_str(__shape__)
                };
            }

            if (this->n == 0) return *this;
            
            T* alias = this->x;
            for(const init_tensor<T>& i: list)          
                alias = std::copy(i.x, i.x + i.n, alias);
            
            return *this;
        }
        
        init_tensor& assign(const T& scalar) {
            if (!this->mem_avail() || !this->shape.empty())
            throw std::runtime_error{"init_tensor::assign: shape unavailable or inconsistent"};
            
            *(this->x) = scalar;
            return *this;
        }
        
        static std::vector<size_t> deduce_shape(const std::initializer_list<init_tensor<T>> list) {
            std::vector<size_t> __shape__;
            
            const size_t list_size = list.size();
            if (list_size == 0) return __shape__;
            
            const init_tensor<T> &first = *list.begin(); 
        
            __shape__.push_back(list_size);
            __shape__.insert(__shape__.end(), first.shape.begin(), first.shape.end());
        
            for(const init_tensor<T>& i: list) {
                if (i.shape != first.shape) 
                    throw std::invalid_argument{"init_tensor::deduce_shape: inconistent shape"};                
            }
        
            return __shape__;
        }

        static std::vector<size_t> deduce_stride(const std::vector<size_t>& shape) {
            const size_t shape_size = shape.size();
            if (shape_size == 0) throw std::runtime_error{"init_tensor::deduce_stride: invalid shape"};

            std::vector<size_t> __stride__(shape_size);
            __stride__[shape_size - 1] = 1;
            if (shape_size > 1) {
                for(int i = shape_size - 2; i >= 0; i--) 
                    __stride__[i] = shape[i + 1] * __stride__[i + 1];
            }

            return __stride__;
        }

        static std::vector<T> flatten(const std::initializer_list<init_tensor<T>> list) {
            std::vector<size_t> __shape__;
            std::vector<T> __flat__;

            const size_t list_size = list.size();
            if (list_size == 0) return __flat__;
            
            const init_tensor<T> &first = *list.begin();
            
            __shape__.push_back(list_size);
            __shape__.insert(__shape__.end(), first.shape.begin(), first.shape.end());
            size_t n = list_size * first.n;
            
            __flat__.resize(n);
            for(const init_tensor<T>& i: list) {
                if (i.shape != first.shape) throw std::invalid_argument{"init_tensor::flatten: inconsistent shape"}; 
                auto start = __flat__.begin(), end = start + i.n;               
                std::copy(i.x, i.x + i.n, __flat__.begin());
            }                          
        }

        ~init_tensor(void) {
            if (this->x) delete[] x;

            this->n = 0;
            this->shape.resize(0);
            this->stride.resize(0);
        }
};

template<typename T>
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

    return;
}

template<typename T>
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