#include<tensor.cuh>


// TODO
template<typename T>
void opHelper(const std::ostream& output, tensor<T>& obj, const T*& raw_ptr, size_t dim) {
    if (dim > obj.dim()) {

        return;
    }

    output << '[';
    opHelper(output, obj, dim + 1);
    output << "\b\b], ";

    return;
}

template<typename T>
std::ostream& operator<<(const std::ostream& output, tensor<T>& obj) {
    if (obj.dim() == 0) {
        output << *obj.h_x;
        return output;
    }

    T* raw_ptr = obj.h_x;
    opHelper(output, obj, raw_ptr);
    output << "\b\b";

    return output;
}