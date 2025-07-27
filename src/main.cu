#include<iostream>
#include<vector>
#include<initializer_list>
#include<tensor.cuh>

using namespace std;

// For now, use these. I'll make it generic later.
template<typename T>
using init_tensor_0D = T;

template<typename T>
using init_tensor_1D = initializer_list<T>;

template<typename T>
using init_tensor_2D = initializer_list<init_tensor_1D<T>>;

template<typename T>
using init_tensor_3D = initializer_list<init_tensor_2D<T>>;

template<typename T>
using init_tensor_4D = initializer_list<init_tensor_3D<T>>;

int main(void) {
    using dtype = double;

    cout << endl;

    init_tensor_1D<dtype> a = {1, 2, 3, 4};
    init_tensor_1D<dtype> b = {1, 2, 3, 4};
    tensor<dtype> x(a), y(b);

    tensor<dtype> z = x * 2;

    cout << "X = " << x << endl;
    cout << "Y = " << y << endl;
    cout << "Z = " << z << endl;

    cout << endl;

    return 0;
} 