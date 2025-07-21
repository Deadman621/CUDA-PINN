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

    init_tensor_2D<dtype> d_inputs =  {{1.0, 2.0, 3.0, 2.5}, {2.0, 5.0, -1.0, 2.0}, {-1.5, 2.7, 3.3, -0.8}};
    init_tensor_2D<dtype> d_weights = {{0.2, 0.8, -0.5, 1.0}, {0.5, -0.91, 0.26, -0.5}, {-0.26, -0.27, 0.17, 0.87}};
    init_tensor_2D<dtype> d_biases = {{2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}, {0.5, 0.5, 0.5}};

    tensor<dtype> inputs(d_inputs), weights(d_weights), biases(d_biases);

    auto layer_outputs = inputs * tensor<dtype>::transpose(weights) /* + biases */;

    cout << endl << layer_outputs << endl;

    return 0;
} 