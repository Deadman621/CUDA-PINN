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
    using dtype = float;

    init_tensor_2D<dtype> a = {{1, 2}, 
                               {3, 4}, 
                               {1, 2}, 
                               {3, 4}};
                               
    init_tensor_2D<dtype> b = {{1, 2}, 
                               {3, 4}};

    tensor<dtype> x = a;
    tensor<dtype> y = b;
    
    tensor<dtype> p = a;

    //p.reshape({1});
    cout << "dimension = " << p.dim() << endl;
    cout << "size = " << p.size() << endl;
    cout << "p.shape = [";
    for(int i = 0; i < p.get_shape().size(); i++)
        cout << p.get_shape()[i] << ", ";
    cout << "\b\b]" << endl;    
    cout << "p.stride = [";
    for(int i = 0; i < p.get_stride().size(); i++)
        cout << p.get_stride()[i] << ", ";
    cout << "\b\b]" << endl;
    cout << "p.h_x = " << p << endl;

    return 0;
}   