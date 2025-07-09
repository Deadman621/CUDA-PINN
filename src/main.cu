#include<iostream>
#include<utils.cuh>
#include<vector>
#include<initializer_list>

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
    typedef float dtype;

    init_tensor_1D<dtype> a = {1, 2, 3, 4};
    init_tensor_2D<dtype> b = {{1, 2}, {3, 4}};
    
    tensor<dtype> v(a);
    tensor<dtype> s(b);
    tensor<dtype> t = 1;
    
    /*     cout << "X = [";
    for(int i = 0; i < t.size(); i++)
    cout << t.raw()[i] << ", ";
    cout << "\b\b]" << endl; */
    
    
    v.reshape({1, 4});
    cout << "dimension = " << v.dim() << endl;
    cout << "size = " << v.size() << endl;
    cout << "V = [";
    for(int i = 0; i < v.get_stride().size(); i++)
        cout << v.get_stride()[i] << ", ";
    cout << "\b\b]" << endl;
    
    return 0;
}   