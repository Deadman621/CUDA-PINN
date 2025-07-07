#include<iostream>
#include<utils.cuh>
#include<vector>
#include<initializer_list>

using namespace std;

// For now, use these. I'll make it generic later.
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

    init_tensor_1D<dtype> b = {1, 2, 3, 4, 5, 6};

    tensor<dtype> t(b);
    cout << "dimension = " << t.dim() << endl;
    cout << "size = " << t.size() << endl;
    cout << "X = [";
    for(int i = 0; i < t.size(); i++)
        cout << t.raw()[i] << ", ";
    cout << "\b\b]" << endl;

    auto x = *b.begin();

    std::vector<int> v = {1, 2, 3};
    auto a = v[0];  

    return 0;
}   