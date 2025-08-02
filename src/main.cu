#include<iostream>
#include<vector>
#include<initializer_list>
#include<tensor.cuh>
#include<memory>
#include<signal.h>

using namespace std;

int main(void) {
    cout << endl;
    using tensor_d = tensor<double>;

    tensor_d a = tensor_d({1, 2, 3});
    tensor_d b = tensor_d({4, 5, 6});
    tensor_d c = tensor_d({7, 8, 9});

    auto result = tensor_d::add(a, b);

    cout << result << endl;

    cout << endl;
    return 0;
} 