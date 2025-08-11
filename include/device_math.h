#pragma once

#include<concepts>

namespace device_math {
    template<typename T>
    concept arithmetic = requires(T a, T b) {
        { a + b } -> std::same_as<T>;
        { a - b } -> std::same_as<T>;
        { a * b } -> std::same_as<T>;
        { a / b } -> std::same_as<T>;
        { a == b } -> std::convertible_to<bool>;
    };

    template<typename T>
    concept device_compatible =
        std::is_trivially_copyable_v<T> &&
        std::is_standard_layout_v<T>;

    template<typename T>
    concept device_arithmetic =
        arithmetic<T> && device_compatible<T>;
}