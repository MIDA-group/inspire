
#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <cmath>
#include <cstdint>

template <typename InType, typename OutType>
OutType QuantizeValue(InType input)
{
    return static_cast<OutType>(input);
}

template <>
auto QuantizeValue<unsigned char, unsigned short>(unsigned char input) -> unsigned short
{
    return static_cast<unsigned short>(input) * ((unsigned short)256U);
}

template <>
auto QuantizeValue<unsigned short, unsigned char>(unsigned short input) -> unsigned char
{
    return static_cast<unsigned char>(input / ((unsigned short)256U));
}

template <>
auto QuantizeValue<float, unsigned char>(float input) -> unsigned char
{
    float scaledInput = floor((double)input * 255.0 + 0.5);
    if(scaledInput <= 0.0f)
        return 0U;
    if(scaledInput >= 255.0f)
        return 255U;
    return static_cast<unsigned char>(scaledInput);
}

template <>
auto QuantizeValue<double, unsigned char>(double input) -> unsigned char
{
    double scaledInput = floor(input * 255.0 + 0.5);
    if(scaledInput <= 0.0)
        return (unsigned char)0U;
    if(scaledInput >= 255.0)
        return (unsigned char)255U;
    return static_cast<unsigned char>(scaledInput);
}

template <>
auto QuantizeValue<float, unsigned short>(float input) -> unsigned short
{
    float scaledInput = floor((double)input * 65535.0 + 0.5);
    if(scaledInput <= 0.0f)
        return (unsigned short)0U;
    if(scaledInput >= 65535.0f)
        return (unsigned short)65535U;
    return static_cast<unsigned short>(scaledInput);
}

template <>
auto QuantizeValue<double, unsigned short>(double input) -> unsigned short
{
    double scaledInput = floor(input * 65535.0 + 0.5);
    if(scaledInput <= 0.0)
        return (unsigned short)0U;
    if(scaledInput >= 65535.0)
        return (unsigned short)65535U;
    return static_cast<unsigned short>(scaledInput);
}

template <typename T>
inline T QuantizedValueMax()
{
    return static_cast<T>(1); // Only holds for floating point values
}

template <>
inline unsigned char QuantizedValueMax<unsigned char>()
{
    return ((unsigned char)255U);
}

template <>
inline unsigned short QuantizedValueMax<unsigned short>()
{
    return ((unsigned short)65535U);
}

template <typename T>
inline T QuantizedValueMin()
{
    return static_cast<T>(0);
}

template <>
inline unsigned char QuantizedValueMin<unsigned char>()
{
    return ((unsigned char)0U);
}

template <>
inline unsigned short QuantizedValueMin<unsigned short>()
{
    return ((unsigned short)0U);
}

// Fixed point numbers
#define USE_FIXED_POINT

#ifdef USE_FIXED_POINT
using FixedPointNumber = int64_t;

constexpr unsigned long long SHIFTED_ONE = 1ULL << 24ULL;

inline FixedPointNumber FixedPointFromDouble(double a)
{
    return static_cast<FixedPointNumber>(a * static_cast<double>(SHIFTED_ONE));
}

inline double DoubleFromFixedPoint(FixedPointNumber a)
{
    return static_cast<double>(a) / static_cast<double>(SHIFTED_ONE);
}
#else
using FixedPointNumber = double;

FixedPointNumber FixedPointFromDouble(double a)
{
    return a;
}

double DoubleFromFixedPoint(FixedPointNumber a)
{
    return a;
}

#endif

#endif

