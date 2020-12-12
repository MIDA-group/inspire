
#ifndef QUASI_RANDOM_GENERATOR_H
#define QUASI_RANDOM_GENERATOR_H

#include "itkSize.h"
#include "itkPoint.h"
#include "itkImage.h"
#include "itkSmartPointer.h"
#include "itkNumericTraits.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

#include <limits>

double QuasiRandomSqrt1D(unsigned int index, double state=0.0)
{
    static double alpha = 1.0/sqrt(2.0);
    double tmp = state + (index+1) * alpha;
    return tmp - (unsigned int)tmp;
}

double QuasiRandom3Sqrt1D(unsigned int index, double state=0.0)
{
    static double alpha = 1.0/sqrt(3.0);
    double tmp = state + (index+1) * alpha;
    return tmp - (unsigned int)tmp;
}

double QuasiRandomPI1D(unsigned int index, double state=0.0)
{
    constexpr double alpha = 1.0/M_PI;
    double tmp = state + (index+1) * alpha;
    return tmp - (unsigned int)tmp;
}

double QuasiRandomPHI1D(unsigned int index, double state=0.0)
{
    constexpr double alpha = 1.0/1.61803398874989484820458683436563;
    double tmp = state + (index+1) * alpha;
    return tmp - (unsigned int)tmp;    
}

inline
unsigned long long XORShiftRNG1D(unsigned long long x)
{
    x ^= x << 13U;
	x ^= x >> 7U;
	x ^= x << 17U;
    return x;
}

inline
double XORShiftRNGDouble1D(unsigned long long x)
{
    constexpr double DENOM = 1.0/(double)std::numeric_limits<unsigned long long>::max();
    x = XORShiftRNG1D(x);
    return DENOM * XORShiftRNG1D(x);
}

template <unsigned int Dim>
inline itk::FixedArray<unsigned long long, Dim> XORShiftRNG(unsigned long long x)
{
    itk::FixedArray<unsigned long long, Dim> result;
    x = XORShiftRNG1D(x);
    //x = XORShiftRNG1D(x);
    //x = XORShiftRNG1D(x);
    for(unsigned int i = 0; i < Dim; ++i)
    {
        x = XORShiftRNG1D(x);
        result[i] = x;
    }

    return result;
}

template <unsigned int Dim>
inline itk::FixedArray<double, Dim> XORShiftRNGDouble(unsigned long long x)
{
    itk::FixedArray<double, Dim> result;
    x = XORShiftRNG1D(x);
    //x = XORShiftRNG1D(x);
    //x = XORShiftRNG1D(x);
    constexpr double DENOM = 1.0/(double)std::numeric_limits<unsigned long long>::max();
    for(unsigned int i = 0; i < Dim; ++i)
    {
        x = XORShiftRNG1D(x);
        result[i] = x * DENOM;
    }

    return result;
}

template <unsigned int Dim>
itk::Vector<double, Dim> MakeQuasiRandomGeneratorAlpha();

template <>
itk::Vector<double, 1U> MakeQuasiRandomGeneratorAlpha<1U>() {
    itk::Vector<double, 1U> result;
    double phi1 = 1.61803398874989484820458683436563;
    result[0] = 1.0/phi1;
    
    return result;
}

template <>
itk::Vector<double, 2U> MakeQuasiRandomGeneratorAlpha<2U>() {
    itk::Vector<double, 2U> result;
    double phi2 = 1.32471795724474602596090885447809;
    result[0] = 1.0/phi2;
    result[1] = pow(1.0/phi2, 2.0);
    
    return result;
}

template <>
itk::Vector<double, 3U> MakeQuasiRandomGeneratorAlpha<3U>() {
    itk::Vector<double, 3U> result;
    double phi3 = 1.220744084605759475361686349108831;
    result[0] = 1.0/phi3;
    result[1] = pow(1.0/phi3, 2.0);
    result[2] = pow(1.0/phi3, 3.0);

    return result;
}

// Add the N-dimensional quasi random generator implementation

template <unsigned int Dim>
class QuasiRandomGenerator : public itk::Object {
public:
    using Self = QuasiRandomGenerator<Dim>;
    using Superclass = itk::Object;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    using ValueType = itk::FixedArray<double, Dim>;

    itkNewMacro(Self);
  
    itkTypeMacro(QuasiRandomGenerator, itk::Object);

    void SetSeed(unsigned int seed) {
        m_Seed = seed;
        Restart();
    }

    void Restart() {
        m_State = XORShiftRNGDouble<Dim>(m_Seed);
    }

    ValueType GetVariate() {
        for(unsigned j = 0; j < Dim; ++j) {
            m_State[j] = fmod((m_State[j] + m_Alpha[j]), 1.0);
        }
        return m_State;
    }

    ValueType GetConstVariate(unsigned int index) const
    {
        ValueType result;
        double index_dbl = static_cast<double>(index);
        for(unsigned j = 0; j < Dim; ++j) {
            //result[j] = fmod(m_State[j] + index_plus_1 * m_Alpha[j], 1.0);
            double tmp = m_State[j] + index_dbl * m_Alpha[j];
            result[j] = tmp - (unsigned int)tmp;
        }
        return result;
        
    }

    void Advance(unsigned int index)
    {
        m_State = GetConstVariate(index);
    }
protected:
    QuasiRandomGenerator() {
        m_Alpha = MakeQuasiRandomGeneratorAlpha<Dim>();
        m_Seed = 42U;
        Restart();
    }

    itk::FixedArray<double, Dim> m_Alpha;
    itk::FixedArray<double, Dim> m_State;
    itk::Size<Dim> m_Size;

    unsigned int m_Seed;
};

#endif
