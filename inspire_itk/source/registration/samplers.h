
#ifndef SAMPLERS_H
#define SAMPLERS_H

#include "itkSize.h"
#include "itkPoint.h"
#include "itkImage.h"
#include "itkSmartPointer.h"
#include "itkImageRegionIterator.h"
#include "itkNumericTraits.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

class RandomSampler {
    private:
        typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
        typedef typename GeneratorType::Pointer GeneratorPointer;
        
        GeneratorPointer m_Generator;
        unsigned int m_TotalIndices;
    public:
    RandomSampler() {
        m_Generator = GeneratorType::New();
        m_TotalIndices = 1;
    }
    void SetTotalIndices(unsigned int totalIndices) {
        m_TotalIndices = totalIndices;
    }
    void SetSeed(unsigned int seed) {
        m_Generator->SetSeed(seed);    
    }
    void Sample(unsigned int n, std::vector<itk::Vector<double, 1U> >& out) {
        out.clear();
        for (unsigned int i = 0; i < n; ++i) {
            double c = m_Generator->GetVariateWithClosedRange();
            out.push_back(c);
        }
    }
    void SampleIndices(unsigned int n, std::vector<unsigned int>& out) {
        out.clear();
        for (unsigned int i = 0; i < n; ++i) {
            unsigned int c = m_Generator->GetIntegerVariate(m_TotalIndices-1);
            out.push_back(c);
        }
    }        

};

// Quasi Monte Carlo Sampler
template <unsigned int Dim>
class QMCSampler {
};

template <>
class QMCSampler<1U>
{   
private:
    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
    typedef typename GeneratorType::Pointer GeneratorPointer;
    
    GeneratorPointer m_Generator;

    double m_Alpha;
    double m_State;
    itk::Size<1U> m_Size;
public:
    QMCSampler() {
        double phi1 = 1.61803398874989484820458683436563;
        m_Alpha = 1.0/phi1;
        m_Generator = GeneratorType::New();
        m_State = 0.5;
    }
    void SetSeed(unsigned int seed) { m_Generator->SetSeed(seed); }
    void SetSize(itk::Size<1U> sz) {
        m_Size = sz;
    }

    void Restart() {
        m_State = m_Generator->GetVariateWithOpenUpperRange();
    }

    void Sample(unsigned int n, std::vector<itk::Vector<double, 1U> >& out) {
        double alpha = m_Alpha;
        double state = m_State;
        out.clear();
        for (unsigned int i = 0; i < n; ++i) {
            state = fmod(state + alpha, 1.0);
            itk::Vector<double, 1U> statev;
            statev[0] = state;
            out.push_back(statev);
        }
        m_State = state;
    }
    void SampleIndices(unsigned int n, std::vector<unsigned int>& out) {
        double alpha = m_Alpha;
        double state = m_State;
        itk::Size<1U> sz = m_Size;
        out.clear();
        for (unsigned int i = 0; i < n; ++i) {
            state = fmod(state + alpha, 1.0);
            unsigned int c = (unsigned int)(state*sz[0]);
            out.push_back(c);
        }
        m_State = state;
    }
};

template <>
class QMCSampler<2U>
{
private:
    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
    typedef typename GeneratorType::Pointer GeneratorPointer;
    
    GeneratorPointer m_Generator;

    itk::Vector<double, 2U> m_Alpha;
    itk::Vector<double, 2U> m_State;
    itk::Size<2U> m_Size;
public:
    QMCSampler() {
        double phi2 = 1.32471795724474602596090885447809;
        m_Alpha[0] = 1.0/phi2;
        m_Alpha[1] = pow(1.0/phi2, 2.0);
        m_State[0] = 0.5;
        m_State[1] = 0.5;
        m_Generator = GeneratorType::New();
    }
    void SetSeed(unsigned int seed) { m_Generator->SetSeed(seed); }
    void SetSize(itk::Size<2U> sz) {
        m_Size = sz;
    }

    void Restart() {
        m_State[0] = m_Generator->GetVariateWithOpenUpperRange();
        m_State[1] = m_State[0];
    }

    void Sample(unsigned int n, std::vector<itk::Vector<double, 2U> >& out) {
        itk::Vector<double, 2U> alpha = m_Alpha;
        itk::Vector<double, 2U> state = m_State;
        out.clear();
        for (unsigned int i = 0; i < n; ++i) {
            state[0] = fmod((state[0] + alpha[0]), 1.0);
            state[1] = fmod((state[1] + alpha[1]), 1.0);

            out.push_back(state);
        }
        
        m_State = state;
    }

    void SampleIndices(unsigned int n, std::vector<unsigned int>& out) {
        itk::Vector<double, 2U> alpha = m_Alpha;
        itk::Vector<double, 2U> state = m_State;
        itk::Size<2U> sz = m_Size;
        out.clear();
        for (unsigned int i = 0; i < n; ++i) {
            state[0] = fmod((state[0] + alpha[0]), 1.0);
            state[1] = fmod((state[1] + alpha[1]), 1.0);

            unsigned int c1 = (unsigned int)(state[0]*sz[0]);
            unsigned int c2 = (unsigned int)(state[1]*sz[1]);

            out.push_back(c2*sz[0] + c1);
        }
        
        m_State = state;
    }
};

template <>
class QMCSampler<3U>
{
private:
    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
    typedef typename GeneratorType::Pointer GeneratorPointer;
    
    GeneratorPointer m_Generator;

    itk::Vector<double, 3U> m_Alpha;
    itk::Vector<double, 3U> m_State;
    itk::Size<3U> m_Size;
public:
    QMCSampler() {
        double phi3 = 1.220744084605759475361686349108831;
        m_Alpha[0] = 1.0/phi3;
        m_Alpha[1] = pow(1.0/phi3, 2.0);
        m_Alpha[2] = pow(1.0/phi3, 3.0);
        m_State[0] = 0.5;
        m_State[1] = 0.5;
        m_State[2] = 0.5;
        m_Generator = GeneratorType::New();
    }
    void SetSeed(unsigned int seed) { m_Generator->SetSeed(seed); }
    void SetSize(itk::Size<3U> sz) {
        m_Size = sz;
    }

    void Restart() {
        m_State[0] = m_Generator->GetVariateWithOpenUpperRange();
        m_State[1] = m_State[0];
        m_State[2] = m_State[0];
    }

    void Sample(unsigned int n, std::vector<itk::Vector<double, 3U> >& out) {
        itk::Vector<double, 3U> alpha = m_Alpha;
        itk::Vector<double, 3U> state = m_State;
        out.clear();
        for (unsigned int i = 0; i < n; ++i) {
            state[0] = fmod((state[0] + alpha[0]), 1.0);
            state[1] = fmod((state[1] + alpha[1]), 1.0);
            state[2] = fmod((state[2] + alpha[2]), 1.0);

            out.push_back(state);
        }
        
        m_State = state;
    }

    void SampleIndices(unsigned int n, std::vector<unsigned int>& out) {
        itk::Vector<double, 3U> alpha = m_Alpha;
        itk::Vector<double, 3U> state = m_State;
        itk::Size<3U> sz = m_Size;
        out.clear();
        //std::cout << sz << std::endl;
        for (unsigned int i = 0; i < n; ++i) {
            state[0] = fmod((state[0] + alpha[0]), 1.0);
            state[1] = fmod((state[1] + alpha[1]), 1.0);
            state[2] = fmod((state[2] + alpha[2]), 1.0);
            unsigned int c1 = (unsigned int)(state[0]*sz[0]);
            unsigned int c2 = (unsigned int)(state[1]*sz[1]);
            unsigned int c3 = (unsigned int)(state[2]*sz[2]);
            unsigned int index = c3*sz[0]*sz[1] + c2*sz[0] + c1;

            //if(i + 1 == n) {
            //    std::cout << "Coord: " << c1 << ", " << c2 << ", " << c3 << ", (" << index << ")" << std::endl;
            //}
            out.push_back(index);
        }

        m_State = state;
    }
};


#endif
