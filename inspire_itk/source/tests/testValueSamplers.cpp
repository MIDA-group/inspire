
#include "../samplers/valueSampler.h"

namespace ValueSamplerTests {

template <typename SamplerType>
void RunValueSampler()
{
    using SamplerPointer = typename SamplerType::Pointer;
    using VectorType = itk::FixedArray<float, 2U>;
    using VectorType2 = itk::FixedArray<unsigned short, 2U>;

    SamplerPointer sampler = SamplerType::New();

    sampler->SetSeed(64U);

    sampler->RestartFromSeed();

    std::vector<VectorType> out1;
    std::vector<VectorType2> out2;

    unsigned int count = 32U;
    sampler->SampleN(out1, count, 0);
    sampler->SampleValues(out2, 256.0, count, 0);

    std::cout << "[";
    for(unsigned int i = 0; i < count; ++i) {
        if(i > 0)
          std::cout << ", ";
        std::cout << out1[i];
    }
    std::cout << "]" << std::endl << "[";
    for(unsigned int i = 0; i < count; ++i) {
        if(i > 0)
          std::cout << ", ";
        std::cout << out2[i];
    }
    std::cout << "]" << std::endl;
}

void RunValueSamplersTests()
{
    RunValueSampler<UniformValueSampler<float, 2U>>();
    RunValueSampler<QuasiRandomValueSampler<float, 2U>>();
}

}
