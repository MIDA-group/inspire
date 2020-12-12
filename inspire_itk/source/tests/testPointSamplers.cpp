
#include "itkImage.h"
#include "../samplers/pointSampler.h"

namespace PointSamplerTests {
using ImageType = itk::Image<float, 2U>;
using ImagePointer = typename ImageType::Pointer;

ImagePointer MakeTestImage() {
    ImageType::RegionType region;
    ImageType::IndexType index;
    ImageType::SizeType size;

    index[0] = 0;
    index[1] = 0;
    size[0] = 64;
    size[1] = 64;

    region.SetIndex(index);
    region.SetSize(size);

    ImagePointer image = ImageType::New();

    image->SetRegions(region);
    image->Allocate();
    image->FillBuffer(0.0f);

    double valacc = 0.0;
    for(unsigned int i = 12; i < 30; ++i) {
        for(unsigned int j = 8; j < 16; ++j) {
            ImageType::IndexType ind;
            ind[0] = j;
            ind[1] = i;
            image->SetPixel(ind, 1.0f);
            valacc += 1.0;
        }
    }

    std::cout << "Mean value [GT]: " << (valacc / (64*64)) << std::endl;

    return image;
}

void RunSamplerTest(typename PointSamplerBase<itk::Image<float, 2U>, itk::Image<bool, 2U>>::Pointer sampler) {
    typedef PointSample<itk::Image<float, 2U>, itk::Image<float, 2U> > PointSampleType;
    std::vector<PointSampleType> samples;

    unsigned int count = 64U;
    sampler->SampleN(0, samples, count, 1);

    double valacc = 0.0;
    for(unsigned int i = 0; i < count; ++i) {
        valacc += samples[i].m_Value;
        std::cout << samples[i].m_Point << " : " << samples[i].m_Value << " : " << samples[i].m_Weight << std::endl;
    }

    valacc /= count;
    std::cout << "Mean Value: " << valacc << std::endl;
}

void RunUniformPointSampler(ImagePointer image) {
    typedef UniformPointSampler<ImageType, itk::Image<bool, 2U>> SamplerType;
    typedef SamplerType::Pointer SamplerPointer;

    SamplerPointer sampler = SamplerType::New();
    sampler->SetImage(image);
    
    sampler->Initialize();

    RunSamplerTest(sampler.GetPointer());
}

void RunQuasiRandomPointSampler(ImagePointer image) {
    typedef QuasiRandomPointSampler<ImageType, itk::Image<bool, 2U>> SamplerType;
    typedef SamplerType::Pointer SamplerPointer;

    SamplerPointer sampler = SamplerType::New();
    sampler->SetImage(image);

    sampler->Initialize();

    RunSamplerTest(sampler.GetPointer());
}

void RunGradientWeightedPointSampler(ImagePointer image) {
    typedef GradientWeightedPointSampler<ImageType, itk::Image<bool, 2U>> SamplerType;
    typedef SamplerType::Pointer SamplerPointer;

    SamplerPointer sampler = SamplerType::New();
    sampler->SetImage(image);
    sampler->SetSigma(1.0);
    sampler->SetTolerance(1e-15);

    sampler->Initialize();

    RunSamplerTest(sampler.GetPointer());
}

void RunPointSamplersTests() {
    ImagePointer image = MakeTestImage();

    RunUniformPointSampler(image);
    RunQuasiRandomPointSampler(image);
    RunGradientWeightedPointSampler(image);
}
};