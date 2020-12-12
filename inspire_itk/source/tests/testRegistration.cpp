
#include "../registration/alphaBSplineRegistration.h"
#include "../samplers/pointSampler.h"
#include "../registration/mcAlphaCutPointToSetDistance.h"

#include "../common/itkImageProcessingTools.h"
#include "itkTimeProbesCollectorBase.h"

namespace TestRegistration
{
using ImageType = itk::Image<float, 2U>;
using ImagePointer = typename ImageType::Pointer;
using TransformType = itk::BSplineTransform<double, 2U, 3U>;
using TransformPointer = typename TransformType::Pointer;
constexpr unsigned int ImageDimension = 2U;
constexpr unsigned int splineOrder = 3U;

TransformPointer CreateBSplineTransform(ImagePointer image, unsigned int numberOfGridNodes)
{
    typename TransformType::PhysicalDimensionsType fixedPhysicalDimensions;
    typename TransformType::MeshSizeType meshSize;

    TransformPointer transform = TransformType::New();

    for (unsigned int i = 0; i < ImageDimension; i++)
    {
        fixedPhysicalDimensions[i] = image->GetSpacing()[i] *
                                     static_cast<double>(
                                         image->GetLargestPossibleRegion().GetSize()[i] - 1);
    }
    meshSize.Fill(numberOfGridNodes - splineOrder);
    transform->SetTransformDomainOrigin(image->GetOrigin());
    transform->SetTransformDomainPhysicalDimensions(fixedPhysicalDimensions);
    transform->SetTransformDomainMeshSize(meshSize);
    transform->SetTransformDomainDirection(image->GetDirection());

    return transform;
}

typename ImageType::Pointer ApplyTransform(ImagePointer refImage, ImagePointer floImage, TransformPointer transform, double bgValue = 0.5)
{
    typedef itk::ResampleImageFilter<
        ImageType,
        ImageType>
        ResampleFilterType;

    typedef itk::IPT<double, ImageDimension> IPT;

    typename ResampleFilterType::Pointer resample = ResampleFilterType::New();

    resample->SetTransform(transform);
    resample->SetInput(floImage);

    resample->SetSize(refImage->GetLargestPossibleRegion().GetSize());
    resample->SetOutputOrigin(refImage->GetOrigin());
    resample->SetOutputSpacing(refImage->GetSpacing());
    resample->SetOutputDirection(refImage->GetDirection());
    resample->SetDefaultPixelValue(bgValue);

    resample->UpdateLargestPossibleRegion();

    return resample->GetOutput();
}

typename PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>::Pointer CreateHybridPointSampler(ImagePointer im, double w1 = 0.5, bool binaryMode = false, unsigned int seed = 1000U)
{
    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;
    PointSamplerPointer sampler1 = QuasiRandomPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New().GetPointer();
    GradientWeightedPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::Pointer sampler2 = GradientWeightedPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New().GetPointer();
    typename HybridPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::Pointer sampler3 = HybridPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New();
    sampler2->SetSigma(1.0);
    sampler2->SetBinaryMode(binaryMode);
    //sampler2->SetTolerance(1e-3);

    sampler3->AddSampler(sampler1, w1);
    sampler3->AddSampler(sampler2.GetPointer(), 1.0-w1);
    sampler3->SetImage(im);
    sampler3->SetSeed(seed);
    //sampler3->SetDitheringOn();
    sampler3->Initialize();

    return sampler3.GetPointer();
}

typename PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>::Pointer CreateUniformPointSampler(ImagePointer im, unsigned int seed = 1000U)
{
    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;

    PointSamplerPointer sampler1 = UniformPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New().GetPointer();
    sampler1->SetImage(im);
    sampler1->Initialize();
    sampler1->SetSeed(seed);

    return sampler1.GetPointer();
}

typename PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>::Pointer CreateQuasiRandomPointSampler(ImagePointer im, unsigned int seed = 1000U)
{
    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;

    PointSamplerPointer sampler1 = QuasiRandomPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New().GetPointer();
    sampler1->SetImage(im);
    sampler1->Initialize();
    sampler1->SetSeed(seed);

    return sampler1.GetPointer();
}

ImagePointer MakeTestImage(unsigned int xpos1, unsigned int ypos1, unsigned xpos2, unsigned int ypos2, unsigned int xsz, unsigned int ysz, unsigned int imsize) {
    ImageType::RegionType region;
    ImageType::IndexType index;
    ImageType::SizeType size;

    index[0] = 0;
    index[1] = 0;
    size[0] = imsize;//64;
    size[1] = imsize;//64;

    region.SetIndex(index);
    region.SetSize(size);

    ImagePointer image = ImageType::New();

    image->SetRegions(region);
    image->Allocate();
    image->FillBuffer(0.0f);

    double valacc = 0.0;
    for(unsigned int i = ypos1; i < ypos1 + ysz; ++i) {
        for(unsigned int j = xpos1; j < xpos1 + xsz; ++j) {
            ImageType::IndexType ind;
            ind[0] = j;
            ind[1] = i;
            image->SetPixel(ind, 1.0f);
            valacc += 1.0;
        }
    }

    for(unsigned int i = ypos1; i < ypos1 + ysz; ++i) {
        for(unsigned int j = xpos1; j < xpos1 + xsz; ++j) {
            ImageType::IndexType ind;
            ind[0] = imsize-j-1;
            ind[1] = imsize-i-1;
            image->SetPixel(ind, 0.75f);
            valacc += 0.75;
        }
    }

    for(unsigned int i = ypos2; i < ypos2 + ysz; ++i) {
        for(unsigned int j = xpos2; j < xpos2 + xsz; ++j) {
            ImageType::IndexType ind;
            ind[0] = j;
            ind[1] = i;
            image->SetPixel(ind, 0.5f);
            valacc += 0.5;
        }
    }

    for(unsigned int i = ypos2; i < ypos2 + ysz; ++i) {
        for(unsigned int j = xpos2; j < xpos2 + xsz; ++j) {
            ImageType::IndexType ind;
            ind[0] = imsize-j-1;
            ind[1] = imsize-i-1;
            image->SetPixel(ind, 0.25f);
            valacc += 0.25;
        }
    }

    std::cout << "Mean value [GT]: " << (valacc / (64*64)) << std::endl;

    return image;
}

double MeanAbsDiff(ImagePointer image1, ImagePointer image2, ImagePointer* imageOut = nullptr)
{
    using IPT = itk::IPT<float, 2U>;

    typename ImageType::Pointer diff = IPT::DifferenceImage(image1, image2);
    if(imageOut != nullptr)
    {
        *imageOut = diff;
    }
    typename IPT::ImageStatisticsData stats = IPT::ImageStatistics(diff);

    return stats.mean;
}


    void RunTest(int argc, char** argv)
    {
        using IPT = itk::IPT<float, 2U>;

        constexpr unsigned int imsize = 128;

        ImagePointer refImage = MakeTestImage(5, 7, 70, 50, 10, 8, imsize);
        ImagePointer floImage = MakeTestImage(9, 13, 80, 55, 8, 10, imsize);

        IPT::SaveImageU8("./reftest.png", refImage);
        IPT::SaveImageU8("./flotest.png", floImage);

        using DistType = MCAlphaCutPointToSetDistance<ImageType, unsigned short>;
        using DistPointer = typename DistType::Pointer;
        
        DistPointer distStructRefImage = DistType::New();
        DistPointer distStructFloImage = DistType::New();

        distStructRefImage->SetSampleCount(8U);
        distStructRefImage->SetImage(refImage);
        distStructRefImage->SetMaxDistance(0);
        distStructRefImage->SetApproximationThreshold(20.0);
        distStructRefImage->SetApproximationFraction(0.1);

        distStructFloImage->SetSampleCount(8U);
        distStructFloImage->SetImage(floImage);
        distStructFloImage->SetMaxDistance(0);
        distStructFloImage->SetApproximationThreshold(20.0);
        distStructFloImage->SetApproximationFraction(0.1);

        distStructRefImage->Initialize();
        distStructFloImage->Initialize();

        using RegistrationType = AlphaBSplineRegistration<ImageType, DistType, 3U>;
        using RegistrationPointer = typename RegistrationType::Pointer;

        RegistrationPointer reg = RegistrationType::New();

        reg->SetDistDataStructRefImage(distStructRefImage);
        reg->SetDistDataStructFloImage(distStructFloImage);

        using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>;
        using PointSamplerPointer = typename PointSamplerType::Pointer;
        PointSamplerPointer sampler1;
        PointSamplerPointer sampler2;

        constexpr unsigned int seed = 1000U;
        constexpr bool gradientWeightedSamplerBinaryMode = false;
        constexpr unsigned int samplerMode = 0; // Select sampling mode here (0: hybrid, 1: quasi random, 2: uniform)

        if(samplerMode == 0)
        {
            constexpr double w = 0.5;
            sampler1 = CreateHybridPointSampler(refImage, w, gradientWeightedSamplerBinaryMode, seed);
            sampler2 = CreateHybridPointSampler(floImage, w, gradientWeightedSamplerBinaryMode, seed);
        } else if(samplerMode == 1)
        {
            sampler1 = CreateQuasiRandomPointSampler(refImage, seed);
            sampler2 = CreateQuasiRandomPointSampler(floImage, seed);
        } else if(samplerMode == 2)
        {
            sampler1 = CreateUniformPointSampler(refImage, seed);
            sampler2 = CreateUniformPointSampler(floImage, seed);
        }       
        
        reg->SetPointSamplerRefImage(sampler1);
        reg->SetPointSamplerFloImage(sampler2);
        constexpr unsigned int gridPointCount = 20;
        constexpr unsigned int sampleCount = 4096;
        constexpr unsigned int iterations = 1000U;
        constexpr double learningRate = 0.8;
        constexpr double momentum = 0.1;
        constexpr double symmetryLambda = 0.05;

        reg->SetTransformRefToFlo(CreateBSplineTransform(refImage, gridPointCount));
        reg->SetTransformFloToRef(CreateBSplineTransform(floImage, gridPointCount));

        reg->SetSampleCountRefToFlo(sampleCount);
        reg->SetSampleCountFloToRef(sampleCount);
        reg->SetLearningRate(learningRate);
        reg->SetMomentum(momentum);
        reg->SetIterations(iterations);
        reg->SetSymmetryLambda(symmetryLambda);

        std::cout << "Initializing" << std::endl;
        reg->Initialize();

        std::cout << "Running" << std::endl;

        itk::TimeProbesCollectorBase chronometer;

        chronometer.Start("Registration");

        reg->Run();

        chronometer.Stop("Registration");
        chronometer.Report(std::cout);

        TransformPointer t1 = reg->GetTransformRefToFlo();

        ImagePointer transformedImage = ApplyTransform(refImage, floImage, t1, 0.0);
        ImagePointer diffImage;

        char buf[64]; // C-strings on the stack like this are not good...
        sprintf(buf, "%.15f", MeanAbsDiff(refImage, transformedImage, &diffImage));
        std::cout << "Before diff: " << MeanAbsDiff(refImage, floImage) << std::endl;
        std::cout << "After diff:  " << buf << std::endl;

        std::cout << "Final Distance: " << reg->GetValue() << std::endl;

        IPT::SaveImageU8("./transformeddiff.png", diffImage);
        IPT::SaveImageU8("./transformedtest.png", transformedImage);
    }
}