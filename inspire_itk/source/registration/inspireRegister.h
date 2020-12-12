
#ifndef INSPIRE_REGISTER_H
#define INSPIRE_REGISTER_H

// C++ standard library
#include <stdio.h>
#include <string.h>
#include <string>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSquaredDifferenceImageFilter.h"

#include "itkTimeProbesCollectorBase.h"

#include "../common/itkImageProcessingTools.h"
#include "itkTextOutput.h"

#include "itkTimeProbesCollectorBase.h"
#include "itkMemoryProbesCollectorBase.h"

// For BSpline transform resampling
#include "itkBSplineResampleImageFunction.h"
#include "itkIdentityTransform.h"
#include "itkBSplineDecompositionImageFilter.h"

#include "itkTransformToDisplacementFieldFilter.h"
#include "itkVectorIndexSelectionCastImageFilter.h"

#include "itkGradientMagnitudeImageFilter.h"
#include "itkDisplacementFieldJacobianDeterminantFilter.h"

#include "alphaLinearRegistration.h"
#include "alphaBSplineRegistration.h"
#include "../samplers/pointSampler.h"
#include "mcAlphaCutPointToSetDistance.h"

// Include JSON library
#include "../nlohmann/json.hpp"

template <typename TTransformType, unsigned int Dim>
struct BSplineRegistrationCallback : public Command
{
    public:

    using TransformType = TTransformType;
    using TransformPointer = typename TransformType::Pointer;

    TransformPointer m_TransformForward;
    TransformPointer m_TransformReverse;

    virtual void SetTransforms(TransformPointer transformForward, TransformPointer transformReverse)
    {
        m_TransformForward = transformForward;
        m_TransformReverse = transformReverse;
    }

    virtual void Invoke()
    {
        ;
    }
};

struct BSplineRegParamInner {
    double learningRate;
    double samplingFraction;
    double momentum;
    double lambdaFactor;
    long long iterations;
    long long controlPoints;
};

// Parameters for 
struct BSplineRegParam
{
    std::string optimizer;
    double samplingFraction;
    unsigned long long downsamplingFactor;
    double smoothingSigma; // rename to smoothing
    std::string smoothingMode;
    unsigned long long alphaLevels;
    bool gradientMagnitude;
    double normalization;
    int histogramEqualization;
    double learningRate;
    double momentum;
    double lambdaFactor;
    double dmax;
    double distancePower;
    double approximationThreshold;
    double approximationFraction;
    bool inwardsMode;
    unsigned long long seed;
    bool enableCallbacks;
    bool verbose;
    std::string samplingMode;
    std::vector<BSplineRegParamInner> innerParams;
};

struct BSplineRegParamOuter
{
    std::vector<BSplineRegParam> paramSets;
};

using json = nlohmann::json;

json readJSON(std::string path) {
    std::ifstream i(path);
    json j;
    i >> j;
    return j;
}

template <typename C, typename T>
void readJSONKey(C& c, std::string key, T *out) {
    if(c.find(key) != c.end()) {
        *out = c[key];
    }
}

BSplineRegParamOuter readConfig(std::string path) {
    BSplineRegParamOuter param;

    if (path.length() == 0)
    {
        return param;
    }

    json jc = readJSON(path);
    std::cout << jc << std::endl;

    for(size_t i = 0; i < jc["paramSets"].size(); ++i) {
        auto m_i = jc["paramSets"][i];

        BSplineRegParam paramSet;
        
        paramSet.samplingFraction = 0.05;
        readJSONKey(m_i, "samplingFraction", &paramSet.samplingFraction);

        paramSet.optimizer = "sgdm";
        readJSONKey(m_i, "optimizer", &paramSet.optimizer);
        paramSet.downsamplingFactor = 1;
        readJSONKey(m_i, "downsamplingFactor", &paramSet.downsamplingFactor);
        paramSet.smoothingSigma = 0.0;
        readJSONKey(m_i, "smoothing", &paramSet.smoothingSigma);
        paramSet.smoothingMode = "gaussian";
        readJSONKey(m_i, "smoothingMode", &paramSet.smoothingMode);
        paramSet.alphaLevels = 7;
        readJSONKey(m_i, "alphaLevels", &paramSet.alphaLevels);
        paramSet.gradientMagnitude = false;
        readJSONKey(m_i, "gradientMagnitude", &paramSet.gradientMagnitude);
        paramSet.normalization = 0.0;
        readJSONKey(m_i, "normalization", &paramSet.normalization);
        paramSet.histogramEqualization = 0;
        readJSONKey(m_i, "histogramEqualization", &paramSet.histogramEqualization);
        paramSet.approximationThreshold = 20.0;
        readJSONKey(m_i, "approximationThreshold", &paramSet.approximationThreshold);
        paramSet.approximationFraction = 0.2;
        readJSONKey(m_i, "approximationFraction", &paramSet.approximationFraction);
        paramSet.learningRate = 1.0;
        readJSONKey(m_i, "learningRate", &paramSet.learningRate);
        paramSet.momentum = 0.1;
        readJSONKey(m_i, "momentum", &paramSet.momentum);
        paramSet.lambdaFactor = 0.01;
        readJSONKey(m_i, "lambdaFactor", &paramSet.lambdaFactor);
        paramSet.dmax = 1.0;
        readJSONKey(m_i, "dmax", &paramSet.dmax);
        paramSet.distancePower = 1.0;
        readJSONKey(m_i, "distancePower", &paramSet.distancePower);
        paramSet.inwardsMode = false;
        readJSONKey(m_i, "inwardsMode", &paramSet.inwardsMode);
        paramSet.samplingMode = "quasi";
        readJSONKey(m_i, "samplingMode", &paramSet.samplingMode);
        paramSet.seed = 1337;
        readJSONKey(m_i, "seed", &paramSet.seed);
        paramSet.enableCallbacks = false;
        readJSONKey(m_i, "enableCallbacks", &paramSet.enableCallbacks);
        paramSet.verbose = false;
        readJSONKey(m_i, "verbose", &paramSet.verbose);

        for(size_t j = 0; j < m_i["innerParams"].size(); ++j) {
            auto m_i_j = m_i["innerParams"][j];

            BSplineRegParamInner innerParam;
           
            innerParam.learningRate = paramSet.learningRate;
            readJSONKey(m_i_j, "learningRate", &innerParam.learningRate);
            innerParam.samplingFraction = paramSet.samplingFraction;
            readJSONKey(m_i_j, "samplingFraction", &innerParam.samplingFraction);
            innerParam.momentum = paramSet.momentum;
            readJSONKey(m_i_j, "momentum", &innerParam.momentum);
            innerParam.lambdaFactor = paramSet.lambdaFactor;
            readJSONKey(m_i_j, "lambdaFactor", &innerParam.lambdaFactor);
            innerParam.iterations = 500;
            readJSONKey(m_i_j, "iterations", &innerParam.iterations);
            innerParam.controlPoints = 15;
            readJSONKey(m_i_j, "controlPoints", &innerParam.controlPoints);

            paramSet.innerParams.push_back(innerParam);
        }

        param.paramSets.push_back(paramSet);
    }

    return param;
}


template <unsigned int ImageDimension = 3U>
class BSplines {
public:
    constexpr static unsigned int Dim = ImageDimension;
typedef double PixelType;
typedef double CoordinateRepType;

typedef typename itk::Image<PixelType, ImageDimension> ImageType;
typedef typename ImageType::Pointer ImagePointer;

typedef typename itk::Vector<PixelType, ImageDimension> VectorPixelType;
typedef typename itk::Image<VectorPixelType, ImageDimension> DisplacementFieldImageType;
typedef typename DisplacementFieldImageType::Pointer DisplacementFieldImagePointer;

typedef typename itk::IPT<double, ImageDimension> IPT;

constexpr static unsigned int splineOrder = 3;

typedef typename itk::BSplineTransform<double, ImageDimension, splineOrder> TransformType;
typedef typename TransformType::Pointer TransformPointer;

using DerivativeType = typename TransformType::DerivativeType;

using CallbackType = BSplineRegistrationCallback<TransformType, ImageDimension>;

template <typename TransformType>
DisplacementFieldImagePointer TransformToDisplacementField(typename TransformType::Pointer transform, ImagePointer reference_image) {
  typedef typename itk::TransformToDisplacementFieldFilter<DisplacementFieldImageType, CoordinateRepType> DisplacementFieldGeneratorType;

  typename DisplacementFieldGeneratorType::Pointer dfield_gen = DisplacementFieldGeneratorType::New();

  dfield_gen->UseReferenceImageOn();
  dfield_gen->SetReferenceImage(reference_image);
  dfield_gen->SetTransform(transform);
  try {
    dfield_gen->Update();
  } catch (itk::ExceptionObject & err) {
    std::cerr << "Error while generating deformation field: " << err << std::endl;
  }

  return dfield_gen->GetOutput();
}

ImagePointer GradientMagnitudeImage(ImagePointer image, double sigma) {
    typedef typename itk::IPT<double, ImageDimension> IPT;
    if(sigma > 0.0) {
        image = IPT::SmoothImage(image, sigma);
    }
    typedef typename itk::GradientMagnitudeImageFilter<ImageType, ImageType> GradientMagnitudeFilterType;
    typename GradientMagnitudeFilterType::Pointer gmFilter = GradientMagnitudeFilterType::New();

    gmFilter->SetInput(image);
    gmFilter->SetUseImageSpacingOn();
    gmFilter->Update();

    return gmFilter->GetOutput();
}

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

void LinearToBSplineTransform(
    ImagePointer image,
    TransformPointer newTransform,
    typename itk::Transform<double, Dim, Dim>::Pointer oldTransform,
    unsigned int numberOfGridNodes)
{
    typedef typename TransformType::ParametersType ParametersType;
    ParametersType parameters(newTransform->GetNumberOfParameters());
    parameters.Fill(0.0);

    DisplacementFieldImagePointer disp =
        TransformToDisplacementField<itk::Transform<double, Dim, Dim> >(oldTransform, newTransform->GetCoefficientImages()[0]);//newTransform->GetCoefficientImages()[0]

    using ImageAdaptorType = itk::VectorIndexSelectionCastImageFilter<DisplacementFieldImageType, ImageType>;
    using ImageAdaptorPointer = typename ImageAdaptorType::Pointer;

    unsigned int counter = 0;
    for (unsigned int k = 0; k < ImageDimension; k++)
    {
        using ParametersImageType = typename TransformType::ImageType;
        using ResamplerType = itk::ResampleImageFilter<ParametersImageType, ParametersImageType>;
        typename ResamplerType::Pointer upsampler = ResamplerType::New();
        using FunctionType = itk::BSplineResampleImageFunction<ParametersImageType, double>;
        typename FunctionType::Pointer function = FunctionType::New();
        using IdentityTransformType = itk::IdentityTransform<double, ImageDimension>;
        typename IdentityTransformType::Pointer identity = IdentityTransformType::New();
        
        ImageAdaptorPointer adaptor = ImageAdaptorType::New();
        adaptor->SetIndex(k);
        adaptor->SetInput(disp);

        upsampler->SetInput(adaptor->GetOutput());
        upsampler->SetInterpolator(function);
        upsampler->SetTransform(identity);
        upsampler->SetSize(newTransform->GetCoefficientImages()[k]->GetLargestPossibleRegion().GetSize());
        upsampler->SetOutputSpacing(
            newTransform->GetCoefficientImages()[k]->GetSpacing());
        upsampler->SetOutputOrigin(
            newTransform->GetCoefficientImages()[k]->GetOrigin());
        upsampler->SetOutputDirection(image->GetDirection());
        using DecompositionType =
            itk::BSplineDecompositionImageFilter<ParametersImageType, ParametersImageType>;
        typename DecompositionType::Pointer decomposition = DecompositionType::New();
        decomposition->SetSplineOrder(splineOrder);
        decomposition->SetInput(upsampler->GetOutput());
        decomposition->Update();
        typename ParametersImageType::Pointer newCoefficients = decomposition->GetOutput();
        // copy the coefficients into the parameter array
        using Iterator = itk::ImageRegionIterator<ParametersImageType>;
        Iterator it(newCoefficients,
                    newTransform->GetCoefficientImages()[k]->GetLargestPossibleRegion());
        while (!it.IsAtEnd())
        {
            parameters[counter++] = it.Get();
            ++it;
        }
    }
    newTransform->SetParameters(parameters);
}

void UpsampleBSplineTransform(ImagePointer image, TransformPointer newTransform, TransformPointer oldTransform, unsigned int numberOfGridNodes)
{
    typedef typename TransformType::ParametersType ParametersType;
    ParametersType parameters(newTransform->GetNumberOfParameters());
    parameters.Fill(0.0);

    unsigned int counter = 0;
    for (unsigned int k = 0; k < ImageDimension; k++)
    {
        using ParametersImageType = typename TransformType::ImageType;
        using ResamplerType = itk::ResampleImageFilter<ParametersImageType, ParametersImageType>;
        typename ResamplerType::Pointer upsampler = ResamplerType::New();
        using FunctionType = itk::BSplineResampleImageFunction<ParametersImageType, double>;
        typename FunctionType::Pointer function = FunctionType::New();
        using IdentityTransformType = itk::IdentityTransform<double, ImageDimension>;
        typename IdentityTransformType::Pointer identity = IdentityTransformType::New();
        upsampler->SetInput(oldTransform->GetCoefficientImages()[k]);
        upsampler->SetInterpolator(function);
        upsampler->SetTransform(identity);
        upsampler->SetSize(newTransform->GetCoefficientImages()[k]->GetLargestPossibleRegion().GetSize());
        upsampler->SetOutputSpacing(
            newTransform->GetCoefficientImages()[k]->GetSpacing());
        upsampler->SetOutputOrigin(
            newTransform->GetCoefficientImages()[k]->GetOrigin());
        upsampler->SetOutputDirection(image->GetDirection());
        using DecompositionType =
            itk::BSplineDecompositionImageFilter<ParametersImageType, ParametersImageType>;
        typename DecompositionType::Pointer decomposition = DecompositionType::New();
        decomposition->SetSplineOrder(splineOrder);
        decomposition->SetInput(upsampler->GetOutput());
        decomposition->Update();
        typename ParametersImageType::Pointer newCoefficients = decomposition->GetOutput();
        // copy the coefficients into the parameter array
        using Iterator = itk::ImageRegionIterator<ParametersImageType>;
        Iterator it(newCoefficients,
                    newTransform->GetCoefficientImages()[k]->GetLargestPossibleRegion());
        while (!it.IsAtEnd())
        {
            parameters[counter++] = it.Get();
            ++it;
        }
    }
    newTransform->SetParameters(parameters);
}

typename PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>::Pointer CreateHybridPointSampler(ImagePointer im, ImagePointer maskImage, double gradientWeightedProb = 0.5, bool binaryMode = false, double sigma = 0.0, unsigned int seed = 1000U)
{
    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;
    PointSamplerPointer sampler1 = QuasiRandomPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::New().GetPointer();
    typename GradientWeightedPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::Pointer sampler2 =
        GradientWeightedPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::New().GetPointer();
    sampler2->SetSigma(sigma);
    sampler2->SetBinaryMode(binaryMode);
    sampler2->SetTolerance(1e-9);
    if(maskImage)
    {
        using MaskPointer = typename itk::Image<bool, Dim>::Pointer;
        MaskPointer maskBin = IPT::ThresholdImage(maskImage, 0.01);
        
        sampler1->SetMaskImage(maskBin);
        sampler2->SetMaskImage(maskBin);
    }

    typename HybridPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::Pointer sampler3 =
        HybridPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::New();

    sampler3->AddSampler(sampler1, 1.0-gradientWeightedProb);
    sampler3->AddSampler(sampler2.GetPointer(), gradientWeightedProb);
    sampler3->SetImage(im);
    sampler3->SetSeed(seed);
    sampler3->Initialize();

    return sampler3.GetPointer();
}

typename PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>::Pointer CreateUniformPointSampler(ImagePointer im, ImagePointer maskImage, unsigned int seed = 1000U)
{
    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;

    PointSamplerPointer sampler1 = UniformPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::New().GetPointer();
    if(maskImage)
    {
        using MaskPointer = typename itk::Image<bool, Dim>::Pointer;
        MaskPointer maskBin = IPT::ThresholdImage(maskImage, 0.01);
        
        sampler1->SetMaskImage(maskBin);
    }

    sampler1->SetImage(im);
    sampler1->Initialize();
    sampler1->SetSeed(seed);

    return sampler1.GetPointer();
}

typename PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>::Pointer CreateQuasiRandomPointSampler(ImagePointer im, ImagePointer maskImage, unsigned int seed = 1000U)
{
    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;

    PointSamplerPointer sampler1 = QuasiRandomPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::New().GetPointer();
    if(maskImage)
    {
        using MaskPointer = typename itk::Image<bool, Dim>::Pointer;
        MaskPointer maskBin = IPT::ThresholdImage(maskImage, 0.01);
        
        sampler1->SetMaskImage(maskBin);
    }

    sampler1->SetImage(im);
    sampler1->Initialize();
    sampler1->SetSeed(seed);

    return sampler1.GetPointer();
}

size_t ImagePixelCount(typename ImageType::Pointer image)
{
    auto region = image->GetLargestPossibleRegion();
    auto sz = region.GetSize();
    size_t cnt = 1;
    for(unsigned int i = 0; i < Dim; ++i)
    {
        cnt *= sz[i];
    }

    return cnt;
}

double ImageDiagonal(typename ImageType::Pointer image)
{
    auto size = image->GetLargestPossibleRegion().GetSize();
    auto spacing = image->GetSpacing();

    double acc = 0.0;
    for (unsigned int i = 0; i < ImageType::ImageDimension; ++i)
    {
        double d_i = size[i] * spacing[i];
        acc += d_i*d_i;
    }

    return sqrt(acc);
}

// Format:
// Default fraction and sigma
// gw
// Specified fraction and default sigma
// gw20
// Specified fraction and specified sigma
// gw20:0.5
bool ParseGradientWeightingParameter(std::string s, double& outFraction, double& outSigma)
{
    if(s.length() < 2)
    {
        return false;
    }

    if(s[0] != 'g' || s[1] != 'w')
    {
        return false;
    }

    if(s.length() == 2)
    {
        // Assume that out_value already contains the default value, return true to signal that gradient weighting is enabled
        return true;
    }

    // Test if a ':' exists
    auto colonPos = s.find(':');
    if (colonPos == std::string::npos)
    {
        // No ':' exists
        outFraction = atof(s.c_str() + 2) / 100.0;
    } else {
        // A ':' exists

        // Extract the fraction string into fractionStr, skip the colon, and put the rest of the string in sigmaStr
        std::string fractionStr = s.substr(2, colonPos-2);
        std::string sigmaStr = s.substr(colonPos+1);

        if (colonPos > 2)
        {
            // If there is a fraction, read it
            outFraction = atof(fractionStr.c_str()) / 100.0;
        }
      
        if (sigmaStr.length() > 0)
        {
            outSigma = atof(sigmaStr.c_str());
        }        
    }

    return true;
}

//
// Linear
// Linear monte carlo alpha registration
//

template <typename LinearTransformType>
void mcalpha_linear_register_func(typename ImageType::Pointer fixedImage, typename ImageType::Pointer movingImage, typename LinearTransformType::Pointer& transformForward, BSplineRegParam param, ImagePointer fixedMask, ImagePointer movingMask, const DerivativeType& parameterScaling, bool verbose=false)
{
    typedef itk::IPT<double, ImageDimension> IPT;

    using DistType = MCAlphaCutPointToSetDistance<ImageType, unsigned short>;
    using DistPointer = typename DistType::Pointer;

    double dmaxRefImage = 0.0;
    double dmaxFloImage = 0.0;
    if(param.dmax > 0)
    {
        dmaxRefImage = ImageDiagonal(fixedImage) * param.dmax;
        dmaxFloImage = ImageDiagonal(movingImage) * param.dmax;
    }

    DistPointer distStructRefImage = DistType::New();
    DistPointer distStructFloImage = DistType::New();

    distStructRefImage->SetSampleCount(param.alphaLevels);
    distStructRefImage->SetImage(fixedImage);
    distStructRefImage->SetMaxDistance(dmaxRefImage);
    distStructRefImage->SetApproximationThreshold(param.approximationThreshold);
    distStructRefImage->SetApproximationFraction(param.approximationFraction);
    distStructRefImage->SetDistancePower(param.distancePower);
    distStructRefImage->SetInwardsMode(param.inwardsMode);

    distStructFloImage->SetSampleCount(param.alphaLevels);
    distStructFloImage->SetImage(movingImage);
    distStructFloImage->SetMaxDistance(dmaxFloImage);
    distStructFloImage->SetApproximationThreshold(param.approximationThreshold);
    distStructFloImage->SetApproximationFraction(param.approximationFraction);
    distStructFloImage->SetDistancePower(param.distancePower);
    distStructFloImage->SetInwardsMode(param.inwardsMode);

    distStructRefImage->Initialize();
    distStructFloImage->Initialize();

    using RegistrationType = AlphaLinearRegistration<ImageType, DistType, LinearTransformType>;
    using RegistrationPointer = typename RegistrationType::Pointer;

    RegistrationPointer reg = RegistrationType::New();

    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;
    
    PointSamplerPointer sampler1; 
    PointSamplerPointer sampler2;
    double gradientWeightedSigma = 0.5;
    
    double gradientWeightedFraction = 0.5;
    bool enableGradientWeightedSampling = ParseGradientWeightingParameter(param.samplingMode, gradientWeightedFraction, gradientWeightedSigma);
    if (enableGradientWeightedSampling)
    {
        if (gradientWeightedFraction < 0)
        {
            gradientWeightedFraction = 0.0;
        }
        if (gradientWeightedFraction > 1)
        {
            gradientWeightedFraction = 1.0;
        }

        sampler1 = CreateHybridPointSampler(fixedImage, fixedMask, gradientWeightedFraction, false, gradientWeightedSigma, param.seed);
        sampler2 = CreateHybridPointSampler(movingImage, movingMask, gradientWeightedFraction, false, gradientWeightedSigma, param.seed);        
    }
    else
    if(param.samplingMode == "quasi")
    {
        sampler1 = CreateQuasiRandomPointSampler(fixedImage, fixedMask, param.seed);
        sampler2 = CreateQuasiRandomPointSampler(movingImage, movingMask, param.seed);
    } else if(param.samplingMode == "uniform")
    {
        sampler1 = CreateUniformPointSampler(fixedImage, fixedMask, param.seed);
        sampler2 = CreateUniformPointSampler(movingImage, movingMask, param.seed);
    }

    reg->SetPointSamplerRefImage(sampler1);
    reg->SetPointSamplerFloImage(sampler2);

    reg->SetDistDataStructRefImage(distStructRefImage);
    reg->SetDistDataStructFloImage(distStructFloImage);

    unsigned int sampleCountRefToFlo = 128;
    unsigned int sampleCountFloToRef = 128;

    reg->SetTransformRefToFlo(transformForward);
    reg->SetParameterScaling(parameterScaling);

    itk::TimeProbesCollectorBase chronometer;
    itk::MemoryProbesCollectorBase memorymeter;

    for (int q = 0; q < param.innerParams.size(); ++q) {
        double lr1 = param.innerParams[q].learningRate;
        double momentum1 = param.innerParams[q].momentum;
        unsigned int iterations = param.innerParams[q].iterations;

        sampleCountRefToFlo = ImagePixelCount(fixedImage) * param.innerParams[q].samplingFraction;
        sampleCountFloToRef = ImagePixelCount(movingImage) * param.innerParams[q].samplingFraction;

        reg->SetSampleCountRefToFlo(sampleCountRefToFlo);
        reg->SetSampleCountFloToRef(sampleCountFloToRef);
        reg->SetIterations(iterations);
        reg->SetLearningRate(lr1);
        reg->SetMomentum(momentum1);

        if (verbose) {
            reg->SetPrintInterval(1U);
        }
        reg->SetTransformRefToFlo(transformForward);
        reg->SetParameterScaling(parameterScaling);
        std::cerr << "Starting initialization of alphaLinearRegistration" << std::endl;

        reg->Initialize();

        std::cerr << "Initialization of alphaLinearRegistration done" << std::endl;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        reg->Run();

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if(verbose) {
            std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
        }

        transformForward = reg->GetTransformRefToFlo();
    }
}

//
// Deformable monte carlo alpha registration
//

void mcalpha_register_func(typename ImageType::Pointer fixedImage, typename ImageType::Pointer movingImage, TransformPointer& transformForward, TransformPointer& transformInverse, BSplineRegParam param, ImagePointer fixedMask, ImagePointer movingMask, bool verbose=false, CallbackType* callback=nullptr)
{
    typedef itk::IPT<double, ImageDimension> IPT;
    
    using DistType = MCAlphaCutPointToSetDistance<ImageType, unsigned short>;
    using DistPointer = typename DistType::Pointer;

    double dmaxRefImage = 0.0;
    double dmaxFloImage = 0.0;
    if(param.dmax > 0)
    {
        dmaxRefImage = ImageDiagonal(fixedImage) * param.dmax;
        dmaxFloImage = ImageDiagonal(movingImage) * param.dmax;
    }

    DistPointer distStructRefImage = DistType::New();
    DistPointer distStructFloImage = DistType::New();

    distStructRefImage->SetSampleCount(param.alphaLevels);
    distStructRefImage->SetImage(fixedImage);
    distStructRefImage->SetMaxDistance(dmaxRefImage);
    distStructRefImage->SetApproximationThreshold(param.approximationThreshold);
    distStructRefImage->SetApproximationFraction(param.approximationFraction);
    distStructRefImage->SetDistancePower(param.distancePower);
    distStructRefImage->SetInwardsMode(param.inwardsMode);

    distStructFloImage->SetSampleCount(param.alphaLevels);
    distStructFloImage->SetImage(movingImage);
    distStructFloImage->SetMaxDistance(dmaxFloImage);
    distStructFloImage->SetApproximationThreshold(param.approximationThreshold);
    distStructFloImage->SetApproximationFraction(param.approximationFraction);
    distStructFloImage->SetDistancePower(param.distancePower);
    distStructFloImage->SetInwardsMode(param.inwardsMode);

    distStructRefImage->Initialize();
    distStructFloImage->Initialize();

    using RegistrationType = AlphaBSplineRegistration<ImageType, DistType, 3U>;
    using RegistrationPointer = typename RegistrationType::Pointer;

    RegistrationPointer reg = RegistrationType::New();

    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;
    
    PointSamplerPointer sampler1; 
    PointSamplerPointer sampler2;
    double gradientWeightedSigma = 0.5;
    
    double gradientWeightedFraction = 0.5;
    bool enableGradientWeightedSampling = ParseGradientWeightingParameter(param.samplingMode, gradientWeightedFraction, gradientWeightedSigma);
    if (enableGradientWeightedSampling)
    {
        if (gradientWeightedFraction < 0)
        {
            gradientWeightedFraction = 0.0;
        }
        if (gradientWeightedFraction > 1)
        {
            gradientWeightedFraction = 1.0;
        }

        sampler1 = CreateHybridPointSampler(fixedImage, fixedMask, gradientWeightedFraction, false, gradientWeightedSigma, param.seed);
        sampler2 = CreateHybridPointSampler(movingImage, movingMask, gradientWeightedFraction, false, gradientWeightedSigma, param.seed);        
    }
    else
    if(param.samplingMode == "quasi")
    {
        sampler1 = CreateQuasiRandomPointSampler(fixedImage, fixedMask, param.seed);
        sampler2 = CreateQuasiRandomPointSampler(movingImage, movingMask, param.seed);
    } else if(param.samplingMode == "uniform")
    {
        sampler1 = CreateUniformPointSampler(fixedImage, fixedMask, param.seed);
        sampler2 = CreateUniformPointSampler(movingImage, movingMask, param.seed);
    }

    reg->SetPointSamplerRefImage(sampler1);
    reg->SetPointSamplerFloImage(sampler2);

    reg->SetDistDataStructRefImage(distStructRefImage);
    reg->SetDistDataStructFloImage(distStructFloImage);

    unsigned int sampleCountRefToFlo = 128;
    unsigned int sampleCountFloToRef = 128;

    reg->SetTransformRefToFlo(transformForward);
    reg->SetTransformFloToRef(transformInverse);

    for (int q = 0; q < param.innerParams.size(); ++q) {
        double lr1 = param.innerParams[q].learningRate;
        double momentum1 = param.innerParams[q].momentum;
        unsigned int iterations = param.innerParams[q].iterations;
        unsigned int controlPoints = param.innerParams[q].controlPoints;

        sampleCountRefToFlo = ImagePixelCount(fixedImage) * param.innerParams[q].samplingFraction;
        sampleCountFloToRef = ImagePixelCount(movingImage) * param.innerParams[q].samplingFraction;

        reg->SetSampleCountRefToFlo(sampleCountRefToFlo);
        reg->SetSampleCountFloToRef(sampleCountFloToRef);
        reg->SetIterations(iterations);
        reg->SetLearningRate(lr1);
        reg->SetMomentum(momentum1);
        reg->SetSymmetryLambda(param.innerParams[q].lambdaFactor);

        if(q > 0) {
            TransformPointer tforNew = CreateBSplineTransform(fixedImage, controlPoints);
            TransformPointer tinvNew = CreateBSplineTransform(movingImage, controlPoints);
            int curNumberOfGridNodes = controlPoints;
            UpsampleBSplineTransform(fixedImage, tforNew, transformForward, curNumberOfGridNodes);
            UpsampleBSplineTransform(movingImage, tinvNew, transformInverse, curNumberOfGridNodes);
            transformForward = tforNew;
            transformInverse = tinvNew;
        }

        if (param.enableCallbacks && callback != nullptr) {
            callback->SetTransforms(transformForward, transformInverse);
            reg->AddCallback(callback);
        }

        if (verbose) {
            reg->SetPrintInterval(1U);
        }
        reg->SetTransformRefToFlo(transformForward);
        reg->SetTransformFloToRef(transformInverse);
        reg->Initialize();

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        reg->Run();

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if(verbose) {
            std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
        }

        transformForward = reg->GetTransformRefToFlo();
        transformInverse = reg->GetTransformFloToRef();
    }
}

// To fix: Normalization and equalization should only take values inside the mask into account
void register_deformable(
    typename ImageType::Pointer fixedImage,
    typename ImageType::Pointer movingImage,
    BSplineRegParamOuter affineParam,
    BSplineRegParamOuter bsplineParam,
    ImagePointer fixedMask,
    ImagePointer movingMask,
    typename itk::CompositeTransform<double, Dim>::Pointer& transformAffineOut,
    TransformPointer& transformForwardOut,
    TransformPointer& transformInverseOut,
    CallbackType* callback) {

    TransformPointer transformForward;
    TransformPointer transformInverse;

    // Do affine registration

    // Setup transformation
    using CompositeTransformType = itk::CompositeTransform<double, Dim>;
    using CompositeTransformPointer = typename CompositeTransformType::Pointer;
    using AffineTransformType = itk::AffineTransform<double, Dim>;
    using AffineTransformPointer = typename AffineTransformType::Pointer;
    using TranslationTransformType = itk::TranslationTransform<double, Dim>;
    using TranslationTransformPointer = typename TranslationTransformType::Pointer;

    CompositeTransformPointer compositeTransformation = CompositeTransformType::New();
    AffineTransformPointer affineTransformation = AffineTransformType::New();
    TranslationTransformPointer offsetTransformation1 = TranslationTransformType::New();
    TranslationTransformPointer offsetTransformation2 = TranslationTransformType::New();
    
    using ParametersType = typename AffineTransformType::ParametersType;
    ParametersType offset1Param(offsetTransformation1->GetNumberOfParameters());
    ParametersType offset2Param(offsetTransformation2->GetNumberOfParameters());
    ParametersType affineTransformationParam(affineTransformation->GetNumberOfParameters());
    auto reg1 = fixedImage->GetLargestPossibleRegion();
    auto reg2 = movingImage->GetLargestPossibleRegion();

    double diag1 = 0.0;
    double diag2 = 0.0;

    for (unsigned int i = 0; i < offsetTransformation1->GetNumberOfParameters(); ++i)
    {
        const double diag1_i = fixedImage->GetSpacing()[i] * reg1.GetSize()[i];
        const double diag2_i = movingImage->GetSpacing()[i] * reg2.GetSize()[i];
        diag1 += diag1_i * diag1_i;
        diag2 += diag2_i * diag2_i;

        auto center1 = -fixedImage->GetSpacing()[i] * ((reg1.GetSize()[i] / 2.0) + reg1.GetIndex()[i]);
        auto center2 = movingImage->GetSpacing()[i] * ((reg2.GetSize()[i] / 2.0) + reg2.GetIndex()[i]);
        offset1Param[i] = center1;
        offset2Param[i] = center2;
    }

    double diag = 0.5 * (sqrt(diag1) + sqrt(diag2));

    offsetTransformation1->SetParameters(offset1Param);
    offsetTransformation2->SetParameters(offset2Param);

    DerivativeType parameterScaling(affineTransformation->GetNumberOfParameters());
    parameterScaling.Fill(1.0);

    affineTransformationParam.Fill(0.0);
    for (unsigned int i = 0; i < Dim; ++i)
    {
        affineTransformationParam[i*Dim+i] = 1.0;
        for (unsigned int j = 0; j < Dim; ++j)
        {
            parameterScaling[i*Dim+j] = 1.0 / diag;
        }
    }
    affineTransformation->SetParameters(affineTransformationParam);

    compositeTransformation->AppendTransform(offsetTransformation2.GetPointer());
    compositeTransformation->AppendTransform(affineTransformation.GetPointer());
    compositeTransformation->AppendTransform(offsetTransformation1.GetPointer());

    compositeTransformation->SetNthTransformToOptimize(0, false);
    compositeTransformation->SetNthTransformToOptimize(1, true);
    compositeTransformation->SetNthTransformToOptimize(2, false);   

    std::cerr << "Created affine transformation" << std::endl;

    for(size_t i = 0; i < affineParam.paramSets.size(); ++i) {
        auto paramSet = affineParam.paramSets[i];

        ImagePointer fixedImagePrime;
        ImagePointer movingImagePrime;
        if (paramSet.smoothingMode == "gaussian") {
            fixedImagePrime = IPT::SmoothImage(fixedImage, paramSet.smoothingSigma);
            movingImagePrime = IPT::SmoothImage(movingImage, paramSet.smoothingSigma);
        } else if (paramSet.smoothingMode == "median") {
            fixedImagePrime = IPT::MedianFilterImage(fixedImage, paramSet.smoothingSigma);
            movingImagePrime = IPT::MedianFilterImage(movingImage, paramSet.smoothingSigma);
        } else {
            std::cerr << "Illegal smoothing mode: " << paramSet.smoothingMode << std::endl;
            assert(false);
        }
        ImagePointer fixedMaskPrime = fixedMask;
        ImagePointer movingMaskPrime = movingMask;
        if(paramSet.downsamplingFactor != 1) {
            fixedImagePrime = IPT::SubsampleImage(fixedImagePrime, paramSet.downsamplingFactor);
            movingImagePrime = IPT::SubsampleImage(movingImagePrime, paramSet.downsamplingFactor);
            fixedMaskPrime = IPT::SubsampleImage(fixedMaskPrime, paramSet.downsamplingFactor);
            movingMaskPrime = IPT::SubsampleImage(movingMaskPrime, paramSet.downsamplingFactor);
        }
        if(paramSet.gradientMagnitude) {
            fixedImagePrime = GradientMagnitudeImage(fixedImagePrime, 0.0);
            movingImagePrime = GradientMagnitudeImage(movingImagePrime, 0.0);
        }
        if(paramSet.normalization >= 0.0)
        {
            fixedImagePrime = IPT::NormalizeImage(fixedImagePrime, IPT::IntensityMinMax(fixedImagePrime, paramSet.normalization));
            movingImagePrime = IPT::NormalizeImage(movingImagePrime, IPT::IntensityMinMax(movingImagePrime, paramSet.normalization));
            fixedImagePrime = IPT::HistogramEqualization(fixedImagePrime, nullptr, paramSet.histogramEqualization);
            movingImagePrime = IPT::HistogramEqualization(movingImagePrime, nullptr, paramSet.histogramEqualization);
        }

        mcalpha_linear_register_func<CompositeTransformType>(fixedImagePrime, movingImagePrime, compositeTransformation, paramSet, fixedMaskPrime, movingMaskPrime, parameterScaling, paramSet.verbose);       
    }

    transformAffineOut = compositeTransformation;

    // Do deformable registration
    for(size_t i = 0; i < bsplineParam.paramSets.size(); ++i) {
        auto paramSet = bsplineParam.paramSets[i];

        ImagePointer fixedImagePrime;
        ImagePointer movingImagePrime;
        if (paramSet.smoothingMode == "gaussian") {
            fixedImagePrime = IPT::SmoothImage(fixedImage, paramSet.smoothingSigma);
            movingImagePrime = IPT::SmoothImage(movingImage, paramSet.smoothingSigma);
        } else if (paramSet.smoothingMode == "median") {
            fixedImagePrime = IPT::MedianFilterImage(fixedImage, paramSet.smoothingSigma);
            movingImagePrime = IPT::MedianFilterImage(movingImage, paramSet.smoothingSigma);
        } else {
            std::cerr << "Illegal smoothing mode: " << paramSet.smoothingMode << std::endl;
            assert(false);
        }
        ImagePointer fixedMaskPrime = fixedMask;
        ImagePointer movingMaskPrime = movingMask;
        if(paramSet.downsamplingFactor != 1) {
            fixedImagePrime = IPT::SubsampleImage(fixedImagePrime, paramSet.downsamplingFactor);
            movingImagePrime = IPT::SubsampleImage(movingImagePrime, paramSet.downsamplingFactor);
            fixedMaskPrime = IPT::SubsampleImage(fixedMaskPrime, paramSet.downsamplingFactor);
            movingMaskPrime = IPT::SubsampleImage(movingMaskPrime, paramSet.downsamplingFactor);
        }
        if(paramSet.gradientMagnitude) {
            fixedImagePrime = GradientMagnitudeImage(fixedImagePrime, 0.0);
            movingImagePrime = GradientMagnitudeImage(movingImagePrime, 0.0);
        }
        if(paramSet.normalization >= 0.0)
        {
            fixedImagePrime = IPT::NormalizeImage(fixedImagePrime, IPT::IntensityMinMax(fixedImagePrime, paramSet.normalization));
            movingImagePrime = IPT::NormalizeImage(movingImagePrime, IPT::IntensityMinMax(movingImagePrime, paramSet.normalization));
            fixedImagePrime = IPT::HistogramEqualization(fixedImagePrime, nullptr, paramSet.histogramEqualization);
            movingImagePrime = IPT::HistogramEqualization(movingImagePrime, nullptr, paramSet.histogramEqualization);
        }

        if(i == 0) {
            TransformPointer tforNew = CreateBSplineTransform(fixedImagePrime, paramSet.innerParams[0].controlPoints);
            TransformPointer tinvNew = CreateBSplineTransform(movingImagePrime, paramSet.innerParams[0].controlPoints);
            LinearToBSplineTransform(fixedImagePrime, tforNew, compositeTransformation.GetPointer(), paramSet.innerParams[0].controlPoints);
            LinearToBSplineTransform(movingImagePrime, tinvNew, compositeTransformation->GetInverseTransform().GetPointer(), paramSet.innerParams[0].controlPoints);
            transformForward = tforNew;
            transformInverse = tinvNew;
            //break;         
        } else {
            TransformPointer tforNew = CreateBSplineTransform(fixedImagePrime, paramSet.innerParams[0].controlPoints);
            TransformPointer tinvNew = CreateBSplineTransform(movingImagePrime, paramSet.innerParams[0].controlPoints);
            UpsampleBSplineTransform(fixedImagePrime, tforNew, transformForward, paramSet.innerParams[0].controlPoints);
            UpsampleBSplineTransform(movingImagePrime, tinvNew, transformInverse, paramSet.innerParams[0].controlPoints);
            transformForward = tforNew;
            transformInverse = tinvNew;            
        }

        mcalpha_register_func(fixedImagePrime, movingImagePrime, transformForward, transformInverse, paramSet, fixedMaskPrime, movingMaskPrime, paramSet.verbose, callback);       
    }

    transformForwardOut = transformForward;
    transformInverseOut = transformInverse;
}

};

#endif
