
#include <thread>
#include <iostream>

#include "../common/itkImageProcessingTools.h"

#include "itkLabelOverlapMeasuresImageFilter.h"
#include "itkHausdorffDistanceImageFilter.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "../registration/inspireRegister.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

#include "itkTimeProbesCollectorBase.h"
#include "itkMemoryProbesCollectorBase.h"

#include "itkMultiThreaderBase.h"

template <unsigned int ImageDimension>
class SynthEvalDeformable
{
    public:

    typedef itk::IPT<double, ImageDimension> IPT;

    typedef itk::Image<double, ImageDimension> ImageType;
    typedef typename ImageType::Pointer ImagePointer;
    typedef itk::Image<unsigned short, ImageDimension> LabelImageType;
    typedef typename LabelImageType::Pointer LabelImagePointer;
    typedef itk::Image<bool, ImageDimension> MaskType;
    typedef typename MaskType::Pointer MaskPointer;

    typedef BSplines<ImageDimension> BSplineFunc;

/*
    static ImagePointer Chessboard(ImagePointer image1, ImagePointer image2, int cells)
    {
        itk::FixedArray<unsigned int, ImageDimension> pattern;
        pattern.Fill(cells);

        typedef itk::CheckerBoardImageFilter<ImageType> CheckerBoardFilterType;
        typename CheckerBoardFilterType::Pointer checkerBoardFilter = CheckerBoardFilterType::New();
        checkerBoardFilter->SetInput1(image1);
        checkerBoardFilter->SetInput2(image2);
        checkerBoardFilter->SetCheckerPattern(pattern);
        checkerBoardFilter->Update();
        return checkerBoardFilter->GetOutput();
    }*/

    static ImagePointer BlackAndWhiteChessboard(ImagePointer refImage, int cells)
    {
        return Chessboard(IPT::ZeroImage(refImage->GetLargestPossibleRegion().GetSize()), IPT::ConstantImage(1.0, refImage->GetLargestPossibleRegion().GetSize()), cells);
    }

    template <typename TransformType>
    static void RandomizeDeformableTransform(typename TransformType::Pointer transform, double magnitude, unsigned int seed)
    {
        auto N = transform->GetNumberOfParameters();
        typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
        typename GeneratorType::Pointer RNG = GeneratorType::New();
        RNG->SetSeed(seed);

        typename TransformType::DerivativeType delta(N);

        for (unsigned int i = 0; i < N; ++i)
        {
            double x = RNG->GetVariateWithClosedRange();
            double y = (-1.0 + 2.0 * x) * magnitude;
            delta[i] = y;
        }

        transform->UpdateTransformParameters(delta, 1.0);
    }

    template <typename TImageType, typename TTransformType>
    static typename TImageType::Pointer ApplyTransform(
        typename TImageType::Pointer refImage,
        typename TImageType::Pointer floImage,
        typename TTransformType::Pointer transform,
        int interpolator = 1,
        typename TImageType::PixelType defaultValue = 0)
    {
        typedef itk::ResampleImageFilter<
            TImageType,
            TImageType>
            ResampleFilterType;

        typename ResampleFilterType::Pointer resample = ResampleFilterType::New();

        resample->SetTransform(transform);
        resample->SetInput(floImage);

        // Linear interpolator (1) is the default
        if (interpolator == 0)
        {
            auto interp = itk::NearestNeighborInterpolateImageFunction<TImageType, double>::New();
            resample->SetInterpolator(interp);
        }
        else if (interpolator == 2)
        {
            auto interp = itk::BSplineInterpolateImageFunction<TImageType, double>::New();//itk::BSplineInterpolationWeightFunction<double, ImageDimension, 3U>::New();
            resample->SetInterpolator(interp);
        }

        resample->SetSize(refImage->GetLargestPossibleRegion().GetSize());
        resample->SetOutputOrigin(refImage->GetOrigin());
        resample->SetOutputSpacing(refImage->GetSpacing());
        resample->SetOutputDirection(refImage->GetDirection());
        resample->SetDefaultPixelValue(defaultValue);

        resample->UpdateLargestPossibleRegion();

        return resample->GetOutput();
    }

    template <typename TImageType>
    static double LabelAccuracy(typename TImageType::Pointer image1, typename TImageType::Pointer image2) {
        typedef itk::LabelOverlapMeasuresImageFilter<TImageType> FilterType;
        typedef typename FilterType::Pointer FilterPointer;

        FilterPointer filter = FilterType::New();

        filter->SetSourceImage(image1);
        filter->SetTargetImage(image2);

        filter->Update();

        return filter->GetTotalOverlap();
    }

    template <typename TImageType>
    static double HausdorffDistance(typename TImageType::Pointer image1, typename TImageType::Pointer image2) {
        typedef itk::HausdorffDistanceImageFilter<TImageType, TImageType> FilterType;
        typedef typename FilterType::Pointer FilterPointer;

        FilterPointer filter = FilterType::New();

        filter->SetInput1(image1);
        filter->SetInput2(image2);
        filter->SetUseImageSpacing(true);
        filter->Update();

        return filter->GetHausdorffDistance();
    }


    static int MainFunc(int argc, char** argv) {
        itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(1);
        itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1);        

        BSplineFunc bsf;

        typedef typename BSplineFunc::TransformType TransformType;
        typedef typename BSplineFunc::TransformType::Pointer TransformPointer;
        // arguments
        // [0] appname
        // [1] dim
        // [2] ref image
        // [3] output path
        // [4] seed
        // [5, 6, ...] (control points, magnitude), (control points, magnitude), ...
        const char* inputPath = argv[2];
        const char* outputPath = argv[3];

        unsigned int seed = atoi(argv[4]);

        std::vector<unsigned int> controlPointCounts;
        std::vector<double> magnitudes;

        unsigned int ai = 5;
        while(ai + 1 < argc) {
            unsigned int cpc = atoi(argv[ai]);
            double mag = atof(argv[ai+1]);
            
            controlPointCounts.push_back(cpc);
            magnitudes.push_back(mag);

            std::cout << "Control points: " << cpc << "; Magnitude: " << mag << std::endl;

            ai += 2;
        }

        ImagePointer refImage = IPT::LoadImage(inputPath);

        TransformPointer randTransform;
        for (unsigned int i = 0; i < controlPointCounts.size(); ++i)
        {
            if (i == 0)
            {
                randTransform = bsf.CreateBSplineTransform(refImage, controlPointCounts[i]);
            } else {
                TransformPointer tNew = bsf.CreateBSplineTransform(refImage, controlPointCounts[i]);
                bsf.UpsampleBSplineTransform(refImage, tNew, randTransform, controlPointCounts[i]);
                randTransform = tNew;
            }
    
            RandomizeDeformableTransform<TransformType>(randTransform, magnitudes[i], seed);

            seed = seed * 17 + 13;
        }

        IPT::SaveTransformFile(outputPath, randTransform);

        return 0;
    }
};

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "No arguments..." << std::endl;
    }
    int ndim = atoi(argv[1]);
    if(ndim == 2) {
        SynthEvalDeformable<2U>::MainFunc(argc, argv);
    } else if(ndim == 3) {
        SynthEvalDeformable<3U>::MainFunc(argc, argv);
    } else {
        std::cout << "Error: Dimensionality " << ndim << " is not supported." << std::endl;
    }
}
