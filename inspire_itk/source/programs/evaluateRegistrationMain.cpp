
/**
 * --- INSPIRE ---
 * Author: Johan Ofverstedt
 * 
 * Program that evaluates registration based on overlap of labels,
 * as well as based on similarity of the intensity values of
 * provided gray-scale values.
 */

#include <iostream>

#include <thread>

#include "../common/itkImageProcessingTools.h"

#include "itkLabelOverlapMeasuresImageFilter.h"
#include "itkHausdorffDistanceImageFilter.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkIdentityTransform.h"

#include "itkTimeProbesCollectorBase.h"
#include "itkMemoryProbesCollectorBase.h"

#include <chrono>

#include "itkVersion.h"
#include "itkMultiThreaderBase.h"

#include "itkPNGImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"

void RegisterIOFactories() {
    itk::PNGImageIOFactory::RegisterOneFactory();
    itk::NiftiImageIOFactory::RegisterOneFactory();
}

struct PerformanceMetrics {
    double totalOverlap;
    double meanTotalOverlap;
    double meanUnionOverlap; // Jaccard
    double absDiff;
    unsigned int labelCount;
};

template <typename TStream>
void printMetrics(TStream& strm, PerformanceMetrics m, std::string name, bool linebreak=true) {
    strm << name << "(totalOverlap: " << m.totalOverlap << ", meanTotalOverlap: " << m.meanTotalOverlap << ", meanUnionOverlap"<< m.meanUnionOverlap << ", absdiff: " << m.absDiff << ").";
    if(linebreak)
        strm << std::endl;
}

template <typename TStream>
void printMetricsCSV(TStream& strm, PerformanceMetrics m, bool linebreak=true, std::string ident = "") {
    strm << m.totalOverlap << ", " << m.meanTotalOverlap << ", " << m.meanUnionOverlap << ", " << m.absDiff << ", " << m.labelCount;
    if(ident.length() > 0)
        strm << ", " << ident;
    if(linebreak)
        strm << std::endl;
}

template <typename TStream>
void printPairOfMetricsCSV(TStream& strm, PerformanceMetrics before, PerformanceMetrics after, bool linebreak=true, std::string ident = "") {
    strm << before.totalOverlap << ", " << before.meanTotalOverlap << ", " << before.meanUnionOverlap << ", " << before.absDiff << ", " << after.totalOverlap << ", " << after.meanTotalOverlap << ", " << after.meanUnionOverlap << ", " << after.absDiff << ", " << after.labelCount;
    if(ident.length() > 0)
        strm << ", " << ident;
    if(linebreak)
        strm << std::endl;
}

    template <typename TImageType, typename TTransformType>
    typename TImageType::Pointer ApplyTransformToImage(
        typename TImageType::Pointer refImage,
        typename TImageType::Pointer floImage,
        typename TTransformType::Pointer transform,
        int interpolator = 1,
        typename TImageType::PixelType defaultValue = 0)
    {
        if(!transform)
        {
            return floImage;
        }

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

template <typename ImageType>
double MeanAbsDifference(
    typename ImageType::Pointer image1,
    typename ImageType::Pointer image2) {
    
    using ImagePointer = typename ImageType::Pointer;
    using ValueType = typename ImageType::ValueType;
    typedef itk::IPT<ValueType, ImageType::ImageDimension> IPT;

    ImagePointer diffImage = IPT::DifferenceImage(image1, image2);
    typename IPT::ImageStatisticsData stats = IPT::ImageStatistics(diffImage);
    return stats.mean;
}

template <typename TImageType>
void LabelAccuracy(
    typename TImageType::Pointer image1,
    typename TImageType::Pointer image2,
    double& totalOverlap,
    double& meanTotalOverlap,
    double& meanUnionOverlap,
    unsigned int& labelCount
    ) {
        
        typedef itk::LabelOverlapMeasuresImageFilter<TImageType> FilterType;
        typedef typename FilterType::Pointer FilterPointer;

        FilterPointer filter = FilterType::New();

        filter->SetSourceImage(image1);
        filter->SetTargetImage(image2);

        filter->Update();

        totalOverlap = filter->GetTotalOverlap();

        typedef typename FilterType::MapType MapType;
        typedef typename FilterType::MapConstIterator MapConstIterator;
    
        double overlapAcc = 0.0;
        unsigned int overlapCount = 0;

        double unionOverlapAcc = 0.0;
        unsigned int unionOverlapCount = 0; 
        unsigned int localLabelCount = 0;

        MapType map = filter->GetLabelSetMeasures();
        for (MapConstIterator mapIt = map.begin(); mapIt != map.end(); ++mapIt) {
            // Do not include the background in the final value.
            if ((*mapIt).first == 0)
            {
                continue;
            }

            ++localLabelCount;
            // Target overlap
            {
                double numerator = (double)(*mapIt).second.m_Intersection;
                double denominator = (double)(*mapIt).second.m_Target;
                if(denominator > 0) {
                    overlapAcc += (numerator/denominator);
                    ++overlapCount;
                }
            }

            {
                double numerator = (double)(*mapIt).second.m_Intersection;
                double denominator = (double)(*mapIt).second.m_Union;
                if(denominator > 0) {
                    unionOverlapAcc += (numerator/denominator);
                    ++unionOverlapCount;
                }
            }
        }

        if (overlapCount > 0)
            meanTotalOverlap = overlapAcc / overlapCount;
        else
            meanTotalOverlap = 0;

        if (unionOverlapCount > 0)
            meanUnionOverlap = unionOverlapAcc / unionOverlapCount;
        else
            meanUnionOverlap = 0;

        labelCount = localLabelCount;
}


template <typename TransformType, typename ImageType, typename LabelImageType>
void DoEvaluateRegistration(
    typename ImageType::Pointer refImage,
    typename ImageType::Pointer floImage,
    typename LabelImageType::Pointer refImageLabel,
    typename LabelImageType::Pointer floImageLabel,
    typename TransformType::Pointer transformForward,
    PerformanceMetrics& metrics
    )
{
    using ImagePointer = typename ImageType::Pointer;
    using LabelImagePointer = typename LabelImageType::Pointer;

    ImagePointer registeredImage = ApplyTransformToImage<ImageType, TransformType>(refImage, floImage, transformForward, 1, 0.0);

    // Compute the before mean absolute difference
    metrics.absDiff = MeanAbsDifference<ImageType>(refImage, registeredImage);

    if(refImageLabel && floImageLabel)
    {       
        LabelImagePointer registeredLabel = ApplyTransformToImage<LabelImageType, TransformType>(refImageLabel, floImageLabel, transformForward, 0, 0);

        LabelAccuracy<LabelImageType>(registeredLabel, refImageLabel, metrics.totalOverlap, metrics.meanTotalOverlap, metrics.meanUnionOverlap, metrics.labelCount);
    } else
    {
        metrics.totalOverlap = 0.0;
        metrics.meanTotalOverlap = 0.0;
        metrics.meanUnionOverlap = 0.0;
        metrics.labelCount = 0;
    }
}


template <unsigned int ImageDimension>
class EvaluateRegistration {
public:
    typedef itk::IPT<double, ImageDimension> IPT;

    typedef itk::Image<double, ImageDimension> ImageType;
    typedef typename ImageType::Pointer ImagePointer;
    typedef itk::Image<unsigned short, ImageDimension> LabelImageType;
    typedef typename LabelImageType::Pointer LabelImagePointer;
    
    static void Evaluate(
        std::string refImagePath,
        std::string floImagePath,
        std::string refLabelPath, 
        std::string floLabelPath,
        std::string ident,
        std::string transformPath
        )
    {
            ImagePointer refImage;
            try {
                refImage = IPT::LoadImage(refImagePath.c_str());

                refImage = RemoveDirectionInformation<ImageType>(refImage);
            }
            catch (itk::ExceptionObject & err)
		    {
                std::cerr << "Error loading reference image: " << refImagePath.c_str() << std::endl;
			    std::cerr << "ExceptionObject caught !" << std::endl;
			    std::cerr << err << std::endl;
		    }
            
            ImagePointer floImage;
            try {
                floImage = IPT::LoadImage(floImagePath.c_str());

                floImage = RemoveDirectionInformation<ImageType>(floImage);
            }
            catch (itk::ExceptionObject & err)
		    {
                std::cerr << "Error loading floating image: " << floImagePath.c_str() << std::endl;
			    std::cerr << "ExceptionObject caught !" << std::endl;
			    std::cerr << err << std::endl;
		    }

            // Load the label images
            LabelImagePointer refImageLabel;
            try {
                refImageLabel = IPT::LoadLabelImage(refLabelPath.c_str());

                refImageLabel = RemoveDirectionInformation<LabelImageType>(refImageLabel);
            }
            catch (itk::ExceptionObject & err)
		    {
                std::cerr << "Error loading reference label: " << refLabelPath.c_str() << std::endl;
			    std::cerr << "ExceptionObject caught !" << std::endl;
			    std::cerr << err << std::endl;
		    }            
            
            LabelImagePointer floImageLabel;
            try {
                floImageLabel = IPT::LoadLabelImage(floLabelPath.c_str());

                floImageLabel = RemoveDirectionInformation<LabelImageType>(floImageLabel);
            }
            catch (itk::ExceptionObject & err)
		    {
                std::cerr << "Error loading floating label: " << floLabelPath.c_str() << std::endl;
			    std::cerr << "ExceptionObject caught !" << std::endl;
			    std::cerr << err << std::endl;
		    }

            /*
            std::cerr << "Reference image information";
            std::cerr << refImage;
            std::cerr << "Floating image information";
            std::cerr << floImage;
            std::cerr << "Reference label information";
            std::cerr << refImageLabel;
            std::cerr << "Floating label information";
            std::cerr << floImageLabel;
            */

            using TransformBaseType = itk::Transform<double, ImageDimension, ImageDimension>;
            using TransformBasePointer = typename TransformBaseType::Pointer;
            using IdentityTransformType = itk::IdentityTransform<double, ImageDimension>;
            using IdentityTransformPointer = typename IdentityTransformType::Pointer;

            IdentityTransformPointer identityTransform = IdentityTransformType::New();
            if (transformPath.length() > 0)
            {
                TransformBasePointer transformForward;
                transformForward = IPT::LoadTransformFile(transformPath.c_str());

                PerformanceMetrics beforeMetrics;
                DoEvaluateRegistration<IdentityTransformType, ImageType, LabelImageType>(
                    refImage, floImage, refImageLabel, floImageLabel, identityTransform, beforeMetrics
                );

                PerformanceMetrics afterMetrics;
                DoEvaluateRegistration<TransformBaseType, ImageType, LabelImageType>(
                    refImage, floImage, refImageLabel, floImageLabel, transformForward, afterMetrics
                );

                printPairOfMetricsCSV(std::cout, beforeMetrics, afterMetrics, true, ident);
            } else {
                PerformanceMetrics metrics;
                DoEvaluateRegistration<IdentityTransformType, ImageType, LabelImageType>(
                    refImage, floImage, refImageLabel, floImageLabel, identityTransform, metrics
                );

                printMetricsCSV(std::cout, metrics, true, ident);
            }
    }

    static void MainFunc(int argc, char** argv)
    {
        if (argc == 8)
        {
            Evaluate(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
        }
        else if (argc == 7)
        {
            Evaluate(argv[2], argv[3], argv[4], argv[5], argv[6], "");
        }
        else
        {
            Evaluate(argv[2], argv[3], argv[4], argv[5], "", "");
        }
    }
};

int main(int argc, char** argv) {
    if(argc < 6) {
        std::cout << "Too few arguments..." << std::endl;
        std::cout << "Use as: EvaluateRegistration dim refimagepath floimagepath reflabelpath flolabelpath (id) (transformpath)" << std::endl;
        return -1;
    } else if(argc > 8) {
        std::cout << "Too many arguments..." << std::endl;
        std::cout << "Use as: EvaluateRegistration dim refimagepath floimagepath reflabelpath flolabelpath (id) (transformpath)" << std::endl;
        return -1;
    }
    int ndim = atoi(argv[1]);
    if(ndim == 2) {
        EvaluateRegistration<2U>::MainFunc(argc, argv);
    } else if(ndim == 3) {
        EvaluateRegistration<3U>::MainFunc(argc, argv);
    } else {
        std::cout << "Error: Dimensionality " << ndim << " is not supported." << std::endl;
    }
}
