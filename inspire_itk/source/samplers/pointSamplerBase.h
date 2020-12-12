
#ifndef POINT_SAMPLER_BASE_H
#define POINT_SAMPLER_BASE_H

#include "itkImage.h"
#include "itkImageRegionConstIterator.h"

#include "quasiRandomGenerator.h"

template <typename ImageType, typename WeightImageType=ImageType>
struct PointSample {
    typedef typename ImageType::PointType PointType;
    typedef typename ImageType::ValueType ValueType;
    typedef typename WeightImageType::ValueType WeightValueType;

    PointType m_Point;
    ValueType m_Value;
    WeightValueType m_Weight;
};

template <typename ImageType, typename MaskImageType, typename WeightImageType=ImageType>
class PointSamplerBase : public itk::Object {
public:
    using Self = PointSamplerBase;
    using Superclass = itk::Object;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    
    typedef typename ImageType::Pointer ImagePointer;
    typedef typename MaskImageType::Pointer MaskImagePointer;
    typedef typename WeightImageType::Pointer WeightImagePointer;

    typedef typename ImageType::ValueType ValueType;
    typedef typename MaskImageType::ValueType MaskValueType;
    typedef typename WeightImageType::ValueType WeightValueType;

    typedef typename ImageType::SpacingType SpacingType;
    typedef typename ImageType::RegionType RegionType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename ImageType::IndexValueType IndexValueType;
    typedef typename ImageType::PointType PointType;
    
    typedef PointSample<ImageType, WeightImageType> PointSampleType;
  
    // Leave out the new macro since this is an abstract base class

    itkTypeMacro(PointSamplerBase, itk::Object);

    virtual void SetImage(ImagePointer image) {
        m_Image = image;
        m_ImageRawPtr = image.GetPointer();
    }

    virtual void SetMaskImage(MaskImagePointer mask) {
        m_Mask = mask;
        m_MaskRawPtr = mask.GetPointer();
    }

    virtual void SetWeightImage(WeightImagePointer weights) {
        m_Weights = weights;
        m_WeightsRawPtr = weights.GetPointer();
    }

    virtual unsigned long long GetSeed() const {
        return m_Seed;
    }

    virtual void SetSeed(unsigned long long seed) {
        m_Seed = seed;
    }

    virtual void EndIteration(unsigned int count)
    {
        m_Seed = XORShiftRNG1D((m_Seed + count)*3U);
    }

    virtual bool IsDitheringOn() const { return m_Dithering; }

    virtual void SetDitheringOff() { m_Dithering = false; }

    virtual void SetDitheringOn() { m_Dithering = true; }

    virtual void Initialize() {
        m_Spacing = m_Image->GetSpacing();

        ComputeMaskBoundingBox();
    };

    // Abstract function for computing a random index given an origin and size
    // of a region of interest.
    virtual IndexType ComputeIndex(unsigned int sampleID, IndexType& origin, SizeType& size) = 0;

    virtual void Sample(unsigned int sampleID, PointSampleType& pointSampleOut, unsigned int attempts = 1) {
        ImageType* image = m_ImageRawPtr;
        MaskImageType* mask = m_MaskRawPtr;
        WeightImageType* weights = m_WeightsRawPtr;

        IndexType index = ComputeIndex(sampleID, m_BBOrigin, m_BBSize);

        bool isMasked = PerformMaskTest(index);
        if(!isMasked) {
            if(attempts > 0) {
                Sample(sampleID, pointSampleOut, attempts - 1U);
                return;
            }

            pointSampleOut.m_Weight = itk::NumericTraits<WeightValueType>::ZeroValue();
            return;
        }

        pointSampleOut.m_Value = m_ImageRawPtr->GetPixel(index);

        image->TransformIndexToPhysicalPoint(index, pointSampleOut.m_Point);

        if(weights)
        {
            pointSampleOut.m_Weight = weights->GetPixel(index);
        }
        else
        {
            pointSampleOut.m_Weight = itk::NumericTraits<WeightValueType>::OneValue();
        }

        DitherPoint(sampleID, pointSampleOut.m_Point);
    }

    virtual void SampleN(unsigned int sampleID, std::vector<PointSampleType>& pointSampleOut, unsigned int count, unsigned int attempts=1) {
        for(unsigned int i = 0; i < count; ++i) {
            PointSampleType pnt;
            Sample(sampleID + i, pnt, attempts);
            pointSampleOut.push_back(pnt);
        }
    }
protected:
    PointSamplerBase() {
        m_Seed = 42U;
        m_ImageRawPtr = nullptr;
        m_MaskRawPtr = nullptr;
        m_WeightsRawPtr = nullptr;
        m_Dithering = false;
    }
    virtual ~PointSamplerBase() {

    }

    void DitherPoint(unsigned int sampleID, PointType& point) {
        if(m_Dithering) {
            itk::FixedArray<double, ImageType::ImageDimension> vec = XORShiftRNGDouble<ImageType::ImageDimension>(m_Seed + sampleID*3U);

            for(unsigned int i = 0; i < ImageType::ImageDimension; ++i) {
                point[i] = point[i] + (vec[i]-0.5)*m_Spacing[i];
            }
        }
    }

    bool PerformMaskTest(IndexType index) {
        if(m_MaskRawPtr) {
            // Assume here that the index is inside the buffer

            return m_MaskRawPtr->GetPixel(index);
        }
        return true;
    }

    // If there is a mask, compute the bounding box of the pixels
    // inside the mask
    void ComputeMaskBoundingBox() {
        if(m_MaskRawPtr) {
            IndexType minIndex;
            IndexType maxIndex;
            minIndex.Fill(itk::NumericTraits<IndexValueType>::max());
            maxIndex.Fill(itk::NumericTraits<IndexValueType>::min());

            typedef itk::ImageRegionConstIterator<MaskImageType> IteratorType;
            IteratorType it(m_MaskRawPtr, m_MaskRawPtr->GetLargestPossibleRegion());

            it.GoToBegin();
            while(!it.IsAtEnd()) {
                if(it.Value()) {
                    IndexType curIndex = it.GetIndex();
                    for(unsigned int i = 0; i < ImageType::ImageDimension; ++i) {
                        if(curIndex[i] < minIndex[i])
                            minIndex[i] = curIndex[i];
                        if(curIndex[i] > maxIndex[i])
                            maxIndex[i] = curIndex[i];
                    }
                }

                ++it;
            }

            m_BBOrigin = minIndex;
            for(unsigned int i = 0; i < ImageType::ImageDimension; ++i) {
                m_BBSize[i] = maxIndex[i]-minIndex[i];
            }
        } else {
            // No mask, just use the bounds of the whole image
            RegionType region = m_ImageRawPtr->GetLargestPossibleRegion();
            m_BBOrigin = region.GetIndex();
            m_BBSize = region.GetSize();
        }
    }

    // Attributes

    // Non-threaded data
    ImagePointer m_Image;
    MaskImagePointer m_Mask;
    WeightImagePointer m_Weights;

    ImageType* m_ImageRawPtr;
    MaskImageType* m_MaskRawPtr;
    WeightImageType* m_WeightsRawPtr;

    unsigned long long m_Seed;
    
    SpacingType m_Spacing;
    bool m_Dithering;
    IndexType m_BBOrigin;
    SizeType m_BBSize;
}; // End of class PointSamplerBase

#endif

