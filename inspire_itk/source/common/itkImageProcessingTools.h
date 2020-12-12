
#ifndef ITK_IMAGE_PROCESSING_TOOLS_H
#define ITK_IMAGE_PROCESSING_TOOLS_H

#include "itkVersion.h"
#include "itkPoint.h"
#include "itkImage.h"
#include "itkSmartPointer.h"
#include "itkImageRegionIterator.h"
#include "itkNumericTraits.h"

#include "itkCovariantVector.h"

#include "itkImageDuplicator.h"

#include "itkCastImageFilter.h"
#include "itkClampImageFilter.h"
#include "itkConstantPadImageFilter.h"
#include "itkShiftScaleImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "itkExtractImageFilter.h"

#include "itkHistogramMatchingImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkAbsoluteValueDifferenceImageFilter.h"
#include "itkChangeInformationImageFilter.h"
#include <itkLabelOverlapMeasuresImageFilter.h>

#include "itkAdditiveGaussianNoiseImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkMedianImageFilter.h"

#include "itkCSVArray2DFileReader.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"
#include <itkChangeInformationImageFilter.h>
#include "itkMultiplyImageFilter.h"

#include <algorithm>

#include "minicsv.h"

// Transforms
#include "itkAffineTransform.h"
#include "itkIdentityTransform.h"
#include "itkCompositeTransform.h"
#include "itkTranslationTransform.h"
#include "itkTransformFactory.h"

// Interpolation headers
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

#include "itkChangeInformationImageFilter.h"

template <typename ImageType>
typename ImageType::Pointer RemoveDirectionInformation(typename ImageType::Pointer image)
{
    using FilterType = itk::ChangeInformationImageFilter<ImageType>;
    using FilterPointer = typename FilterType::Pointer;

    FilterPointer filter = FilterType::New();
    filter->SetInput(image);

/*
    typename ImageType::PointType origin;
    for (unsigned int i = 0; i < ImageType::ImageDimension; ++i)
    {
        origin[i] = 0;
    }

    filter->SetOutputOrigin(origin);
    filter->ChangeOriginOn();
*/
    auto rot = image->GetDirection();
    rot.SetIdentity();
    filter->SetOutputDirection(rot);
    filter->ChangeDirectionOn();

    filter->Update();

    auto result = filter->GetOutput();

    result->SetRequestedRegion(result->GetLargestPossibleRegion());
    result->SetBufferedRegion(result->GetLargestPossibleRegion());

    return result;
}

namespace itk
{

template <typename Src, typename Dest>
inline itk::SmartPointer<Dest> CastSmartPointer(itk::SmartPointer<Src> ptr)
{
    itk::SmartPointer<Dest> result = itk::SmartPointer<Dest>((Dest *)ptr.GetPointer());
    return result;
}

template <typename TInputImage, typename TOutputImage>
inline typename TOutputImage::Pointer CastImage(
    typename TInputImage::Pointer inputImage)
{

    typedef itk::ClampImageFilter<TInputImage, TOutputImage> CastFilter;
    typedef typename CastFilter::Pointer CastFilterPointer;

    CastFilterPointer castFilter = CastFilter::New();
    castFilter->SetInput(inputImage);
    castFilter->Update();

    return castFilter->GetOutput();
}

template <typename ImagePixelType, unsigned int Dim>
typename itk::Image<ImagePixelType, Dim>::Pointer ConvertImageToIntegerFormat(typename itk::Image<double, Dim>::Pointer image)
{
    typedef double ValueType;
    typedef itk::Image<double, Dim> ImageType;

    typedef itk::Image<ImagePixelType, Dim> ImageUType;

    typedef itk::ShiftScaleImageFilter<ImageType, ImageType> RescaleFilter;
    typedef typename RescaleFilter::Pointer RescaleFilterPointer;

    typedef itk::ClampImageFilter<ImageType, ImageUType> CastFilter;
    typedef typename CastFilter::Pointer CastFilterPointer;

    ValueType minVal = (ValueType)itk::NumericTraits<ImagePixelType>::min();
    ValueType maxVal = (ValueType)itk::NumericTraits<ImagePixelType>::max();//+0.9999999999;
    ValueType span = (maxVal - minVal);
    ValueType halfInt = (itk::NumericTraits<ValueType>::One / 2);

    RescaleFilterPointer rescaleFilter = RescaleFilter::New();
    rescaleFilter->SetInput(image);
    rescaleFilter->SetShift(0);
    //rescaleFilter->SetShift((halfInt / span) + minVal);
    rescaleFilter->SetScale(span);
    rescaleFilter->Update();

    CastFilterPointer castFilter = CastFilter::New();
    castFilter->SetInput(rescaleFilter->GetOutput());
    castFilter->Update();

    return castFilter->GetOutput();
}

template <typename ImagePixelType, unsigned int Dim>
typename itk::Image<ImagePixelType, Dim>::Pointer ConvertImageF32ToIntegerFormat(typename itk::Image<float, Dim>::Pointer image)
{
    typename itk::Image<double, Dim>::Pointer casted = CastImage<itk::Image<float, Dim>, itk::Image<double, Dim> >(image);
    return ConvertImageToIntegerFormat<ImagePixelType, Dim>(casted);
    /*
    typedef itk::CastImageFilter<itk::Image<double, Dim>, itk::Image<float, Dim> > CastFilter;
    typedef typename CastFilter::Pointer CastFilterPointer;

    CastFilterPointer castFilter = CastFilter::New();
    castFilter->SetInput(image);
    castFilter->Update();

    return ConvertImageToIntegerFormat(castFilter->GetOutput());*/
}

template <typename ImagePixelType, unsigned int Dim>
typename itk::Image<double, Dim>::Pointer ConvertImageFromIntegerFormat(
    typename itk::Image<ImagePixelType, Dim>::Pointer image)
{
    typedef double ValueType;
    typedef itk::Image<double, Dim> ImageType;

    typedef itk::Image<ImagePixelType, Dim> ImageUType;

    typedef itk::CastImageFilter<ImageUType, ImageType> CastFilter;
    typedef typename CastFilter::Pointer CastFilterPointer;

    typedef itk::ShiftScaleImageFilter<ImageType, ImageType> RescaleFilter;
    typedef typename RescaleFilter::Pointer RescaleFilterPointer;

    CastFilterPointer castFilter = CastFilter::New();
    castFilter->SetInput(image);
    castFilter->Update();

    ValueType minVal = (ValueType)itk::NumericTraits<ImagePixelType>::min();
    ValueType maxVal = (ValueType)itk::NumericTraits<ImagePixelType>::max();
    ValueType span = (maxVal - minVal);

    RescaleFilterPointer rescaleFilter = RescaleFilter::New();
    rescaleFilter->SetInput(castFilter->GetOutput());
    rescaleFilter->SetShift(-minVal);
    rescaleFilter->SetScale(itk::NumericTraits<ValueType>::One / span);
    rescaleFilter->Update();

    return rescaleFilter->GetOutput();
}

template <typename ImageType>
inline void PrintStatistics(
    typename ImageType::Pointer image,
    std::string name)
{

    typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;
    typename StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New();
    statisticsImageFilter->SetInput(image);
    statisticsImageFilter->Update();

    std::cout << "--- Image " << name << " statistics ---";
    std::cout << "Origin:   " << image->GetLargestPossibleRegion().GetIndex() << std::endl;
    std::cout << "Size:     " << image->GetLargestPossibleRegion().GetSize() << std::endl;
    std::cout << "Spacing:  " << image->GetSpacing() << std::endl;
    std::cout << "Direction:" << image->GetDirection() << std::endl;
    std::cout << "Mean:     " << statisticsImageFilter->GetMean() << std::endl;
    std::cout << "Std. Dev: " << statisticsImageFilter->GetSigma() << std::endl;
    std::cout << "Min:      " << statisticsImageFilter->GetMinimum() << std::endl;
    std::cout << "Max:      " << statisticsImageFilter->GetMaximum() << std::endl;
}

template <typename PixelType, unsigned int Dim>
inline typename itk::Image<PixelType, Dim - 1>::Pointer ExtractSlice(
    typename itk::Image<PixelType, Dim>::Pointer image,
    unsigned int dim,
    int ind)
{
    typedef itk::Image<PixelType, Dim> InputImageType;
    typedef itk::Image<PixelType, Dim - 1> OutputImageType;

    typedef itk::ExtractImageFilter<InputImageType, OutputImageType> FilterType;
    typedef typename FilterType::Pointer FilterPointer;

    auto region = image->GetLargestPossibleRegion();
    typedef typename InputImageType::IndexType IndexType;
    typedef typename InputImageType::SizeType SizeType;

    IndexType index = region.GetIndex();
    SizeType size = region.GetSize();

    index[dim] = ind;
    size[dim] = 0;

    region.SetIndex(index);
    region.SetSize(size);

    FilterPointer filter = FilterType::New();
    filter->SetExtractionRegion(region);
    filter->SetInput(image);
    filter->SetDirectionCollapseToIdentity(); // This is required.
    filter->Update();

    return filter->GetOutput();
}

template <typename T>
void VectorToCSV(const char *path, std::vector<T> &values)
{
    FILE *f = fopen(path, "wb");

    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
            fprintf(f, ",");

        fprintf(f, "%.7f", values[i]);
    }

    fclose(f);
}

template <typename ValueType, unsigned int Dim>
class IPT
{
  public:
    typedef itk::Image<ValueType, Dim> ImageType;
    typedef typename ImageType::Pointer ImagePointer;

    typedef itk::Image<unsigned char, Dim> ImageU8Type;
    typedef typename ImageU8Type::Pointer ImageU8Pointer;
    typedef itk::Image<unsigned short, Dim> ImageU16Type;
    typedef typename ImageU16Type::Pointer ImageU16Pointer;

    typedef itk::Image<unsigned short, Dim> LabelImageType;
    typedef typename LabelImageType::Pointer LabelImagePointer;

    typedef typename itk::Image<bool, Dim> BinaryImageType;
    typedef typename BinaryImageType::Pointer BinaryImagePointer;

    typedef typename ImageType::PointType PointType;
    typedef typename ImageType::PointValueType PointValueType;

    typedef std::vector<PointType> PointSetType;

    typedef typename ImageType::RegionType RegionType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename ImageType::SpacingType SpacingType;

    typedef itk::Vector<PointValueType, Dim> VectorType;
    typedef itk::CovariantVector<PointValueType, Dim> CovariantVectorType;

    typedef itk::Transform<PointValueType, Dim, Dim> BaseTransformType;
    typedef typename BaseTransformType::Pointer BaseTransformPointer;
    typedef itk::CompositeTransform<PointValueType, Dim> CompositeTransformType;
    typedef typename CompositeTransformType::Pointer CompositeTransformPointer;

    //
    // Routines for computing image sizes and centers.
    //

    static SizeType GetImageSize(
        ImagePointer image)
    {
        RegionType region = image->GetLargestPossibleRegion();

        SizeType size = region.GetSize();

        return size;
    }

  private:
    template <typename SourceType, typename DestType>
    static DestType MakeVT(
        SourceType src)
    {
        DestType result;

        for (unsigned int i = 0; i < Dim; ++i)
        {
            result[i] = src[i];
        }

        return result;
    }

  public:
    //
    //
    // Routines for creating vector-equivalent classes (VectorType, CovariantVectorType, PointType, SizeType, ...)
    //
    //

    static VectorType MakeVector(
        const PointValueType *p)
    {
        return MakeVT<const PointValueType *, VectorType>(p);
    }

    static VectorType MakeVector(
        CovariantVectorType v)
    {
        return MakeVT<CovariantVectorType, VectorType>(v);
    }

    static VectorType MakeVector(
        PointType v)
    {
        return MakeVT<PointType, VectorType>(v);
    }

    static CovariantVectorType MakeCovariantVector(
        const PointValueType *p)
    {
        return MakeVT<const PointValueType *, CovariantVectorType>(p);
    }

    static CovariantVectorType MakeCovariantVector(
        PointType p)
    {
        return MakeVT<PointType, CovariantVectorType>(p);
    }

    static CovariantVectorType MakeCovariantVector(
        VectorType p)
    {
        return MakeVT<VectorType, CovariantVectorType>(p);
    }

    static PointType MakePoint(
        const PointValueType *p)
    {
        return MakeVT<const PointValueType *, PointType>(p);
    }

    static VectorType ReLUVector(
        VectorType v)
    {
        for (unsigned int i = 0; i < Dim; ++i)
        {
            if (v[i] < 0)
                v[i] = 0;
        }

        return v;
    }

    static CovariantVectorType ReLUCovariantVector(
        CovariantVectorType v)
    {
        for (unsigned int i = 0; i < Dim; ++i)
        {
            if (v[i] < 0)
                v[i] = 0;
        }

        return v;
    }

    static PointType ReLUPoint(
        PointType v)
    {
        for (unsigned int i = 0; i < Dim; ++i)
        {
            if (v[i] < 0)
                v[i] = 0;
        }

        return v;
    }

    static VectorType RoundVector(
        VectorType v)
    {
        for (unsigned int i = 0; i < Dim; ++i)
        {
            v[i] = static_cast<PointValueType>(itk::Math::Round<long long, PointValueType>(v[i]));
        }

        return v;
    }

    static CovariantVectorType RoundCovariantVector(
        CovariantVectorType v)
    {
        for (unsigned int i = 0; i < Dim; ++i)
        {
            v[i] = static_cast<PointValueType>(itk::Math::Round<long long, PointValueType>(v[i]));
        }

        return v;
    }

    static PointType RoundPoint(
        PointType v)
    {
        for (unsigned int i = 0; i < Dim; ++i)
        {
            v[i] = static_cast<PointValueType>(itk::Math::Round<long long, PointValueType>(v[i]));
        }

        return v;
    }

    static VectorType AbsVector(
        VectorType v)
    {
        for (unsigned int i = 0; i < Dim; ++i)
        {
            v[i] = itk::Math::abs(v[i]);
        }

        return v;
    }

    static CovariantVectorType AbsCovariantVector(
        CovariantVectorType v)
    {
        for (unsigned int i = 0; i < Dim; ++i)
        {
            v[i] = itk::Math::abs(v[i]);
        }

        return v;
    }

    static PointType AbsPoint(
        PointType v)
    {
        for (unsigned int i = 0; i < Dim; ++i)
        {
            v[i] = itk::Math::abs(v[i]);
        }

        return v;
    }

    static SizeType MakeSize(
        CovariantVectorType v)
    {
        SizeType result;

        for (size_t i = 0; i < Dim; ++i)
        {
            result[i] = itk::Math::ceil(v[i]); //1 + static_cast<itk::SizeValueType>(v[i]);
        }

        return result;
    }

    static SizeType MakeSize(
        VectorType v)
    {
        SizeType result;

        for (size_t i = 0; i < Dim; ++i)
        {
            result[i] = itk::Math::ceil(v[i]); //1 + static_cast<itk::SizeValueType>(v[i]);
        }

        return result;
    }

    static PointType ComputeImageCenter(
        ImagePointer image,
        CovariantVectorType offset,
        bool considerSpacing)
    {

        PointType result;

        RegionType region = image->GetLargestPossibleRegion();
        SpacingType spacing = image->GetSpacing();

        IndexType index = region.GetIndex();
        SizeType size = region.GetSize();

        if (considerSpacing)
        {
            for (unsigned int i = 0; i < Dim; ++i)
            {
                result[i] = (double)(index[i] + spacing[i] * size[i] * 0.5 + offset[i]);
            }
        }
        else
        {
            for (unsigned int i = 0; i < Dim; ++i)
            {
                result[i] = (double)(index[i] + size[i] * 0.5 + offset[i]);
            }
        }

        return result;
    }

    static PointType ComputeImageCenter(
        ImagePointer image,
        bool considerSpacing)
    {

        CovariantVectorType offset;
        offset.Fill(0.0);

        return ComputeImageCenter(image, offset, considerSpacing);
    }

    static PointValueType ComputeImageDiagonalSize(
        ImagePointer image,
        bool considerSpacing)
    {

        PointValueType result = 0;

        typename itk::Image<ValueType, Dim>::RegionType region = image->GetLargestPossibleRegion();

        typename itk::Image<ValueType, Dim>::SizeType size = region.GetSize();

        if (considerSpacing)
        {
            for (unsigned int i = 0; i < Dim; ++i)
            {
                PointValueType sideLength = size[i] * image->GetSpacing()[i];
                result += itk::Math::sqr(static_cast<PointValueType>(sideLength));
            }
        }
        else
        {
            for (unsigned int i = 0; i < Dim; ++i)
            {
                result += itk::Math::sqr(static_cast<PointValueType>(size[i]));
            }
        }

        return sqrt(result);
    }

    //
    // Routines for image creation / duplication.
    //

    static ImagePointer ConstantImage(ValueType c, SizeType sz)
    {
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::RegionType RegionType;

        RegionType region;

        IndexType index;
        index.Fill(0);

        region.SetIndex(index);
        region.SetSize(sz);

        ImagePointer image = ImageType::New();

        image->SetRegions(region);

        image->Allocate(false);

        image->FillBuffer(c);

        return image;
    }

    static ImagePointer ZeroImage(SizeType sz)
    {
        return ConstantImage(0, sz);
    }

    static PointType ImageGeometricalCenterPoint(
        ImagePointer image)
    {
    const typename ImageType::RegionType & region =
      image->GetLargestPossibleRegion();
    const typename ImageType::IndexType & index =
      region.GetIndex();
    const typename ImageType::SizeType & size =
      region.GetSize();

    PointType point;

    typedef typename PointType::ValueType CoordRepType;

    typedef ContinuousIndex< CoordRepType,
                             Dim >  ContinuousIndexType;

    typedef typename ContinuousIndexType::ValueType ContinuousIndexValueType;

    ContinuousIndexType centerIndex;

    for ( unsigned int k = 0; k < Dim; k++ )
      {
      centerIndex[k] =
        static_cast< ContinuousIndexValueType >( index[k] )
        + static_cast< ContinuousIndexValueType >( size[k] - 1 ) / 2.0;
      }

    image->TransformContinuousIndexToPhysicalPoint(centerIndex, point);
    return point;

    }

    static ImagePointer CircularMask(
        typename ImageType::RegionType region,
        const SpacingType &spacing,
        ValueType zero,
        ValueType one)
    {

        ImagePointer image = ImageType::New();

        typename ImageType::SizeType size = region.GetSize();

        PointType center;
        ValueType radius;
        ValueType radiusRec;

        image->SetRegions(region);
        image->SetSpacing(spacing);
        image->Allocate();

        center = ImageGeometricalCenterPoint(image);

        typename ImageType::IndexType index;

        for(unsigned int i = 0; i < Dim; ++i) {
            index[i] = size[i] / 2;
            ValueType rad = spacing[i] * size[i] / 2.0;
            if(i == 0 || rad < radius) {
                radius = rad;
            }
        }

        if(radius == 0.0) {
            radiusRec = 0.0;
        } else {
            radiusRec = 1.0 / radius;
        }

        itk::ImageRegionIterator<ImageType> writeIter(image, image->GetLargestPossibleRegion());
        writeIter.GoToBegin();

        PointType point;
        VectorType v;

        while (!writeIter.IsAtEnd())
        {
            image->TransformIndexToPhysicalPoint(writeIter.GetIndex(), point);

            for (unsigned int i = 0; i < Dim; ++i)
            {
                v[i] = (point[i] - center[i]) * radiusRec;
            }

            auto norm = v.GetNorm();
            if (norm <= 1.0)
                writeIter.Set(one);
            else
                writeIter.Set(zero);

            ++writeIter;
        }

        return image;
    }

    static ValueType HannFunc(
        ValueType t)
    {
        ValueType tt = 0.5 + 0.5 * t;
        return 0.5 * (1.0 - cos(itk::Math::twopi * tt));
    }

   static ImagePointer HannMask(
        typename ImageType::RegionType region,
        const SpacingType &spacing,
        ValueType exponent = 1.0)
    {

        ImagePointer image = ImageType::New();

        typename ImageType::SizeType size = region.GetSize();

        PointType center;
        ValueType radius;
        ValueType radiusRec;

        image->SetRegions(region);
        image->SetSpacing(spacing);
        image->Allocate();

        center = ImageGeometricalCenterPoint(image);

        typename ImageType::IndexType index;

        for(unsigned int i = 0; i < Dim; ++i) {
            index[i] = size[i] / 2;
            ValueType rad = spacing[i] * size[i] / 2.0;
            if(i == 0 || rad < radius) {
                radius = rad;
            }
        }

        if(radius == 0.0) {
            radiusRec = 0.0;
        } else {
            radiusRec = 1.0 / radius;
        }

        itk::ImageRegionIterator<ImageType> writeIter(image, image->GetLargestPossibleRegion());
        writeIter.GoToBegin();

        PointType point;
        VectorType v;

        while (!writeIter.IsAtEnd())
        {
            image->TransformIndexToPhysicalPoint(writeIter.GetIndex(), point);

            for (unsigned int i = 0; i < Dim; ++i)
            {
                v[i] = (point[i] - center[i]) * radiusRec;
            }

            auto norm = v.GetNorm();
            if (norm <= 1.0)
                writeIter.Set(pow(HannFunc(norm), exponent));
            else
                writeIter.Set(0.0);

            ++writeIter;
        }

        return image;
    }

    static ImagePointer CircularMask2(
        typename ImageType::RegionType region,
        const SpacingType &spacing,
        ValueType zero,
        ValueType one)
    {

        ImagePointer image = ImageType::New();

        typename ImageType::IndexType start = region.GetIndex();
        typename ImageType::SizeType size = region.GetSize();

        PointType center;
        ValueType radius[Dim];

        for (unsigned int i = 0; i < Dim; ++i)
        {
            center[i] = 0.5 * ((double)size[i] - (double)start[i]);
            radius[i] = 0.5 * ((double)size[i] - (double)start[i]);

            if (radius[i] > 0.0)
            {
                radius[i] = 1.0 / radius[i];
            }
            else
            {
                radius[i] = 0.0;
            }
        }

        image->SetRegions(region);
        image->SetSpacing(spacing);
        image->Allocate();

        itk::ImageRegionIterator<ImageType> writeIter(image, image->GetLargestPossibleRegion());
        writeIter.GoToBegin();

        PointType point;

        while (!writeIter.IsAtEnd())
        {
            image->TransformIndexToPhysicalPoint(writeIter.GetIndex(), point);

            for (unsigned int i = 0; i < Dim; ++i)
            {
                point[i] = (point[i] - center[i]) * radius[i];
            }

            VectorType v = MakeVector(point);

            //if(point.EuclideanDistanceTo(center) <= 1.0)
            if (v.GetNorm() <= 1.0)
                writeIter.Set(one);
            else
                writeIter.Set(zero);

            ++writeIter;
        }

        return image;
    }

    static ImagePointer RectMask(
        typename ImageType::RegionType region,
        const SpacingType &spacing,
        ValueType one)
    {

        ImagePointer image = ImageType::New();

        image->SetRegions(region);
        image->SetSpacing(spacing);
        image->Allocate();

        itk::ImageRegionIterator<ImageType> writeIter(image, image->GetLargestPossibleRegion());
        writeIter.GoToBegin();

        PointType point;

        while (!writeIter.IsAtEnd())
        {
            writeIter.Set(one);
            ++writeIter;
        }

        return image;
    }

    static ImagePointer CloneImage(ImagePointer image)
    {
        typedef itk::ImageDuplicator<ImageType> DuplicatorType;
        typedef typename DuplicatorType::Pointer DuplicatorPointer;

        DuplicatorPointer dup = DuplicatorType::New();

        dup->SetInputImage(image);

        dup->Update();

        return dup->GetOutput();
    }

    static LabelImagePointer CloneLabelImage(LabelImagePointer image)
    {
        typedef itk::ImageDuplicator<LabelImageType> DuplicatorType;
        typedef typename DuplicatorType::Pointer DuplicatorPointer;

        DuplicatorPointer dup = DuplicatorType::New();

        dup->SetInputImage(image);

        dup->Update();

        return dup->GetOutput();
    }

    static ImagePointer SubsampleImage(
        ImagePointer image,
        unsigned int factor)
    {
        if(!image)
            return image;
        if (factor <= 1)
            return image;

        typedef itk::ShrinkImageFilter<ImageType, ImageType> FilterType;
        typedef typename FilterType::Pointer FilterPointer;

        FilterPointer filter = FilterType::New();
        filter->SetInput(image);
        filter->SetShrinkFactors(factor);

        filter->Update();

        return filter->GetOutput();
    }

    static LabelImagePointer SubsampleLabelImage(
        LabelImagePointer image,
        unsigned int factor)
    {
        if (factor <= 0)
            return image;

        typedef itk::ShrinkImageFilter<LabelImageType, LabelImageType> FilterType;
        typedef typename FilterType::Pointer FilterPointer;

        FilterPointer filter = FilterType::New();
        filter->SetInput(image);
        filter->SetShrinkFactors(factor);

        filter->Update();

        return filter->GetOutput();
    }

    static ImagePointer LoadImageU8(
        const char *path)
    {
        typedef itk::ImageFileReader<ImageU8Type> ReaderType;
        typedef typename ReaderType::Pointer ReaderPointer;

        ReaderPointer reader = ReaderType::New();

        reader->SetFileName(path);
        reader->Update();

        return ConvertImageFromU8Format(reader->GetOutput());
    }

    static void SaveImageU8(
        std::string path,
        ImagePointer image)
    {

        typedef itk::ImageFileWriter<ImageU8Type> WriterType;
        typedef typename WriterType::Pointer WriterPointer;

        ImageU8Pointer imageu8 = ConvertImageToIntegerFormat<unsigned char>(image);

        WriterPointer writer = WriterType::New();

        writer->SetFileName(path.c_str());
        writer->SetInput(imageu8);
        writer->Update();
    }

    static void SaveImageU16(
        std::string path,
        ImagePointer image)
    {

        typedef itk::ImageFileWriter<ImageU16Type> WriterType;
        typedef typename WriterType::Pointer WriterPointer;

        ImageU16Pointer imageu16 = ConvertImageToIntegerFormat<unsigned short>(image);

        WriterPointer writer = WriterType::New();

        writer->SetFileName(path.c_str());
        writer->SetInput(imageu16);
        writer->Update();
    }

    static void SaveImage(
        std::string path,
        ImagePointer image,
        bool format16U)
    {
        if(format16U)
            SaveImageU16(path, image);
        else
            SaveImageU8(path, image);
    }

    template <typename ImagePixelType>
    static typename itk::Image<ImagePixelType, Dim>::Pointer ReadImage(const char *path)
    {
        typedef itk::ImageFileReader<itk::Image<ImagePixelType, Dim>> ReaderType;
        typedef typename ReaderType::Pointer ReaderPointer;

        ReaderPointer reader = ReaderType::New();

        reader->SetFileName(path);
        reader->Update();

        return reader->GetOutput();
    }

    static ImagePointer LoadImage(
        const char *path)
    {

        typedef itk::ImageIOBase::IOComponentType ScalarPixelType;

        itk::ImageIOBase::Pointer imageIO =
            itk::ImageIOFactory::CreateImageIO(
                path, itk::ImageIOFactory::ReadMode);
        if (!imageIO)
        {
            std::cout << "Failed to create image io object to label image: " << path << "." << std::endl;
            //itkExceptionMacro ("Failed to read image information for file: " << path << ".");
            //throw itkExceptionObject("Failed to read image information.");
            std::cout << "Failed to read image information for file: " << path << "." << std::endl;
            assert(false);
        }

        imageIO->SetFileName(path);
        imageIO->ReadImageInformation();

        const ScalarPixelType pixelType = imageIO->GetComponentType();

        switch (pixelType)
        {
            case ImageIOBase::UNKNOWNCOMPONENTTYPE: {
                std::cout << "UNKNOWNCOMPONENTTYPE not supported." << std::endl;
                break;
            }
            case itk::ImageIOBase::UCHAR:
            {
                typedef unsigned char ImagePixelType;
                return ConvertImageFromIntegerFormat<ImagePixelType>(ReadImage<ImagePixelType>(path));
            }
            case itk::ImageIOBase::CHAR:
            {
                typedef signed char ImagePixelType;
                return ConvertImageFromIntegerFormat<ImagePixelType>(ReadImage<ImagePixelType>(path));
            }
            case itk::ImageIOBase::USHORT:
            {
                typedef unsigned short ImagePixelType;
                return ConvertImageFromIntegerFormat<ImagePixelType>(ReadImage<ImagePixelType>(path));
            }
            case itk::ImageIOBase::SHORT:
            {
                typedef signed short ImagePixelType;
                return ConvertImageFromIntegerFormat<ImagePixelType>(ReadImage<ImagePixelType>(path));
            }
            case itk::ImageIOBase::UINT:
            {
                typedef unsigned int ImagePixelType;
                return ConvertImageFromIntegerFormat<ImagePixelType>(ReadImage<ImagePixelType>(path));
            }
            case itk::ImageIOBase::INT:
            {
                typedef signed int ImagePixelType;
                return ConvertImageFromIntegerFormat<ImagePixelType>(ReadImage<ImagePixelType>(path));
            }
            case ImageIOBase::ULONG: {
                std::cout << "ULONG datatype not supported." << std::endl;
                break;
            }
            case ImageIOBase::LONG: {
                std::cout << "LONG datatype not supported." << std::endl;
                break;
            }
            case itk::ImageIOBase::FLOAT:
        {
            typedef float ImagePixelType;
            return  CastImage<itk::Image<ImagePixelType, Dim>, itk::Image<ValueType, Dim> >(ReadImage<ImagePixelType>(path));
           
            //std::cout << "Float-typed images not supported yet: " << path << "." << std::endl;

            //assert(false);
            //itkExceptionMacro ("Float-typed images not supported yet: " << path << ".");
        }
        case itk::ImageIOBase::DOUBLE:
        {
            return ReadImage<ValueType>(path);
        }
        }

        return nullptr;
    }

    static typename itk::Image<unsigned short, Dim>::Pointer LoadLabelImage(
        const char *path)
    {
        typedef unsigned short PixelType;
        typedef itk::ImageIOBase::IOComponentType ScalarPixelType;

        itk::ImageIOBase::Pointer imageIO =
            itk::ImageIOFactory::CreateImageIO(
                path, itk::ImageIOFactory::ReadMode);
        if (!imageIO)
        {
            std::cout << "Failed to create image io object to label image: " << path << "." << std::endl;
            //itkExceptionMacro ("Failed to read image information for file: " << path << ".");
            //throw itkExceptionObject("Failed to read image information.");
            assert(false);
        }

        imageIO->SetFileName(path);
        imageIO->ReadImageInformation();

        const ScalarPixelType pixelType = imageIO->GetComponentType();

        switch (pixelType)
        {
            case ImageIOBase::UNKNOWNCOMPONENTTYPE: {
                std::cout << "UNKNOWNCOMPONENTTYPE not supported." << std::endl;
                break;
            }
            case itk::ImageIOBase::UCHAR:
            {
                typedef unsigned char ImagePixelType;
                return CastImage<itk::Image<ImagePixelType, Dim>, itk::Image<PixelType, Dim>>(ReadImage<ImagePixelType>(path));
            }
            case itk::ImageIOBase::CHAR:
            {
                typedef signed char ImagePixelType;
                return CastImage<itk::Image<ImagePixelType, Dim>, itk::Image<PixelType, Dim>>(ReadImage<ImagePixelType>(path));
            }
            case itk::ImageIOBase::USHORT:
            {
                typedef unsigned short ImagePixelType;
                return CastImage<itk::Image<ImagePixelType, Dim>, itk::Image<PixelType, Dim>>(ReadImage<ImagePixelType>(path));
            }
            case itk::ImageIOBase::SHORT:
            {
                typedef signed short ImagePixelType;
                return CastImage<itk::Image<ImagePixelType, Dim>, itk::Image<PixelType, Dim>>(ReadImage<ImagePixelType>(path));
            }
            case itk::ImageIOBase::UINT:
            {
                typedef unsigned int ImagePixelType;
                return CastImage<itk::Image<ImagePixelType, Dim>, itk::Image<PixelType, Dim>>(ReadImage<ImagePixelType>(path));
            }
            case itk::ImageIOBase::INT:
            {
                typedef signed int ImagePixelType;
                return CastImage<itk::Image<ImagePixelType, Dim>, itk::Image<PixelType, Dim>>(ReadImage<ImagePixelType>(path));
            }
            case ImageIOBase::ULONG: {
                std::cout << "ULONG datatype not supported." << std::endl;
                break;
            }
            case ImageIOBase::LONG: {
                std::cout << "LONG datatype not supported." << std::endl;
                break;
            }
            case itk::ImageIOBase::FLOAT:
        {
            typedef float ImagePixelType;
            return CastImage<itk::Image<ImagePixelType, Dim>, itk::Image<PixelType, Dim>>(ReadImage<ImagePixelType>(path));//ConvertImageF32ToIntegerFormat<PixelType, Dim>(ReadImage<ImagePixelType>(path));
            //std::cout << "Float-typed images not supported yet: " << path << "." << std::endl;
            //assert(false);
            //itkExceptionMacro ("Float-typed images not supported yet: " << path << ".");
        }
        case itk::ImageIOBase::DOUBLE:
        {
            typedef double ImagePixelType;
            return CastImage<itk::Image<ImagePixelType, Dim>, itk::Image<PixelType, Dim>>(ReadImage<ImagePixelType>(path));//itk::ConvertImageToIntegerFormat<PixelType, Dim>(ReadImage<ImagePixelType>(path));
            //std::cout << "Float-typed images not supported yet: " << path << "." << std::endl;
            //assert(false);
        }
        }

        return nullptr;
    }

    static void SaveLabelImage(
        std::string path,
        LabelImagePointer image)
    {
        typedef itk::ImageFileWriter<LabelImageType> WriterType;
        typedef typename WriterType::Pointer WriterPointer;

        WriterPointer writer = WriterType::New();

        writer->SetFileName(path.c_str());
        writer->SetInput(image);
        writer->Update();
    }

    //
    // Creation and specification of image interpolators.
    //

    enum ImwarpInterpolationMode
    {
        kImwarpInterpNearest,
        kImwarpInterpLinear,
        kImwarpInterpCubic,
        kImwarpInterpBSpline,
    };

    static typename itk::InterpolateImageFunction<ImageType, PointValueType>::Pointer MakeInterpolator(ImwarpInterpolationMode mode)
    {
        typedef itk::InterpolateImageFunction<ImageType, PointValueType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointer;

        InterpolatorPointer ptr;

        if (mode == kImwarpInterpNearest)
        {
            ptr = itk::NearestNeighborInterpolateImageFunction<ImageType, PointValueType>::New();
        }
        else if (mode == kImwarpInterpLinear)
        {
            ptr = itk::LinearInterpolateImageFunction<ImageType, PointValueType>::New();
        }
        else if (mode == kImwarpInterpCubic)
        {
            typedef itk::BSplineInterpolateImageFunction<ImageType, PointValueType> BSplineInterpolatorType;
            typedef typename BSplineInterpolatorType::Pointer BSplineInterpolatorPointer;

            BSplineInterpolatorPointer bsplinePtr = itk::BSplineInterpolateImageFunction<ImageType, PointValueType>::New();
            bsplinePtr->SetSplineOrder(3);
            ptr = bsplinePtr;
        }
        else if (mode == kImwarpInterpBSpline)
        {
            ptr = itk::BSplineInterpolateImageFunction<ImageType, PointValueType>::New();
        }
        else
        {
            assert(false);
        }

        return ptr;
    }

    static typename itk::InterpolateImageFunction<LabelImageType, PointValueType>::Pointer MakeLabelInterpolator()
    {
        typedef itk::InterpolateImageFunction<LabelImageType, PointValueType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointer;

        InterpolatorPointer ptr;

        ptr = itk::NearestNeighborInterpolateImageFunction<LabelImageType, PointValueType>::New();

        return ptr;
    }

    //
    // Format convertion routines.
    //

    template <typename ImagePixelType = unsigned char>
    static typename itk::Image<ImagePixelType, Dim>::Pointer ConvertImageToIntegerFormat(ImagePointer image)
    {

        typedef itk::Image<ImagePixelType, Dim> ImageUType;

        typedef itk::ShiftScaleImageFilter<ImageType, ImageType> RescaleFilter;
        typedef typename RescaleFilter::Pointer RescaleFilterPointer;

        typedef itk::ClampImageFilter<ImageType, ImageUType> CastFilter;
        typedef typename CastFilter::Pointer CastFilterPointer;

        ValueType minVal = (ValueType)itk::NumericTraits<ImagePixelType>::min();
        ValueType maxVal = (ValueType)itk::NumericTraits<ImagePixelType>::max();
        ValueType span = (maxVal - minVal);
//std::cout << "minVal: " << minVal << ", maxVal: " << maxVal << ", span: " << span << std::endl;
        ValueType halfInt = (itk::NumericTraits<ValueType>::One / 2);
//(halfInt / span) + minVal
        RescaleFilterPointer rescaleFilter = RescaleFilter::New();
        rescaleFilter->SetInput(image);
        rescaleFilter->SetShift(0);
        rescaleFilter->SetScale(span);
        rescaleFilter->Update();

        CastFilterPointer castFilter = CastFilter::New();
        castFilter->SetInput(rescaleFilter->GetOutput());
        castFilter->Update();

        return castFilter->GetOutput();
    }

    template <typename ImagePixelType = unsigned char>
    static ImagePointer ConvertImageFromIntegerFormat(
        typename itk::Image<ImagePixelType, Dim>::Pointer image)
    {
        typedef itk::Image<ImagePixelType, Dim> ImageUType;

        typedef itk::CastImageFilter<ImageUType, ImageType> CastFilter;
        typedef typename CastFilter::Pointer CastFilterPointer;

        typedef itk::ShiftScaleImageFilter<ImageType, ImageType> RescaleFilter;
        typedef typename RescaleFilter::Pointer RescaleFilterPointer;

        CastFilterPointer castFilter = CastFilter::New();
        castFilter->SetInput(image);
        castFilter->Update();

        ValueType minVal = (ValueType)itk::NumericTraits<ImagePixelType>::min();
        ValueType maxVal = (ValueType)itk::NumericTraits<ImagePixelType>::max();
        ValueType span = (maxVal - minVal);
	// Something wrong HERE!
        RescaleFilterPointer rescaleFilter = RescaleFilter::New();
        rescaleFilter->SetInput(castFilter->GetOutput());
        rescaleFilter->SetShift(0);
        rescaleFilter->SetScale(itk::NumericTraits<ValueType>::One / span);
        rescaleFilter->Update();

        return rescaleFilter->GetOutput();
/*        ValueType minVal = (ValueType)itk::NumericTraits<ImagePixelType>::min();
        ValueType maxVal = (ValueType)itk::NumericTraits<ImagePixelType>::max();
        ValueType span = (maxVal - minVal);
	// Something wrong HERE!
        RescaleFilterPointer rescaleFilter = RescaleFilter::New();
        rescaleFilter->SetInput(castFilter->GetOutput());
        rescaleFilter->SetShift(-minVal);
        rescaleFilter->SetScale(itk::NumericTraits<ValueType>::One / span);
        rescaleFilter->Update();

        return rescaleFilter->GetOutput();
*/
    }

    //
    // Noise generation routines.
    //

    static ImagePointer AdditiveNoise(ImagePointer image, double stdDev = 1.0, double mean = 0.0, unsigned int seed = 0, bool clampOutputFlag = true)
    {
        typedef itk::AdditiveGaussianNoiseImageFilter<ImageType, ImageType> FilterType;
        typedef typename FilterType::Pointer FilterPointer;

        if (stdDev == 0.0)
            return image;

        FilterPointer ngen = FilterType::New();

        ngen->SetInput(image);
        ngen->SetMean(mean);
        ngen->SetStandardDeviation(stdDev);
        ngen->SetSeed(seed);
        ngen->SetNumberOfThreads(1);

        ngen->Update();

        ImagePointer noisyImage = ngen->GetOutput();

        if (clampOutputFlag)
        {
            typedef itk::ClampImageFilter<ImageType, ImageType> ClampFilterType;
            typedef typename ClampFilterType::Pointer ClampFilterPointer;

            ClampFilterPointer clampFilter = ClampFilterType::New();

            clampFilter->SetInput(noisyImage);

            clampFilter->SetBounds(0.0, 1.0);

            clampFilter->Update();

            return clampFilter->GetOutput();
        }
        else
        {
            return noisyImage;
        }
    }

    static typename itk::Image<bool, Dim>::Pointer ThresholdImage(
        ImagePointer image,
        ValueType threshold)
    {
        typedef itk::BinaryThresholdImageFilter<ImageType, itk::Image<bool, Dim>> ThresholderType;

        typename ThresholderType::Pointer thresholder = ThresholderType::New();

        thresholder->SetInput(image);
        thresholder->SetInsideValue(true);
        thresholder->SetOutsideValue(false);

        thresholder->SetLowerThreshold(threshold);

        thresholder->Update(); // Finalize thresholding

        return thresholder->GetOutput();
    }

    static LabelImagePointer ThresholdToLabelImage(
        ImagePointer image,
        ValueType threshold)
    {
        typedef itk::BinaryThresholdImageFilter<ImageType, LabelImageType> ThresholderType;

        typename ThresholderType::Pointer thresholder = ThresholderType::New();

        thresholder->SetInput(image);
        thresholder->SetInsideValue(1U);
        thresholder->SetOutsideValue(0U);

        thresholder->SetLowerThreshold(threshold);

        thresholder->Update(); // Finalize thresholding

        return thresholder->GetOutput();
    }

    static unsigned int GetPixelCount(
        ImagePointer image)
    {
        typename ImageType::RegionType region = image->GetLargestPossibleRegion();
        
        unsigned int pixelCount = 1;
        for(unsigned int i = 0; i < Dim; ++i) {
            pixelCount = pixelCount * region.GetSize()[i];
        }

        return pixelCount;        
    }

    struct MinMaxSpan
    {
        ValueType minimum;
        ValueType maximum;   
        ValueType span;     
    };

    static MinMaxSpan IntensityMinMax(ImagePointer image, double p)
    {
        assert(p < 0.5);

        MinMaxSpan result;

        result.minimum = itk::NumericTraits<ValueType>::max();
        result.maximum = itk::NumericTraits<ValueType>::min();
        result.span = itk::NumericTraits<ValueType>::max();

        std::vector<ValueType> values;

        itk::ImageRegionConstIterator<ImageType> reader(image, image->GetLargestPossibleRegion());
        reader.GoToBegin();
       
        values.reserve(GetPixelCount(image));

        while (!reader.IsAtEnd())
        {
            values.push_back(reader.Get());

            ++reader;
        }

        std::sort(values.begin(), values.end());

        if (values.size() == 0)
        {
            result.minimum = itk::NumericTraits<ValueType>::ZeroValue();
            result.maximum = itk::NumericTraits<ValueType>::OneValue();
            result.span = result.maximum;
        }
        else
        {
            size_t lowPos = (size_t)(p * values.size());
            size_t hiPos = (size_t)((1.0 - p) * values.size());

            if (lowPos >= values.size())
                lowPos = values.size() - 1;
            if (hiPos >= values.size())
                hiPos = values.size() - 1;

            result.minimum = values[lowPos];
            result.maximum = values[hiPos];
            result.span = result.maximum - result.minimum;
        }

        assert(result.maximum >= result.minimum);

        if (result.span <= 0)
        {
            result.span = 1.0;
        }

        return result;
    }

    static MinMaxSpan IntensityMinMax(ImagePointer image, double p, BinaryImagePointer mask)
    {
        assert(p < 0.5);

        MinMaxSpan result;

        result.minimum = itk::NumericTraits<ValueType>::max();
        result.maximum = itk::NumericTraits<ValueType>::min();
        result.span = itk::NumericTraits<ValueType>::max();

        std::vector<ValueType> values;

        itk::ImageRegionConstIterator<ImageType> reader(image, image->GetLargestPossibleRegion());
        reader.GoToBegin();

        itk::ImageRegionConstIterator<BinaryImageType> maskReader(mask, mask->GetLargestPossibleRegion());
        maskReader.GoToBegin();
       
        values.reserve(GetPixelCount(image));

        while (!reader.IsAtEnd())
        {
            if(maskReader.Get())
            {
                values.push_back(reader.Get());
            }

            ++reader;
            ++maskReader;
        }

        std::sort(values.begin(), values.end());

        if (values.size() == 0)
        {
            result.minimum = itk::NumericTraits<ValueType>::ZeroValue();
            result.maximum = itk::NumericTraits<ValueType>::OneValue();
            result.span = result.maximum;
        }
        else
        {
            size_t lowPos = (size_t)(p * values.size());
            size_t hiPos = (size_t)((1.0 - p) * values.size());

            if (lowPos >= values.size())
                lowPos = values.size() - 1;
            if (hiPos >= values.size())
                hiPos = values.size() - 1;

            result.minimum = values[lowPos];
            result.maximum = values[hiPos];
            result.span = result.maximum - result.minimum;
        }

        assert(result.maximum >= result.minimum);

        if (result.span <= 0)
        {
            result.span = 1.0;
        }

        return result;   
    }

    // Assumes that the image is normalized before-hand
    static ImagePointer HistogramEqualization(
        ImagePointer image,
        BinaryImagePointer mask,
        int bins
    )
    {
        if (bins <= 0)
        {
            return image;
        }

        ImagePointer resultImage = CloneImage(image);

        itk::ImageRegionIterator<ImageType> reader(image, image->GetLargestPossibleRegion());
        reader.GoToBegin();

        std::vector<unsigned int> hist;

        for(unsigned int i = 0; i < bins + 1; ++i)
        {
            hist.push_back(0);
        }

        unsigned int totalCount = 0;
        if (mask)
        {
            itk::ImageRegionIterator<BinaryImageType> maskReader(mask, mask->GetLargestPossibleRegion());
            
            maskReader.GoToBegin();

            while (!reader.IsAtEnd())
            {
                assert(!maskReader.IsAtEnd());

                if (maskReader.Get())
                {
                    unsigned int index = (unsigned int)(reader.Get() * bins + 0.5);
                    ++hist[index];
                    ++totalCount;
                }
                ++reader;
                ++maskReader;
            }
        } else {
            while (!reader.IsAtEnd())
            {
                unsigned int index = (unsigned int)(reader.Get() * bins + 0.5);
                ++hist[index];
                ++totalCount;
                ++reader;
            }
        }

        unsigned int hist_min_index = hist.size();
        unsigned int counter = 0;
        for (unsigned int i = 0; i < hist.size(); ++i)
        {
            counter += hist[i];
            hist[i] = counter;

            if (hist_min_index == hist.size() && counter > 0)
            {
                hist_min_index = i;
            }
        }
        unsigned int hist_min = hist[hist_min_index];
        double denom = (double)(totalCount-hist_min);
        if (totalCount == hist_min)
        {
            denom = 1.0;
        }

        itk::ImageRegionIterator<ImageType> writer(resultImage, resultImage->GetLargestPossibleRegion());
        if (mask)
        {
            itk::ImageRegionIterator<BinaryImageType> maskReader(mask, mask->GetLargestPossibleRegion());
            
            maskReader.GoToBegin();
            
            while (!writer.IsAtEnd())
            {
                assert(!maskReader.IsAtEnd());

                if (maskReader.Get())
                {
                    unsigned int index = (unsigned int)(writer.Get() * bins + 0.5);
                    double f = (hist[index]-hist_min) / denom;
                    writer.Set(f);
                }
                ++writer;
                ++maskReader;
            }
        } else {
            while (!writer.IsAtEnd())
            {
                unsigned int index = (unsigned int)(writer.Get() * bins + 0.5);
                double f = (hist[index]-hist_min) / denom;
                writer.Set(f);
                ++writer;
            }
        }

        return resultImage;
    }

    static ImagePointer NormalizeImage(
        ImagePointer image,
        MinMaxSpan mms)
    {
        image = CloneImage(image);

        itk::ImageRegionIterator<ImageType> writer(image, image->GetLargestPossibleRegion());
        writer.GoToBegin();
        
        while (!writer.IsAtEnd())
        {
            ValueType val = writer.Get();

            if (val <= mms.minimum)
                val = itk::NumericTraits<ValueType>::ZeroValue();
            else if (val >= mms.maximum)
                val = itk::NumericTraits<ValueType>::OneValue();
            else
                val = static_cast<ValueType>((val - mms.minimum) / (double)(mms.span));

            writer.Set(val);

            ++writer;
        }

        return image;
    }

/*    static ImagePointer PercentileRescaleImage(ImagePointer image, double p)
    {
        assert(p < 0.5);
        
        ImagePointer result = ImageType::New();
        result->SetDirection(image->GetDirection());
        result->SetSpacing(image->GetSpacing());
        result->SetRegions(image->GetLargestPossibleRegion());

        result->Allocate();

        std::vector<ValueType> values;

        itk::ImageRegionConstIterator<ImageType> reader(image, image->GetLargestPossibleRegion());
        reader.GoToBegin();

        itk::ImageRegionIterator<ImageType> writer(result, result->GetLargestPossibleRegion());
        writer.GoToBegin();

        values.reserve(GetPixelCount(image));

        while (!reader.IsAtEnd())
        {
            values.push_back(reader.Get());

            ++reader;
        }

        std::sort(values.begin(), values.end());

        ValueType low;
        ValueType hi;
        ValueType span;

        if (values.size() == 0)
        {
            low = 0;
            hi = 1;
        }
        else
        {
            size_t lowPos = (size_t)(p * values.size());
            size_t hiPos = (size_t)((1.0 - p) * values.size());

            if (lowPos >= values.size())
                lowPos = values.size() - 1;
            if (hiPos >= values.size())
                hiPos = values.size() - 1;

            low = values[lowPos];
            hi = values[hiPos];
        }

        assert(hi >= low);

        span = hi - low;
        if (span <= 0)
        {
            span = 1.0;
            low = 0.0;
            hi = 0.0;
        }

        reader.GoToBegin();

        while (!reader.IsAtEnd())
        {
            assert(!writer.IsAtEnd());

            ValueType val = reader.Get();

            if (val <= low)
                val = (ValueType)0;
            else if (val >= hi)
                val = (ValueType)1;
            else
                val = (ValueType)((val - low) / (double)(span));

            writer.Set(val);

            ++reader;
            ++writer;
        }

        return result;
    }

    static ImagePointer PercentileRescaleImage(
        ImagePointer image,
        double p,
        typename itk::Image<bool, Dim>::Pointer mask)
    {
        assert(p < 0.5);

        ImagePointer result = ImageType::New();
        result->SetDirection(image->GetDirection());
        result->SetSpacing(image->GetSpacing());
        result->SetRegions(image->GetLargestPossibleRegion());

        result->Allocate();

        std::vector<ValueType> values;

        itk::ImageRegionConstIterator<ImageType> reader(image, image->GetLargestPossibleRegion());
        reader.GoToBegin();
        itk::ImageRegionConstIterator<itk::Image<bool, Dim>> maskReader(mask, mask->GetLargestPossibleRegion());
        maskReader.GoToBegin();

        itk::ImageRegionIterator<ImageType> writer(result, result->GetLargestPossibleRegion());
        writer.GoToBegin();

        values.reserve(GetPixelCount(image));

        while (!reader.IsAtEnd())
        {
            assert(maskReader.IsAtEnd() == false);

            if (maskReader.Get())
                values.push_back(reader.Get());

            ++maskReader;
            ++reader;
        }

        std::sort(values.begin(), values.end());

        ValueType low;
        ValueType hi;
        ValueType span;

        if (values.size() == 0)
        {
            low = 0;
            hi = 1;
        }
        else
        {
            size_t lowPos = (size_t)(p * values.size());
            size_t hiPos = (size_t)((1.0 - p) * values.size());

            if (lowPos >= values.size())
                lowPos = values.size() - 1;
            if (hiPos >= values.size())
                hiPos = values.size() - 1;

            low = values[lowPos];
            hi = values[hiPos];
        }

        span = hi - low;

        assert(hi >= low);
        if (span <= 0)
        {
            span = 1.0;
            low = 0.0;
            hi = 0.0;
        }

        reader.GoToBegin();
        maskReader.GoToBegin();

        while (!reader.IsAtEnd())
        {
            assert(!writer.IsAtEnd());
            assert(!maskReader.IsAtEnd());

            ValueType val = reader.Get();
            bool maskVal = maskReader.Get();

            if (maskVal)
            {
                if (val <= low)
                    val = (ValueType)0;
                else if (val >= hi)
                    val = (ValueType)1;
                else
                    val = (ValueType)((val - low) / (double)(span));
            } else {
                val = 0;
            }

            writer.Set(val);

            ++maskReader;
            ++reader;
            ++writer;
        }

        return result;
    }*/

    static ImagePointer MultiplyImageByConstant(
        ImagePointer image,
        ValueType factor)
    {
        typedef itk::MultiplyImageFilter< ImageType, ImageType, ImageType > FilterType;
        typename FilterType::Pointer filter = FilterType::New();
        filter->SetInput( image );
        filter->SetConstant( factor );      
        filter->Update();  

        return filter->GetOutput();
    }

    static LabelImagePointer MultiplyLabelImageByConstant(
        LabelImagePointer image,
        ValueType factor)
    {
        ImagePointer tmpImage = itk::ConvertImageFromIntegerFormat<unsigned short, Dim>(image);

        ImagePointer multImage = MultiplyImageByConstant(tmpImage, factor);

        return itk::ConvertImageToIntegerFormat<unsigned short, Dim>(multImage);
    }

    static ImagePointer DifferenceImage(
        ImagePointer image1,
        ImagePointer image2)
    {
        typedef itk::AbsoluteValueDifferenceImageFilter<ImageType, ImageType, ImageType> AbsDiffFilterType;

        typename AbsDiffFilterType::Pointer diffFilter = AbsDiffFilterType::New();
        diffFilter->SetInput1(image1);
        diffFilter->SetInput2(image2);

        diffFilter->Update();

        return diffFilter->GetOutput();
    }

    struct ImageStatisticsData {
        ValueType min;
        ValueType max;
        ValueType mean;
        ValueType sigma;
    };

    static ImageStatisticsData ImageStatistics(ImagePointer image) {
        ImageStatisticsData result;

        typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;
        typename StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New();
        statisticsImageFilter->SetInput(image);
        statisticsImageFilter->Update();

        result.min = statisticsImageFilter->GetMinimum();
        result.max = statisticsImageFilter->GetMaximum();
        result.mean = statisticsImageFilter->GetMean();
        result.sigma = statisticsImageFilter->GetSigma();

        return result;
    }

    //
    //
    // Point set routines.
    //
    //

    //
    // Routines for computing image edge vertices (corners).
    //

  private:
    static void ComputeImageEdgeVerticesRecursive(
        PointSetType &points,
        PointType &p,
        IndexType &index,
        SizeType &size,
        const SpacingType &spacing,
        unsigned int d)
    {
        if (d >= Dim)
        {
            points.push_back(p);
        }
        else
        {
            p[d] = static_cast<PointValueType>(index[d]);

            ComputeImageEdgeVerticesRecursive(points, p, index, size, spacing, d + 1);

            // Add (size - 1) to the point's position to locate it
            // on the last pixel.

            if (size[d] > static_cast<PointValueType>(0))
                p[d] += static_cast<PointValueType>((size[d] - 1) * spacing[d]);

            ComputeImageEdgeVerticesRecursive(points, p, index, size, spacing, d + 1);
        }
    }

    static void ComputeImageEdgeVerticesRecursive(
        PointSetType &points,
        PointType &p,
        IndexType &index,
        SizeType &size,
        unsigned int d)
    {
        if (d >= Dim)
        {
            points.push_back(p);
        }
        else
        {
            p[d] = static_cast<PointValueType>(index[d]);

            ComputeImageEdgeVerticesRecursive(points, p, index, size, d + 1);

            // Add (size - 1) to the point's position to locate it
            // on the last pixel.

            if (size[d] > static_cast<PointValueType>(0))
                p[d] += static_cast<PointValueType>((size[d] - 1));

            ComputeImageEdgeVerticesRecursive(points, p, index, size, d + 1);
        }
    }

  public:
    static void ComputeImageEdgeVertices(
        ImagePointer image,
        PointSetType &points,
        bool considerSpacing)
    {
        RegionType region = image->GetLargestPossibleRegion();

        IndexType index = region.GetIndex();
        SizeType size = region.GetSize();

        PointType p;

        if (considerSpacing)
            ComputeImageEdgeVerticesRecursive(points, p, index, size, image->GetSpacing(), 0U);
        else
            ComputeImageEdgeVerticesRecursive(points, p, index, size, 0U);
    }

    static PointValueType ComputeImageEdgeVertexMeanDistance(
        PointSetType &points1,
        PointSetType &points2)
    {
        assert(points1.size() == points2.size());

        PointValueType mean = 0;

        for (size_t i = 0; i < points1.size(); ++i)
        {
            PointType p1 = points1[i];
            PointType p2 = points2[i];

            mean += p1.EuclideanDistanceTo(p2);
        }

        return mean / static_cast<PointValueType>(points1.size());
    }

    static void PrintPointSet(const PointSetType &ps)
    {
        std::cout << "{";
        for (size_t i = 0; i < ps.size(); ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << ps[i];
        }
        std::cout << "}" << std::endl;
    }

    static void PointSetBoundingBox(
        PointSetType &points,
        PointType &minOut,
        PointType &maxOut)
    {
        if (points.size() == 0)
        {
            minOut.Fill(0);
            maxOut.Fill(0);
        }
        else
        {
            minOut = points[0];
            maxOut = points[0];

            for (size_t i = 1; i < points.size(); ++i)
            {
                PointType p = points[i];

                for (unsigned int j = 0; j < Dim; ++j)
                {
                    if (minOut[j] > p[j])
                        minOut[j] = p[j];
                    if (maxOut[j] < p[j])
                        maxOut[j] = p[j];
                }
            }
        }
    }

    static double PointSetMeanEuclideanDistance(
        PointSetType &a,
        PointSetType &b)
    {
        double acc = 0.0;

        assert(a.size() == b.size());

        double sizeRec = a.size();
        if (a.size() > 0)
        {
            sizeRec = 1.0 / sizeRec;
        }

        for (size_t i = 0; i < a.size(); ++i)
        {
            acc += a[i].EuclideanDistanceTo(b[i]);
        }

        return acc * sizeRec;
    }

    /**
     * O(n^2) asymmetric SMD computation from the reference to target point-set.
     **/
    static double PointSetASMD(
        PointSetType &ref,
        PointSetType &target)
    {
        double acc = 0.0;

        double sizeRec = ref.size();
        if (ref.size() > 0)
        {
            sizeRec = 1.0 / sizeRec;
        }
        if (target.size() == 0)
        {
            return 0.0;
        }
//#define POINTSET_ASMD_DEBUG_PRINTOUTS
#ifdef POINTSET_ASMD_DEBUG_PRINTOUTS
        for (size_t i = 0; i < ref.size(); ++i)
        {
            std::cout << "From " << ref[i] << "..." << std::endl;

            double cur = ref[i].EuclideanDistanceTo(target[0]);
            std::cout << "To " << target[0] << (cur) << std::endl;

            for (size_t j = 1; j < target.size(); ++j)
            {
                double d = ref[i].EuclideanDistanceTo(target[j]);
                if (d < cur)
                {
                    cur = d;
                    std::cout << "To " << target[j] << (d) << " [NEW BEST]" << std::endl;
                }
                else
                {
                    std::cout << "To " << target[j] << (d) << std::endl;
                }
            }
            acc += cur;
        }
#else
        for (size_t i = 0; i < ref.size(); ++i)
        {
            double cur = ref[i].EuclideanDistanceTo(target[0]);
            for (size_t j = 1; j < target.size(); ++j)
            {
                double d = ref[i].EuclideanDistanceTo(target[j]);
                if (d < cur)
                {
                    cur = d;
                }
            }
            acc += cur;
        }
#endif

        return acc * sizeRec;
    }

    //
    // Transformation routines.
    //

    static PointSetType TransformPointSet(
        typename itk::Transform<PointValueType, Dim, Dim>::Pointer transform,
        PointSetType &points)
    {
        PointSetType result;

        for (size_t i = 0; i < points.size(); ++i)
        {
            result.push_back(transform->TransformPoint(points[i]));
        }

        return result;
    }

    //static Point

    static PointSetType TranslatePointSet(
        VectorType offset,
        PointSetType &points)
    {
        PointSetType result;

        for (size_t i = 0; i < points.size(); ++i)
        {
            result.push_back(points[i] + offset);
        }

        return result;
    }

    static PointSetType &TranslatePointSetInPlace(
        CovariantVectorType offset,
        PointSetType &points)
    {
        for (size_t i = 0; i < points.size(); ++i)
        {
            points[i] += +offset;
        }

        return points;
    }

    static BaseTransformPointer MakeTranslationTransformation(
        CovariantVectorType translation)
    {
        typedef itk::TranslationTransform<PointValueType, Dim> TranslationTransformType;
        typedef typename TranslationTransformType::Pointer TranslationTransformPointer;
        typedef typename TranslationTransformType::ParametersType ParametersType;

        TranslationTransformPointer ttp = TranslationTransformType::New();
        ParametersType param(ttp->GetNumberOfParameters());
        for (unsigned int i = 0; i < Dim; ++i)
        {
            param[i] = translation[i];
        }

        ttp->SetParameters(param);

        return itk::CastSmartPointer<TranslationTransformType, BaseTransformType>(ttp);
    }

    static BaseTransformPointer CompositeTransformWithTranslation(
        typename itk::Transform<PointValueType, Dim, Dim>::Pointer transform,
        CovariantVectorType translation)
    {
        typedef itk::CompositeTransform<PointValueType, Dim> CompositeTransformType;
        typedef typename CompositeTransformType::Pointer CompositeTransformPointer;

        bool anyNonZero = false;
        for (unsigned int i = 0; i < Dim; ++i)
        {
            if (!itk::Math::AlmostEquals(translation[i], -0))
            {
                anyNonZero = true;
                break;
            }
        }

        if (!anyNonZero)
            return transform;

        CompositeTransformPointer compositeTransform = CompositeTransformType::New();
        compositeTransform->AddTransform(transform);
        compositeTransform->AddTransform(MakeTranslationTransformation(translation));

        return itk::CastSmartPointer<CompositeTransformType, BaseTransformType>(compositeTransform);
    }

    static ImagePointer TransformImage(
        ImagePointer inputImage,
        typename itk::Transform<double, Dim, Dim>::Pointer transform,
        PointType originPoint,
        SizeType outputSize,
        ImwarpInterpolationMode interpMode,
        ValueType bgValue = 0)
    {

        typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleFilterType;
        typedef typename ResampleFilterType::Pointer ResampleFilterPointer;

        ResampleFilterPointer resampler = ResampleFilterType::New();
        resampler->SetInput(inputImage);

        transform = CompositeTransformWithTranslation(transform, MakeCovariantVector(originPoint));

        itk::Transform<double, Dim, Dim> *transformRawPtr = transform.GetPointer();

        resampler->SetTransform(transformRawPtr);

        resampler->SetSize(outputSize);
        resampler->SetOutputOrigin(inputImage->GetOrigin());
        resampler->SetOutputSpacing(inputImage->GetSpacing());
        resampler->SetOutputDirection(inputImage->GetDirection());
        resampler->SetDefaultPixelValue(bgValue);

        resampler->SetInterpolator(MakeInterpolator(interpMode));

        resampler->Update();

        return resampler->GetOutput();
    }

    static ImagePointer TransformImage(
        ImagePointer inputImage,
        ImagePointer refImage,
        typename itk::Transform<double, Dim, Dim>::Pointer transform,
        ImwarpInterpolationMode interpMode,
        ValueType bgValue = 0)
    {
        typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleFilterType;
        typedef typename ResampleFilterType::Pointer ResampleFilterPointer;

        ResampleFilterPointer resampler = ResampleFilterType::New();
        resampler->SetInput(inputImage);

        itk::Transform<double, Dim, Dim> *transformRawPtr = transform.GetPointer();

        resampler->SetTransform(transformRawPtr);

        resampler->SetReferenceImage(refImage);
        resampler->SetUseReferenceImage(true);

        resampler->SetDefaultPixelValue(bgValue);

        resampler->SetInterpolator(MakeInterpolator(interpMode));

        resampler->Update();

        return resampler->GetOutput();
    }

    static LabelImagePointer TransformLabelImage(
        LabelImagePointer inputImage,
        typename itk::Transform<double, Dim, Dim>::Pointer transform,
        PointType originPoint,
        SizeType outputSize,
        unsigned short bgValue = 0)
    {

        typedef itk::ResampleImageFilter<LabelImageType, LabelImageType> ResampleFilterType;
        typedef typename ResampleFilterType::Pointer ResampleFilterPointer;

        ResampleFilterPointer resampler = ResampleFilterType::New();
        resampler->SetInput(inputImage);

        transform = CompositeTransformWithTranslation(transform, MakeCovariantVector(originPoint));

        itk::Transform<double, Dim, Dim> *transformRawPtr = transform.GetPointer();

        resampler->SetTransform(transformRawPtr);

        resampler->SetSize(outputSize);
        resampler->SetOutputOrigin(inputImage->GetOrigin());
        resampler->SetOutputSpacing(inputImage->GetSpacing());
        resampler->SetOutputDirection(inputImage->GetDirection());
        resampler->SetDefaultPixelValue(bgValue);

        resampler->SetInterpolator(MakeLabelInterpolator());

        resampler->Update();

        return resampler->GetOutput();
    }

    static LabelImagePointer TransformLabelImage(
        LabelImagePointer inputImage,
        LabelImagePointer refImage,
        typename itk::Transform<double, Dim, Dim>::Pointer transform,
        unsigned short bgValue = 0)
    {
        typedef itk::ResampleImageFilter<LabelImageType, LabelImageType> ResampleFilterType;
        typedef typename ResampleFilterType::Pointer ResampleFilterPointer;

        ResampleFilterPointer resampler = ResampleFilterType::New();
        resampler->SetInput(inputImage);

        itk::Transform<double, Dim, Dim> *transformRawPtr = transform.GetPointer();

        resampler->SetTransform(transformRawPtr);

        resampler->SetReferenceImage(refImage);
        resampler->SetUseReferenceImage(true);

        resampler->SetDefaultPixelValue(bgValue);

        resampler->SetInterpolator(MakeLabelInterpolator());

        resampler->Update();

        return resampler->GetOutput();
    }

    static ImagePointer TransformImageAutoCrop(
        ImagePointer image,
        BaseTransformPointer transform,
        ImwarpInterpolationMode interpMode,
        PointSetType *outPointSet,
        ValueType bgValue = 0)
    {

        PointSetType inputPointSet;
        ComputeImageEdgeVertices(image, inputPointSet, true);

        PointSetType outputPointSet = TransformPointSet(transform->GetInverseTransform(), inputPointSet);

        PointType minPoint;
        PointType maxPoint;

        PointSetBoundingBox(outputPointSet, minPoint, maxPoint);

        VectorType sizeVec = maxPoint - minPoint;

        SizeType size = MakeSize(sizeVec);

        if (outPointSet)
        {
            *outPointSet = TranslatePointSet(-MakeVector(minPoint), outputPointSet);
        }

        return TransformImage(image, transform, minPoint, size, interpMode, bgValue);
    }

    static ImagePointer TranslateImage(
        ImagePointer image,
        CovariantVectorType t,
        ValueType bgValue = 0)
    {
        t = RoundCovariantVector(t);

        SizeType origSize = GetImageSize(image);
        SizeType extSize = MakeSize(AbsCovariantVector(t));
        SizeType newSize = origSize + extSize;

        PointType origin;
        origin.Fill(0);

        for (unsigned int i = 0; i < Dim; ++i)
        {
            if (t[i] < 0.0)
                t[i] = 0.0;
        }

        return TransformImage(image,
                              MakeTranslationTransformation(-t),
                              origin,
                              newSize,
                              kImwarpInterpNearest,
                              bgValue);
    }

    static void RoundPointSet(
        PointSetType &points)
    {
        for (size_t i = 0; i < points.size(); ++i)
        {
            for (unsigned int d = 0; d < Dim; ++d)
            {
                points[i][d] = (PointValueType)((long long)(points[i][d] + 0.5));
            }
        }
    }

    struct OverlapScore
    {
        int label;
        double jaccard;
        double dice;
    };

    static std::vector<OverlapScore> ComputeLabelOverlap(
        LabelImagePointer labelImage1,
        LabelImagePointer labelImage2)
    {
        typedef itk::LabelOverlapMeasuresImageFilter<LabelImageType> FilterType;
        typedef typename FilterType::Pointer FilterPointer;

        FilterPointer filter = FilterType::New();

        filter->SetSourceImage(labelImage1);
        filter->SetTargetImage(labelImage2);

        filter->Update();

        typedef typename FilterType::MapType MapType;

        std::vector<OverlapScore> result;

        MapType labelMap = filter->GetLabelSetMeasures();
        typename MapType::const_iterator it;
        for (it = labelMap.begin(); it != labelMap.end(); ++it)
        {
            if ((*it).first == 0)
            { // Ignore background
                continue;
            }

            OverlapScore score;
            score.label = it->first;
            score.jaccard = filter->GetUnionOverlap(score.label);
            score.dice = filter->GetMeanOverlap(score.label);

            result.push_back(score);
        }

        OverlapScore allScore;
        allScore.label = 0;
        allScore.jaccard = filter->GetUnionOverlap();
        allScore.dice = filter->GetMeanOverlap();
        result.push_back(allScore);
	
        return result;
    }

    static std::vector<OverlapScore> ComputeMaskOverlap(
        ImagePointer refImage,
        ImagePointer floatingImage,
        ValueType threshold)
    {
        typedef itk::Image<bool, Dim> MaskImageType;
        typedef typename MaskImageType::Pointer MaskImagePointer;

        MaskImagePointer refImageBool = ThresholdImage(refImage, threshold);
        MaskImagePointer floatingImageBool = ThresholdImage(floatingImage, threshold);

        LabelImagePointer refImageLabel = itk::CastImage<MaskImageType, LabelImageType>(refImageBool);
        LabelImagePointer floatingImageLabel = itk::CastImage<MaskImageType, LabelImageType>(floatingImageBool);

        std::vector<OverlapScore> result = ComputeLabelOverlap(refImageLabel, floatingImageLabel);
        if(result.size() > 0) {
            result[0].label = 65534U;
        }

        return result;
    }

    static std::vector<OverlapScore> ComputeTotalOverlap(
        LabelImagePointer labelImage1,
        LabelImagePointer labelImage2)
    {
        labelImage1 = CloneLabelImage(labelImage1);
        labelImage2 = CloneLabelImage(labelImage2);

        SaturateLabelImage(labelImage1, 0U, 1U);
        SaturateLabelImage(labelImage2, 0U, 1U);

        std::vector<OverlapScore> result = ComputeLabelOverlap(labelImage1, labelImage2);

        if(result.size() > 0) {
            result[0].label = 65535U;
        }

        return result;
    }

    static void SaveLabelOverlapCSV(
        const char *path,
        std::vector<OverlapScore> &scores,
        bool writeHeaders = false)
    {
        FILE *f = fopen(path, "wb");

        if (f)
        {
            if (writeHeaders)
            {
                fprintf(f, "Label,Jaccard,Dice\n");
            }
            for (size_t i = 0; i < scores.size(); ++i)
            {
                fprintf(f, "%d,%.7f,%.7f\n", scores[i].label, scores[i].jaccard, scores[i].dice);
            }

            fclose(f);
        }
    }

    static ImagePointer SmoothImage(
        ImagePointer image,
        double sigma)
    {
        if(sigma <= 0.0)
            return image;

        typedef itk::DiscreteGaussianImageFilter<
            ImageType, ImageType>
            filterType;

        // Create and setup a Gaussian filter
        typename filterType::Pointer gaussianFilter = filterType::New();
        gaussianFilter->SetInput(image);
	gaussianFilter->SetUseImageSpacingOn();
        gaussianFilter->SetMaximumKernelWidth(128);
        gaussianFilter->SetVariance(sigma * sigma);
        gaussianFilter->Update();

        return gaussianFilter->GetOutput();
    }

    static ImagePointer MedianFilterImage(
        ImagePointer image,
        double radius)
    {
        if(radius <= 0.0)
            return image;

        typedef itk::MedianImageFilter<
            ImageType, ImageType>
            FilterType;

        // Create and setup a Gaussian filter
        typename FilterType::Pointer filter = FilterType::New();

        auto spacing = image->GetSpacing();
        typename FilterType::InputSizeType rad;
        for (unsigned int i = 0; i < Dim; ++i) {
            rad[i] = static_cast<typename ImageType::SizeValueType>(spacing[i] * radius + 0.5);
            if(rad[i] == 0) {
                return image;
            }
        }

        filter->SetInput(image);
        filter->SetRadius(rad);
        filter->Update();

        return filter->GetOutput();
    }

    static void SaturateImage(
        ImagePointer image,
        ValueType lower,
        ValueType upper)
    {
        itk::ImageRegionIterator<ImageType> writeIter(image, image->GetRequestedRegion());

        writeIter.GoToBegin();

        while (!writeIter.IsAtEnd())
        {
            if (writeIter.Get() < lower)
                writeIter.Set(lower);
            if (writeIter.Get() > upper)
                writeIter.Set(upper);

            ++writeIter;
        }
    }

    static void SaturateLabelImage(
        LabelImagePointer image,
        unsigned short lower,
        unsigned short upper)
    {
        itk::ImageRegionIterator<LabelImageType> writeIter(image, image->GetRequestedRegion());

        writeIter.GoToBegin();

        while (!writeIter.IsAtEnd())
        {
            if (writeIter.Get() < lower)
                writeIter.Set(lower);
            if (writeIter.Get() > upper)
                writeIter.Set(upper);

            ++writeIter;
        }
    }

    static ImagePointer SubtractImage(
        ImagePointer image1,
        ImagePointer image2)
    {
        typedef itk::SubtractImageFilter<ImageType, ImageType> FilterType;
        typedef typename FilterType::Pointer FilterPointer;

        FilterPointer f = FilterType::New();

        f->SetInput1(image1);
        f->SetInput2(image2);

        f->Update();

        return f->GetOutput();
    }

    static ImagePointer BandpassFilterImage(
        ImagePointer image,
        double lo_sigma,
        double hi_sigma)
    {
        ImagePointer i1 = SmoothImage(image, hi_sigma);
        ImagePointer i2;
        if (lo_sigma > 0.0)
            i2 = SmoothImage(image, lo_sigma);
        else
            i2 = image;

        ImagePointer i3 = SubtractImage(i2, i1);
        SaturateImage(i3, 0.0, 1.0);

        return i3;
    }

    static ImagePointer MatchHistogram(
        ImagePointer reference,
        ImagePointer source,
        itk::SizeValueType bins,
        itk::SizeValueType matchPoints)
    {
        typedef itk::HistogramMatchingImageFilter<ImageType, ImageType> FilterType;
        typedef typename FilterType::Pointer FilterPointer;

        FilterPointer f = FilterType::New();

        f->SetReferenceImage(reference);
        f->SetSourceImage(source);
        f->SetNumberOfHistogramLevels(bins);
        f->SetNumberOfMatchPoints(matchPoints);

        f->Update();

        return f->GetOutput();
    }
// TODO: COMPLETE THIS
    static PointSetType LoadPointSet(
        const char *path,
        bool skipHeaderRow = false)
    {
        PointSetType points;

        itk::CSVFile csv = itk::ReadCSV(path);
        if(csv.success == false) {
            std::cout << "Failed to read file." << std::endl;
        }

        size_t i = skipHeaderRow ? 1U : 0U;
        for(; i < csv.contents.size(); ++i) {
            std::vector<std::string> row = csv.contents[i];

            PointType p;
            if(row.size() == 0U) {
                continue;
            }
            if(row.size() == Dim + 1) { // Assume a row header
                for (unsigned int j = 0; j < Dim; ++j)
                {
                    if(row[j+1].empty()) {
                        p[j] = NAN;
                    } else {
                        p[j] = atof(row[j+1].c_str());
                    }
                }
            } else if(row.size() == Dim) { // No row header
                for (unsigned int j = 0; j < Dim; ++j)
                {
                    if(row[j].empty()) {
                        p[j] = NAN;
                    } else {
                        p[j] = atof(row[j].c_str());
                    }
                }
            } else {
                std::cout << "Row " << i << " contains an illegal number of entries" << std::endl;
                return points;
            }
             points.push_back(p);            
        }

        return points;
    }
/*
        typedef itk::CSVArray2DFileReader<double> ReaderType;
        typedef typename ReaderType::Pointer ReaderPointer;

        typedef itk::CSVArray2DDataObject<double> DataFrameObjectType;
        typedef typename DataFrameObjectType::Pointer DataFrameObjectPointer;

        ReaderPointer reader = ReaderType::New();

        reader->SetFileName(path);

        reader->SetFieldDelimiterCharacter(',');
        reader->HasColumnHeadersOff();
        reader->HasRowHeadersOff();
        reader->UseStringDelimiterCharacterOff();

        std::cout << "Before parsing" << std::endl;

        reader->Parse();

        reader->Update();

        DataFrameObjectPointer dfo = reader->GetOutput();
        std::cout << "File loaded" << std::endl;
        unsigned int i = 0;
        while (true)
        {
            try
            {
                std::vector<double> vec = dfo->GetRow(i);
                if (vec.empty())
                    break;
                PointType p;
                for (unsigned int j = 0; j < Dim; ++j)
                {
                    p[j] = vec.at(j);
                }
                points.push_back(p);
                ++i;
            }
            catch (itk::ExceptionObject &e)
            {
                std::cout << e << std::endl;
                break;
            }
        }
        //ArrayRawPointer arr = reader->GetOutput();
        //ArrayType matrix = arr->GetMatrix();

        //matrix->GetData(row, column);
        // arr->GetData(row, column)

        //unsigned int columns = reeader->GetColumnHeaders().size();
        //unsigned int rows = reeader->GetRowHeaders().size();

        return points;
    }*/

    static void SavePointSet(
        const char* path,
        PointSetType& pointSet)
    {
        FILE *f = fopen(path, "wb");

        if(f) {
            for(size_t i = 0; i < pointSet.size(); ++i) {
                fprintf(f, "%.7f", pointSet[i][0]);
                for(unsigned int j = 1; j < Dim; ++j) {
                    fprintf(f, ",%.7f", pointSet[i][j]);
                }

                if(i + 1 < pointSet.size()) {
                    fprintf(f, "\n");
                }
            }

            fclose(f);
        }
    }

    enum SpacingMode
    {
        kDefaultSpacingMode,
        kRemoveSpacingMode,
        kResampleSpacingMode
    };

    static ImagePointer RemoveSpacing(
        ImagePointer image,
        SpacingMode mode = kDefaultSpacingMode)
    {
        typename ImageType::SpacingType spacing = image->GetSpacing();

        bool all_are_unit = true;
        for (unsigned int i = 0; i < Dim; ++i)
        {
            if (!itk::Math::AlmostEquals<double, double>(spacing[i], 1.0))
            {
                all_are_unit = false;
                break;
            }
        }

        if (all_are_unit)
            return image;

        if (mode == kResampleSpacingMode)
        {
            typedef itk::AffineTransform<double, Dim> AffineTransformType;
            typedef typename AffineTransformType::Pointer AffineTransformPointer;

            AffineTransformPointer transform = AffineTransformType::New();

            typename AffineTransformType::ParametersType parameters(Dim * Dim + Dim);

            parameters.Fill(0);

            transform->SetParameters(parameters);

            transform->SetIdentity();

            VectorType scaleVec;
            for (unsigned int i = 0; i < Dim; ++i)
            {
                scaleVec[i] = 1.0 / spacing[i];
            }
            transform->Scale(scaleVec);

            PointType origin;
            origin.Fill(0.0);
            typename ImageType::SizeType sz = GetImageSize(image);

            for (unsigned int i = 0; i < Dim; ++i)
            {
                sz[i] = (itk::SizeValueType)(sz[i] * (spacing[i]));
            }

            ImagePointer result = IPT::TransformImage(image, itk::CastSmartPointer<itk::AffineTransform<double, Dim>, itk::Transform<double, Dim, Dim>>(transform), origin, sz, kImwarpInterpLinear);

            return RemoveSpacing(result, kRemoveSpacingMode);
        }
        else if (mode == kRemoveSpacingMode)
        {
            for (unsigned int i = 0; i < Dim; ++i)
            {
                spacing[i] = 1;
            }

            typedef itk::ChangeInformationImageFilter<ImageType> FilterType;
            typedef typename FilterType::Pointer FilterPointer;

            FilterPointer filter = FilterType::New();

            filter->SetInput(image);
            filter->SetOutputSpacing(spacing);
            filter->ChangeSpacingOn();

            filter->Update();

            return filter->GetOutput();
        }
        else
        {
            return image;
        }
    }

    static LabelImagePointer RemoveLabelSpacing(
        LabelImagePointer image,
        SpacingMode mode = kDefaultSpacingMode)
    {
        typename LabelImageType::SpacingType spacing = image->GetSpacing();

        bool all_are_unit = true;
        for (unsigned int i = 0; i < Dim; ++i)
        {
            if (!itk::Math::AlmostEquals<double, double>(spacing[i], 1.0))
            {
                all_are_unit = false;
                break;
            }
        }

        if (all_are_unit)
            return image;

        if (mode == kResampleSpacingMode)
        {
            typedef itk::AffineTransform<double, Dim> AffineTransformType;
            typedef typename AffineTransformType::Pointer AffineTransformPointer;

            AffineTransformPointer transform = AffineTransformType::New();

            typename AffineTransformType::ParametersType parameters(Dim * Dim + Dim);

            parameters.Fill(0);

            transform->SetParameters(parameters);

            transform->SetIdentity();

            VectorType scaleVec;
            for (unsigned int i = 0; i < Dim; ++i)
            {
                scaleVec[i] = 1.0 / spacing[i];
            }
            transform->Scale(scaleVec);

            PointType origin;
            origin.Fill(0.0);
            typename LabelImageType::SizeType sz = image->GetLargestPossibleRegion().GetSize(); //GetImageSize(image);

            for (unsigned int i = 0; i < Dim; ++i)
            {
                sz[i] = (itk::SizeValueType)(sz[i] * (spacing[i]));
            }

            LabelImagePointer result = IPT::TransformLabelImage(image, itk::CastSmartPointer<itk::AffineTransform<double, Dim>, itk::Transform<double, Dim, Dim>>(transform), origin, sz);

            return RemoveLabelSpacing(result, kRemoveSpacingMode);
        }
        else if (mode == kRemoveSpacingMode)
        {
            for (unsigned int i = 0; i < Dim; ++i)
            {
                spacing[i] = 1;
            }

            typedef itk::ChangeInformationImageFilter<LabelImageType> FilterType;
            typedef typename FilterType::Pointer FilterPointer;

            FilterPointer filter = FilterType::New();

            filter->SetInput(image);
            filter->SetOutputSpacing(spacing);
            filter->ChangeSpacingOn();

            filter->Update();

            return filter->GetOutput();
        }
        else
        {
            return image;
        }
    }

    static bool StringHasPrefix(const char* str, const char* prefix) {
        while(*prefix) {
            if(*str) {
                if(*str != *prefix) {
                    return false;
                } else {
                    ++str;
                    ++prefix;
                }                
            } else {
                return false;
            }
        }
        return true;
    }

    // TODO: Finish the transform extraction functions
    // BEGIN HERE

    static unsigned int GetTransformCount(
        BaseTransformPointer transform)
    {
        if(!transform)
            return 0U;
        if(StringHasPrefix(transform->GetNameOfClass(), "CompositeTransform")) {
            CompositeTransformPointer tf = static_cast<CompositeTransformType*>(transform.GetPointer());

            return tf->GetNumberOfTransforms();
        } else {
            return 1U;
        }
    }

    static BaseTransformPointer GetNthTransform(
        BaseTransformPointer transform,
        unsigned int index)
    {
        if(StringHasPrefix(transform->GetNameOfClass(), "CompositeTransform")) {
            CompositeTransformPointer tf = static_cast<CompositeTransformType*>(transform.GetPointer());
            return tf->GetNthTransform(index);
        } else if(index == 0) {
            return transform;
        } else {
            return nullptr;
        }
    }

    //static BaseTransformPointer ConvertTransformToAffine(
    //    BaseTransformPointer transform)
    //{

    //}

    // END HERE

    static void SaveTransformFile(
        const char* path,
        BaseTransformPointer transform)
    {
        typedef itk::TransformFileWriterTemplate<double> TransformWriterType;
        TransformWriterType::Pointer writer = TransformWriterType::New();

        writer->SetInput(transform);

        writer->SetFileName(path);

        writer->Update();
    }

    static BaseTransformPointer LoadTransformFile(
        const char* path)
    {
        typedef itk::TransformFileReaderTemplate<double> TransformReaderType;
        TransformReaderType::Pointer reader = TransformReaderType::New();

        reader->SetFileName(path);
        
        reader->Update();

        typedef typename TransformReaderType::TransformListType* const TransformListRawPointer;
        TransformListRawPointer transforms = reader->GetTransformList();
        
        typedef itk::CompositeTransform<double, Dim> CompositeTransformType;
        typedef typename CompositeTransformType::Pointer CompositeTransformPointer;

        auto it = transforms->begin();
        auto itEnd = transforms->end();

        if(it == itEnd) {
            CompositeTransformPointer composite = CompositeTransformType::New();
            return static_cast< BaseTransformType* >(composite.GetPointer() );
        } else {
            auto itPlusOne = it;
            ++itPlusOne;
            if(itPlusOne == itEnd) {
                return static_cast< BaseTransformType* >((*it).GetPointer());
            } else {
                CompositeTransformPointer composite = CompositeTransformType::New();
                for(; it != itEnd; ++it) {
                    composite->AddTransform(static_cast<BaseTransformType*>((*it).GetPointer()));
                }
                return static_cast< BaseTransformType* >(composite.GetPointer());
            }
        }
    }
};

template <typename PixelType, unsigned short Dim>
inline void Save3DImageSlicesU8(
    const char *path,
    const char *name,
    const char *type,
    typename itk::Image<PixelType, Dim>::Pointer image)
{
    assert(Dim == 3U);

    char buf[512];
    typename itk::Image<PixelType, Dim>::SizeType sz = image->GetLargestPossibleRegion().GetSize();
    typedef PixelType ValueType;

    sprintf(buf, "%s%s_xy.%s", path, name, type);

    itk::IPT<ValueType, Dim - 1>::SaveImageU8(buf, itk::ExtractSlice<ValueType, Dim>(image, 2, sz[2] / 2));

    sprintf(buf, "%s%s_xz.%s", path, name, type);

    itk::IPT<ValueType, Dim - 1>::SaveImageU8(buf, itk::ExtractSlice<ValueType, Dim>(image, 1, sz[1] / 2));

    sprintf(buf, "%s%s_yz.%s", path, name, type);

    itk::IPT<ValueType, Dim - 1>::SaveImageU8(buf, itk::ExtractSlice<ValueType, Dim>(image, 0, sz[0] / 2));
}

template <unsigned short Dim>
inline void Save3DLabelImageSlices(
    const char *path,
    const char *name,
    const char *type,
    typename itk::Image<unsigned short, Dim>::Pointer image)
{
    assert(Dim == 3U);

    char buf[512];

    typename itk::Image<unsigned short, Dim>::SizeType sz = image->GetLargestPossibleRegion().GetSize();
    typedef double ValueType;

    sprintf(buf, "%s%s_label_xy.%s", path, name, type);

    itk::IPT<ValueType, Dim - 1>::SaveLabelImage(buf, itk::ExtractSlice<unsigned short, Dim>(image, 2, sz[2] / 2));

    sprintf(buf, "%s%s_label_xz.%s", path, name, type);

    itk::IPT<ValueType, Dim - 1>::SaveLabelImage(buf, itk::ExtractSlice<unsigned short, Dim>(image, 1, sz[1] / 2));

    sprintf(buf, "%s%s_label_yz.%s", path, name, type);

    itk::IPT<ValueType, Dim - 1>::SaveLabelImage(buf, itk::ExtractSlice<unsigned short, Dim>(image, 0, sz[0] / 2));
}

/**
 * Parsing routines for reading strings of numbers of the form '4x2x1' and '2.0x1.0x0.0'
 * for reading sequences of numbers as command-line parameters.
 */

inline std::vector<std::string> SplitOnX(
    const std::string& s)
{
    std::vector<std::string> result;

    size_t cur = 0;
    size_t start = 0;
    size_t len = s.length();
    while(cur < len) {
        char c = s[cur];
        if(c == 'x') {
            result.push_back(s.substr(start, cur-start));
            start = cur + 1;
        }
        ++cur;
    }

    if(start < cur) {
        result.push_back(s.substr(start, cur-start));
    }
    
    return result;
}

inline std::vector<unsigned int> ParseSamplingFactors(
    const std::string& s)
    {
        std::vector<unsigned int> result;

        std::vector<std::string> tokens = SplitOnX(s);

        for(size_t i = 0; i < tokens.size(); ++i) {
            result.push_back( static_cast<unsigned int>( atoi(tokens[i].c_str() ) ) );
        }

        return result;
    }
inline std::vector<double> ParseSmoothingSigmas(
    const std::string& s)
    {
        std::vector<double> result;

        std::vector<std::string> tokens = SplitOnX(s);

        for(size_t i = 0; i < tokens.size(); ++i)
        {
            result.push_back(atof(tokens[i].c_str()));
        }

        return result;
    }
}

inline double VectorMean(
    std::vector<double>& v)
{
    double s = 0.0;

    for(size_t i = 0; i < v.size(); ++i)
    {
        s += v[i];
    }

    if(v.size() > 0)
    {
        return s / v.size();
    } else {
        return s;
    }
}

inline double VectorStdDev(
    std::vector<double>& v,
    double mean)
{
    double s = 0.0;

    for(size_t i = 0; i < v.size(); ++i)
    {
        double x = v[i] - mean;
        s += x * x;
    }

    if(v.size() <= 1)
    {
        return 0.0;
    } else {
        return sqrt(s / (v.size() - 1));
    }
}

#endif
