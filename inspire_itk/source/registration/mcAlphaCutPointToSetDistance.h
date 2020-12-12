
/**
 * Implementation of a Monte Carlo framework (insert reference here) for computing the fuzzy alpha-cut-based point-to-set distance
 * ("Linear time distances between fuzzy sets with applications to pattern matching and classification",
 * by J. Lindblad and N. Sladoje, IEEE Transactions on Image Processing, 2013).
 *
 * Author: Johan Ofverstedt
 */

#include <itkImage.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <cmath>
#include <algorithm>

#include "../common/quantization.h"
#include "../samplers/valueSampler.h"

// Debug flags
//#define DEBUG_NODE_COUNTING_ENABLED

// Linear interpolation. alpha=0 : a, alpha=1 : b
#define LERP(a, b, alpha) ((a) + alpha * ((b)-(a)))

// A namespace collecting auxilliary data-structures and functions
// used by the Monte Carlo distance framework.

namespace MCDSInternal {

template <unsigned int Dim>
struct ValuedCornerPoints;

template <>
struct ValuedCornerPoints<1U> {
  itk::Vector<double, 2U> m_Values;
};

template <>
struct ValuedCornerPoints<2U> {
  itk::Vector<double, 4U> m_Values;
};

template <>
struct ValuedCornerPoints<3U> {
  itk::Vector<double, 8U> m_Values;
};

template <typename T, unsigned int Dim>
struct CornerPoints;

template <typename T>
struct CornerPoints<T, 1U> {
  static constexpr unsigned int size = 2U;
  itk::FixedArray<itk::FixedArray<T, 1U>, size> m_Points;
};

template <typename T>
struct CornerPoints<T, 2U> {
  static constexpr unsigned int size = 4U;
  itk::FixedArray<itk::FixedArray<T, 2U>, size> m_Points;
};

template <typename T>
struct CornerPoints<T, 3U> {
  static constexpr unsigned int size = 8U;
  itk::FixedArray<itk::FixedArray<T, 3U>, size> m_Points;
};

template <typename T, unsigned int Dim>
void ComputeCornersRec(unsigned int cur, unsigned int& pos, itk::FixedArray<T, Dim>& index, CornerPoints<T, Dim>& out) {
  if(cur == 0) {
    out.m_Points[pos++] = index;
  } else {
    ComputeCornersRec(cur-1, pos, index, out);
    index[cur-1] = itk::NumericTraits<T>::OneValue();
    ComputeCornersRec(cur-1, pos, index, out);
    index[cur-1] = itk::NumericTraits<T>::ZeroValue();
  }

}

template <typename T, unsigned int Dim>
CornerPoints<T, Dim> ComputeCorners() {
  CornerPoints<T, Dim> res;
  itk::FixedArray<T, Dim> index;
  index.Fill(itk::NumericTraits<T>::ZeroValue());
  unsigned int pos = 0;
  ComputeCornersRec(Dim, pos, index, res);
  
  return res;
}

inline double PerformEaseOut(double value, double easeOutThreshold)
{
  // The threshold is an arbitrary threshold below which the magnitude of
  // the gradient begins to be proportionally scaled to ease out gradients near
  // objects.
  if(value < easeOutThreshold)
  {
    return (value / easeOutThreshold);
  }
  else 
  {
    return 1.0;
  }
}

template <unsigned int ImageDimension>
inline double InterpolateDistancesWithGrad(itk::Vector<double, ImageDimension> frac, ValuedCornerPoints<ImageDimension>& distanceValues, itk::Vector<double, ImageDimension>& grad, double easeOutThreshold=0.0);

// Linear interpolation
template <>
inline double InterpolateDistancesWithGrad<1U>(itk::Vector<double, 1U> frac, ValuedCornerPoints<1U>& distanceValues, itk::Vector<double, 1U>& grad, double easeOutThreshold) {
  double xx = frac[0];
  //double ixx = 1.0 - xx;
  double value = LERP(distanceValues.m_Values[0], distanceValues.m_Values[1], xx); //ixx * distanceValues.m_Values[0] + xx * distanceValues.m_Values[1];

  // Compute the scale to apply (initially at 1.0, since if no easing is applied,
  // the gradient should just be the average of two differences).
  double scale = PerformEaseOut(value, easeOutThreshold);

  grad[0] = (distanceValues.m_Values[1] - distanceValues.m_Values[0]) * scale;

  return value;
}

// Bilinear interpolation
template <>
inline double InterpolateDistancesWithGrad<2U>(itk::Vector<double, 2U> frac, ValuedCornerPoints<2U>& distanceValues, itk::Vector<double, 2U>& grad, double easeOutThreshold) {
  double xx = frac[0];
  double yy = frac[1];
  //double ixx = 1.0 - xx;
  //double iyy = 1.0 - yy;

  double v_00 = distanceValues.m_Values[0];
  double v_10 = distanceValues.m_Values[1];
  double v_01 = distanceValues.m_Values[2];
  double v_11 = distanceValues.m_Values[3];

  const double step_10_00 = v_10 - v_00;
  const double step_11_01 = v_11 - v_01;
  const double step_01_00 = v_01 - v_00;
  const double step_11_10 = v_11 - v_10;
  
  // Compute the value with bilinear interpolation
  //const double val1 = LERP(v_00, v_10, xx);
  //const double val2 = LERP(v_01, v_11, xx);
  const double val1 = v_00 + step_10_00 * xx;
  const double val2 = v_01 + step_11_01 * xx;
  const double value = LERP(val1, val2, yy);
  //const double value = v_00 * ixx * iyy + v_10 * xx * iyy + v_01 * ixx * yy + v_11 * xx * yy;
  
  // Compute the scale to apply (initially at 0.5, since if no easing is applied,
  // the gradient should just be the average of two differences).
  double scale = 0.5 * PerformEaseOut(value, easeOutThreshold);
  
  // Compute the gradient vector at the mid-point between the grid points.
  grad[0] = (step_10_00 + step_11_01) * scale;
  grad[1] = (step_01_00 + step_11_10) * scale;
  
  return value;
}

// Trilinear interpolation
template <>
inline double InterpolateDistancesWithGrad<3U>(itk::Vector<double, 3U> frac, ValuedCornerPoints<3U>& distanceValues, itk::Vector<double, 3U>& grad, double easeOutThreshold) {
  double xx = frac[0];
  double yy = frac[1];
  double zz = frac[2];
  double ixx = 1.0 - xx;
  double iyy = 1.0 - yy;
  double izz = 1.0 - zz;

  double v_000 = distanceValues.m_Values[0];
  double v_100 = distanceValues.m_Values[1];
  double v_010 = distanceValues.m_Values[2];
  double v_110 = distanceValues.m_Values[3];
  double v_001 = distanceValues.m_Values[4];
  double v_101 = distanceValues.m_Values[5];
  double v_011 = distanceValues.m_Values[6];
  double v_111 = distanceValues.m_Values[7];

  double v_00 = v_000 * ixx + v_100 * xx;
  double v_01 = v_001 * ixx + v_101 * xx;
  double v_10 = v_010 * ixx + v_110 * xx;
  double v_11 = v_011 * ixx + v_111 * xx;

  double v_0 = v_00 * iyy + v_10 * yy;
  double v_1 = v_01 * iyy + v_11 * yy;

  double value = v_0 * izz + v_1 * zz;

  /*
   v = v_0 * izz + v_1 * zz
   =>
   v = (v_00 * iyy + v_10 * yy) * izz +
         (v_01 * iyy + v_11 * yy) * zz =
   ((v_000 * ixx + v_100 * xx) * iyy + (v_010 * ixx + v_110 * xx) * yy) * izz +
     ((v_001 * ixx + v_101 * xx) * iyy + (v_011 * ixx + v_111 * xx) * yy) * zz =

   df/dx = ((-v_000 + v_100) * iyy + (-v_010 + v_110) * yy) * izz +
     ((-v_001 + v_101) * iyy + (-v_011 + v_111) * yy) * zz =     
  ((v_100 - v_000) * iyy + (v_110 - v_010) * yy) * izz + ((v_101 - v_001) * iyy + (v_111 - v_011) * yy) * zz

   df/dy = -(v_000 * ixx + v_100 * xx) * izz + (v_010 * ixx + v_110 * xx) * izz +
     -(v_001 * ixx + v_101 * xx) * zz + (v_011 * ixx + v_111 * xx) * zz =
   ((v_010 - v_000) * ixx + (v_110 - v_100) * xx) * izz + ((v_011 - v_001) * ixx + (v_111 - v_101) * xx) * zz

   df/dz = -((v_000 * ixx + v_100 * xx) * iyy + (v_010 * ixx + v_110 * xx) * yy) + 
     ((v_001 * ixx + v_101 * xx) * iyy + (v_011 * ixx + v_111 * xx) * yy) =
     ((v_001 - v_000) * ixx + (v_101 - v_100) * xx) * iyy + ((v_011 - v_010) * ixx + (v_111 - v_110) * xx) * yy

  */

  //grad[0] = ((v_100 - v_000) * iyy + (v_110 - v_010) * yy) * izz + ((v_101 - v_001) * iyy + (v_111 - v_011) * yy) * zz;
  //grad[1] = ((v_010 - v_000) * ixx + (v_110 - v_100) * xx) * izz + ((v_011 - v_001) * ixx + (v_111 - v_101) * xx) * zz;
  //grad[2] = ((v_001 - v_000) * ixx + (v_101 - v_100) * xx) * iyy + ((v_011 - v_010) * ixx + (v_111 - v_110) * xx) * yy;
  
  // Compute the scale to apply (initially at 0.25, since if no easing is applied,
  // the gradient should just be the average of four differences).
  double scale = 0.25 * PerformEaseOut(value, easeOutThreshold);

  // Compute the gradient vector at the mid-point between the grid points.
  grad[0] = (((v_100 - v_000) + (v_110 - v_010)) + ((v_101 - v_001) + (v_111 - v_011))) * scale;
  grad[1] = (((v_010 - v_000) + (v_110 - v_100)) + ((v_011 - v_001) + (v_111 - v_101))) * scale;
  grad[2] = (((v_001 - v_000) + (v_101 - v_100)) + ((v_011 - v_010) + (v_111 - v_110))) * scale;
  
  return value;
}

template <unsigned int ImageDimension>
inline unsigned int PixelCount(itk::Size<ImageDimension> &sz)
{
  unsigned int acc = sz[0];
  for (unsigned int i = 1; i < ImageDimension; ++i)
  {
    acc *= sz[i];
  }
  return acc;
}

template <unsigned int ImageDimension>
inline unsigned int LargestDimension(itk::Size<ImageDimension> &sz)
{
  unsigned int res = 0;
  unsigned int s = sz[0];
  for (unsigned int i = 1U; i < ImageDimension; ++i)
  {
    if (sz[i] > s)
    {
      res = i;
      s = sz[i];
    }
  }
  return res;
}

// Computes recursively the exact number of nodes required for a given image.
template <typename IndexType, typename SizeType, unsigned int ImageDimension>
unsigned int MaxNodeIndex(IndexType index, SizeType size, unsigned int nodeIndex) {
      if(PixelCount(size) <= 1U) {
        return nodeIndex;
      }

      IndexType midIndex = index;
      unsigned int selIndex = LargestDimension<ImageDimension>(size);
      unsigned int maxSz = size[selIndex];

      midIndex[selIndex] = midIndex[selIndex] + maxSz / 2;
      SizeType sz1 = size;
      SizeType sz2 = size;

      sz1[selIndex] = sz1[selIndex] / 2;
      sz2[selIndex] = size[selIndex] - sz1[selIndex];

      unsigned int nodeIndex1 = nodeIndex * 2;
      unsigned int nodeIndex2 = nodeIndex * 2 + 1;

      return MaxNodeIndex<IndexType, SizeType, ImageDimension>(midIndex, sz2, nodeIndex2);
}

template <typename IndexType, typename SizeType, unsigned int ImageDimension, typename SpacingType>
inline double LowerBoundDistance(IndexType pnt, IndexType rectOrigin, SizeType rectSz, SpacingType sp)
{
  double d = 0;
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    long long pnt_i = (long long)pnt[i];
    long long lowEdgePos_i = (long long)rectOrigin[i];
    long long highEdgePos_i = (long long)(rectOrigin[i] + rectSz[i]);

    double d1_i = (double)(lowEdgePos_i - pnt_i);
    double d2_i = (double)(pnt_i - highEdgePos_i);
    double d_i = std::max(std::max(d1_i, d2_i), 0.0) * sp[i];
    d += d_i*d_i;
  } 
  return d;
}

template <typename IndexType, typename SizeType, unsigned int ImageDimension, typename SpacingType>
inline double LowerBoundDistanceApprox(IndexType pnt, IndexType rectOrigin, SizeType rectSz, SpacingType sp, double threshold, double fraction)
{
  double d = 0;
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    long long pnt_i = (long long)pnt[i];
    long long lowEdgePos_i = (long long)rectOrigin[i];
    long long highEdgePos_i = lowEdgePos_i + ((long long)(rectSz[i]) - 1LL);

    double d1_i = (double)(lowEdgePos_i - pnt_i);
    double d2_i = (double)(pnt_i - highEdgePos_i);
    double d_i = std::max(std::max(d1_i, d2_i), 0.0) * sp[i];
    d += d_i*d_i;
  }

  if (d > threshold * threshold) {
    // Convert to Euclidean distance
    double d_e = sqrt(d);
    // Apply relaxation of the bound
    d_e = fraction*(d_e-threshold) + threshold;
    // Convert back to squared Euclidean distance
    d = d_e*d_e;
  }

  return d;
}

template <typename T>
inline unsigned int PruneLevelsLinear(const T* values, unsigned int start, unsigned int end, T val) {
  for(; start < end; --end) {
    if(values[end-1] <= val) {
      break;
    }
  }
  return end;
}
};

//
// Evaluation context containing the auxilliary data-structures required
// to sample intensity values and compute the value and gradient in
// the Monte Carlo framework.
//
// In a multi-threaded scenario, each thread must command its own
// private eval context.
//
// TODO: Allow the seed to be set for each eval context...
template <typename TImageType, typename TInternalValueType = unsigned short>
class MCDSEvalContext : public itk::Object {
public:
  using ImageType = TImageType;
  using InternalValueType = TInternalValueType;

  using Self = MCDSEvalContext;
  using Superclass = itk::Object;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;   

  static constexpr unsigned int ImageDimension = ImageType::ImageDimension;

  typedef typename ImageType::Pointer ImagePointer;
  typedef typename ImageType::RegionType RegionType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::IndexValueType IndexValueType;
  typedef typename ImageType::SizeValueType SizeValueType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename itk::ContinuousIndex<double, ImageType::ImageDimension> ContinousIndexType;
  typedef typename ImageType::SpacingType SpacingType;
  typedef typename ImageType::ValueType ValueType;
  typedef typename ImageType::PointType PointType;

  typedef MCDSInternal::CornerPoints<IndexValueType, ImageType::ImageDimension> CornersType;

  typedef ValueSamplerBase<double, 1U> ValueSamplerType;
  typedef typename ValueSamplerType::Pointer ValueSamplerPointer;
  
  itkNewMacro(Self);

  itkTypeMacro(MCDSEvalContext, itk::Object);

  std::unique_ptr<double[]> m_Table;
  std::vector<InternalValueType> m_InwardsValues;
  std::vector<InternalValueType> m_ComplementValues;

  ValueSamplerPointer m_Sampler;
  unsigned int m_Samples;
  
  virtual unsigned int GetSampleCount() const
  {
    return m_Samples;
  }

  virtual void SetSampleCount(unsigned int samples)
  {
    m_Samples = samples;
  }

  virtual ValueSamplerPointer GetSampler() const
  {
    return m_Sampler;
  }
  
  virtual void SetSampler(ValueSamplerPointer sampler)
  {
    m_Sampler = sampler;
  }

  virtual void RestartSampler()
  {
    m_Sampler->RestartFromSeed();
  }

  virtual void Initialize()
  {
    m_InwardsValues.reserve(m_Samples);
    m_ComplementValues.reserve(m_Samples);
    m_Table = std::move(std::unique_ptr<double[]>(new double[m_Samples * CornersType::size]));
  }
protected:
  MCDSEvalContext() = default;
};

template <typename TImageType, typename TInternalValueType = unsigned short>
class MCAlphaCutPointToSetDistance : public itk::Object
{
public:
    using Self = MCAlphaCutPointToSetDistance;
    using Superclass = itk::Object;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
   
    itkNewMacro(Self);

    itkTypeMacro(MCAlphaCutPointToSetDistance, itk::Object);

    using ImageType = TImageType;
    using InternalValueType = TInternalValueType;

  static constexpr unsigned int ImageDimension = ImageType::ImageDimension;

  typedef typename ImageType::Pointer ImagePointer;
  typedef typename ImageType::RegionType RegionType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::IndexValueType IndexValueType;
  typedef typename ImageType::SizeValueType SizeValueType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename itk::ContinuousIndex<double, ImageType::ImageDimension> ContinousIndexType;
  typedef typename ImageType::SpacingType SpacingType;
  typedef typename ImageType::ValueType ValueType;
  typedef itk::FixedArray<InternalValueType, 2U> NodeValueType;
  typedef typename ImageType::PointType PointType;

  typedef itk::Image<bool, ImageDimension> MaskImageType;
  typedef typename MaskImageType::Pointer MaskImagePointer;
  typedef itk::NearestNeighborInterpolateImageFunction<MaskImageType, double> InterpolatorType;
  typedef typename InterpolatorType::Pointer InterpolatorPointer;

  typedef ValueSamplerBase<double, 1U> ValueSamplerType;
  typedef typename ValueSamplerType::Pointer ValueSamplerPointer;

  typedef MCDSInternal::CornerPoints<IndexValueType, ImageType::ImageDimension> CornersType;
  
  typedef MCDSEvalContext<ImageType, InternalValueType> EvalContextType;
  typedef typename EvalContextType::Pointer EvalContextPointer;

  void SetImage(ImagePointer image)
  {
    m_Image = image;
    m_RawImagePtr = image.GetPointer();
  }

  void SetMaskImage(MaskImagePointer maskImage) {
    m_MaskImage = maskImage;
  }

  double GetMaxDistance() const
  {
    return m_MaxDistance;
  }

  void SetMaxDistance(double dmax) {
    assert(dmax >= 0.0);

    m_MaxDistance = dmax;
  }

  double GetApproximationThreshold() const
  {
    return m_ApproximationDistanceThreshold;
  }

  void SetApproximationThreshold(double distanceThreshold)
  {
    assert(distanceThreshold >= 0.0);

    m_ApproximationDistanceThreshold = distanceThreshold;
  }

  double GetApproximationFraction() const
  {
    return m_ApproximationDistanceFraction;
  }

  void SetApproximationFraction(double distanceFraction)
  {
    assert(distanceFraction >= 0.0);

    m_ApproximationDistanceFraction = distanceFraction;
  }

  double GetDistancePower() const
  {
    return m_DistancePower;
  }

  void SetDistancePower(double power)
  {
    m_DistancePower = power;
  }

  bool GetInwardsMode() const
  {
    return m_InwardsMode;
  }

  void SetInwardsMode(bool mode)
  {
    m_InwardsMode = mode;
  }

  unsigned int GetSampleCount() const
  {
    return m_SampleCount;
  }

  void SetSampleCount(unsigned int count)
  {
    m_SampleCount = count;
  }

  void SetSamplingMode(ValueSamplerTypeEnum samplerType)
  {
    m_ValueSamplerType = samplerType;
  }

  EvalContextPointer MakeEvalContext()
  {
    EvalContextPointer cxt = EvalContextType::New();
    cxt->SetSampleCount(m_SampleCount);

    ValueSamplerPointer valueSampler;
    if(m_ValueSamplerType == ValueSamplerTypeUniform)
    {
      valueSampler = UniformValueSampler<double, 1U>::New().GetPointer();
    }
    else if(m_ValueSamplerType == ValueSamplerTypeQuasiRandom)
    {
      valueSampler = QuasiRandomValueSampler<double, 1U>::New().GetPointer();
    }
    cxt->SetSampler(valueSampler);

    cxt->Initialize();
    return cxt;
  }

  // Builds the kd-tree and initializes data-structures
  void Initialize()
  {
    // Compute height
    constexpr unsigned int dim = ImageType::ImageDimension;
    RegionType region = m_Image->GetLargestPossibleRegion();
    SizeType sz = region.GetSize();
    if(m_MaxDistance <= 0.0) {
      m_MaxDistance = 0.0;
      for(unsigned int i = 0; i < ImageDimension; ++i) {
        m_MaxDistance += sz[i]*sz[i];
      }
    }

    unsigned int nodeCount = MCDSInternal::MaxNodeIndex<IndexType, SizeType, ImageDimension>(region.GetIndex(), sz, 1);
    m_Array = std::move(std::unique_ptr<NodeValueType[]>(new NodeValueType[nodeCount]));

    if(MCDSInternal::PixelCount(sz) > 0)
      BuildTreeRec(1, region.GetIndex(), sz);

    m_Corners = MCDSInternal::ComputeCorners<IndexValueType, ImageType::ImageDimension>();

#ifdef DEBUG_NODE_COUNTING_ENABLED
    m_DebugVisitCount = 0;
#endif

    m_RawImagePtr = m_Image.GetPointer();

    if(m_MaskImage) {
      m_MaskInterpolator = InterpolatorType::New();
      m_MaskInterpolator->SetInputImage(m_MaskImage);
    }
  }

  bool ValueAndDerivative(
    EvalContextPointer evalContext,
    unsigned int pointIndex,
    PointType point,
    ValueType h,
    double& valueOut,
    itk::Vector<double, ImageDimension>& gradOut) const {

    ImageType* image = m_RawImagePtr;

    // If we have a mask, check if we are inside the mask image bounds, and inside the mask
    if(m_MaskImage) {
      if(m_MaskInterpolator->IsInsideBuffer(point)) {
        if(!m_MaskInterpolator->Evaluate(point))
          return false;
      } else {
        return false;
      }
    }

    ContinousIndexType cIndex;
    IndexType pntIndex; // Generate the index
    itk::Vector<double, ImageDimension> frac;

    bool flag = image->TransformPhysicalPointToContinuousIndex(point, cIndex);
    for(unsigned int i = 0; i < ImageDimension; ++i) {
      pntIndex[i] = (long long)cIndex[i];
      frac[i] = cIndex[i] - (double)pntIndex[i];
    }
    
    // Sampling

    InternalValueType hQ = QuantizeValue<ValueType, InternalValueType>(h);

    evalContext->m_InwardsValues.clear();
    evalContext->m_ComplementValues.clear();
    for(unsigned int i = 0; i < evalContext->m_Samples; ++i)
    {
      itk::FixedArray<double, 1U> val;
      evalContext->m_Sampler->Sample(val, pointIndex, i, evalContext->m_Samples);
      
      InternalValueType valQ = QuantizeValue<ValueType, InternalValueType>(val[0]);
      if (m_InwardsMode)
      {
          if(hQ > 0)
            evalContext->m_InwardsValues.push_back(valQ % hQ);
          else
            evalContext->m_InwardsValues.push_back(0);
      }
      else
      {
        if(valQ <= hQ)
        {
            evalContext->m_InwardsValues.push_back(valQ);
        }
        else
        {
            evalContext->m_ComplementValues.push_back(QuantizedValueMax<InternalValueType>() - valQ);
        }
      }
    }
    
    // For now, just remove the complement values when in inwards mode... this is not optimal.
    /*if (m_InwardsMode)
    {
      evalContext->m_ComplementValues.clear();
      if (evalContext->m_InwardsValues.size() == 0)
      {
        return false;
      }
    }*/

    std::sort(evalContext->m_InwardsValues.begin(), evalContext->m_InwardsValues.end());
    std::sort(evalContext->m_ComplementValues.begin(), evalContext->m_ComplementValues.end());

    RegionType region = image->GetLargestPossibleRegion();

    bool isFullyInside = true;
    for(unsigned int j = 0; j < ImageDimension; ++j) {
      if(pntIndex[j] < 0 || pntIndex[j] + 1 >= region.GetSize()[j]) {
        isFullyInside = false;
        break;
      }
    }

    unsigned int inwardsStart = 0;
    unsigned int complementStart = 0;

    ValueType minInVal = QuantizedValueMax<InternalValueType>();
    ValueType minCoVal = QuantizedValueMax<InternalValueType>();

    if(isFullyInside)
    {
      for(unsigned int i = 0; i < CornersType::size; ++i)
      {     
        IndexType cindex = pntIndex;
        for(unsigned int j = 0; j < ImageDimension; ++j)
        {
          cindex[j] = cindex[j] + m_Corners.m_Points[i][j];
        }

        InternalValueType valIn = QuantizeValue<ValueType, InternalValueType>(image->GetPixel(cindex));
        InternalValueType valCo = QuantizedValueMax<InternalValueType>() - valIn;
        if(valIn < minInVal)
          minInVal = valIn;
        if(valCo < minCoVal)
          minCoVal = valCo;
      }

      for(; inwardsStart < evalContext->m_InwardsValues.size(); ++inwardsStart)
      {
        if(evalContext->m_InwardsValues[inwardsStart] > minInVal)
          break;
      }
      for(; complementStart < evalContext->m_ComplementValues.size(); ++complementStart)
      {
        if(evalContext->m_ComplementValues[complementStart] > minCoVal)
          break;
      }
    }

    unsigned int sampleCount = evalContext->m_InwardsValues.size() + evalContext->m_ComplementValues.size();   

    if(isFullyInside && (inwardsStart < evalContext->m_InwardsValues.size() || complementStart < evalContext->m_ComplementValues.size()))
    {    
      double dmax = m_MaxDistance;
      double dmaxSq = dmax * dmax;
      for(unsigned int i = 0; i < CornersType::size; ++i) {
        double* dists_i = evalContext->m_Table.get() + (sampleCount * i);

        unsigned int j = 0;
        for(; j < inwardsStart; ++j)
          dists_i[j] = 0.0;
        for(; j < evalContext->m_InwardsValues.size(); ++j)
          dists_i[j] = dmaxSq;
        for(; j < evalContext->m_InwardsValues.size()+complementStart; ++j)
          dists_i[j] = 0.0;
        for(; j < sampleCount; ++j)
          dists_i[j] = dmaxSq;
      }

      Search(pntIndex, inwardsStart, complementStart, evalContext.GetPointer());

      MCDSInternal::ValuedCornerPoints<ImageDimension> cornerValues;

      const double localDistancePower = m_DistancePower;

      for(unsigned int i = 0; i < CornersType::size; ++i)
      {
        cornerValues.m_Values[i] = 0.0;
        double* dists_i = evalContext->m_Table.get() + (sampleCount * i);

        for(unsigned int j = 0; j < sampleCount; ++j)
        {
          cornerValues.m_Values[i] += pow(dists_i[j], m_DistancePower * 0.5); // Instead of sqrt(dists_i[j]) for arbitrary power.
        }

        cornerValues.m_Values[i] = cornerValues.m_Values[i] / sampleCount;
      }

      valueOut = MCDSInternal::InterpolateDistancesWithGrad<ImageDimension>(frac, cornerValues, gradOut);

      // Apply spacing to gradient
      typedef typename ImageType::SpacingType SpacingType;
      SpacingType spacing = m_Image->GetSpacing();
    
      for(unsigned int i = 0; i < ImageDimension; ++i)
        gradOut[i] /= spacing[i];
    } else {
      valueOut = 0.0;
      for(unsigned int i = 0; i < ImageDimension; ++i)
        gradOut[i] = 0.0;
    }

    return true;
  }
#ifdef DEBUG_NODE_COUNTING_ENABLED  
  mutable size_t m_DebugVisitCount;
#endif
protected:
  MCAlphaCutPointToSetDistance()
  {
    m_ValueSamplerType = ValueSamplerTypeQuasiRandom;
    m_MaxDistance = 0.0;
    m_ApproximationDistanceThreshold = 20.0;
    m_ApproximationDistanceFraction = 0.1;
    m_DistancePower = 1.0;
    m_InwardsMode = false;
  }

  ImagePointer m_Image;
  ImageType* m_RawImagePtr;
  MaskImagePointer m_MaskImage;
  InterpolatorPointer m_MaskInterpolator;
  std::unique_ptr<NodeValueType[]> m_Array;
  unsigned int m_SampleCount;
  double m_MaxDistance;
  double m_ApproximationDistanceThreshold;
  double m_ApproximationDistanceFraction;
  double m_DistancePower;
  bool m_InwardsMode;
  CornersType m_Corners;
  ValueSamplerTypeEnum m_ValueSamplerType;

  struct StackNode
  {
    IndexType m_Index;
    SizeType m_Size;
    unsigned int m_NodeIndex;
    unsigned int m_InStart;
    unsigned int m_InEnd;
    unsigned int m_CoStart;
    unsigned int m_CoEnd;
  };

  void BuildTreeRec(unsigned int nodeIndex, IndexType index, SizeType sz)
  {
    constexpr unsigned int dim = ImageType::ImageDimension;

    typedef itk::ImageRegionConstIterator<ImageType> IteratorType;
    NodeValueType* data = m_Array.get();

    unsigned int szCount = MCDSInternal::PixelCount<dim>(sz);

    if (szCount == 1U)
    {
      NodeValueType nv;
      if(m_MaskImage) {
        if(m_MaskImage->GetPixel(index)) {
          nv[0] = QuantizeValue<ValueType, InternalValueType>(m_RawImagePtr->GetPixel(index));
          nv[1] = QuantizedValueMax<InternalValueType>() - nv[0];
          data[nodeIndex - 1] = nv;
        } else {
          data[nodeIndex - 1].Fill(QuantizedValueMin<InternalValueType>());
        }
      } else {
        nv[0] = QuantizeValue<ValueType, InternalValueType>(m_RawImagePtr->GetPixel(index));
        nv[1] = QuantizedValueMax<InternalValueType>() - nv[0];
        data[nodeIndex - 1] = nv;
      }
    }
    else
    {
      IndexType midIndex = index;
      unsigned int selIndex = MCDSInternal::LargestDimension<dim>(sz);
      unsigned int maxSz = sz[selIndex];

      midIndex[selIndex] = midIndex[selIndex] + maxSz / 2;
      SizeType sz1 = sz;
      SizeType sz2 = sz;

      sz1[selIndex] = sz1[selIndex] / 2;
      sz2[selIndex] = sz[selIndex] - sz1[selIndex];

      unsigned int nodeIndex1 = nodeIndex * 2;

      BuildTreeRec(nodeIndex1, index, sz1);
      BuildTreeRec(nodeIndex1+1, midIndex, sz2);

      NodeValueType n1 = *(data + (nodeIndex1 - 1));
      NodeValueType n2 = *(data + nodeIndex1);

      NodeValueType* dataCur = data + (nodeIndex-1);

      // Compute the maximum of the two nodes, for each channel
      for (unsigned int i = 0; i < 2U; ++i)
      {
        if (n2[i] > n1[i])
          (*dataCur)[i] = n2[i];
        else
          (*dataCur)[i] = n1[i];
      }
    }
  }

  void Search(
      IndexType index,
      unsigned int inwardsStart, unsigned int complementStart, EvalContextType* evalContext) const
  {
    InternalValueType* inwardsValues = evalContext->m_InwardsValues.data();
    InternalValueType* complementValues = evalContext->m_ComplementValues.data();
    unsigned int inwardsCount = evalContext->m_InwardsValues.size();
    unsigned int complementCount = evalContext->m_ComplementValues.size();

    typedef typename ImageType::SpacingType SpacingType;

    SpacingType spacing = m_Image->GetSpacing();

    double* distTable = evalContext->m_Table.get();
    NodeValueType* data = m_Array.get(); // Node data

    unsigned int sampleCount = m_SampleCount;

    double threshold = m_ApproximationDistanceThreshold;
    //double thresholdSq = threshold*threshold;
    double fraction = 1.0 + m_ApproximationDistanceFraction;
    //double fractionSq = fraction*fraction;

    // Stack
    StackNode stackNodes[33];
    StackNode curStackNode;
    unsigned int stackIndex = 0;

    // Initialize the stack state
    curStackNode.m_Index = m_Image->GetLargestPossibleRegion().GetIndex();
    curStackNode.m_Size = m_Image->GetLargestPossibleRegion().GetSize();
    curStackNode.m_NodeIndex = 1;
    curStackNode.m_InStart = inwardsStart;
    curStackNode.m_InEnd = MCDSInternal::PruneLevelsLinear(inwardsValues, inwardsStart, inwardsCount, data[0][0]);
    curStackNode.m_CoStart = complementStart;
    curStackNode.m_CoEnd = MCDSInternal::PruneLevelsLinear(complementValues, complementStart, complementCount, data[0][1]);

    // All elements eliminated before entering the loop. Search is over. Return.
    if(curStackNode.m_InStart == curStackNode.m_InEnd && curStackNode.m_CoStart == curStackNode.m_CoEnd)
      return;
    
    itk::FixedArray<itk::Point<double, ImageDimension>, CornersType::size> corners;
    IndexType cornerIndices[CornersType::size];

    for (unsigned int i = 0; i < CornersType::size; ++i)
    {
      for (unsigned int j = 0; j < ImageDimension; ++j)
      {
        auto cornerIndex = m_Corners.m_Points[i][j] + index[j];
        cornerIndices[i][j] = cornerIndex;
        corners[i][j] = static_cast<double>(cornerIndex) * spacing[j];
      }
    }

#ifdef DEBUG_NODE_COUNTING_ENABLED
    unsigned int visitCount = 0;
#endif
    while(true)
    {
      unsigned int npx = curStackNode.m_Size[0];
      for(unsigned int i = 1; i < ImageDimension; ++i) {
        npx *= curStackNode.m_Size[i];
      }

      unsigned int inStartLocal = curStackNode.m_InStart;
      unsigned int inEndLocal = curStackNode.m_InEnd;
      unsigned int coStartLocal = curStackNode.m_CoStart;
      unsigned int coEndLocal = curStackNode.m_CoEnd;

      // Is the node a leaf - compute distances
      if (npx == 1U)
      {
#ifdef DEBUG_NODE_COUNTING_ENABLED
        ++visitCount;
#endif

        itk::Point<double, ImageDimension> leafPoint;
        for(unsigned int j = 0; j < ImageDimension; ++j)
          leafPoint[j] = static_cast<double>(curStackNode.m_Index[j]*spacing[j]);

        // Compare d with all the distances recorded for the alpha levels (which are still in play)
        for (unsigned int i = 0; i < CornersType::size; ++i)
        {
          const double d = corners[i].SquaredEuclideanDistanceTo(leafPoint);
          double* distTable_i = distTable + (sampleCount * i);
          double* coDistTable_i = distTable_i + inwardsCount;

          for (unsigned int j = coEndLocal; coStartLocal < j; --j)
          {
            unsigned int tabInd = j-1;//+inwardsCount
            double cur_j = coDistTable_i[tabInd];
            if (d < cur_j)
            {
              coDistTable_i[tabInd] = d;
            } else {
              break;
            }
          }          

          for (unsigned int j = inEndLocal; inStartLocal < j; --j)
          {
            unsigned int tabInd = j-1;
            double cur_j = distTable_i[tabInd];
            if (d < cur_j)
            {
              distTable_i[tabInd] = d;
            } else {
              break;
            }
          }
        }
      }
      else
      { // Continue traversing the tree
        // Compute (approximate) lower bounds on distance from each nearest grid-point to the node bounding box
        IndexType innerNodeInd = curStackNode.m_Index;
        SizeType innerNodeSz = curStackNode.m_Size;
        double lowerBoundDistances[CornersType::size];
        for (unsigned int i = 0; i < CornersType::size; ++i)
        {
          double* distTable_i = distTable + (sampleCount * i);
          double* coDistTable_i = distTable_i + inwardsCount;

          lowerBoundDistances[i] = MCDSInternal::LowerBoundDistanceApprox<IndexType, SizeType, ImageDimension>(
            cornerIndices[i], innerNodeInd, innerNodeSz, spacing, threshold, fraction
          );
        }

        // Eliminate inwards values based on distance bounds
        for (; inStartLocal < inEndLocal; ++inStartLocal)
        {
          bool anyMayImprove = false;
          for (unsigned int i = 0; i < CornersType::size; ++i)
          {
            double cur_best = distTable[inStartLocal + (sampleCount * i)];
            if (lowerBoundDistances[i] < cur_best)
            {
              anyMayImprove = true;
              break;
            }
          }
          if (anyMayImprove)
            break;
        }

        // Eliminate complement values based on distance bounds
        for (; coStartLocal < coEndLocal; ++coStartLocal)
        {
          bool anyMayImprove = false;
          for (unsigned int i = 0; i < CornersType::size; ++i)
          {
            double cur_best = distTable[coStartLocal+inwardsCount + (sampleCount * i)];
            if (lowerBoundDistances[i] < cur_best)
            {
              anyMayImprove = true;
              break;
            }
          }
          if (anyMayImprove)
            break;
        }

        // If all alpha levels are eliminated, backtrack
        // without doing the work to compute the node split
        // and tree value look-ups.
        if (inStartLocal == inEndLocal && coStartLocal == coEndLocal)
        {
          if(stackIndex == 0)
            break;
          curStackNode = stackNodes[--stackIndex];
          continue;
        }

        IndexType midIndex = innerNodeInd;
        unsigned int selIndex = MCDSInternal::LargestDimension<ImageDimension>(innerNodeSz);
        unsigned int maxSz = innerNodeSz[selIndex];
        unsigned int halfMaxSz = maxSz / 2;

        midIndex[selIndex] = midIndex[selIndex] + halfMaxSz;
        SizeType sz1 = innerNodeSz;
        SizeType sz2 = innerNodeSz;

        sz1[selIndex] = halfMaxSz;
        sz2[selIndex] = maxSz - halfMaxSz;

        unsigned int nodeIndex1 = curStackNode.m_NodeIndex * 2;

        unsigned int inEndLocal1 = MCDSInternal::PruneLevelsLinear(inwardsValues, inStartLocal, inEndLocal, data[nodeIndex1-1][0]);
        unsigned int coEndLocal1 = MCDSInternal::PruneLevelsLinear(complementValues, coStartLocal, coEndLocal, data[nodeIndex1-1][1]);
        unsigned int inEndLocal2 = MCDSInternal::PruneLevelsLinear(inwardsValues, inStartLocal, inEndLocal, data[nodeIndex1][0]);
        unsigned int coEndLocal2 = MCDSInternal::PruneLevelsLinear(complementValues, coStartLocal, coEndLocal, data[nodeIndex1][1]);

        if (index[selIndex] < midIndex[selIndex])
        {
          if(inStartLocal < inEndLocal1 || coStartLocal < coEndLocal1)
          {
            curStackNode.m_Index = innerNodeInd;
            curStackNode.m_Size = sz1;
            curStackNode.m_NodeIndex = nodeIndex1;
            curStackNode.m_InStart = inStartLocal;
            curStackNode.m_InEnd = inEndLocal1;
            curStackNode.m_CoStart = coStartLocal;
            curStackNode.m_CoEnd = coEndLocal1;
            if(inStartLocal < inEndLocal2 || coStartLocal < coEndLocal2)
            {
              stackNodes[stackIndex].m_Index = midIndex;
              stackNodes[stackIndex].m_Size = sz2;
              stackNodes[stackIndex].m_NodeIndex = nodeIndex1+1;
              stackNodes[stackIndex].m_InStart = inStartLocal;
              stackNodes[stackIndex].m_InEnd = inEndLocal2;
              stackNodes[stackIndex].m_CoStart = coStartLocal;
              stackNodes[stackIndex].m_CoEnd = coEndLocal2;
              ++stackIndex;
            }
            continue;
          }
          else if(inStartLocal < inEndLocal2 || coStartLocal < coEndLocal2)
          {
            curStackNode.m_Index = midIndex;
            curStackNode.m_Size = sz2;
            curStackNode.m_NodeIndex = nodeIndex1+1;
            curStackNode.m_InStart = inStartLocal;
            curStackNode.m_InEnd = inEndLocal2;
            curStackNode.m_CoStart = coStartLocal;
            curStackNode.m_CoEnd = coEndLocal2;
            continue;
          }      
        }
        else
        {
          if(inStartLocal < inEndLocal2 || coStartLocal < coEndLocal2)
          {
            curStackNode.m_Index = midIndex;
            curStackNode.m_Size = sz2;
            curStackNode.m_NodeIndex = nodeIndex1+1;
            curStackNode.m_InStart = inStartLocal;
            curStackNode.m_InEnd = inEndLocal2;
            curStackNode.m_CoStart = coStartLocal;
            curStackNode.m_CoEnd = coEndLocal2;
            if(inStartLocal < inEndLocal1 || coStartLocal < coEndLocal1)
            {
              stackNodes[stackIndex].m_Index = innerNodeInd;
              stackNodes[stackIndex].m_Size = sz1;
              stackNodes[stackIndex].m_NodeIndex = nodeIndex1;
              stackNodes[stackIndex].m_InStart = inStartLocal;
              stackNodes[stackIndex].m_InEnd = inEndLocal1;
              stackNodes[stackIndex].m_CoStart = coStartLocal;
              stackNodes[stackIndex].m_CoEnd = coEndLocal1;
              ++stackIndex;
            }
            continue;
          } else if(inStartLocal < inEndLocal1 || coStartLocal < coEndLocal1) 
          {
            curStackNode.m_Index = innerNodeInd;
            curStackNode.m_Size = sz1;
            curStackNode.m_NodeIndex = nodeIndex1;
            curStackNode.m_InStart = inStartLocal;
            curStackNode.m_InEnd = inEndLocal1;
            curStackNode.m_CoStart = coStartLocal;
            curStackNode.m_CoEnd = coEndLocal1;
            continue;
          }       
      } // End of else branch
      }

      // If we arrive here, we need to pop a stack node
      // unless the stack is empty, which would trigger a
      // termination of the loop.
      if(stackIndex == 0)
        break;
      curStackNode = stackNodes[--stackIndex];
    } // End main "recursion" loop

#ifdef DEBUG_NODE_COUNTING_ENABLED
    m_DebugVisitCount += visitCount;
#endif
  } // End of Search function

}; // End of class
