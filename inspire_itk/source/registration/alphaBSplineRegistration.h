
// The main component of a registration framework
// based on distance measures between fuzzy sets.

#ifndef ALPHA_BSPLINE_REGISTRATION_H
#define ALPHA_BSPLINE_REGISTRATION_H

#include "itkImage.h"
#include "itkDomainThreader.h"
#include "itkThreadedIndexedContainerPartitioner.h"
#include "itkBSplineTransform.h"

#include <atomic>

#include "../common/quantization.h"
#include "../common/command.h"

// Point sampler methods
#include "../samplers/pointSamplerBase.h"

// Computes the (numerical) spatial derivatives matrix of a transformation w.r.t. a given point and step size.
// First index (row) represents the index of the input dimension being changed
// Second index (column) represents the index of the output dimension changing accordingly
template <typename TTransformType, unsigned int Dim>
itk::Matrix<double, Dim, Dim> NumericalSpatialDerivatives(
    TTransformType* transform,
    itk::Point<double, Dim> p,
    double stepSize = 1e-6
    )
{
    using PointType = itk::Point<double, Dim>;
    itk::Matrix<double, Dim, Dim> result;
    double denomFactor = 1.0/(2.0 * stepSize);

    for (unsigned int i = 0; i < Dim; ++i)
    {
        PointType pForward = p;
        pForward[i] = pForward[i] + stepSize;
        
        PointType pBackward = p;
        pBackward[i] = pBackward[i] - stepSize;

        PointType tpForward = transform->TransformPoint(pForward);
        PointType tpBackward = transform->TransformPoint(pBackward);

        for (unsigned int j = 0; j < Dim; ++j)
        {
            result(i, j) = (tpForward[j]-tpBackward[j]) * denomFactor;
        }
    }

    return result;
}

// Computes the (numerical) spatial derivatives matrix of a transformation w.r.t. a given point and step size.
// First index (row) represents the index of the input dimension being changed
// Second index (column) represents the index of the output dimension changing accordingly
template <typename TTransformType, unsigned int Dim>
itk::Matrix<double, Dim, Dim> NumericalSpatialDerivativesForward(
    TTransformType* transform,
    itk::Point<double, Dim> p,
    itk::Point<double, Dim> tpMid,
    double stepSize = 1e-6
    )
{
    using PointType = itk::Point<double, Dim>;
    itk::Matrix<double, Dim, Dim> result;
    double denomFactor = 1.0/stepSize;

    for (unsigned int i = 0; i < Dim; ++i)
    {
        PointType pForward = p;
        pForward[i] = pForward[i] + stepSize;
        
        PointType tpForward = transform->TransformPoint(pForward);
        if (tpForward == pForward)
        {
            for (unsigned int j = 0; j < Dim; ++j)
            {
                result(i, j) = 0.0;
            }
            result(i, i) = 1.0;
        } else {
            for (unsigned int j = 0; j < Dim; ++j)
            {
                result(i, j) = (tpForward[j]-tpMid[j]) * denomFactor;
            }
        }
    }

    return result;
}

template <typename TTransformType, unsigned int Dim>
bool ApplyBSplineTransformToPointWithSpatialNumericalDerivatives(
    TTransformType* transform,
    itk::Point<double, Dim> p,
    itk::Point<double, Dim> &outputPoint,
    itk::Matrix<double, Dim, Dim> &derivatives,
    typename TTransformType::WeightsType &weights,
    typename TTransformType::ParameterIndexArrayType &indices,
    double stepSize = 1e-6
    )
{
    using PointType = itk::Point<double, Dim>;
    double denomFactor = 1.0/stepSize;

    PointType points[Dim];
    bool isInside[Dim];

    for (unsigned int i = 0; i < Dim; ++i)
    {
        PointType pForward = p;
        pForward[i] = pForward[i] + stepSize;
        
        transform->TransformPoint(pForward, points[i], weights, indices, isInside[i]);
    }
    bool outputPointInside;
    transform->TransformPoint(p, outputPoint, weights, indices, outputPointInside);
    if (!outputPointInside)
    {
        derivatives.SetIdentity();
    } else {
        for (unsigned int i = 0; i < Dim; ++i)
        {
            if (isInside[i])
            {
                for (unsigned int j = 0; j < Dim; ++j)
                {
                    derivatives(i, j) = (points[i][j]-outputPoint[j]) * denomFactor;
                }
            } else {
                for (unsigned int j = 0; j < Dim; ++j)
                {
                    derivatives(i, j) = 0.0;
                }
            }
        }
    }

    return outputPointInside;
}

template <typename TImageType, typename TDistType, unsigned int TSplineOrder>
struct AlphaBSplineRegistrationThreadState
{
    using ImageType = TImageType;
    using ImagePointer = typename ImageType::Pointer;

    constexpr static unsigned int ImageDimension = ImageType::ImageDimension;
    constexpr static unsigned int SplineOrder = TSplineOrder;

	using TransformType = itk::BSplineTransform<double, ImageDimension, SplineOrder>;
	using TransformPointer = typename TransformType::Pointer;
    using DerivativeType = typename TransformType::DerivativeType;

    using WeightsType = typename TransformType::WeightsType;
    using ParameterIndexArrayType = typename TransformType::ParameterIndexArrayType;

    using DistType = TDistType;
    using DistPointer = typename DistType::Pointer;

    using DistEvalContextType = typename DistType::EvalContextType;
    using DistEvalContextPointer = typename DistEvalContextType::Pointer;

    std::unique_ptr<FixedPointNumber[]> m_DerivativeRefToFlo;
    std::unique_ptr<FixedPointNumber[]> m_DerivativeFloToRef;
    std::unique_ptr<FixedPointNumber[]> m_WeightsRefToFlo;
    std::unique_ptr<FixedPointNumber[]> m_WeightsFloToRef;

    WeightsType m_ParamWeightsRefToFlo;
    WeightsType m_ParamWeightsFloToRef;

    ParameterIndexArrayType m_ParamIndicesRefToFlo;
    ParameterIndexArrayType m_ParamIndicesFloToRef;

    FixedPointNumber m_DistanceRefToFlo;
    FixedPointNumber m_DistanceFloToRef;
    FixedPointNumber m_WeightRefToFlo;
    FixedPointNumber m_WeightFloToRef;

    unsigned int m_ParamNumRefToFlo;
    unsigned int m_ParamNumFloToRef;

    unsigned int m_SupportSizeRefToFlo;
    unsigned int m_SupportSizeFloToRef;

    DistEvalContextPointer m_DistEvalContextRefImage;
    DistEvalContextPointer m_DistEvalContextFloImage;

    void Initialize(unsigned int paramNumRefToFlo, unsigned int paramNumFloToRef, unsigned int supportSizeRefToFlo, unsigned int supportSizeFloToRef, DistEvalContextPointer distEvalContextRefImage, DistEvalContextPointer distEvalContextFloImage)
    {
        m_ParamNumRefToFlo = paramNumRefToFlo;
        m_ParamNumFloToRef = paramNumFloToRef;
        m_SupportSizeRefToFlo = supportSizeRefToFlo;
        m_SupportSizeFloToRef = supportSizeFloToRef;

        m_DistEvalContextRefImage = distEvalContextRefImage;
        m_DistEvalContextFloImage = distEvalContextFloImage;

        m_DerivativeRefToFlo.reset(new FixedPointNumber[m_ParamNumRefToFlo]);
        m_WeightsRefToFlo.reset(new FixedPointNumber[m_ParamNumRefToFlo]);       
        m_DerivativeFloToRef.reset(new FixedPointNumber[m_ParamNumFloToRef]);
        m_WeightsFloToRef.reset(new FixedPointNumber[m_ParamNumFloToRef]);

        m_ParamWeightsRefToFlo.SetSize(supportSizeRefToFlo);
        m_ParamWeightsFloToRef.SetSize(supportSizeFloToRef);
        m_ParamIndicesRefToFlo.SetSize(supportSizeRefToFlo);
        m_ParamIndicesFloToRef.SetSize(supportSizeFloToRef);
    }

    inline void StartIteration()
    {
        //m_WeightRefToFlo = 0;
        //m_WeightFloToRef = 0;
        //m_DistanceRefToFlo = 0;
        //m_DistanceFloToRef = 0;

        // Set derivative and weight accumulators to zero
        memset(m_DerivativeRefToFlo.get(), 0, sizeof(FixedPointNumber) * m_ParamNumRefToFlo);
        memset(m_WeightsRefToFlo.get(), 0, sizeof(FixedPointNumber) * m_ParamNumRefToFlo);
        memset(m_DerivativeFloToRef.get(), 0, sizeof(FixedPointNumber) * m_ParamNumFloToRef);
        memset(m_WeightsFloToRef.get(), 0, sizeof(FixedPointNumber) * m_ParamNumFloToRef);
    }
};

template <class TAssociate, class TImageType, class TTransformType, class TDistType>
class AlphaBSplineRegistrationVADThreader : public itk::DomainThreader<itk::ThreadedIndexedContainerPartitioner, TAssociate>
{
public:
  using Self = AlphaBSplineRegistrationVADThreader;
  using Superclass = itk::DomainThreader<itk::ThreadedIndexedContainerPartitioner, TAssociate>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  // The domain is an index range.
  using DomainType = typename Superclass::DomainType;

  using ImageType = TImageType;
  using TransformType = TTransformType;
  using TransformPointer = typename TransformType::Pointer;

  using DistType = TDistType;
  using DistPointer = typename DistType::Pointer;

  using ThreadStateType = AlphaBSplineRegistrationThreadState<ImageType, DistType, 3U>;

  using PointType = typename ImageType::PointType;

  using DerivativeType = typename TransformType::DerivativeType;
  using WeightsType = typename TransformType::WeightsType;
  using ParameterIndexArrayType = typename TransformType::ParameterIndexArrayType;

  using DistEvalContextType = typename DistType::EvalContextType;
  using DistEvalContextPointer = typename DistEvalContextType::Pointer;

  using PointSamplerType = typename TAssociate::PointSamplerType;
  using PointSamplerPointer = typename PointSamplerType::Pointer;
  using PointSampleType = typename TAssociate::PointSampleType;

  constexpr static unsigned int ImageDimension = TAssociate::ImageDimension;

  // This creates the ::New() method for instantiating the class.
  itkNewMacro(Self);

protected:
  // We need a constructor for the itkNewMacro.
  AlphaBSplineRegistrationVADThreader()
  {
      ;
  }

private:
  std::atomic<unsigned int> m_AtomicPointIndex;

  void
  BeforeThreadedExecution() override
  {
      //std::vector<ThreadStateType>& threadStates = this->m_Associate->m_ThreadData;
    // Resize our per-thread data structures to the number of threads that we
    // are actually going to use.  At this point the number of threads that
    // will be used have already been calculated and are available.  The number
    // of threads used depends on the number of cores or processors available
    // on the current system.  It will also be truncated if, for example, the
    // number of cells in the CellContainer is smaller than the number of cores
    // available.
    //const itk::ThreadIdType numberOfThreads = this->GetNumberOfThreadsUsed();
    // Check here that we have enough thread states
    const itk::ThreadIdType numberOfThreads = this->GetNumberOfWorkUnitsUsed();

    this->m_Associate->InitializeThreadState(numberOfThreads);

    m_AtomicPointIndex = 0U;
  }

  void
  ThreadedExecution(const DomainType & subDomain, const itk::ThreadIdType threadId) override
  {
    // Here we set the number of points which will be sampled in one batch by this threader.
    constexpr unsigned int BATCH_SIZE = 32U;

    // Store local versions of the data
    ThreadStateType* state = &this->m_Associate->m_ThreadData[threadId];
    itk::IndexValueType refToFloSampleCount = static_cast<itk::IndexValueType>(this->m_Associate->m_RefToFloSampleCount);
    itk::IndexValueType floToRefSampleCount = static_cast<itk::IndexValueType>(this->m_Associate->m_FloToRefSampleCount);

    unsigned int supportSizeRefToFlo = state->m_SupportSizeRefToFlo;
    unsigned int supportSizeFloToRef = state->m_SupportSizeFloToRef;

    TransformType* transformRefToFlo = this->m_Associate->m_TransformRefToFloRawPtr;
    TransformType* transformFloToRef = this->m_Associate->m_TransformFloToRefRawPtr;
    PointSamplerType* pointSamplerRef = this->m_Associate->m_PointSamplerRefImage.GetPointer();
    PointSamplerType* pointSamplerFlo = this->m_Associate->m_PointSamplerFloImage.GetPointer();

    FixedPointNumber* derivativeRefToFlo = state->m_DerivativeRefToFlo.get();
    FixedPointNumber* derivativeFloToRef = state->m_DerivativeFloToRef.get();
    FixedPointNumber* weightsRefToFlo = state->m_WeightsRefToFlo.get();
    FixedPointNumber* weightsFloToRef = state->m_WeightsFloToRef.get();

    DistType* distStructRefImage = this->m_Associate->m_DistDataStructRefImage.GetPointer();
    DistType* distStructFloImage = this->m_Associate->m_DistDataStructFloImage.GetPointer();
    DistEvalContextType* distEvalContextRefImage = state->m_DistEvalContextRefImage.GetPointer();
    DistEvalContextType* distEvalContextFloImage = state->m_DistEvalContextFloImage.GetPointer();

    WeightsType& paramWeightsRefToFlo = state->m_ParamWeightsRefToFlo;
    WeightsType& paramWeightsFloToRef = state->m_ParamWeightsFloToRef;
    ParameterIndexArrayType& paramIndicesRefToFlo = state->m_ParamIndicesRefToFlo;
    ParameterIndexArrayType& paramIndicesFloToRef = state->m_ParamIndicesFloToRef;

    // Initialize the thread state for this iteration
    state->StartIteration();

    FixedPointNumber distanceRefToFloAcc = 0;
    FixedPointNumber weightRefToFloAcc = 0;
    FixedPointNumber distanceFloToRefAcc = 0;
    FixedPointNumber weightFloToRefAcc = 0;

    PointSampleType pointSample;

    const double lambda = this->m_Associate->m_SymmetryLambda;

    unsigned int endPointIndex = 0U;
    unsigned int totalCount = refToFloSampleCount + floToRefSampleCount;

    // Either use the dynamic load allocation system, or fall back
    // on the static domain-partitioning mode.
    constexpr bool USE_DYNAMIC_LOAD_ALLOCATION = true;
    
    while(endPointIndex < totalCount)
    {
        unsigned int loopPointIndex;
        unsigned int end1;
        unsigned int end2;

        if(USE_DYNAMIC_LOAD_ALLOCATION)
        {
            loopPointIndex = this->m_AtomicPointIndex.fetch_add(BATCH_SIZE); //, std::memory_order_acq_rel
            endPointIndex = loopPointIndex + BATCH_SIZE;
            if(endPointIndex > totalCount)
            {
                endPointIndex = totalCount;
            }

            end1 = endPointIndex;
        }
        else
        {
            loopPointIndex = subDomain[0];
            end1 = subDomain[1]+1;
            endPointIndex = totalCount;
        }

    end2 = end1;
    if(end1>refToFloSampleCount)
    {
        end1 = refToFloSampleCount;
    }

    for(; loopPointIndex < end1; ++loopPointIndex)
    {
        // Reference to Floating sample
        pointSamplerRef->Sample(loopPointIndex, pointSample);
        //state->m_DistEvalContextFloImage->RestartSampler();

            ComputePointValueAndDerivative(
                pointSample,
                distStructFloImage,
                distEvalContextFloImage,
                transformRefToFlo,
                transformFloToRef,
                distanceRefToFloAcc,
                weightRefToFloAcc,
                derivativeRefToFlo,
                derivativeFloToRef,
                weightsRefToFlo,
                weightsFloToRef,
                paramWeightsRefToFlo,
                paramWeightsFloToRef,
                paramIndicesRefToFlo,
                paramIndicesFloToRef,
                supportSizeRefToFlo,
                supportSizeFloToRef,
                lambda,
                loopPointIndex);
    }
    for(; loopPointIndex < end2; ++loopPointIndex)
    {
        unsigned int pointIndex = loopPointIndex-refToFloSampleCount;// + floToRefSampleCount*localIteration;
        // Floating to Reference sample
        pointSamplerFlo->Sample(pointIndex, pointSample);
        //state->m_DistEvalContextRefImage->RestartSampler();

        ComputePointValueAndDerivative(
                pointSample,
                distStructRefImage,
                distEvalContextRefImage,
                transformFloToRef,
                transformRefToFlo,
                distanceFloToRefAcc,
                weightFloToRefAcc,
                derivativeFloToRef,
                derivativeRefToFlo,
                weightsFloToRef,
                weightsRefToFlo,
                paramWeightsFloToRef,
                paramWeightsRefToFlo,
                paramIndicesFloToRef,
                paramIndicesRefToFlo,
                supportSizeFloToRef,
                supportSizeRefToFlo,
                lambda,
                pointIndex);
      }

    }

    state->m_DistanceRefToFlo = distanceRefToFloAcc;
    state->m_DistanceFloToRef = distanceFloToRefAcc;
    state->m_WeightRefToFlo = weightRefToFloAcc;
    state->m_WeightFloToRef = weightFloToRefAcc;

    unsigned int refSamples = state->m_DistEvalContextRefImage->GetSampleCount();
    state->m_DistEvalContextRefImage->GetSampler()->EndIteration(refSamples * refToFloSampleCount);
    unsigned int floSamples = state->m_DistEvalContextFloImage->GetSampleCount();
    state->m_DistEvalContextFloImage->GetSampler()->EndIteration(floSamples * floToRefSampleCount);
  }

  void
  AfterThreadedExecution() override
  {
    const itk::ThreadIdType numberOfThreads = this->GetNumberOfWorkUnitsUsed();

    this->m_Associate->m_ThreadsUsed = numberOfThreads;
    this->m_Associate->m_PointSamplerRefImage->EndIteration(this->m_Associate->m_RefToFloSampleCount);
    this->m_Associate->m_PointSamplerFloImage->EndIteration(this->m_Associate->m_FloToRefSampleCount);
  }

  inline static void ComputePointValueAndDerivative(
      PointSampleType& pointSample,
      DistType* distStruct,
      DistEvalContextType* distEvalContext,
      TransformType* tfor,
      TransformType* trev,
      FixedPointNumber& value,
      FixedPointNumber& weight,
      FixedPointNumber* dfor,
      FixedPointNumber* drev,
      FixedPointNumber* wfor,
      FixedPointNumber* wrev,
      WeightsType& splineWeightsFor,
      WeightsType& splineWeightsRev,
      ParameterIndexArrayType& parameterIndicesFor,
      ParameterIndexArrayType& parameterIndicesRev,
      unsigned int supportSizeFor,
      unsigned int supportSizeRev,
      double lambda,
      unsigned int pointIndex
      )
  {
    constexpr unsigned int Dim = TAssociate::ImageDimension;

	PointType transformedPoint;
	PointType returnedPoint;
	bool isInside1;
    bool isInside;

    unsigned int paramPerDimFor = tfor->GetNumberOfParametersPerDimension();
    unsigned int paramPerDimRev = trev->GetNumberOfParametersPerDimension();

	tfor->TransformPoint(pointSample.m_Point, transformedPoint, splineWeightsFor, parameterIndicesFor, isInside1);
	if (!isInside1)
		return;

    itk::Matrix<double, Dim, Dim> trevSpatialDerivatives;// = NumericalSpatialDerivativesForward<TransformType, Dim>(trev, transformedPoint, returnedPoint);
    isInside = ApplyBSplineTransformToPointWithSpatialNumericalDerivatives<TransformType, Dim>(
        trev,
        transformedPoint,
        returnedPoint,
        trevSpatialDerivatives,
        splineWeightsRev,
        parameterIndicesRev,
        1e-6
    );

    itk::Vector<double, ImageDimension> symLossVec;
    double symLossValue;
    
    // Compute symmetry loss (value and vector)
    if(!isInside)
    {
	    symLossVec = (transformedPoint - pointSample.m_Point);
    } else
    {
	    symLossVec = (returnedPoint - pointSample.m_Point);
    }
   symLossValue = 0.5 * symLossVec.GetSquaredNorm();

    // Compute the point-to-set distance and gradient here

    double localValue = 0.0;
    itk::Vector<double, Dim> grad;
    bool flag = distStruct->ValueAndDerivative(
        distEvalContext,
        pointIndex,
        transformedPoint,
        pointSample.m_Value,
        localValue,
        grad);

    double w = pointSample.m_Weight;
    double valueW;

    if(!flag || !isInside) {

        valueW = 0.0;
        //if(!isInside)
        //{
        //    return;
        //}
        return;
    } else {
        valueW = pointSample.m_Weight;
    }

    const double invLambda = 1.0;// - lambda;
    const double lambdaWeighted = lambda * w;
    const double invLambdaWeighted = invLambda*w;
    const double invLambdaValueWeighted = invLambda*valueW;
    const double doubleValue = invLambdaValueWeighted * localValue + lambdaWeighted * symLossValue;
    const double doubleWeight = invLambdaValueWeighted + lambdaWeighted;
    value += FixedPointFromDouble(doubleValue);
    weight += FixedPointFromDouble(doubleWeight);

	// Compute jacobian for metric and symmetry loss
	for (unsigned int dim = 0; dim < Dim; ++dim)
	{
		unsigned int offFor = dim * paramPerDimFor;
		const double gradVal = grad[dim];
		const double sltGradVal = symLossVec[dim];

        const double lambdaWeightedSymmetryLossGradVal = lambdaWeighted * sltGradVal;
        const double invLambdaValueWGradVal = invLambdaValueWeighted * gradVal;
        typename WeightsType::ValueType* weightsFor = &splineWeightsFor[0];
        typename ParameterIndexArrayType::ValueType* indicesFor = &parameterIndicesFor[0];

        double sltGradValAcc = 0.0;
        double wacc = 0.0;
        for (unsigned int j = 0; j < Dim; ++j)
        {
            // Check order of trevSpatialDerivatives parameters
            const double tsd_j = trevSpatialDerivatives(dim, j);
            // For 1-norm
            //wacc += fabs(tsd_j);
            // For 2-norm
            wacc += tsd_j*tsd_j;
            const double sltGradVal_j = tsd_j * symLossVec[j];
            sltGradValAcc += sltGradVal_j;
        }
        // For 2-norm
        wacc = sqrt(wacc);

        const double forwardSymmetryGradContribution = lambdaWeighted * sltGradValAcc;
        const double forwardSymmetryWeightContribution = lambdaWeighted * wacc;

        const double dforUnscaled = invLambdaValueWGradVal + forwardSymmetryGradContribution;
        const double wforUnscaled = invLambdaWeighted + forwardSymmetryWeightContribution;
		for (unsigned int mu = 0; mu < supportSizeFor; ++mu)
		{
			unsigned int parInd = offFor + indicesFor[mu];
            double sw = weightsFor[mu];

			dfor[parInd] -= FixedPointFromDouble(dforUnscaled * sw);
			wfor[parInd] += FixedPointFromDouble(wforUnscaled * sw);
		}

        if(isInside)
        {
            typename WeightsType::ValueType* weightsRev = &splineWeightsRev[0];
            typename ParameterIndexArrayType::ValueType* indicesRev = &parameterIndicesRev[0];

    		unsigned int offRev = dim * paramPerDimRev;
		    for (unsigned int mu = 0; mu < supportSizeRev; ++mu)
		    {
			    unsigned int parIndRev = offRev + indicesRev[mu];//parameterIndicesRev[mu];
			    double swInv = weightsRev[mu];//splineWeightsRev[mu];

                drev[parIndRev] -= FixedPointFromDouble(lambdaWeightedSymmetryLossGradVal * swInv);
                wrev[parIndRev] += FixedPointFromDouble(lambdaWeighted * swInv);
		    }
        }
	}
  }
};

template <class TAssociate, class TImageType, class TTransformType, class TDistType>
class AlphaBSplineRegistrationStepThreader : public itk::DomainThreader<itk::ThreadedIndexedContainerPartitioner, TAssociate>
{
public:
  using Self = AlphaBSplineRegistrationStepThreader;
  using Superclass = itk::DomainThreader<itk::ThreadedIndexedContainerPartitioner, TAssociate>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using DomainType = typename Superclass::DomainType;

  using ImageType = TImageType;

  using DistType = TDistType;
  using DistPointer = typename DistType::Pointer;

  using ThreadStateType = AlphaBSplineRegistrationThreadState<ImageType, DistType, 3U>;
  using TransformType = TTransformType;
  using TransformPointer = typename TransformType::Pointer;

  using DerivativeType = typename TransformType::DerivativeType;
  using WeightsType = typename TransformType::WeightsType;
  using ParameterIndexArrayType = typename TransformType::ParameterIndexArrayType;

  // This creates the ::New() method for instantiating the class.
  itkNewMacro(Self);

protected:
  // We need a constructor for the itkNewMacro.
  AlphaBSplineRegistrationStepThreader() = default;

private:
  std::atomic<unsigned int> m_AtomicPointIndex;
  char pad[60];
  unsigned int m_TotalCount;

  void
  BeforeThreadedExecution() override
  {
    //const itk::ThreadIdType numberOfThreads = this->GetNumberOfThreadsUsed();
    const itk::ThreadIdType numberOfThreads = this->GetNumberOfWorkUnitsUsed();

    m_AtomicPointIndex = 0U;
    m_TotalCount = this->m_Associate->m_TransformRefToFlo->GetNumberOfParameters() + this->m_Associate->m_TransformFloToRef->GetNumberOfParameters();
  }

  void
  ThreadedExecution(const DomainType & subDomain, const itk::ThreadIdType threadId) override
  {
    // Here we set the number of transform parameter which will be processed in one batch by this threader.
    constexpr unsigned int BATCH_SIZE = 8U;

    // The number of threads used for the previous step
    unsigned int threadsUsed = this->m_Associate->m_ThreadsUsed;
    unsigned int paramNumRefToFlo = this->m_Associate->m_TransformRefToFlo->GetNumberOfParameters();
    const double momentum = this->m_Associate->m_Momentum;
    const double invMomentum = 1.0 - momentum;
    constexpr double eps = 0.01;

    DerivativeType& derRefToFlo = this->m_Associate->m_DerivativeRefToFlo;
    DerivativeType& derFloToRef = this->m_Associate->m_DerivativeFloToRef;
    ThreadStateType* threadState = this->m_Associate->m_ThreadData.get();

    if(threadId == 0)
    {
        FixedPointNumber refToFloValue = 0;
        FixedPointNumber floToRefValue = 0;
        FixedPointNumber refToFloWeight = 0;
        FixedPointNumber floToRefWeight = 0;
        for(unsigned int i = 0; i < threadsUsed; ++i) {
            refToFloValue += threadState[i].m_DistanceRefToFlo;
            floToRefValue += threadState[i].m_DistanceFloToRef;
            refToFloWeight += threadState[i].m_WeightRefToFlo;
            floToRefWeight += threadState[i].m_WeightFloToRef;
        }

        this->m_Associate->m_Value = 0.5 * (
          DoubleFromFixedPoint(refToFloValue)/(DoubleFromFixedPoint(refToFloWeight) + eps) +
          DoubleFromFixedPoint(floToRefValue)/(DoubleFromFixedPoint(floToRefWeight) + eps));
    }

    unsigned int localPointIndex = 0U;
    unsigned int localTotalCount = this->m_TotalCount;

    while(localPointIndex < localTotalCount)
    {
        unsigned int startPointIndex = this->m_AtomicPointIndex.fetch_add(BATCH_SIZE, std::memory_order_acq_rel);
        unsigned int endPointIndex = startPointIndex + BATCH_SIZE;
        if(endPointIndex > localTotalCount)
        {
            endPointIndex = localTotalCount;
        }
        localPointIndex = endPointIndex;

    itk::IndexValueType ii = startPointIndex;
    itk::IndexValueType end1 = endPointIndex;

    if(end1 > paramNumRefToFlo)
        end1 = paramNumRefToFlo;

    itk::IndexValueType end2 = endPointIndex;

    for (; ii < end1; ++ii)
    {
        itk::IndexValueType localIndex = ii;

        FixedPointNumber derVal = 0;
        FixedPointNumber weightVal = 0;
        for(unsigned int i = 0; i < threadsUsed; ++i) {
            derVal += threadState[i].m_DerivativeRefToFlo[localIndex];
            weightVal += threadState[i].m_WeightsRefToFlo[localIndex];
        }

        const double prevVal = derRefToFlo[localIndex];
        const double normalized = DoubleFromFixedPoint(derVal) / (DoubleFromFixedPoint(weightVal)+eps);
        derRefToFlo[localIndex] = prevVal * momentum + invMomentum * normalized;
    }
    for (; ii < end2; ++ii)
    {
        itk::IndexValueType localIndex = ii - paramNumRefToFlo;

        FixedPointNumber derVal = 0;
        FixedPointNumber weightVal = 0;
        for(unsigned int i = 0; i < threadsUsed; ++i) {
            derVal += threadState[i].m_DerivativeFloToRef[localIndex];
            weightVal += threadState[i].m_WeightsFloToRef[localIndex];
        }

        const double prevVal = derFloToRef[localIndex];
        const double normalized = DoubleFromFixedPoint(derVal) / (DoubleFromFixedPoint(weightVal)+eps);
        derFloToRef[localIndex] = prevVal * momentum + invMomentum * normalized;
    }

    }
  }

  void
  AfterThreadedExecution() override
  {
    //const itk::ThreadIdType numberOfThreads = this->GetNumberOfThreadsUsed();
    const itk::ThreadIdType numberOfThreads = this->GetNumberOfWorkUnitsUsed();

    this->m_Associate->m_ThreadsUsed = numberOfThreads;
  }
};

template <typename TImageType, typename TDistType, unsigned int TSplineOrder=3>
class AlphaBSplineRegistration : public itk::Object {
public:
    using Self = AlphaBSplineRegistration<TImageType, TDistType, TSplineOrder>;
    using Superclass = itk::Object;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    
    constexpr static unsigned int ImageDimension = TImageType::ImageDimension;
    constexpr static unsigned int SplineOrder = TSplineOrder;

    using ImageType = TImageType;
    typedef typename ImageType::Pointer ImagePointer;
    using DistType = TDistType;
    
    typedef typename ImageType::ValueType ValueType;
    
    typedef typename ImageType::SpacingType SpacingType;
    typedef typename ImageType::RegionType RegionType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename ImageType::IndexValueType IndexValueType;
    typedef typename ImageType::PointType PointType;

    using DistPointer = typename DistType::Pointer;

    using DistEvalContextType = typename DistType::EvalContextType;
    using DistEvalContextPointer = typename DistEvalContextType::Pointer;

	using TransformType = itk::BSplineTransform<double, ImageDimension, SplineOrder>;
	using TransformPointer = typename TransformType::Pointer;
    using DerivativeType = typename TransformType::DerivativeType;

    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, ImageDimension> , ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;
    using PointSampleType = PointSample<ImageType, ImageType>;

    using VADThreaderType = AlphaBSplineRegistrationVADThreader<Self, ImageType, TransformType, DistType>;
    using VADThreaderPointer = typename VADThreaderType::Pointer;
    using StepThreaderType = AlphaBSplineRegistrationStepThreader<Self, ImageType, TransformType, DistType>;
    using StepThreaderPointer = typename StepThreaderType::Pointer;

    using ThreadStateType = AlphaBSplineRegistrationThreadState<ImageType, DistType, 3U>;

    itkNewMacro(Self);

    itkTypeMacro(AlphaBSplineRegistration, itk::Object);

    virtual TransformPointer GetTransformRefToFlo() const
    {
        return m_TransformRefToFlo;
    }

    virtual TransformPointer GetTransformFloToRef() const
    {
        return m_TransformFloToRef;
    }

    virtual void SetTransformRefToFlo(TransformPointer transform)
    {
        m_TransformRefToFlo = transform;
        m_TransformRefToFloRawPtr = m_TransformRefToFlo.GetPointer();
    }

    virtual void SetTransformFloToRef(TransformPointer transform)
    {
        m_TransformFloToRef = transform;
        m_TransformFloToRefRawPtr = m_TransformFloToRef.GetPointer();
    }

    virtual void SetPointSamplerRefImage(PointSamplerPointer sampler)
    {
        m_PointSamplerRefImage = sampler;
    }
    
    virtual void SetPointSamplerFloImage(PointSamplerPointer sampler)
    {
        m_PointSamplerFloImage = sampler;
    }

    virtual void SetDistDataStructRefImage(DistPointer dist)
    {
        m_DistDataStructRefImage = dist;
    }

    virtual void SetDistDataStructFloImage(DistPointer dist)
    {
        m_DistDataStructFloImage = dist;
    }

    virtual void SetLearningRate(double learningRate)
    {
        m_LearningRate = learningRate;
    }

    virtual void SetMomentum(double momentum)
    {
        m_Momentum = momentum;
    }

    virtual void SetSymmetryLambda(double symmetryLambda)
    {
        m_SymmetryLambda = symmetryLambda;
    }

    virtual void SetIterations(unsigned int iterations)
    {
        m_Iterations = iterations;
    }

    virtual void SetSampleCountRefToFlo(unsigned int count)
    {
        m_RefToFloSampleCount = count;
    }

    virtual void SetSampleCountFloToRef(unsigned int count)
    {
        m_FloToRefSampleCount = count;
    }

    virtual void AddCallback(Command* cmd)
    {
        m_Callbacks.push_back(cmd);
    }

    virtual void SetPrintInterval(unsigned int interval) {
        m_PrintInterval = interval;
    }

    virtual double GetValue() const
    {
        return m_Value;
    }

    virtual void Initialize()
    {
        assert(m_TransformRefToFlo.GetPointer() != nullptr);
        assert(m_TransformFloToRef.GetPointer() != nullptr);

        m_DerivativeRefToFlo.SetSize(m_TransformRefToFlo->GetNumberOfParameters());
        m_DerivativeFloToRef.SetSize(m_TransformFloToRef->GetNumberOfParameters());
        m_DerivativeRefToFlo.Fill(0.0);
        m_DerivativeFloToRef.Fill(0.0);

//    void Initialize(unsigned int paramNumRefToFlo, unsigned int paramNumFloToRef, unsigned int supportSize, DistEvalContextPointer distEvalContextRefImage, DistEvalContextPointer distEvalContextFloImage)

        m_ThreadsAllocated = 0U;

        unsigned int globalMaximumThreads = itk::MultiThreaderBase::GetGlobalMaximumNumberOfThreads();

        m_ThreadData.reset(new ThreadStateType[globalMaximumThreads]); //new ThreadStateType[128U]);

        InitializeThreadState(1);
    }

    virtual void InitializeThreadState(unsigned int threads)
    {
        assert(itk::MultiThreaderBase::GetGlobalMaximumNumberOfThreads() >= threads);

        for(unsigned int i = m_ThreadsAllocated; i < threads; ++i)
        {
            unsigned int supSizeRefToFlo = m_TransformRefToFlo->GetNumberOfAffectedWeights();
            unsigned int supSizeFloToRef = m_TransformFloToRef->GetNumberOfAffectedWeights();

            DistEvalContextPointer evalContext1 = m_DistDataStructRefImage->MakeEvalContext();
            DistEvalContextPointer evalContext2 = m_DistDataStructFloImage->MakeEvalContext();
            m_ThreadData[i].Initialize(
                m_TransformRefToFlo->GetNumberOfParameters(),
                m_TransformFloToRef->GetNumberOfParameters(),
                supSizeRefToFlo,
                supSizeFloToRef,
                evalContext1,
                evalContext2
            );
        }
        if (threads > m_ThreadsAllocated)
        {
            m_ThreadsAllocated = threads;
        }
    }

    virtual void Run()
    {
        unsigned int iterations = m_Iterations;
        for(unsigned int i = 0; i < iterations; ++i)
        {
            //std::cout << "Iteration: " << i << std::endl;

            // Compute the distance value and derivatives for a sampled subset of the image
            typename VADThreaderType::DomainType completeDomain1;
            completeDomain1[0] = 0;
            completeDomain1[1] = this->m_RefToFloSampleCount + this->m_FloToRefSampleCount - 1;
            this->m_VADThreader->SetMaximumNumberOfThreads(itk::MultiThreaderBase::GetGlobalMaximumNumberOfThreads());
            this->m_VADThreader->SetNumberOfWorkUnits(itk::MultiThreaderBase::GetGlobalMaximumNumberOfThreads());
            this->m_VADThreader->Execute(this, completeDomain1);

            // Aggregate, normalize, and apply a step counter to the gradient direction
            
            typename StepThreaderType::DomainType completeDomain2;
            completeDomain2[0] = 0;
            completeDomain2[1] = m_TransformRefToFlo->GetNumberOfParameters() + m_TransformFloToRef->GetNumberOfParameters() - 1;
            this->m_StepThreader->SetMaximumNumberOfThreads(itk::MultiThreaderBase::GetGlobalMaximumNumberOfThreads());
            this->m_StepThreader->SetNumberOfWorkUnits(itk::MultiThreaderBase::GetGlobalMaximumNumberOfThreads());
            this->m_StepThreader->Execute(this, completeDomain2);           

            m_TransformRefToFlo->UpdateTransformParameters(m_DerivativeRefToFlo, m_LearningRate);
            m_TransformFloToRef->UpdateTransformParameters(m_DerivativeFloToRef, m_LearningRate);

            for(size_t k = 0; k < m_Callbacks.size(); ++k)
            {
                m_Callbacks[k]->Invoke();
            }

            if(m_PrintInterval > 0U && i % m_PrintInterval == 0)
            {
                std::cout << "[" << i << "] Value: " << m_Value << std::endl;
            }
        }
        //chronometer.Report(std::cout);
    }   
protected:
    AlphaBSplineRegistration()
    {
        m_Value = 0.0;
        m_Momentum = 0.1;
        m_LearningRate = 1.0;
        m_SymmetryLambda = 0.05;
        m_ThreadsAllocated = 0;

        m_RefToFloSampleCount = 4096;
        m_FloToRefSampleCount = 4096;

        m_Iterations = 300;
        m_PrintInterval = 0;

        m_VADThreader = VADThreaderType::New();
        m_StepThreader = StepThreaderType::New();
    }

    // Transformations
    TransformPointer m_TransformRefToFlo;
    TransformPointer m_TransformFloToRef;
    TransformType* m_TransformRefToFloRawPtr;
    TransformType* m_TransformFloToRefRawPtr;

    // Point samplers
    PointSamplerPointer m_PointSamplerRefImage;
    PointSamplerPointer m_PointSamplerFloImage;

    // Target distance data structures
    DistPointer m_DistDataStructRefImage;
    DistPointer m_DistDataStructFloImage;

    // Objective function and optimization parameters
    double m_Momentum;
    double m_LearningRate;
    double m_SymmetryLambda;

    // Sampling parameters
    unsigned int m_RefToFloSampleCount;
    unsigned int m_FloToRefSampleCount;

    // Output and reporting settings
    unsigned int m_PrintInterval;

    // Current state
    double m_Value;

    // Thread-local data for value and derivatives computation
    std::unique_ptr<ThreadStateType[]> m_ThreadData;

    // Threads used is assigned by the VAD threader and used by the Step threader
    unsigned int m_ThreadsUsed;
    // The number of thread states which have been initialized
    unsigned int m_ThreadsAllocated;

    // Threaders
    VADThreaderPointer m_VADThreader;
    StepThreaderPointer m_StepThreader;

    unsigned int m_Iterations;

    std::vector<Command*> m_Callbacks;

    // Current/last derivative
    DerivativeType m_DerivativeRefToFlo;
    DerivativeType m_DerivativeFloToRef;

    friend class AlphaBSplineRegistrationVADThreader<Self, ImageType, TransformType, DistType>;
    friend class AlphaBSplineRegistrationStepThreader<Self, ImageType, TransformType, DistType>;
};

#endif
