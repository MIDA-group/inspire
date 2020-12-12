
// The main component of a registration framework
// based on distance measures between fuzzy sets.

#ifndef ALPHA_LINEAR_REGISTRATION_H
#define ALPHA_LINEAR_REGISTRATION_H

#include "itkImage.h"
#include "itkDomainThreader.h"
#include "itkThreadedIndexedContainerPartitioner.h"
#include "itkTransform.h"
#include "itkRigid2DTransform.h"

#include <atomic>

#include "../common/quantization.h"
#include "../common/command.h"

// Point sampler methods
#include "../samplers/pointSamplerBase.h"

template <typename TTransformType>
void TransformStepOne(
    TTransformType* transform,
    unsigned int paramIndex,
    unsigned int paramCount,
    double step,
    itk::Array2D<double>& out) {
    
    using DerivativeType = typename TTransformType::DerivativeType;

    DerivativeType derivative(paramCount);

    derivative.Fill(0.0);
    derivative[paramIndex] = step;

    auto transform1 = transform->Clone();
    auto transform2 = transform->Clone();

    transform1->UpdateTransformParameters(derivative, 1.0);

    auto inverse1 = transform1->GetInverseTransform();

    transform2->UpdateTransformParameters(derivative, -1.0);

    auto inverse2 = transform2->GetInverseTransform();

    if (!inverse1 || !inverse2) {
        std::cerr << "Not invertible" << std::endl;
        return;
    }

    auto param1 = inverse1->GetParameters();
    auto param2 = inverse2->GetParameters();

    //double acc = 0.0;
    for (unsigned int i = 0; i < paramCount; ++i)
    {
        double diff_i = (param1[i] - param2[i]) / (2.0 * step);
        /*if (diff_i >= 0.0)
            diff_i = 1.0;
        else
            diff_i = -1.0;*/
        //acc += fabs(diff_i);
        out.SetElement(i, paramIndex, diff_i);
    }
}

template <typename TTransformType>
void InverseDerivativeMatrix(TTransformType* transform, double step, itk::Array2D<double>& out) {
    unsigned int count = transform->GetNumberOfParameters();

    out.Fill(0.0);

    for (unsigned int i = 0; i < count; ++i) {
        TransformStepOne<TTransformType>(transform, i, count, step, out);
    }

    //std::cerr << out << std::endl;
}

template <typename TImageType, typename TDistType, typename TTransformType>
struct AlphaLinearRegistrationThreadState
{
    using ImageType = TImageType;
    using ImagePointer = typename ImageType::Pointer;

    constexpr static unsigned int ImageDimension = ImageType::ImageDimension;

	using TransformType = TTransformType;
	using TransformPointer = typename TransformType::Pointer;
    using DerivativeType = typename TransformType::DerivativeType;

    using JacobianType = typename TransformType::JacobianType;

    using DistType = TDistType;
    using DistPointer = typename DistType::Pointer;

    using DistEvalContextType = typename DistType::EvalContextType;
    using DistEvalContextPointer = typename DistEvalContextType::Pointer;

    std::unique_ptr<FixedPointNumber[]> m_DerivativeRefToFlo;
    std::unique_ptr<FixedPointNumber[]> m_DerivativeFloToRef;
    std::unique_ptr<FixedPointNumber[]> m_WeightsRefToFlo;
    std::unique_ptr<FixedPointNumber[]> m_WeightsFloToRef;

    JacobianType m_ParamJacobianRefToFlo;
    JacobianType m_ParamJacobianFloToRef;

    FixedPointNumber m_DistanceRefToFlo;
    FixedPointNumber m_DistanceFloToRef;
    FixedPointNumber m_WeightRefToFlo;
    FixedPointNumber m_WeightFloToRef;

    unsigned int m_ParamNumRefToFlo;
    unsigned int m_ParamNumFloToRef;

    DistEvalContextPointer m_DistEvalContextRefImage;
    DistEvalContextPointer m_DistEvalContextFloImage;

    void Initialize(
        unsigned int paramNumRefToFlo,
        unsigned int paramNumFloToRef,
        DistEvalContextPointer distEvalContextRefImage,
        DistEvalContextPointer distEvalContextFloImage)
    {
        m_ParamNumRefToFlo = paramNumRefToFlo;
        m_ParamNumFloToRef = paramNumFloToRef;

        m_DistEvalContextRefImage = distEvalContextRefImage;
        m_DistEvalContextFloImage = distEvalContextFloImage;

        m_DerivativeRefToFlo.reset(new FixedPointNumber[m_ParamNumRefToFlo]);
        m_WeightsRefToFlo.reset(new FixedPointNumber[m_ParamNumRefToFlo]);       
        m_DerivativeFloToRef.reset(new FixedPointNumber[m_ParamNumFloToRef]);
        m_WeightsFloToRef.reset(new FixedPointNumber[m_ParamNumFloToRef]);

        m_ParamJacobianRefToFlo.SetSize(TImageType::ImageDimension, m_ParamNumRefToFlo);
        m_ParamJacobianFloToRef.SetSize(TImageType::ImageDimension, m_ParamNumFloToRef);
    }

    inline void StartIteration()
    {
        // Set derivative and weight accumulators to zero
        memset(m_DerivativeRefToFlo.get(), 0, sizeof(FixedPointNumber) * m_ParamNumRefToFlo);
        memset(m_WeightsRefToFlo.get(), 0, sizeof(FixedPointNumber) * m_ParamNumRefToFlo);
        memset(m_DerivativeFloToRef.get(), 0, sizeof(FixedPointNumber) * m_ParamNumFloToRef);
        memset(m_WeightsFloToRef.get(), 0, sizeof(FixedPointNumber) * m_ParamNumFloToRef);
    }
};

template <class TAssociate, class TImageType, class TTransformType, class TDistType>
class AlphaLinearRegistrationVADThreader : public itk::DomainThreader<itk::ThreadedIndexedContainerPartitioner, TAssociate>
{
public:
  using Self = AlphaLinearRegistrationVADThreader;
  using Superclass = itk::DomainThreader<itk::ThreadedIndexedContainerPartitioner, TAssociate>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  // The domain is an index range.
  using DomainType = typename Superclass::DomainType;

  using ImageType = TImageType;
  using TransformType = TTransformType;
  using TransformPointer = typename TransformType::Pointer;
  using InverseTransformType = TransformType;
  using InverseTransformPointer = TransformPointer;
  //using InverseTransformType = typename TTransformType::InverseTransformBaseType;
  //using InverseTransformPointer = typename TTransformType::InverseTransformBasePointer;

  using DistType = TDistType;
  using DistPointer = typename DistType::Pointer;

  using ThreadStateType = AlphaLinearRegistrationThreadState<ImageType, DistType, TransformType>;

  using PointType = typename ImageType::PointType;

  using DerivativeType = typename TransformType::DerivativeType;
  using JacobianType = typename TransformType::JacobianType;

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
  AlphaLinearRegistrationVADThreader()
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

    TransformPointer transformRefToFloCopy = this->m_Associate->m_TransformRefToFlo->Clone();
    InverseTransformPointer transformFloToRefCopy = this->m_Associate->m_TransformFloToRef->Clone();

    TransformType* transformRefToFlo = transformRefToFloCopy.GetPointer();
    InverseTransformType* transformFloToRef = transformFloToRefCopy.GetPointer();//this->m_Associate->m_TransformFloToRefRawPtr;
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

    JacobianType& paramJacobianRefToFlo = state->m_ParamJacobianRefToFlo;
    JacobianType& paramJacobianFloToRef = state->m_ParamJacobianFloToRef;

    // Initialize the thread state for this iteration
    state->StartIteration();

    FixedPointNumber distanceRefToFloAcc = 0;
    FixedPointNumber weightRefToFloAcc = 0;
    FixedPointNumber distanceFloToRefAcc = 0;
    FixedPointNumber weightFloToRefAcc = 0;

    PointSampleType pointSample;

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
                distanceRefToFloAcc,
                weightRefToFloAcc,
                derivativeRefToFlo,
                weightsRefToFlo,
                paramJacobianRefToFlo,
                loopPointIndex);
    }
    for(; loopPointIndex < end2; ++loopPointIndex)
    {
        unsigned int pointIndex = loopPointIndex-refToFloSampleCount;// + floToRefSampleCount*localIteration;
        // Floating to Reference sample
        pointSamplerFlo->Sample(pointIndex, pointSample);

        ComputePointValueAndDerivative(
                pointSample,
                distStructRefImage,
                distEvalContextRefImage,
                transformFloToRef,
                distanceFloToRefAcc,
                weightFloToRefAcc,
                derivativeFloToRef,
                weightsFloToRef,
                paramJacobianFloToRef,
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
      itk::Transform<double, ImageDimension, ImageDimension>* tfor,
      FixedPointNumber& value,
      FixedPointNumber& weight,
      FixedPointNumber* dfor,
      FixedPointNumber* wfor,
      JacobianType& jacobianFor,
      unsigned int pointIndex
      )
  {
    constexpr unsigned int Dim = TAssociate::ImageDimension;

	PointType transformedPoint;

	transformedPoint = tfor->TransformPoint(pointSample.m_Point);

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

    if(!flag) {
        return;
    }
    
    double w = pointSample.m_Weight;
    double valueW = pointSample.m_Weight;

    const double doubleValue = valueW * localValue;
    const double doubleWeight = valueW;
    value += FixedPointFromDouble(doubleValue);
    weight += FixedPointFromDouble(doubleWeight);

    const unsigned int paramCount = tfor->GetNumberOfParameters();

    // Compute jacobian
    tfor->ComputeJacobianWithRespectToParameters(pointSample.m_Point, jacobianFor);

	// Compute jacobian for metric
    for (unsigned int mu = 0; mu < paramCount; ++mu)
	{
        double d_mu = 0.0;
        double w_mu = 0.0;

	    for (unsigned int dim = 0; dim < Dim; ++dim)
    	{
	    	const double gradVal = grad[dim];

            const double valueWGradVal = doubleWeight * gradVal;
            double sw = jacobianFor.GetElement(dim, mu);

            d_mu += valueWGradVal * sw;
            w_mu += doubleWeight;// * fabs(sw);
		}

		dfor[mu] -= FixedPointFromDouble(d_mu);
		wfor[mu] += FixedPointFromDouble(w_mu);
	}
  }
};

template <class TAssociate, class TImageType, class TTransformType, class TDistType>
class AlphaLinearRegistrationStepThreader : public itk::DomainThreader<itk::ThreadedIndexedContainerPartitioner, TAssociate>
{
public:
  using Self = AlphaLinearRegistrationStepThreader;
  using Superclass = itk::DomainThreader<itk::ThreadedIndexedContainerPartitioner, TAssociate>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using DomainType = typename Superclass::DomainType;

  using ImageType = TImageType;

  using DistType = TDistType;
  using DistPointer = typename DistType::Pointer;

  using TransformType = TTransformType;
  using TransformPointer = typename TransformType::Pointer;
  using InverseTransformType = TransformType;
  using InverseTransformPointer = TransformPointer;
  //using InverseTransformType = typename TTransformType::InverseTransformBaseType;
  //using InverseTransformPointer = typename TTransformType::InverseTransformBasePointer;

  using ThreadStateType = AlphaLinearRegistrationThreadState<ImageType, DistType, TransformType>;

  using DerivativeType = typename TransformType::DerivativeType;

  // This creates the ::New() method for instantiating the class.
  itkNewMacro(Self);

protected:
  // We need a constructor for the itkNewMacro.
  AlphaLinearRegistrationStepThreader() = default;

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
    DerivativeType& derRefToFloMemory = this->m_Associate->m_DerivativeMemoryRefToFlo;
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

        const double prevVal = derRefToFloMemory[localIndex];
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

template <typename TImageType, typename TDistType, typename TTransformType>
class AlphaLinearRegistration : public itk::Object {
public:
    using Self = AlphaLinearRegistration<TImageType, TDistType, TTransformType>;
    using Superclass = itk::Object;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    
    constexpr static unsigned int ImageDimension = TImageType::ImageDimension;

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

	using TransformType = TTransformType;
	using TransformPointer = typename TransformType::Pointer;
    using InverseTransformType = TransformType;
    using InverseTransformPointer = TransformPointer;
    //using InverseTransformType = typename TTransformType::InverseTransformBaseType;
    //using InverseTransformPointer = typename TTransformType::InverseTransformBasePointer;
    using DerivativeType = typename TransformType::DerivativeType;

    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, ImageDimension> , ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;
    using PointSampleType = PointSample<ImageType, ImageType>;

    using VADThreaderType = AlphaLinearRegistrationVADThreader<Self, ImageType, TransformType, DistType>;
    using VADThreaderPointer = typename VADThreaderType::Pointer;
    using StepThreaderType = AlphaLinearRegistrationStepThreader<Self, ImageType, TransformType, DistType>;
    using StepThreaderPointer = typename StepThreaderType::Pointer;

    using ThreadStateType = AlphaLinearRegistrationThreadState<ImageType, DistType, TransformType>;

    itkNewMacro(Self);

    itkTypeMacro(AlphaLinearRegistration, itk::Object);

    virtual TransformPointer GetTransformRefToFlo() const
    {
        return m_TransformRefToFlo;
    }

    virtual InverseTransformPointer GetTransformFloToRef() const
    {
        return m_TransformFloToRef;
    }

    virtual void SetTransformRefToFlo(TransformPointer transform)
    {
        m_TransformRefToFlo = transform;
        m_TransformRefToFloRawPtr = m_TransformRefToFlo.GetPointer();

        m_DerivativeScaling.SetSize(m_TransformRefToFlo->GetNumberOfParameters());
        m_DerivativeScaling.Fill(1.0);

        m_TransformFloToRef = dynamic_cast<TransformType *>((transform->GetInverseTransform()).GetPointer());
        // What to do here?
        if(!m_TransformFloToRef) {
            m_TransformFloToRef = transform->Clone();
        }

        m_TransformFloToRefRawPtr = m_TransformFloToRef.GetPointer();
    }

    virtual void SetParameterScaling(const DerivativeType& derivativeScaling)
    {
        for (unsigned int i = 0; i < m_TransformRefToFlo->GetNumberOfParameters(); ++i)
        {
            m_DerivativeScaling[i] = derivativeScaling[i];
        }
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
        m_DerivativeMemoryRefToFlo.SetSize(m_TransformRefToFlo->GetNumberOfParameters());
        m_DerivativeMemoryRefToFlo.Fill(0.0);

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
            std::cerr << "Allocating thread " << i << std::endl;
            DistEvalContextPointer evalContext1 = m_DistDataStructRefImage->MakeEvalContext();
            DistEvalContextPointer evalContext2 = m_DistDataStructFloImage->MakeEvalContext();

            m_ThreadData[i].Initialize(
                m_TransformRefToFlo->GetNumberOfParameters(),
                m_TransformFloToRef->GetNumberOfParameters(),
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
            InverseTransformPointer invMaybe = dynamic_cast<TransformType *>((m_TransformRefToFlo->GetInverseTransform()).GetPointer());//m_TransformRefToFlo->GetInverseTransform();
            bool invertible = false;
            if (invMaybe)
            {
                m_TransformFloToRef = invMaybe;
                m_TransformFloToRefRawPtr = m_TransformFloToRef.GetPointer();
                invertible = true;
            }

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

            // Make a prototype that allocates an Array2d every time... fix later

            unsigned int count1 = m_TransformRefToFlo->GetNumberOfParameters();
            unsigned int count2 = m_TransformFloToRef->GetNumberOfParameters();

            if (invertible) {
                itk::Array2D<double> invMatrix(count1, count2);
                InverseDerivativeMatrix<TransformType>(m_TransformRefToFlo.GetPointer(), 1e-7, invMatrix);

                for (unsigned int j = 0; j < count1; ++j) {
                    double acc = 0.0;
                    for (unsigned int k = 0; k < count2; ++k) {
                        double w_k = invMatrix.GetElement(k, j);
                        acc += w_k * m_DerivativeFloToRef[k];
                    }
                    m_DerivativeRefToFlo[j] = 0.5 * (m_DerivativeRefToFlo[j] + acc);
                }
            } else {
                std::cerr << "Not invertible" << std::endl;
            }

            // Scale parameters
            
            for (unsigned int j = 0; j < m_TransformRefToFlo->GetNumberOfParameters(); ++j)
            {
                m_DerivativeMemoryRefToFlo[j] = m_DerivativeRefToFlo[j];
                m_DerivativeRefToFlo[j] *= m_DerivativeScaling[j];
            }

            m_TransformRefToFlo->UpdateTransformParameters(m_DerivativeRefToFlo, m_LearningRate);
            
            for(size_t k = 0; k < m_Callbacks.size(); ++k)
            {
                m_Callbacks[k]->Invoke();
            }

            if(m_PrintInterval > 0U && i % m_PrintInterval == 0)
            {
                std::cout << m_TransformRefToFlo->GetParameters() << std::endl;
                std::cout << "[" << i << "] Value: " << m_Value << std::endl;
            }
        }

        // Compute the final inverse (flo to ref) transformation
        InverseTransformPointer finalInvMaybe = dynamic_cast<TransformType *>((m_TransformRefToFlo->GetInverseTransform()).GetPointer());
        if (finalInvMaybe)
        {
            m_TransformFloToRef = finalInvMaybe;
            m_TransformFloToRefRawPtr = m_TransformFloToRef.GetPointer();
        }
    }   
protected:
    AlphaLinearRegistration()
    {
        m_Value = 0.0;
        m_Momentum = 0.1;
        m_LearningRate = 1.0;
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
    InverseTransformPointer m_TransformFloToRef;
    TransformType* m_TransformRefToFloRawPtr;
    InverseTransformType* m_TransformFloToRefRawPtr;

    // Point samplers
    PointSamplerPointer m_PointSamplerRefImage;
    PointSamplerPointer m_PointSamplerFloImage;

    // Target distance data structures
    DistPointer m_DistDataStructRefImage;
    DistPointer m_DistDataStructFloImage;

    // Objective function and optimization parameters
    double m_Momentum;
    double m_LearningRate;

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
    DerivativeType m_DerivativeMemoryRefToFlo;
    
    DerivativeType m_DerivativeScaling;

    friend class AlphaLinearRegistrationVADThreader<Self, ImageType, TransformType, DistType>;
    friend class AlphaLinearRegistrationStepThreader<Self, ImageType, TransformType, DistType>;
};

#endif
