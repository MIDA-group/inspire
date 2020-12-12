
/**
 * --- INSPIRE ---
 * Author: Johan Ofverstedt
 * 
 * The core application in INSPIRE. Registers two images
 * first (optionally) with affine registration, followed
 * by deformable parametric B-spline registration based
 * on d_{\alpha AMD} distances.
 */

#include <thread>
#include <fstream>
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

#include <chrono>

#include "itkVersion.h"
#include "itkMultiThreaderBase.h"

#include "itkPNGImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"

#include "../common/progress.h"

void RegisterIOFactories() {
    itk::PNGImageIOFactory::RegisterOneFactory();
    itk::NiftiImageIOFactory::RegisterOneFactory();
}

struct ProgramConfig {
    std::string affineConfigPath;
    std::string configPath;
    unsigned int workers;
    bool rerun;
    std::string refImagePath;
    std::string floImagePath;
    std::string refMaskPath;
    std::string floMaskPath;
    std::string outPathAffineForward;
    std::string outPathAffineReverse;
    std::string outPathForward;
    std::string outPathReverse;

    // Animation related
    std::string  animOutPath;
    std::string  animFormat;
    bool         anim16BitMode;
    std::string  animRenderMode;
    unsigned int animFreq;
    unsigned int animDownsampling;
    std::string  animRefImagePath;
    std::string  animFloImagePath;
};

void printTitle()
{
    std::cout << "INSPIRE (Intensity and Spatial Information-Based Deformable Image Registration)" << std::endl;
}
void printHelp(const char* binPath)
{
    const char *spc = "  ";
    std::cout << "Usage: " << binPath << " dim (parameters)" << std::endl;
    std::cout << spc << "-ref : reference image path [REQUIRED]" << std::endl;
    std::cout << spc << "-flo : floating image path [REQUIRED]" << std::endl;
    std::cout << spc << "-out_path_forward : path of the output deformable transformation (ref to flo) [REQUIRED]" << std::endl;
    std::cout << spc << "-out_path_reverse : path of the output deformable transformation (flo to ref) [REQUIRED]" << std::endl;
    std::cout << spc << "-cfg : (or deform_cfg) path of the config file for the deformable registration  [REQUIRED]" << std::endl;
    std::cout << spc << "-affine_cfg : path of the config file for the affine registration" << std::endl;
    std::cout << spc << "-ref_mask : reference mask image path" << std::endl;
    std::cout << spc << "-flo_mask : floating mask image path" << std::endl;
    std::cout << spc << "-out_path_affine_forward : path of the output affine transformation (ref to flo)" << std::endl;
    std::cout << spc << "-out_path_affine_reverse : path of the output affine transformation (flo to ref)" << std::endl;
    std::cout << spc << "-workers : number of worker threads to use" << std::endl;
    //TODO: Add animation system parameters here
}

/**
 * Command line arguments
 * binpath dim cfgPath refImagePath floImagePath outPathForward outPathReverse (-refmask refMaskPath) (-flomaskpath floMaskPath) (-workers 6)
 */

bool readKeyValuePairForProgramConfig(int argc, char** argv, int startIndex, ProgramConfig& cfg) {
    assert(startIndex + 1 < argc);

    std::string key = argv[startIndex];
    std::string value = argv[startIndex + 1];

    if (key == "-ref") {
        cfg.refImagePath = value;
    } else if (key == "-flo") {
        cfg.floImagePath = value;
    } else if (key == "-affine_cfg") {
        cfg.affineConfigPath = value;
    } else if ((key == "-cfg") || (key == "-deform_cfg")) {
        cfg.configPath = value;
    } else if (key == "-ref_mask") {
        cfg.refMaskPath = value;
    } else if (key == "-flo_mask") {
        cfg.floMaskPath = value;
    } else if (key == "-workers") {
        cfg.workers = atoi(value.c_str());
    } else if (key == "-rerun") {
        cfg.rerun = atoi(value.c_str()) != 0;
    } else if (key == "-out_path_affine_forward") {
        cfg.outPathAffineForward = value;
    } else if (key == "-out_path_affine_reverse") {
        cfg.outPathAffineReverse = value;
    } else if ((key == "-out_path_forward") || (key == "-out_path_deform_forward")) {
        cfg.outPathForward = value;
    } else if ((key == "-out_path_reverse") || (key == "-out_path_deform_reverse")) {
        cfg.outPathReverse = value;
    } // animation related parameters start here 
    else if (key == "-anim_out_path")
    {
        cfg.animOutPath = value;
    }
    else if (key == "-anim_freq")
    {
        cfg.animFreq = atoi(value.c_str());
    }
    else if (key == "-anim_16_bit_mode")
    {
        cfg.anim16BitMode = atoi(value.c_str()) != 0;
    }
    else if (key == "-anim_render_mode")
    {
        cfg.animRenderMode = value;
    }
    else if (key == "-anim_format")
    {
        cfg.animFormat = value;
    }
    else if (key == "-anim_downsampling")
    {
        cfg.animDownsampling = atoi(value.c_str());
    }
    else if (key == "-anim_ref")
    {
        cfg.animRefImagePath = value;
    }
    else if (key == "-anim_flo")
    {
        cfg.animFloImagePath = value;
    }
    else
    {
        // Error
        std::cerr << "Parameter " << key << " with value " << value << " is unsupported." << std::endl;
        return false;
    }
    
    return true;
}

ProgramConfig readProgramConfigFromArgs(int argc, char** argv, bool& success) {
    ProgramConfig res;

    //res.affineConfigPath = argv[2];
    //res.configPath = argv[3];
    //res.refImagePath = argv[4];
    //res.floImagePath = argv[5];

    // Defaults for optional parameters
    res.workers = 6;
    res.rerun = true;
    res.refMaskPath = "";
    res.floMaskPath = "";

    // Animation defaults
    res.animOutPath = "";
    res.animFormat = "png";
    res.anim16BitMode = false;
    res.animRenderMode = "floating";
    res.animFreq = 50;
    res.animDownsampling = 1;
    res.animRefImagePath = "";
    res.animFloImagePath = "";
    
    for (int i = 2; i+1 < argc; i += 2) {
        if(!readKeyValuePairForProgramConfig(argc, argv, i, res))
        {
            success = false;
            break;
        }
    }

    return res;
}

bool checkFile(std::string path)
{
    std::ifstream file(path.c_str());
    if (!file)
    {
        return false;
    }
    else
    {
        return true;
    }
}

// Animation callback


template <typename TImageType, typename TTransformType>
typename TImageType::Pointer AnimApplyTransformToImage(
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

template <typename TImageType, typename TTransformType, unsigned int Dim>
struct RegistrationAnimationCallback : public BSplineRegistrationCallback<TTransformType, Dim>
{
    public:

    using ImageType = TImageType;
    using ImagePointer = typename ImageType::Pointer;

    using TransformType = TTransformType;
    using TransformPointer = typename TransformType::Pointer;

    using IPT = itk::IPT<double, Dim>;

    // Inherited
    //TransformPointer m_TransformForward;
    //TransformPointer m_TransformReverse;

    std::string m_Path;
    std::string m_Format;
    unsigned int m_Counter;
    unsigned int m_Freq;
    bool m_Format16BitMode;
    unsigned int m_Downsampling;
    std::string m_RenderMode;
    ImagePointer m_RefImage;
    ImagePointer m_FloImage;

    unsigned int m_WriteCounter;

    RegistrationAnimationCallback(
        std::string path,
        std::string format,
        bool format16BitMode,
        std::string renderMode,
        unsigned int freq,
        unsigned int downsampling,
        ImagePointer refImage,
        ImagePointer floImage)
    {
        m_Counter = 0;
        m_WriteCounter = 1;

        m_Path = path;
        m_Format = format;
        m_Format16BitMode = format16BitMode;
        m_RenderMode = renderMode;
        m_Downsampling = downsampling;

        if (freq == 0)
        {
            m_Freq = 1;
        }
        else
        {
            m_Freq = freq;
        }

        m_RefImage = refImage;
        m_FloImage = floImage;
    }

    virtual void Invoke()
    {
        if (m_Counter % m_Freq == 0)
        {
            std::string fname = m_Path;

            char buf[256];
            sprintf(buf, "frame%05d.%s", m_WriteCounter, m_Format.c_str());
            
            fname += buf;

            // This can be optimized to not first produce a full resolution image and then subsample it...

            ImagePointer frame = AnimApplyTransformToImage<ImageType, TransformType>(m_RefImage, m_FloImage, this->m_TransformForward, 0, 0);
            if (m_RenderMode == "difference")
            {
                frame = IPT::DifferenceImage(m_RefImage, frame);
            } else if (m_RenderMode == "composite")
            {
                // TO be done
            } else if (m_RenderMode == "alternating")
            {
                if (m_WriteCounter % 2 == 0)
                {
                    frame = m_RefImage;
                }
            } else if (m_RenderMode == "floating" || m_RenderMode == "")
            {
                ;
            }
            
            frame = IPT::SubsampleImage(frame, m_Downsampling);

            if (m_Format16BitMode)
            {
                IPT::SaveImageU16(fname.c_str(), frame);
            }
            else
            {
                IPT::SaveImageU8(fname.c_str(), frame);
            }
            
            ++m_WriteCounter;
        }

        ++m_Counter;
    }
};

template <unsigned int ImageDimension>
class RegisterDeformableProgram
{
    public:

    typedef itk::IPT<double, ImageDimension> IPT;

    typedef itk::Image<double, ImageDimension> ImageType;
    typedef typename ImageType::Pointer ImagePointer;
    typedef itk::Image<bool, ImageDimension> MaskType;
    typedef typename MaskType::Pointer MaskPointer;

    typedef BSplines<ImageDimension> BSplineFunc;

    static bool Run(
        ProgramConfig cfg,
        BSplineRegParamOuter& affineParams,
        BSplineRegParamOuter& params)
    {
        bool success = true;

        if (!cfg.rerun)
        {
            bool allfine = true;

            if (!allfine || (cfg.outPathAffineForward != "" && !checkFile(cfg.outPathAffineForward)))
            {
                allfine = false;
            }
            if (!allfine || (cfg.outPathAffineReverse != "" && !checkFile(cfg.outPathAffineReverse)))
            {
                allfine = false;
            }
            if (!allfine || (cfg.outPathForward != "" && !checkFile(cfg.outPathForward)))
            {
                allfine = false;
            }
            if (!allfine || (cfg.outPathReverse != "" && !checkFile(cfg.outPathReverse)))
            {
                allfine = false;
            }

            if (allfine)
            {
                return success;
            }
        }

        BSplineFunc bsf;

        typedef typename BSplineFunc::TransformType TransformType;
        typedef typename BSplineFunc::TransformType::Pointer TransformPointer;
	
        ImagePointer refImage;
        try {
            refImage = IPT::LoadImage(cfg.refImagePath.c_str());

            refImage = RemoveDirectionInformation<ImageType>(refImage);
        }
        catch (itk::ExceptionObject & err)
	    {
            std::cerr << "Error loading reference image: " << cfg.refImagePath.c_str() << std::endl;
	        std::cerr << "ExceptionObject caught !" << std::endl;
		    std::cerr << err << std::endl;
            success = false;
            return success;
		}
            
        ImagePointer floImage;
        try {
            floImage = IPT::LoadImage(cfg.floImagePath.c_str());

            floImage = RemoveDirectionInformation<ImageType>(floImage);
        }
        catch (itk::ExceptionObject & err)
		{
            std::cerr << "Error loading floating image: " << cfg.floImagePath.c_str() << std::endl;
		    std::cerr << "ExceptionObject caught !" << std::endl;
		    std::cerr << err << std::endl;
            success = false;
            return success;
	    }

        // Print information about images
        std::cerr << "Reference image information" << std::endl;
        std::cerr << refImage << std::endl;
        std::cerr << "Floating image information" << std::endl;
        std::cerr << floImage << std::endl;

        ImagePointer refImageMask;
        if (cfg.refMaskPath != "") {
            try {
                refImageMask = IPT::LoadImage(cfg.refMaskPath.c_str());
            }
            catch (itk::ExceptionObject & err)
            {
                std::cerr << "Error loading reference mask: " << cfg.refMaskPath.c_str() << std::endl;
	            std::cerr << "ExceptionObject caught !" << std::endl;
	            std::cerr << err << std::endl;
                success = false;
                return success;
	        }
        }
        ImagePointer floImageMask;
        if (cfg.floMaskPath != "") {
            try {
                floImageMask = IPT::LoadImage(cfg.floMaskPath.c_str());
            }
            catch (itk::ExceptionObject & err)
            {
                std::cerr << "Error loading reference mask: " << cfg.floMaskPath.c_str() << std::endl;
	            std::cerr << "ExceptionObject caught !" << std::endl;
	            std::cerr << err << std::endl;
                success = false;
                return success;
	        }
        }

        using CompositeTranformType = itk::CompositeTransform<double, ImageDimension>;
        using CompositeTranformPointer = typename CompositeTranformType::Pointer;

        CompositeTranformPointer affineTransform;
        TransformPointer forwardTransform;
        TransformPointer inverseTransform;

        using RegistrationAnimationCallbackType = RegistrationAnimationCallback<ImageType, TransformType, ImageDimension>;
        
        // Setup animation

        RegistrationAnimationCallbackType* animationCallback = nullptr;

        if (cfg.animOutPath != "")
        {
            ImagePointer animRefImage = refImage;
            ImagePointer animFloImage = floImage;

            if (cfg.animRefImagePath != "")
            {
                try {
                    animRefImage = IPT::LoadImage(cfg.animRefImagePath.c_str());
                }
                catch (itk::ExceptionObject & err)
                {
                    std::cerr << "Error loading animation reference image: " << cfg.animRefImagePath.c_str() << std::endl;
	                std::cerr << "ExceptionObject caught !" << std::endl;
	                std::cerr << err << std::endl;
                    success = false;
                    return success;
	            }                
            }
            if (cfg.animFloImagePath != "")
            {
                try {
                    animFloImage = IPT::LoadImage(cfg.animFloImagePath.c_str());
                }
                catch (itk::ExceptionObject & err)
                {
                    std::cerr << "Error loading animation floating image: " << cfg.animFloImagePath.c_str() << std::endl;
	                std::cerr << "ExceptionObject caught !" << std::endl;
	                std::cerr << err << std::endl;
                    success = false;
                    return success;
	            }                
            }

            animationCallback = new RegistrationAnimationCallbackType(
                cfg.animOutPath,
                cfg.animFormat,
                cfg.anim16BitMode,
                cfg.animRenderMode,
                cfg.animFreq,
                cfg.animDownsampling,
                animRefImage,
                animFloImage);
        }

        std::cerr << "Starting registration" << std::endl;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        bsf.register_deformable(
            refImage,
            floImage,
            affineParams,
            params,
            refImageMask,
            floImageMask,
            affineTransform,
            forwardTransform,
            inverseTransform,
            animationCallback);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();

        // Remove the mask images (if they exist)
        refImageMask = nullptr;
        floImageMask = nullptr;

        if (animationCallback != nullptr)
        {
            delete animationCallback;
        }

        if(cfg.outPathForward.length() > 260) {
            std::cerr << "Error. Too long forward transform path." << std::endl;
            success = false;
            return success;
        }

        if(cfg.outPathAffineForward != "") {
            try {
                IPT::SaveTransformFile(cfg.outPathAffineForward.c_str(), affineTransform.GetPointer());
            }
            catch (itk::ExceptionObject & err)
            {
                std::cerr << "Error saving forward affine transformation file." << std::endl;
	            std::cerr << err << std::endl;
                success = false;
	    	}
        }

        if(cfg.outPathAffineReverse != "") {
            try {
                auto inverse = affineTransform->GetInverseTransform();
                if (inverse) {
                    IPT::SaveTransformFile(cfg.outPathAffineReverse.c_str(), affineTransform->GetInverseTransform().GetPointer());
                } else {
                    std::cerr << "Error saving reverse affine transformation file. The transformation is not invertible." << std::endl;
                    success = false;
                }
            }
            catch (itk::ExceptionObject & err)
            {
                std::cerr << "Error saving reverse affine transformation file." << std::endl;
	            std::cerr << err << std::endl;
                success = false;
	    	}
        }

        try {
            IPT::SaveTransformFile(cfg.outPathForward.c_str(), forwardTransform.GetPointer());
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "Error saving forward transformation file." << std::endl;
	        std::cerr << err << std::endl;
            success = false;
		}

        try {
            IPT::SaveTransformFile(cfg.outPathReverse.c_str(), inverseTransform.GetPointer());
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "Error saving reverse transformation file." << std::endl;
	        std::cerr << err << std::endl;
            success = false;
		}

        std::cout << "(Registration) Time elapsed: " << elapsed << "[s]" << std::endl;
        
        return success;
    }

    static int MainFunc(int argc, char** argv) {
        RegisterIOFactories();

        itk::TimeProbesCollectorBase chronometer;
        itk::MemoryProbesCollectorBase memorymeter;

        std::cout << "--- RegisterDeformable ---" << std::endl;

        chronometer.Start("Registration");
        memorymeter.Start("Registration");

        bool success = true;
        ProgramConfig config = readProgramConfigFromArgs(argc, argv, success);
        if (!success)
        {
            return -1;
        }
        std::cout << "Program config read..." << std::endl;
        BSplineRegParamOuter affineParams = readConfig(config.affineConfigPath);
        std::cout << "Affine registration config read..." << std::endl;
        BSplineRegParamOuter params = readConfig(config.configPath);
        std::cout << "Deformable Registration config read..." << std::endl;

        // Threading
        itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(config.workers);
        itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(config.workers);

        success = Run(config, affineParams, params);

        chronometer.Stop("Registration");
        memorymeter.Stop("Registration");

        //chronometer.Report(std::cout);
        //memorymeter.Report(std::cout);

        return (success ? 0 : -1);
    }
};

int main(int argc, char** argv) {
    printTitle();
    if(argc < 2) {
        printHelp(argv[0]);
        return -1;
    }
    int ndim = atoi(argv[1]);
    if(ndim == 2) {
        RegisterDeformableProgram<2U>::MainFunc(argc, argv);
    } else if(ndim == 3) {
        RegisterDeformableProgram<3U>::MainFunc(argc, argv);
    } else {
        std::cout << "Error: Dimensionality " << ndim << " is not supported." << std::endl;
    }
}
