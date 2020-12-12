
#include <stdio.h>
#include <string.h>
#include <string>

#include "../common/itkImageProcessingTools.h"
#include "itkTextOutput.h"

enum ImwarpInterpolationMode
{
    kImwarpInterpNearest,
    kImwarpInterpLinear,
    kImwarpInterpCubic,
};

enum SpacingMode {
	kDefaultSpacing,
	kRemoveSpacing,
	kResampleSpacing,
};

struct TransformProgramParam
{
	std::string transformPath;
	
	std::string referenceImagePath;

	std::string inputPath;
	std::string outputPath;

	ImwarpInterpolationMode interpolationMode;
	SpacingMode spacingMode;

	bool format16U;
	std::string bgValue;
	double scalingFactor;
	double divideFactor;
};

template <unsigned int Dim>
static void DoTransform(TransformProgramParam &param)
{
	// Types

	typedef itk::IPT<double, Dim> IPT;

	typedef typename IPT::ImageType ImageType;
	typedef typename IPT::ImagePointer ImagePointer;
	typedef itk::Image<unsigned short, Dim> LabelImageType;

	//itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
	//itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1);
    //itk::OutputWindow::SetInstance(itk::TextOutput::New());

	typename IPT::SpacingMode spacingMode = IPT::kDefaultSpacingMode;
	if (param.spacingMode == kDefaultSpacing)
		spacingMode = IPT::kDefaultSpacingMode;
	else if (param.spacingMode == kRemoveSpacing)
		spacingMode = IPT::kRemoveSpacingMode;
	else if (param.spacingMode == kResampleSpacing)
		spacingMode = IPT::kResampleSpacingMode;

	typename itk::Transform<double, Dim, Dim>::Pointer transform = IPT::LoadTransformFile(param.transformPath.c_str());

	ImagePointer refImage;

	// Load reference image if exists
	if (param.referenceImagePath != "")
	{
		refImage = IPT::RemoveSpacing(IPT::LoadImage(param.referenceImagePath.c_str()), spacingMode);
		itk::PrintStatistics<ImageType>(refImage, "Reference");
	}
	else
	{
		std::cout << "Error: No reference image provided." << std::endl;
		return;
	}

	typedef typename IPT::ImageStatisticsData ISD;

	ImagePointer floatingImage = IPT::RemoveSpacing(IPT::LoadImage(param.inputPath.c_str()), spacingMode);

	if(param.scalingFactor != 1.0 || param.divideFactor != 1.0) {
		const double localScalingFactor = param.scalingFactor / param.divideFactor;
		itk::PrintStatistics<ImageType>(floatingImage, "Floating Before Scaling");
		floatingImage = IPT::MultiplyImageByConstant(floatingImage, localScalingFactor);
		itk::PrintStatistics<ImageType>(floatingImage, "Floating After Scaling");
	} else {
		itk::PrintStatistics<ImageType>(floatingImage, "Floating");
	}

	double bgVal;
	if (param.bgValue == "min")
	{
		ISD stats = IPT::ImageStatistics(floatingImage);

		bgVal = stats.min;
	}
	else if (param.bgValue == "max")
	{
		ISD stats = IPT::ImageStatistics(floatingImage);

		bgVal = stats.max;
	}
	else if (param.bgValue == "mean")
	{
		ISD stats = IPT::ImageStatistics(floatingImage);

		bgVal = stats.mean;
	}
	else if (param.bgValue == "refmin")
	{
		ISD stats = IPT::ImageStatistics(refImage);

		bgVal = stats.min;
	}
	else if (param.bgValue == "refmax")
	{
		ISD stats = IPT::ImageStatistics(refImage);

		bgVal = stats.max;
	}
	else if (param.bgValue == "refmean")
	{
		ISD stats = IPT::ImageStatistics(refImage);

		bgVal = stats.mean;
	}
	else
	{
		bgVal = atof(param.bgValue.c_str());
	}

	ImagePointer finalImage = IPT::TransformImage(
		floatingImage,
		refImage,
		transform,
		static_cast<typename IPT::ImwarpInterpolationMode>(param.interpolationMode),
		bgVal);

	IPT::SaturateImage(finalImage, 0.0, 1.0);

	itk::PrintStatistics<ImageType>(finalImage, "Transformed");

	if (param.format16U)
	{
		IPT::SaveImageU16(param.outputPath.c_str(), finalImage);
	}
	else
	{
		IPT::SaveImageU8(param.outputPath.c_str(), finalImage);
	}
}

int main(int argc, char **argv)
{
	TransformProgramParam param;

	unsigned int dim = 2U;

	// Defaults
	param.format16U = false;
	param.bgValue = "mean";

	param.interpolationMode = kImwarpInterpLinear;
	param.spacingMode = kDefaultSpacing;
	param.scalingFactor = 1.0;
	param.divideFactor = 1.0;

	// Parameters
	for (int pi = 1; pi + 1 < argc; pi += 2)
	{
		const char* mod = argv[pi];
		const char* arg = argv[pi + 1];

		if (strcmp(mod, "-ref") == 0)
		{
			param.referenceImagePath = arg;
		}
		else if (strcmp(mod, "-transform") == 0)
		{
			param.transformPath = arg;
		}
		else if (strcmp(mod, "-in") == 0)
		{
			param.inputPath = arg;
		}
		else if (strcmp(mod, "-out") == 0)
		{
			param.outputPath = arg;
		}
		else if (strcmp(mod, "-interpolation") == 0)
		{
			std::string mode = arg;
			if(mode == "nearest") {
				param.interpolationMode = kImwarpInterpNearest;
			} else if(mode == "linear") {
				param.interpolationMode = kImwarpInterpLinear;
			} else if(mode == "cubic") {
				param.interpolationMode = kImwarpInterpCubic;
			} else {
				std::cout << "Illegal interpolation mode: '" << mode << "', allowed modes are: nearest, linear, cubic." << std::endl;
				return -1;
			}
		}
		else if (strcmp(mod, "-spacing_mode") == 0)
		{
			std::string mode = arg;

			if(mode == "default") {
				param.spacingMode = kDefaultSpacing;
			} else if(mode == "remove") {
				param.spacingMode = kRemoveSpacing;
			} else if(mode == "resample") {
				param.spacingMode = kResampleSpacing;
			} else {
				std::cout << "Illegal spacing mode: '" << mode << "', allowed modes are: default, remove, resample." << std::endl;
				return -1;
			}
		}
		else if (strcmp(mod, "-scaling_factor") == 0)
		{
			param.scalingFactor = atof(arg);
		}
		else if (strcmp(mod, "-divide_factor") == 0)
		{
			param.divideFactor = atof(arg);
		}
		else if (strcmp(mod, "-16bit") == 0)
		{
			param.format16U = (0 != atoi(arg));
		}
		else if (strcmp(mod, "-bg") == 0)
		{
			param.bgValue = arg;
		}
		else if (strcmp(mod, "-dim") == 0)
		{
			dim = (unsigned int)atoi(arg);
			if(dim < 2 || dim > 3) {
				std::cout << "Illegal number of dimensions: '" << dim << "', only 2 or 3 supported." << std::endl;
				return -1;
			}
		}
	}

	if (dim == 3U)
	{
		DoTransform<3U>(param);
	}
	else if(dim == 2U)
	{
		DoTransform<2U>(param);
	}

	return 0;
}
