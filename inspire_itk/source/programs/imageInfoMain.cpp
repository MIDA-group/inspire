
/**
 * --- INSPIRE ---
 * Author: Johan Ofverstedt
 * 
 * Utility program that displays information about an image.
 */

#include <thread>
#include <fstream>
#include <iostream>

#include "../common/itkImageProcessingTools.h"

#include "itkVersion.h"
#include "itkMultiThreaderBase.h"

#include "itkPNGImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"

void RegisterIOFactories() {
    itk::PNGImageIOFactory::RegisterOneFactory();
    itk::NiftiImageIOFactory::RegisterOneFactory();
}

template <unsigned int ImageDimension>
class ImageInfoProgram
{
    public:

    typedef itk::IPT<double, ImageDimension> IPT;

    typedef itk::Image<double, ImageDimension> ImageType;
    typedef typename ImageType::Pointer ImagePointer;

    static int MainFunc(int argc, char** argv) {
        RegisterIOFactories();

        std::cout << "--- ImageInfo ---" << std::endl;

        // Threading
        itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(1);
        itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1);

        ImagePointer image;
        try {
            image = IPT::LoadImage(argv[2]);
        }
        catch (itk::ExceptionObject & err)
		{
            std::cerr << "Error loading image: " << argv[2] << std::endl;
		    std::cerr << "ExceptionObject caught !" << std::endl;
		    std::cerr << err << std::endl;
            return -1;
	    }

        std::cout << image << std::endl;

        return 0;
    }
};

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "No arguments..." << std::endl;
    }
    int ndim = atoi(argv[1]);
    if(ndim == 2) {
        ImageInfoProgram<2U>::MainFunc(argc, argv);
    } else if(ndim == 3) {
        ImageInfoProgram<3U>::MainFunc(argc, argv);
    } else {
        std::cout << "Error: Dimensionality " << ndim << " is not supported." << std::endl;
    }
}
