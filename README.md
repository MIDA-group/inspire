# inspire
INSPIRE: Intensity and Spatial Information-Based Deformable Image Registration

---

INSPIRE is a deformable image registration method (and tool) for 2D and 3D images.
Paper reference: Goes Here

The implementation is based on ITK, a general-purpose library for image processing and analysis, optimisation, registration, segmentation and much more.

To build INSPIRE, follow these steps:
Download and install ITK (github.com/InsightSoftwareConsortium/ITK)

Clone the inspire repo, into, e.g. my_path/inspire
Create a new directory my_path/inspire-build
From my_path/inspire-build do: "cmake -DCMAKE\_BUILD\_TYPE=Release ../inspire/inspire\_itk"
make

Now you are ready to register images (assuming that they have already been registered up to affine transformations).

From inspire-build for 2d images, call INSPIRE:
./InspireRegister 2 -ref my\_reference\_image.tif -flo my\_floating\_image.tif -deform_cfg ../inspire/inspire\_itk/config/default\_configuration.json \\
    -out\_path\_deform\_forward tforward.txt -out\_path\_deform\_reverse treverse.txt

For 3d images:
./InspireRegister 3 -ref my\_reference\_image.tif -flo my\_floating\_image.tif -deform_cfg ../inspire/inspire\_itk/config/default\_configuration.json \\
    -out\_path\_deform\_forward tforward.txt -out\_path\_deform\_reverse treverse.txt
    
To apply the transformation to images, use the following command:
./InspireTransform -dim 2 -16bit 1 -interpolation linear -transform tforward.txt -ref my\_reference\_image.tif -in my\_floating\_image.tif -out my\_output\_image.tif

Supported interpolation modes: (nearest, linear, cubic)


