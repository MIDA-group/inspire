# INSPIRE: Intensity and Spatial Information-Based Deformable Image Registration

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

INSPIRE is configured by editing configuration files written in the standard JSON format.

It has an hierarchical structure. For each level of the configuration file, you can specify some preprocessing settings that should be applied to the image, and general settings. Then in each sublevel of the configuration you can specify the number of control-points in the B-spline grid, iterations, learning rate, momentum.
Here is a default configuration which goes up to 48 control points in the final level of the registration process, starting out at only 9.

{

  "paramSets": [
  
      {"optimizer": "sgdm", "samplingFraction": 0.01, "downsamplingFactor": 4, "smoothing": 3.0, "samplingMode": "gw50:2", "smoothingMode": "gaussian", "dmax": 0.2, "alphaLevels": 7, "normalization": 0.001, "lambdaFactor": 0.01, "seed": 2000, "enableCallbacks": false, "verbose": false, "innerParams": [
      
          {"iterations": 1000, "controlPoints": 9, "learningRate": 5, "momentum": 0.9}, {"iterations": 1000, "controlPoints": 14, "learningRate": 3, "momentum": 0.9}]},
          
      {"optimizer": "sgdm", "samplingFraction": 0.01, "downsamplingFactor": 2, "smoothing": 1.0, "samplingMode": "gw50:2", "smoothingMode": "gaussian", "dmax": 0.1, "alphaLevels": 7, "normalization": 0.001, "lambdaFactor": 0.01, "seed": 2001, "enableCallbacks": false, "verbose": false, "innerParams": [
      
          {"iterations": 1000, "controlPoints": 24, "learningRate": 2, "momentum": 0.3}]},
          
      {"optimizer": "sgdm", "samplingFraction": 0.01, "downsamplingFactor": 1, "smoothing": 0.0, "samplingMode": "gw50:2", "smoothingMode": "gaussian", "dmax": 0.05, "alphaLevels": 7, "normalization": 0.001, "lambdaFactor": 0.01, "seed": 2002, "enableCallbacks": false, "verbose": false, "innerParams": [
      
          {"iterations": 500, "controlPoints": 48, "learningRate": 1, "momentum": 0.3}]}
          
]

}

