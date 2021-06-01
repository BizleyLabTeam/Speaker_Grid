## Calibration

Fisheye calibration using Matlab (**camera_calibrator_estimateAlignment_2021_05_31.m**)


To correct any image, use the following:

undistortedImage = undistortFisheyeImage(originalImage, cameraParams.Intrinsics);

where camaraParams is loaded from **CalibResults_2021_05_31.mat**
