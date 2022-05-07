# CMSC733-Project3
Buildings built in minutes - An SfM Approach

## Introduction
We reconstruct a 3D scene and simultaneously obtained the camera poses with respect to the scene, with a given set of 6 images from a monocular camera and
their feature point correspondences. Following are the steps involved:
* Feature detection and finding correspondences
* Estimating Fundamental Matrix
* Essential Matrix and solving for camera poses
* Linear Triangulation and recovering correct pose
* Non Linear Triangulation
* Linear PnP, RANSAC and Non linear optimization
* Bundle Adjustment

## How to run the code
- Change the directory to the folder where Wrapper.py is located. Eg.     
```
cd ./Code 
```

- Run the Wrapper.py file using the following command:    
```
python3 Wrapper.py --InputDir ../Data/input --OutputDir ../Data/output --NumImages 6
```