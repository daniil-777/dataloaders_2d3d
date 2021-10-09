# Python Toolbox for the NYU Depth Dataset V2

## Description
Python script for using the [NYU Depth Dataset V2][nyuv2].
This script allows to load color/depth images and extract point cloud.


## Data
Please, load the labeled dataset:


```diff
wget horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
```

[nyuv2]: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

## Run

You shold specify path to the nyu_depth_v2_labeled.mat file in preprocess.py Path variable. To load the data simply run the code :
```diff
python preprocess.py
```

## Features

main function that can be used for data loader
```
preprocess_labeled_dataset
```

Generates rotation matrix by sampling uniformly along x/y/z axis and translation vector 

```
def transform_pc
```
Samples uniformly 1000 2d points and corresponding 3d points
```
def p2d3d_sample_from_image
```
Generates point cloud from the color and depth
```
def extract_pcd_o3d_from_rgbd
```
Generates point cloud from the depth image
```
def extract_pcd_o3d_from_depth
```

