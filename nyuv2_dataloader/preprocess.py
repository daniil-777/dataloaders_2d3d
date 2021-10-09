#!/usr/bin/env python3

from pathlib import Path
from nyuv2 import *
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import re
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# import numpy as np
import matplotlib as mpl
from scipy.spatial.transform import Rotation

# import matplotlib.pyplot as plt
mpl.use('tkagg')


# DATASET_DIR = Path('dataset')
DATASET_DIR = Path('/Volumes/Ubuntu/trace_lab/datasets')


def plot_color(ax, color, title="Color"):
    """Displays a color image from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(color)

def plot_depth(ax, depth, title="Depth"):
    """Displays a depth map from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(depth, cmap='Spectral')


def transform_pc(pointcloud):
    """Generates rotation matrix by sampling uniformly along x/y/z axis 
    and translation vector 


    Args:
        pointcloud (np.array): point cloud

    Returns:
        [type]: Rotation, translation, point cloud
    """
    factor = 1
    anglex = np.random.uniform() * np.pi / factor
    angley = np.random.uniform() * np.pi / factor
    anglez = np.random.uniform() * np.pi / factor

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                    [0, cosx, -sinx],
                    [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                    [0, 1, 0],
                    [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                    [sinz, cosz, 0],
                    [0, 0, 1]])
    R_ab = Rx.dot(Ry).dot(Rz)
    R_ba = R_ab.T
    translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                np.random.uniform(-0.5, 0.5)])
    translation_ba = -R_ba.dot(translation_ab)

    pointcloud1 = pointcloud.T
    pointcloud1 = np.squeeze(pointcloud1)

    rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
    pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)
    answer = {}
    answer['R'] = rotation_ab.as_matrix()
    answer['t'] = translation_ab
    answer['pointcloud'] = pointcloud2
    return answer


def p2d3d_sample_from_image(depth, cam_matrix):
    """samples uniformly 1000 2d points and corresponding 3d points

    Args:
        depth (np.array): tensor of depth
        cam_matrix (np.array): tensor of camera intinsics

    Returns:
        [dict]: dictionary with p2d/p3d coordinates
    """
    cx = cam_matrix[0, 2]
    cy = cam_matrix[1, 2]
    fx = cam_matrix[0, 0]
    fy = cam_matrix[1, 1]

    rows, cols = depth.shape
    #sample points uniformly
    ind_x = np.random.choice(rows, 1000)
    ind_y = np.random.choice(cols, 1000)

    p3d_z = depth[ind_x, ind_y]
    p3d_x = (ind_x - cx) * p3d_z / fx
    p3d_y = (ind_x - cy) * p3d_z / fy
    answer = {}
    answer['p2d_coords'] = np.dstack((p3d_x, p3d_y))
    answer['p3d_coords'] = np.dstack((p3d_x, p3d_y, p3d_z))

    return answer
    # bool_array = np.full((rows, cols), False)
    # bool_array[ind_x, ind_y] = True

    # c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    # # valid = (depth > 0) & (depth < 255) & bool_array
    # valid = bool_array

    # x = 

    # z = np.where(valid, depth / 256.0, np.nan)
    # x = np.where(valid, z * (c - cx) / fx, 0)
    # y = np.where(valid, z * (r - cy) / fy, 0)
    # return np.dstack((x, y, z))



def extract_pcd_o3d_from_rgbd(color, depth):
    """generates point cloud from the color and depth
    Args:
        color (np.array): tensor of color
        depth (np.array): tensor of depth

    Returns:
        np.array: 3d point cloud
    """
    color = o3d.geometry.Image(color)
    depth = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_nyu_format(color, depth)
    points3d = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    points3d = np.array(points3d.points)
    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return points3d
    

def extract_pcd_o3d_from_depth(depth):
    """generates point cloud from the depth image

    Args:
        depth ([type]): [description]

    Returns:
        [type]: [description]
    """
    points3d = o3d.geometry.PointCloud.create_from_depth_image(depth,  o3d.camera.PinholeCameraIntrinsic(
               o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    points3d = np.array(points3d.points)
    return points3d


def preprocess_labeled_dataset():
    cam = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    cam = np.array(cam.intrinsic_matrix)

    print("cam: ", cam)

    labeled = LabeledDataset(DATASET_DIR / 'nyu_depth_v2_labeled.mat')
    # labeled = LabeledDataset('/Volumes/Ubuntu/trace_lab/datasets/nyu_depth_v2_labeled.mat')

    color, depth = labeled[0] #extract data item from the NYU dataset
    color = np.array(color)
    depth = np.array(depth)

    # selected_2d, selected_3d = p2d3d_sample_from_image(depth, cam)
    selected = p2d3d_sample_from_image(depth, cam)
    selected_2d, selected_3d = selected['p2d_coords'], selected['p3d_coords']

    transformation = transform_pc(selected_3d)
    R, t, point_3d_rotated = transformation['R'], transformation['t'], transformation['pointcloud']


    print("selected_2d: ", selected_2d)
    print("selected_3d: ", selected_3d)
    print("point_3d_rotated: ", point_3d_rotated)
    print("R: ", R)
    print("t: ", t)


   
    labeled.close()


if __name__ == "__main__":
    preprocess_labeled_dataset()
# test_raw_dataset()
