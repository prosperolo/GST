#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import glob
import os
import sys
from PIL import Image
import imageio
import torch
import cv2
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from copy import deepcopy

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    color: np.array = None
    mask: np.array = None
    px: float = None
    py: float = None
    crop_info: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasInfosHumman(rgb_paths, mask_paths, poses, intrins, distcoeff, camera_idx, random_background_color=True, bg_color=0, tight_crop=False):
    cam_infos = []

    for i, c2w in enumerate(poses):
        current_intrins = deepcopy(intrins[i])
        fx = current_intrins[0, 0]
        fy = current_intrins[1, 1]
        cx = current_intrins[0, 2] 
        cy = current_intrins[1, 2]

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        # w2c = np.transpose(np.matmul(w2c, camera_transform_matrix))
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = rgb_paths[i]
        
        if random_background_color:
            random_color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        else:
            random_color = np.ones(3, dtype=np.uint8) * bg_color

        image = imageio.imread(image_path)
        h, w, _ = image.shape
        if distcoeff is not None:
            cur_dist = distcoeff[i]
            image = cv2.undistort(image, current_intrins, cur_dist)

        mask = imageio.imread(mask_paths[i])
        if len(mask.shape) < 3:
            mask = mask[:, :, None]

        image = np.where(mask, image, random_color)

        image, mask, (h_min, w_min, crop_size) = crop_cmu_image(image, mask, cx, cy, tight_crop=tight_crop)
        
        cx = cx - w_min
        cy = cy - h_min

        image = Image.fromarray(image)
        if len(mask.shape) < 3 or mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=-1)
        mask = Image.fromarray(mask*(255//np.max(mask)))

        FovX = focal2fov(fx, crop_size)
        FovY = focal2fov(fy, crop_size)

        cx = (cx - crop_size//2) / (crop_size/2)
        cy = (cy - crop_size//2) / (crop_size/2)

        crop_info = np.array([h_min, w_min, crop_size, h, w])

        cam_infos.append(CameraInfo(uid=Path(image_path).stem + str(camera_idx[i]), R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=camera_idx[i], width=image.size[0], height=image.size[1], 
                        color=random_color, mask=mask, px=cx, py=cy, crop_info=crop_info))
        
    return cam_infos


def adjust_for_boundaries(c_min, width, image_size):
    if c_min < 0:
        return 0
    elif c_min + width > image_size:
        return image_size - width
    else:
        return c_min


def crop_human_image(image, mask):
    image_h, image_w, c = image.shape
    image_size = min(image_h, image_w)
    h, w, c = mask.nonzero()
    if len(h) > 0:
        h_max = np.max(h)
        h_min = np.min(h)
    else:
        h_max = image_h
        h_min = 0
    if len(w) > 0:
        w_max = np.max(w)
        w_min = np.min(w)
    else:
        w_max = image_w 
        w_min = 0
    delta_h = h_max - h_min
    delta_w = w_max - w_min
    diff = abs(delta_h - delta_w)
    size = max(image_h // 4, delta_h, delta_w)
    if delta_h > delta_w:
        w_min = w_min - diff // 2
        w_min = adjust_for_boundaries(w_min, size, image_size)
    else:
        h_min = h_min - diff // 2
        h_min = adjust_for_boundaries(h_min, size, image_size)

    image = image[h_min:h_min+size, w_min:w_min+size, :]
    mask = mask[h_min:h_min+size, w_min:w_min+size, :]
    return image, mask, (h_min, w_min, size)


def crop_cmu_image(image, mask, cx, cy, tight_crop=True):
    if tight_crop:
        return crop_cmu_image_tight(image, mask)
    else:
        return crop_cmu_square(image, mask, cx, cy)


def crop_cmu_square(image, mask, cx, cy):
    image_h, image_w, c = image.shape
    h, w, c = mask.nonzero()
    size = 2*round(min(cy, image_h-cy))
    w_min = round(cx) - size//2
    h_min = round(cy) - size//2
    image = image[h_min:h_min+size, w_min:w_min+size, :]
    mask = mask[h_min:h_min+size, w_min:w_min+size, :]
    return image, mask, (h_min, w_min, size)


def crop_cmu_image_tight(image, mask):
    image_h, image_w, c = image.shape
    h, w, c = mask.nonzero()
    if len(w) > 0:
        w_max = np.max(w)
        w_min = np.min(w)
        w_diff = w_max - w_min
    else:
        w_max = image_w // 2
        w_min = image_w // 2
        w_diff = 0
    if len(h) > 0:
        h_max = np.max(h)
        h_min = np.min(h)
        h_diff = h_max - h_min
    else:
        h_max = image_h // 2
        h_min = image_h // 2
        h_diff = 0
    center_w = (w_max + w_min) // 2
    center_h = (h_max + h_min) // 2
    size = max(image_h // 4, w_diff, h_diff)
    w_min = center_w - size // 2
    h_min = center_h - size // 2
    w_min = adjust_for_boundaries(w_min, size, image_w)
    h_min = adjust_for_boundaries(h_min, size, image_h)
    image = image[h_min:h_min+size, w_min:w_min+size, :]
    mask = mask[h_min:h_min+size, w_min:w_min+size, :]
    return image, mask, (h_min, w_min, size)


def readCamerasFromNpy(folder_path, 
                       w2c_Rs_rmo=None, 
                       w2c_Ts_rmo=None, 
                       focals_folder_path=None):
    # Set every_5th_in for the testing set
    cam_infos = []
    # Transform fov from degrees to radians
    fname_order_path = os.path.join(folder_path, "frame_order.txt")
    c2w_T_rmo_path = os.path.join(folder_path, "c2w_T_rmo.npy")
    c2w_R_rmo_path = os.path.join(folder_path, "c2w_R_rmo.npy")
    if focals_folder_path is None:
        focals_folder_path = folder_path
    focal_lengths_path = os.path.join(focals_folder_path, "focal_lengths.npy")

    with open(fname_order_path, "r") as f:
        fnames = f.readlines()
    fnames = [fname.split("\n")[0] for fname in fnames]

    if w2c_Ts_rmo is None:
        c2w_T_rmo = np.load(c2w_T_rmo_path)
    if w2c_Rs_rmo is None:
        c2w_R_rmo = np.load(c2w_R_rmo_path)
    focal_lengths = np.load(focal_lengths_path)[:, 0, :]

    camera_transform_matrix = np.eye(4)
    camera_transform_matrix[0, 0] *= -1
    camera_transform_matrix[1, 1] *= -1
    
    # assume shape 128 x 128
    image_side = 128

    for f_idx, fname in enumerate(fnames):

        w2c_template = np.eye(4)
        if w2c_Rs_rmo is None:
            w2c_R = np.transpose(c2w_R_rmo[f_idx])
        else:
            w2c_R = w2c_Rs_rmo[f_idx]
        if w2c_Ts_rmo is None:
            w2c_T = - np.matmul(c2w_T_rmo[f_idx], w2c_R)
        else:
            w2c_T = w2c_Ts_rmo[f_idx]
        w2c_template[:3, :3] = w2c_R
        # at this point the scene scale is approx. that of shapenet cars
        w2c_template[3:, :3] = w2c_T

        # Pytorch3D cameras have (x left, y right, z away axes)
        # need to transform to COLMAP / OpenCV (x right, y down, z forward)
        # transform axes and transpose to column major order
        w2c = np.transpose(np.matmul(w2c_template, camera_transform_matrix))

        # get the world-to-camera transform and set R, T
        # w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_name = fname.split(".png")

        focal_lengths_ndc = focal_lengths[f_idx]
        focal_lengths_px = focal_lengths_ndc * image_side / 2

        FovY = focal2fov(focal_lengths_px[1], image_side) 
        FovX = focal2fov(focal_lengths_px[0], image_side)

        cam_infos.append(CameraInfo(uid=fname, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                        image_path=None, image_name=image_name, width=image_side, height=image_side))
        
    return cam_infos

def readSrnSceneInfo(path, eval, num_pts=1000):
    print("Reading filepaths")
    rgb_paths = sorted(glob.glob(os.path.join(path, "rgb", "*")))
    pose_paths = sorted(glob.glob(os.path.join(path, "pose", "*")))
    assert len(rgb_paths) == len(pose_paths), "Unequal number of paths"
    print("Got {} images in dataset".format(len(rgb_paths)))
    train_idxs = [i for i in range(len(rgb_paths))]
    test_idxs = [i for i in range(len(rgb_paths)) if i not in train_idxs]
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTxt(rgb_paths, pose_paths, train_idxs)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTxt(rgb_paths, pose_paths, test_idxs)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    # always instantiate a new random point cloud
    # Since this data set has no colmap data, we start with random points
    print(f"Generating random point cloud ({num_pts})...")
    
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.random((num_pts, 3)) * 0.5 - 0.25
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "SRN": readSrnSceneInfo
}