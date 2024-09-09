import imageio
import glob
import os
from tqdm import tqdm
import json
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import copy
import smplx
import yaml
import cv2
from PIL import Image
import imageio

from .dataset_readers import readCamerasInfosHumman
from utils.general_utils import PILtoTorch, PILtoTorchHMR, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World, fov2focal


HUMMAN_DATASET_ROOT = None # Change this to your data directory
assert HUMMAN_DATASET_ROOT is not None, "Update the location of the Humman Dataset"
CACHE_DIR = os.path.join(os.environ.get("HOME"), ".cache")
CACHE_DIR_4DHUMANS = os.path.join(CACHE_DIR, "4DHumans")

@dataclass
class Camera:
    R: torch.tensor
    T: torch.tensor


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def compute_3d_bounds(vertices):
    # obtain the original bounds for point sampling
    min_xyz = np.min(vertices, axis=0)
    max_xyz = np.max(vertices, axis=0)
    min_xyz -= 0.05
    max_xyz += 0.05
    world_bounds = np.stack([min_xyz, max_xyz], axis=0)
    return world_bounds

def get_mask(vertices, H, W, K, R, T):
    world_bounds = compute_3d_bounds(vertices)
    
    pose = np.concatenate([R, T[:, None]], axis=1)
    
    bound_mask = get_bound_2d_mask(world_bounds, K, pose, H, W)
    return bound_mask


class HuMMan(Dataset):
    def __init__(self, cfg,
                 dataset_name="train"):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name

        self.base_path = os.path.join(HUMMAN_DATASET_ROOT)

        if hasattr(self.cfg.data, "overfit"):
            self.overfit = self.cfg.data.overfit 
        else:
            self.overfit = False

        if hasattr(self.cfg.data, "num_training_images"):
            self.num_training_images = cfg.data.num_training_images
        else:
            self.num_training_images = 4

        if self.overfit:
            dataset_name = "train"

        filename = "train.txt" if self.dataset_name == "train" else "test.txt"

        with open(os.path.join(self.base_path, filename)) as file:
            self.folders = [line.rstrip() for line in file]

        if self.dataset_name != "train":
            self.num_training_images = 1000

        if hasattr(self.cfg.data, "undistort"):
            self.undistort = self.cfg.data.undistort
        else:
            self.undistort = False

        self.cameras_jsons = {}
        
        self.projection_matrix = getProjectionMatrix(
            znear=cfg.data.znear, zfar=cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)
        
        self.smpl = smplx.create(
            model_path=os.path.join(CACHE_DIR_4DHUMANS, "data/smpl/SMPL_NEUTRAL.pkl"),
            model_type='smpl',
            gender='neutral')
        
        self.items = self.process_folders()
        
    def __len__(self):
        return len(self.items)
    
    @staticmethod
    def get_numpy(hfile, key):
        val = hfile.get(key)
        return np.array(val)
    
    def load_joints(self, folder_name):
        joints_path = os.path.join(self.base_path, folder_name, "processed_joints.npy")
        joints_array = np.load(joints_path)
        return joints_array

    def load_annots(self, folder_name):
        annots_path = os.path.join(self.base_path, folder_name, "annots.npy")
        annots = np.load(annots_path, allow_pickle=True).item()
        cams = annots["cams"]
        ims = annots["ims"]
        return cams, ims

    def parse_cameras(self, folder):
        json_path = os.path.join(self.base_path, folder, "cameras.json")
        with open(json_path, "r") as jfile:
            jdata = json.load(jfile)
        parsed_cameras = {}
        for camera_id, values in jdata.items():
            R = values["R"]
            T = values["T"]
            K = values["K"]
            cam_matrix = np.zeros((4, 4))
            cam_matrix[:3, :3] = np.array(R)
            cam_matrix[:3, 3] = np.array(T).reshape(-1)
            cam_matrix[3, 3] = 1
            cam_matrix = np.linalg.inv(cam_matrix)
            parsed_cameras[camera_id] = {
                "extrins": cam_matrix,
                "intrins": np.array(K)
            }
        return parsed_cameras

    def process_folders(self):
        items = []
        for folder in self.folders:
            folder_id = eval(''.join(c for c in folder if c.isdigit()).lstrip("0"))
            parsed_cameras = self.parse_cameras(folder)
            images_files = sorted(os.listdir(os.path.join(self.base_path, folder, "kinect_color", "kinect_000")))
            for filename in images_files[::6]:
                # joints_3d = np.zeros((32, 3))
                smpl_params_path = os.path.join(self.base_path, folder, 'smpl_params', filename.split(".")[0]+".npz")
                with np.load(smpl_params_path) as smpl_params:
                    global_orient = smpl_params['global_orient']
                    body_pose = smpl_params['body_pose']
                    betas = smpl_params['betas']
                    transl = smpl_params['transl']
                
                smpl_params = {
                    "global_orient": global_orient,
                    "body_pose": body_pose,
                    "betas": betas,
                    "transl": transl
                }

                # compute SMPL vertices in the world coordinate system
                output = self.smpl(
                    betas=torch.Tensor(betas).view(1, 10),
                    body_pose=torch.Tensor(body_pose).view(1, 23, 3),
                    global_orient=torch.Tensor(global_orient).view(1, 1, 3),
                    transl=torch.Tensor(transl).view(1, 3),
                    return_verts=True
                )
                joints_3d = output.vertices.detach().numpy().squeeze()
                
                images_info = []
                for i, camera_id in enumerate(os.listdir(os.path.join(self.base_path, folder, "kinect_color"))):
                    frame_path = os.path.join(self.base_path, folder, "kinect_color", camera_id, filename)
                    mask_path = os.path.join(self.base_path, folder, "kinect_mask", camera_id, filename)
                    camera_id_name = "_color_".join(camera_id.split("_"))
                    camera_id = ''.join(c for c in camera_id if c.isdigit()).lstrip("0")
                    camera_id = eval(camera_id) if len(camera_id) > 0 else 0
                    info_dict = {
                        "camera_id": camera_id,
                        "camera_pose": parsed_cameras[camera_id_name]["extrins"],
                        "intrinsics": parsed_cameras[camera_id_name]["intrins"],
                        "frame_path": frame_path,
                        "mask_path": mask_path 
                    }
                    images_info.append(info_dict)
                    if i == 0:
                        images_info.append(copy.deepcopy(info_dict))
                items.append((images_info, joints_3d, folder_id, smpl_params))
        return items

    def load_example_id(self, index,
                        trans = np.array([0.0, 0.0, 0.0]), scale=1.0):
        data_paths, points_3d, subject, smpl_params = self.items[index]

        all_rgbs = []
        all_world_view_transforms = []
        all_full_proj_transforms = []
        all_camera_centers = []
        all_view_to_world_transforms = []
        cam_ids = []
        background_colors = []
        masks = []
        all_focals_pixels = []
        pps_pixels = []
        crop_infos = []
        sherf_masks = []

        cam_ids = []
        cam_poses = []
        rgb_paths = []
        intrinsics = []
        masks_paths = []
        if self.overfit:
            indices = torch.randperm(len(data_paths)-1)
            indices = [0, 0] + indices.tolist()
        else:
            if self.dataset_name == "training":
                indices = torch.randperm(len(data_paths))
            else:
                indices = torch.arange(len(data_paths))
            indices = torch.concatenate((indices[:1], indices))
            indices = indices.tolist()
        indices = indices[:(self.num_training_images + 1)]
        data_paths = [data_paths[i] for i in indices]
        
        for item in data_paths:
            cam_ids.append(item["camera_id"])
            cam_poses.append(item["camera_pose"])
            rgb_paths.append(item["frame_path"])
            masks_paths.append(item["mask_path"])
            intrinsics.append(item["intrinsics"])

        random_background_color = self.cfg.data.random_background_color if hasattr(self.cfg.data, "random_background_color") else False
        background_color = 255 if hasattr(self.cfg.data, "white_background") and self.cfg.data.white_background == True else 0
        tight_crop = self.cfg.data.get("cropped", False)
        cam_infos = readCamerasInfosHumman(rgb_paths, masks_paths, cam_poses, intrinsics, None, cam_ids, random_background_color=random_background_color, bg_color=background_color, tight_crop=tight_crop)

        for i, cam_info in enumerate(cam_infos):
            R = cam_info.R
            T = cam_info.T
            if self.cfg.data.get("hmr_preprocessing", False) and i==0:
                all_rgbs.append(PILtoTorchHMR(cam_info.image, 
                (self.cfg.data.training_resolution, self.cfg.data.training_resolution))[:3, :, :])
            else:
                all_rgbs.append(PILtoTorch(cam_info.image, 
                (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])

            h_min, w_min, crop_size, h, w = cam_info.crop_info
            sherf_mask = get_mask(points_3d, h, w, intrinsics[i], np.transpose(R), T)
            
            sherf_mask = sherf_mask[h_min:h_min+crop_size, w_min:w_min+crop_size]
            sherf_mask = Image.fromarray(np.repeat(sherf_mask[:, :, None], 3, axis=-1)*(255//np.max(sherf_mask)))
            sherf_masks.append(PILtoTorch(sherf_mask, (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])

            world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
            view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

            projection_matrix = getProjectionMatrix(
                znear=self.cfg.data.znear,
                zfar=self.cfg.data.zfar,
                fovX=cam_info.FovX,
                fovY=cam_info.FovY,
                pX=cam_info.px,
                pY=cam_info.py
            ).transpose(0, 1)

            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            all_world_view_transforms.append(world_view_transform)
            all_view_to_world_transforms.append(view_world_transform)
            all_full_proj_transforms.append(full_proj_transform)
            all_camera_centers.append(camera_center)
            background_colors.append(torch.tensor(cam_info.color/255, dtype=torch.float32))
            masks.append(PILtoTorch(cam_info.mask, 
                                                        (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :][0, ...])
            all_focals_pixels.append(torch.tensor([fov2focal(cam_info.FovX, self.cfg.data.training_resolution),
                                                                fov2focal(cam_info.FovY, self.cfg.data.training_resolution)]))
            pps_pixels.append(torch.tensor([cam_info.px * self.cfg.data.training_resolution / 2,
                                                        cam_info.py * self.cfg.data.training_resolution / 2]))
            crop_infos.append(torch.tensor(cam_info.crop_info))
            

        all_world_view_transforms = torch.stack(all_world_view_transforms)
        all_view_to_world_transforms = torch.stack(all_view_to_world_transforms)
        all_full_proj_transforms = torch.stack(all_full_proj_transforms)
        all_camera_centers = torch.stack(all_camera_centers)
        all_rgbs = torch.stack(all_rgbs)
        cam_ids = torch.tensor(cam_ids)
        background_colors = torch.stack(background_colors)
        masks = torch.stack(masks)
        all_focals_pixels = torch.stack(all_focals_pixels)
        pps_pixels = torch.stack(pps_pixels)
        crop_infos = torch.stack(crop_infos)
        sherf_masks =  torch.stack(sherf_masks)

        ret = {
            "gt_images": all_rgbs,
            "world_view_transforms": all_world_view_transforms,
            "view_to_world_transforms": all_view_to_world_transforms,
            "full_proj_transforms": all_full_proj_transforms,
            "camera_centers": all_camera_centers,
            "points_3d": torch.tensor(points_3d),
            "cam_ids": cam_ids,
            "background_color": background_colors,
            "gt_masks": masks,
            "subject": torch.tensor(subject),
            "focals_pixels": all_focals_pixels,
            "pps_pixels": pps_pixels,
            "crops_info": crop_infos,
        }
        ret["sherf_masks"] = sherf_masks
        if smpl_params is not None:
            ret["global_orient"] = torch.tensor(smpl_params["global_orient"])
            ret["body_pose"] = torch.tensor(smpl_params["body_pose"])
            ret["betas"] = torch.tensor(smpl_params["betas"])
            ret["transl"] = torch.tensor(smpl_params["transl"])
        return ret

    def get_example_id(self, index):
        return str(index)

    
    def get_source_cw2wT(self, source_cameras_view_to_world):
        qs = []
        for c_idx in range(source_cameras_view_to_world.shape[0]):
            qs.append(matrix_to_quaternion(source_cameras_view_to_world[c_idx, :3, :3].transpose(0, 1)))
        return torch.stack(qs, dim=0)

    def __getitem__(self, index):
        images_and_camera_poses = self.load_example_id(index)

        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])

        return images_and_camera_poses