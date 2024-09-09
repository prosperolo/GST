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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from utils.general_utils import matrix_to_quaternion
import torch


WARNED = False


def get_extra_cameras_for_batch(batch_data, projection_matrix, num_cameras, relative_cams=False, mode="interp", radius=1, t=np.array([0., 0., 0.])):
    world_view_transforms = []
    view_world_transforms = []
    camera_centers = []

    if mode == "interp":
        cam_ids = batch_data["cam_ids"]
        cam_matrices = batch_data["world_view_transforms"].squeeze().cpu()
        cam_ids = cam_ids.squeeze().tolist()
        unique_ids = set(cam_ids) 
        num_extra_cameras = num_cameras // len(list(unique_ids))
        cam_matrices = [np.linalg.inv(c.transpose(0, 1)) for c in cam_matrices]
        loop_cameras_c2w_cmo = get_interpolated_cameras(cam_matrices, num_extra_cameras)
    elif mode == "loop":
        loop_cameras_c2w_cmo = get_loop_cameras(num_imgs_in_loop=num_cameras, radius=radius, t=t)
    else:
        raise ValueError

    # append the first example as conditioning
    for loop_camera_c2w_cmo in loop_cameras_c2w_cmo:
        view_world_transform = torch.from_numpy(loop_camera_c2w_cmo).transpose(0, 1).float()
        world_view_transform = torch.from_numpy(loop_camera_c2w_cmo).inverse().transpose(0, 1).float()
        camera_center = view_world_transform[3, :3].clone()

        camera_centers.append(camera_center)
        world_view_transforms.append(world_view_transform)
        view_world_transforms.append(view_world_transform)

    world_view_transforms = torch.stack(world_view_transforms)
    view_world_transforms = torch.stack(view_world_transforms)
    camera_centers = torch.stack(camera_centers)

    full_proj_transforms = world_view_transforms.bmm(projection_matrix.unsqueeze(0).expand(
        world_view_transforms.shape[0], 4, 4))

    images_and_camera_poses = {
            "world_view_transforms": world_view_transforms,
            "view_to_world_transforms": view_world_transforms,
            "full_proj_transforms": full_proj_transforms,
            "camera_centers": camera_centers
            }

    if relative_cams:
        images_and_camera_poses = make_poses_relative_to_first(images_and_camera_poses)
    images_and_camera_poses["source_cv2wT_quat"] = get_source_cw2wT(
                images_and_camera_poses["view_to_world_transforms"])

    for k, v in images_and_camera_poses.items():
        images_and_camera_poses[k] = v.unsqueeze(0)

    return images_and_camera_poses


def make_poses_relative_to_first(images_and_camera_poses):
    inverse_first_camera = images_and_camera_poses["world_view_transforms"][0].inverse().clone()
    for c in range(images_and_camera_poses["world_view_transforms"].shape[0]):
        images_and_camera_poses["world_view_transforms"][c] = torch.bmm(
                                            inverse_first_camera.unsqueeze(0),
                                            images_and_camera_poses["world_view_transforms"][c].unsqueeze(0)).squeeze(0)
        images_and_camera_poses["view_to_world_transforms"][c] = torch.bmm(
                                            images_and_camera_poses["view_to_world_transforms"][c].unsqueeze(0),
                                            inverse_first_camera.inverse().unsqueeze(0)).squeeze(0)
        images_and_camera_poses["full_proj_transforms"][c] = torch.bmm(
                                            inverse_first_camera.unsqueeze(0),
                                            images_and_camera_poses["full_proj_transforms"][c].unsqueeze(0)).squeeze(0)
        images_and_camera_poses["camera_centers"][c] = images_and_camera_poses["world_view_transforms"][c].inverse()[3, :3]
    return images_and_camera_poses


def get_source_cw2wT(source_cameras_view_to_world):
    qs = []
    for c_idx in range(source_cameras_view_to_world.shape[0]):
        qs.append(matrix_to_quaternion(source_cameras_view_to_world[c_idx, :3, :3].transpose(0, 1)))
    return torch.stack(qs, dim=0)


def visualize_cameras(ax, cameras, colors=['r', 'g', 'b'], ids=None):
    translations = np.array([cam[:3, 3] for cam in cameras])
    if np.max(translations) > 0:
        scale = np.max(translations) / 10 
    else:
        scale = 1
    vectors = np.array([[0., 0., scale, 1.], [0., scale, 0., 1.], [scale, 0., 0., 1.]]) 
    origin = np.array([0., 0., 0., 1.])
    for cam_i, camera in enumerate(cameras):
        for i, vector in enumerate(vectors):
            color = colors[i]
            transformed_vector = np.matmul(camera, vector)
            transformed_origin = np.matmul(camera, origin)
            ax.plot([transformed_origin[0], transformed_vector[0]], [transformed_origin[1], transformed_vector[1]], [transformed_origin[2], transformed_vector[2]], color=color)
        if ids is not None:
            ax.text(transformed_origin[0], transformed_origin[1], transformed_origin[2], str(ids[cam_i].item()), color="black")


def visualize_points(ax, xyz, marker='o', color="blue", ids=None):
    xs = xyz[:, 0]
    ys = xyz[:, 1]
    zs = xyz[:, 2]
    ax.scatter(xs, ys, zs, c=color, marker=marker)
    if ids is not None:
        for i in range(len(ids)):
            ax.text(xs[i], ys[i], zs[i], str(ids[i]), color="black")


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def get_interpolated_cameras(cameras_in, extra_cameras_per_pair=3):
    cameras_in = [c for c in cameras_in]
    cameras_in.append(cameras_in[0])
    pairs = [(cameras_in[i], cameras_in[i+1]) for i in range(len(cameras_in)-1)]
    interp_matrices = []
    for first, second in pairs:
        first_R = first[:3, :3]
        second_R = second[:3, :3]
        first_t = first[:3, 3]
        second_t = second[:3, 3]
        interp_rots = interpolate_rots(first_R, second_R, extra_cameras_per_pair)
        interp_ts = interpolate_translations(first_t, second_t, extra_cameras_per_pair)
        for i in range(interp_ts.shape[0]):
            matrix = np.zeros((4, 4))
            matrix[:3, :3] = interp_rots[i]
            matrix[:3, 3] = interp_ts[i]
            matrix[3, 3] = 1.
            interp_matrices.append(matrix)
    return interp_matrices


def interpolate_translations(first_t, second_t, num):
    diff = second_t - first_t
    steps = np.linspace(0, 1, num+2)
    interp_t = [first_t + step*diff for step in steps]
    return np.array(interp_t)[1:]


def interpolate_rots(first_R, second_R, num):
    matrices = np.stack((first_R, second_R))
    matrices = R.from_matrix(matrices)
    slerp = Slerp([0, 1], matrices)
    interp_rots = slerp(np.linspace(0, 1, num+2))
    return interp_rots.as_matrix()[1:]


def get_loop_cameras(num_imgs_in_loop, radius=2.0, t=np.array([0., 0., 0.])):
    all_cameras_c2w_cmo = []

    for i in range(num_imgs_in_loop):
        azimuth_angle = - np.pi * 2 * i / num_imgs_in_loop
        elevation_angle = np.pi / 25 #* np.sin(np.pi * i / num_imgs_in_loop)
        x = np.cos(azimuth_angle) * radius * np.cos(elevation_angle)
        y = np.sin(elevation_angle) * radius
        z = np.sin(azimuth_angle) * radius * np.cos(elevation_angle) 

        camera_T_c2w = np.array([x, y, z], dtype=np.float32)

        # in COLMAP / OpenCV convention: z away from camera, y down, x right
        camera_z = - camera_T_c2w / radius
        up = np.array([0, 1, 0], dtype=np.float32)
        camera_x = np.cross(up, camera_z)
        camera_x = camera_x / np.linalg.norm(camera_x)
        camera_y = np.cross(camera_z, camera_x)

        # camera_T_c2w -= np.array([0., 1., 0.])
        camera_T_c2w += t
        camera_c2w_cmo = np.hstack([camera_x[:, None], 
                                    camera_y[:, None], 
                                    camera_z[:, None], 
                                    camera_T_c2w[:, None]])
        camera_c2w_cmo = np.vstack([camera_c2w_cmo, np.array([0, 0, 0, 1], dtype=np.float32)[None, :]])

        all_cameras_c2w_cmo.append(camera_c2w_cmo)
    return all_cameras_c2w_cmo


def get_figure_of_8_cameras(num_imgs_in_loop, radius=2.0):

    all_cameras_c2w_cmo = []

    init_phase_elevation = - np.pi / 6

    for i in range(num_imgs_in_loop):
        azimuth_angle = np.pi / 3.5 * np.sin(4 * i * np.pi / num_imgs_in_loop)
        elevation_angle = (np.pi / 5) * np.sin(init_phase_elevation + i * np.pi / num_imgs_in_loop)
        x = np.cos(azimuth_angle) * radius * np.cos(elevation_angle)
        y = np.sin(azimuth_angle) * radius * np.cos(elevation_angle)
        z = np.sin(elevation_angle) * radius

        camera_T_c2w = np.array([x, y, z], dtype=np.float32)

        # in COLMAP / OpenCV convention: z away from camera, y down, x right
        camera_z = - camera_T_c2w / radius
        up = np.array([0, 0, -1], dtype=np.float32)
        camera_x = np.cross(up, camera_z)
        camera_x = camera_x / np.linalg.norm(camera_x)
        camera_y = np.cross(camera_z, camera_x)

        camera_c2w_cmo = np.hstack([camera_x[:, None], 
                                    camera_y[:, None], 
                                    camera_z[:, None], 
                                    camera_T_c2w[:, None]])
        camera_c2w_cmo = np.vstack([camera_c2w_cmo, np.array([0, 0, 0, 1], dtype=np.float32)[None, :]])

        all_cameras_c2w_cmo.append(camera_c2w_cmo)

    return all_cameras_c2w_cmo
