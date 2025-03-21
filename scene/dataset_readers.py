#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import re
from tqdm import tqdm
import os
import pickle
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.transformation_utils import axis_angle_to_matrix, matrix_to_axis_angle, axis_angle_to_quaternion
from utils.cos_similarity_utils import find_most_similar
import numpy as np
import torch
import json
import imageio
import cv2
import random
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from smpl.smpl_numpy import SMPL
from smplx.body_models import SMPLX
from data.dna_rendering.dna_rendering_sample_code.SMCReader import SMCReader
import trimesh

body_parts_smplx = {'l_leg': [1, 4, 7, 10],
              'r_leg': [2, 5, 8, 11],
              'center_body': [0, 3, 6, 9, 12, 15, 22, 23, 24],
              'l_arm': [13, 16, 18, 20],
              'r_arm': [14, 17, 19, 21],
                }
body_parts_smpl = {'l_leg': [1, 4, 7, 10],
              'r_leg': [2, 5, 8, 11],
              'center_body': [0, 3, 6, 9, 12, 15],
              'l_arm': [13, 16, 18, 20, 22],
              'r_arm': [14, 17, 19, 21, 23],
                }


def convert_to_camera_to_world(R, T):
    R_wc, _ = cv2.Rodrigues(np.array(R))
    R_cw = R_wc.T                       
    T_cw = -R_cw @ np.array(T).reshape(3, 1) 
    transform_matrix = np.eye(4)      
    transform_matrix[:3, :3] = R_cw     
    transform_matrix[:3, 3] = T_cw.flatten()
    return transform_matrix

class CameraInfo(NamedTuple):
    uid: int
    pose_id: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    bkgd_mask: np.array
    bound_mask: np.array
    width: int
    height: int
    smpl_param: dict
    world_vertex: np.array
    world_bound: np.array
    big_pose_smpl_param: dict
    big_pose_world_vertex: np.array
    big_pose_world_bound: np.array
    pair_t: list
    jump_note: list
    dataset_type: str

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    frame_len: int

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

##################################   ZJUMoCapRefine   ##################################

def get_camera_extrinsics_zju_mocap_refine(view_index, val=False, camera_view_num=36):
    def norm_np_arr(arr):
        return arr / np.linalg.norm(arr)

    def lookat(eye, at, up):
        zaxis = norm_np_arr(at - eye)
        xaxis = norm_np_arr(np.cross(zaxis, up))
        yaxis = np.cross(xaxis, zaxis)
        _viewMatrix = np.array([
            [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
            [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
            [-zaxis[0], -zaxis[1], -zaxis[2], np.dot(zaxis, eye)],
            [0       , 0       , 0       , 1     ]
        ])
        return _viewMatrix
    
    def fix_eye(phi, theta):
        camera_distance = 3
        return np.array([
            camera_distance * np.sin(theta) * np.cos(phi),
            camera_distance * np.sin(theta) * np.sin(phi),
            camera_distance * np.cos(theta)
        ])

    if val:
        eye = fix_eye(np.pi + 2 * np.pi * view_index / camera_view_num + 1e-6, np.pi/2 + np.pi/12 + 1e-6).astype(np.float32) + np.array([0, 0, -0.8]).astype(np.float32)
        at = np.array([0, 0, -0.8]).astype(np.float32)

        extrinsics = lookat(eye, at, np.array([0, 0, -1])).astype(np.float32)
    return extrinsics

def readCamerasZJUMoCapRefine(path, output_view, white_background, image_scaling=0.5, split='train', args=None, mono_test=False, render_novel_pose=False, sim_type=None, pose_step=-1, novel_view_vis=False):
    cam_infos = []

    if mono_test and render_novel_pose:
        if split == 'train':
            pose_start = 0
            pose_interval = 300
            pose_num = 1
        elif split == 'test':
            pose_start = 300
            pose_interval = 1
            pose_num = 200
    elif mono_test:
        if split == 'train':
            pose_start = 0
            pose_interval = 300
            pose_num = 1
        elif split == 'test':
            pose_start = 0
            pose_interval = 1
            pose_num = 300
    elif render_novel_pose:
        if split == 'train':
            pose_start = 0
            pose_interval = 300
            pose_num = 1
        elif split == 'test':
            pose_start = 300
            pose_interval = 20
            pose_num = 10
    else:
        if split == 'train':
            pose_start = 0
            pose_interval = 5
            pose_num = 60
        elif split == 'test':
            pose_start = 0
            pose_interval = 50
            pose_num = 6


    ann_file = os.path.join(path, 'annots.npy')
    annots = np.load(ann_file, allow_pickle=True).item()
    cams = annots['cams']
    ims = np.array([
        np.array(ims_data['ims'])[output_view]
        for ims_data in annots['ims']
    ])

    cam_inds = np.array([
        np.arange(len(ims_data['ims']))[output_view]
        for ims_data in annots['ims']
    ])

    smpl_model = SMPL(sex='neutral', model_dir='./assets/SMPL_NEUTRAL.pkl')

    # SMPL in canonical space
    big_pose_smpl_param = {}
    big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
    big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['shapes'] = np.zeros((1,10)).astype(np.float32)
    big_pose_smpl_param['poses'] = np.zeros((1,72)).astype(np.float32)
    big_pose_smpl_param['poses'][0, 5] = 45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 8] = -45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 23] = -30/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 26] = 30/180*np.array(np.pi)

    big_pose_xyz, _ = smpl_model(big_pose_smpl_param['poses'], big_pose_smpl_param['shapes'].reshape(-1))
    big_pose_xyz = (np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose()) + big_pose_smpl_param['Th']).astype(np.float32)

    # obtain the original bounds for point sampling
    big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
    big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
    big_pose_min_xyz -= 0.05
    big_pose_max_xyz += 0.05
    big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

    if render_novel_pose and split != 'train':
        assert pose_step != -1, "please set pose step!"
        pose_embedding = []
        for pose_index in range(0, 300, 5): # training pose

            i_n = int(os.path.basename(ims[pose_index][0])[:-4])
            smpl_param_path = os.path.join(path, "smpl_params", '{}.npy'.format(i_n))
            smpl_param = np.load(smpl_param_path, allow_pickle=True).item()
            smpl_param['poses'] = np.array(smpl_param['poses']).astype(np.float32)

            cur_pose_embedding = torch.tensor(smpl_param['poses']).reshape(1, -1, 3)
            cur_mat = axis_angle_to_matrix(cur_pose_embedding.squeeze())
            for i in range(pose_step):
                lst_model_idx = max(0, pose_index-5*(i+1))
                i_n_lst = int(os.path.basename(ims[lst_model_idx][0])[:-4])
                lst_smpl_param_path = os.path.join(path, "smpl_params", '{}.npy'.format(i_n_lst))
                lst_smpl_param = np.load(lst_smpl_param_path, allow_pickle=True).item()
                lst_smpl_param['poses'] = np.array(lst_smpl_param['poses']).astype(np.float32)
                lst_pose_embedding = torch.tensor(lst_smpl_param['poses']).reshape(1, -1, 3).squeeze()
                lst_mat = axis_angle_to_matrix(lst_pose_embedding) #(B,3,3)
                # calculate delta pose
                cur_delta_mat = torch.bmm(cur_mat, torch.linalg.inv(lst_mat)) #(B,3,3)
                cur_delta_pose = matrix_to_axis_angle(cur_delta_mat)
                # update embedding
                cur_pose_embedding = torch.cat([cur_pose_embedding, cur_delta_pose.unsqueeze(0)], dim=0)
                cur_mat = lst_mat
            cur_pose_embedding = cur_pose_embedding.transpose(0, 1)
            cur_pose_embedding = cur_pose_embedding.reshape(cur_pose_embedding.shape[0], -1)
            pose_embedding.append(cur_pose_embedding.unsqueeze(0))
        pose_embedding = torch.cat(pose_embedding, dim=0)  # [100, 25, 3*(1+step)]
        
        pose_body_parts = {}
        for name in body_parts_smpl:
            pose_part = pose_embedding[:, body_parts_smpl[name], :]
            pose_body_parts[name] = pose_part


    idx = 0
    pair_t_lst = []
    for pose_index in tqdm(range(pose_start, pose_start+pose_num*pose_interval, pose_interval)):
        
        i = int(os.path.basename(ims[pose_index][0])[:-4])
        vertices_path = os.path.join(path, 'smpl_vertices', '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)

        smpl_param_path = os.path.join(path, "smpl_params", '{}.npy'.format(i))
        smpl_param = np.load(smpl_param_path, allow_pickle=True).item()
        Rh = smpl_param['Rh']
        smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
        smpl_param['Th'] = smpl_param['Th'].astype(np.float32)
        smpl_param['shapes'] = smpl_param['shapes'].astype(np.float32)
        smpl_param['poses'] = smpl_param['poses'].astype(np.float32)

        pair_t = []
        jump_note = []

        if render_novel_pose and split != 'train':
            jump_note = [-1 for i in range(len(pose_body_parts))]
            cur_pose_embedding = torch.tensor(smpl_param['poses']).reshape(1, -1, 3)   # no global no hand
            cur_mat = axis_angle_to_matrix(cur_pose_embedding.squeeze())
            for i in range(pose_step):
                lst_model_idx = max(0, pose_index-5*(i+1))
                i_n_lst = int(os.path.basename(ims[lst_model_idx][0])[:-4])
                lst_smpl_param_path = os.path.join(path, "smpl_params", '{}.npy'.format(i_n_lst))
                lst_smpl_param = np.load(lst_smpl_param_path, allow_pickle=True).item()
                lst_smpl_param['poses'] = np.array(lst_smpl_param['poses']).astype(np.float32)
                lst_pose_embedding = torch.tensor(lst_smpl_param['poses']).reshape(1, -1, 3).squeeze()
                lst_mat = axis_angle_to_matrix(lst_pose_embedding) #(B,3,3)
                # calculate delta pose
                cur_delta_mat = torch.bmm(cur_mat, torch.linalg.inv(lst_mat)) #(B,3,3)
                cur_delta_pose = matrix_to_axis_angle(cur_delta_mat)
                # update embedding
                cur_pose_embedding = torch.cat([cur_pose_embedding, cur_delta_pose.unsqueeze(0)], dim=0)
                cur_mat = lst_mat
            cur_pose_embedding = cur_pose_embedding.transpose(0, 1)
            cur_pose_embedding = cur_pose_embedding.reshape(cur_pose_embedding.shape[0], -1).unsqueeze(0)

            p_idx = 0
            for name in body_parts_smpl:
                cur_pose_part = cur_pose_embedding[:, body_parts_smpl[name], :]
                if sim_type == 'axis-angle':
                    pose_body_part = pose_body_parts[name]
                    maxsim_t, sim = find_most_similar(cur_pose_part, pose_body_part)
                elif sim_type == 'quater':
                    cur_pose_part = axis_angle_to_quaternion(cur_pose_part)
                    pose_body_part = axis_angle_to_quaternion(pose_body_parts[name])
                    maxsim_t, sim = find_most_similar(cur_pose_part, pose_body_part)
                elif sim_type == 'euclidean':
                    pose_body_part = pose_body_parts[name]
                    maxsim_t, sim = find_most_similar(cur_pose_part, pose_body_part, sim_type='euclidean')
                maxsim, maxt = maxsim_t     

                if len(pair_t_lst) == 0:      
                    selected_value = maxt[0]
                    pair_t.append(maxt[0])
                else:
                    t_lst = pair_t_lst[p_idx]
                    distances = torch.abs(maxt - t_lst)
                    neighbor_num = len(distances[distances<3])
                    print(t_lst)
                    print(maxt[distances<3])
                    print(neighbor_num)

                    if neighbor_num > 1:
                        w_ = maxsim[distances < 3][1]/(maxsim[distances < 3][0]+maxsim[distances < 3][1])
                        selected_value = w_ * maxt[distances < 3][0] + (1 - w_) * maxt[distances < 3][1]
                    elif neighbor_num == 1:
                        selected_value = maxt[distances < 3][0]
                    else:
                        # jitter
                        jump_note[p_idx] = t_lst
                        selected_value = maxt[0]
                        
                    pair_t.append(selected_value)

                p_idx+=1

            pair_t_lst = pair_t


        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bound = np.stack([min_xyz, max_xyz], axis=0)

        for view_index in range(len(output_view)):

            if novel_view_vis:
                view_index_look_at = view_index
                view_index = 0

            # Load image, mask, K, D, R, T
            image_path = os.path.join(path, ims[pose_index][view_index].replace('\\', '/'))
            image_name = ims[pose_index][view_index].split('.')[0]
            image = np.array(imageio.imread(image_path).astype(np.float32)/255.)

            msk_path = image_path.replace('images', 'mask').replace('jpg', 'png')
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)

            if not novel_view_vis:
                cam_ind = cam_inds[pose_index][view_index]
                K = np.array(cams['K'][cam_ind])
                D = np.array(cams['D'][cam_ind])
                R = np.array(cams['R'][cam_ind])
                T = np.array(cams['T'][cam_ind]) / 1000.

                image = cv2.undistort(image, K, D)
                msk = cv2.undistort(msk, K, D)
            else:
                pose = np.matmul(np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]]), get_camera_extrinsics_zju_mocap_refine(view_index_look_at, val=True))
                R = pose[:3,:3]
                T = pose[:3, 3].reshape(-1, 1)
                cam_ind = cam_inds[pose_index][view_index]
                K = np.array(cams['K'][cam_ind])

            image[msk == 0] = 1 if white_background else 0

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3,3:4] = T

            # get the world-to-camera transform and set R, T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Reduce the image resolution by ratio, then remove the back ground
            ratio = image_scaling
            if ratio != 1.:
                H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                K[:2] = K[:2] * ratio

            image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")

            focalX = K[0,0]
            focalY = K[1,1]
            FovX = focal2fov(focalX, image.size[0])
            FovY = focal2fov(focalY, image.size[1])

            # get bounding mask and bcakground mask
            bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])
            bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.byte))

            bkgd_mask = Image.fromarray(np.array(msk*255.0, dtype=np.byte))

            cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask, 
                            bound_mask=bound_mask, width=image.size[0], height=image.size[1], 
                            smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound, 
                            big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz, 
                            big_pose_world_bound=big_pose_world_bound, pair_t=pair_t, jump_note=jump_note, dataset_type='zju'))

            idx += 1
            
    return cam_infos, pose_interval*pose_num

def readZJUMoCapRefineInfo(path, white_background, output_path, eval, args, mono_test=False, render_novel_pose=False, sim_type=None, pose_step=-1):
    train_view = [0, 3, 6, 9, 12, 15, 18, 21]
    test_view = [i for i in range(0, 23)]
    test_view = [i for i in test_view if i not in train_view]
    if mono_test:
        train_view = [8]
        test_view = [8]
    print("Reading Training Transforms")
    train_cam_infos, frame_len = readCamerasZJUMoCapRefine(path, train_view, white_background, image_scaling=0.5, split='train', args=args, mono_test=mono_test, render_novel_pose=render_novel_pose)
    print("Reading Test Transforms")
    test_cam_infos, frame_len = readCamerasZJUMoCapRefine(path, test_view, white_background, image_scaling=0.5, split='test', args=args, mono_test=mono_test, render_novel_pose=render_novel_pose, sim_type=sim_type, pose_step=pose_step)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    ply_path = os.path.join('output', output_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 6890 #100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = train_cam_infos[0].big_pose_world_vertex

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
                           ply_path=ply_path,
                           frame_len=frame_len)
    return scene_info

##################################   DNARendering   ##################################

def readCamerasDNARendering(path, output_view, white_background, image_scaling=0.25, \
                            split='train', args=None, mono_test=False, render_novel_pose=False, sim_type=None, pose_step=-1):
    cam_infos = []

    if mono_test and render_novel_pose:
        if split == 'train':
            pose_start = 0
            pose_interval = 150
            pose_num = 1
        elif split == 'test':
            pose_start = 110
            pose_interval = 1
            pose_num = 40
    elif mono_test:
        if split == 'train':
            pose_start = 0
            pose_interval = 100
            pose_num = 1
        elif split == 'test':
            pose_start = 0
            pose_interval = 1
            pose_num = 100
    elif render_novel_pose:
        if split == 'train':
            pose_start = 0
            pose_interval = 150
            pose_num = 1
        elif split == 'test':
            pose_start = 110
            pose_interval = 5
            pose_num = 8
    else:
        if split == 'train':
            pose_start = 0
            pose_interval = 1
            pose_num = 100
        elif split == 'test':
            pose_start = 0
            pose_interval = 5
            pose_num = 20


    data_name = path.split('/')[-1]
    main_path = os.path.join(path, data_name+'.smc')
    smc_reader = SMCReader(main_path)
    annots_file_path = main_path.replace('main', 'annotations').split('.')[0] + '_annots.smc'
    smc_annots_reader = SMCReader(annots_file_path)

    gender = smc_reader.actor_info['gender']
    smpl_model = {}
    smpl_model[gender] = SMPLX('./assets/models/smplx/', smpl_type='smplx',
                                gender=gender, use_face_contour=True, flat_hand_mean=False, use_pca=False, 
                                num_pca_comps=24, num_betas=10,
                                num_expression_coeffs=10,
                                ext='npz')

    # SMPL in canonical space
    big_pose_smpl_param = {}
    big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
    big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['global_orient'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['betas'] = np.zeros((1,10)).astype(np.float32)
    big_pose_smpl_param['body_pose'] = np.zeros((1,63)).astype(np.float32)
    big_pose_smpl_param['jaw_pose'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['left_hand_pose'] = np.zeros((1,45)).astype(np.float32)
    big_pose_smpl_param['right_hand_pose'] = np.zeros((1,45)).astype(np.float32)
    big_pose_smpl_param['leye_pose'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['reye_pose'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['expression'] = np.zeros((1,10)).astype(np.float32)
    big_pose_smpl_param['transl'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['body_pose'][0, 2] = 45/180*np.array(np.pi)
    big_pose_smpl_param['body_pose'][0, 5] = -45/180*np.array(np.pi)
    big_pose_smpl_param['body_pose'][0, 20] = -30/180*np.array(np.pi)
    big_pose_smpl_param['body_pose'][0, 23] = 30/180*np.array(np.pi)

    big_pose_smpl_param_tensor= {}
    for key in big_pose_smpl_param.keys():
        big_pose_smpl_param_tensor[key] = torch.from_numpy(big_pose_smpl_param[key])

    body_model_output = smpl_model[gender](
        global_orient=big_pose_smpl_param_tensor['global_orient'],
        betas=big_pose_smpl_param_tensor['betas'],
        body_pose=big_pose_smpl_param_tensor['body_pose'],
        jaw_pose=big_pose_smpl_param_tensor['jaw_pose'],
        left_hand_pose=big_pose_smpl_param_tensor['left_hand_pose'],
        right_hand_pose=big_pose_smpl_param_tensor['right_hand_pose'],
        leye_pose=big_pose_smpl_param_tensor['leye_pose'],
        reye_pose=big_pose_smpl_param_tensor['reye_pose'],
        expression=big_pose_smpl_param_tensor['expression'],
        transl=big_pose_smpl_param_tensor['transl'],
        return_full_pose=True,
    )

    big_pose_smpl_param['poses'] = body_model_output.full_pose.detach()
    if args.smpl_type == 'simple_smplx':
        big_pose_smpl_param['poses'] = big_pose_smpl_param['poses'][:, :75]
    big_pose_smpl_param['shapes'] = np.concatenate([big_pose_smpl_param['betas'], big_pose_smpl_param['expression']], axis=-1)
    big_pose_xyz = np.array(body_model_output.vertices.detach()).reshape(-1,3).astype(np.float32)
    
    # obtain the original bounds for point sampling
    big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
    big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
    big_pose_min_xyz -= 0.05
    big_pose_max_xyz += 0.05
    big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

    if render_novel_pose:
        assert pose_step != -1, "please set pose step!"
        pose_embedding = []
        for pose_index in range(0, 100, 1):
            model_path = os.path.join(path, 'model')    
            current_model = model_path+f'/{str(pose_index).zfill(6)}.npz'
            loaded_data = np.load(current_model)
            smpl_param = {key: loaded_data[key] for key in loaded_data if key != 'xyz'}
            cur_pose_embedding = torch.tensor(smpl_param['poses'][:, 0:75]).reshape(1, -1, 3)   # no global no hand
            cur_mat = axis_angle_to_matrix(cur_pose_embedding.squeeze())
            for i in range(pose_step):
                lst_model_idx = max(0, pose_index-i-1)
                lst_loaded_data = np.load(model_path+f'/{str(lst_model_idx).zfill(6)}.npz')
                lst_smpl_param = {key: lst_loaded_data[key] for key in lst_loaded_data if key != 'xyz'}
                lst_pose_embedding = torch.tensor(lst_smpl_param['poses'][:, 0:75]).reshape(1, -1, 3).squeeze()
                lst_mat = axis_angle_to_matrix(lst_pose_embedding) #(B,3,3)
                # calculate delta pose
                cur_delta_mat = torch.bmm(cur_mat, torch.linalg.inv(lst_mat)) #(B,3,3)
                cur_delta_pose = matrix_to_axis_angle(cur_delta_mat)
                # update embedding
                cur_pose_embedding = torch.cat([cur_pose_embedding, cur_delta_pose.unsqueeze(0)], dim=0)
                cur_mat = lst_mat
            cur_pose_embedding = cur_pose_embedding.transpose(0, 1)
            cur_pose_embedding = cur_pose_embedding.reshape(cur_pose_embedding.shape[0], -1)
            pose_embedding.append(cur_pose_embedding.unsqueeze(0))
        pose_embedding = torch.cat(pose_embedding, dim=0)  # [100, 25, 3*(1+step)]
        
        pose_body_parts = {}
        for name in body_parts_smplx:
            pose_part = pose_embedding[:, body_parts_smplx[name], :]
            pose_body_parts[name] = pose_part

    idx = 0
    pair_t_lst = []
    
    for pose_index in range(pose_start, pose_start+pose_num*pose_interval, pose_interval):

        model_path = os.path.join(path, 'model')    
        os.makedirs(model_path, exist_ok=True)
        current_model = model_path+f'/{str(pose_index).zfill(6)}.npz'

        if not os.path.exists(current_model):
            print('creating model '+f'{str(pose_index).zfill(6)}.npz')

            smpl_dict = smc_annots_reader.get_SMPLx(Frame_id=pose_index)
            smpl_data = {}
            smpl_data['global_orient'] = smpl_dict['fullpose'][0].reshape(-1)
            smpl_data['body_pose'] = smpl_dict['fullpose'][1:22].reshape(-1)
            smpl_data['jaw_pose'] = smpl_dict['fullpose'][22].reshape(-1)
            smpl_data['leye_pose'] = smpl_dict['fullpose'][23].reshape(-1)
            smpl_data['reye_pose'] = smpl_dict['fullpose'][24].reshape(-1)
            smpl_data['left_hand_pose'] = smpl_dict['fullpose'][25:40].reshape(-1)
            smpl_data['right_hand_pose'] = smpl_dict['fullpose'][40:55].reshape(-1)
            smpl_data['transl'] = smpl_dict['transl'].reshape(-1)
            smpl_data['betas'] = smpl_dict['betas'].reshape(-1)
            smpl_data['expression'] = smpl_dict['expression'].reshape(-1)

            # load smpl data
            smpl_param = {
                'global_orient': np.expand_dims(smpl_data['global_orient'].astype(np.float32), axis=0),
                'transl': np.expand_dims(smpl_data['transl'].astype(np.float32), axis=0),
                'body_pose': np.expand_dims(smpl_data['body_pose'].astype(np.float32), axis=0),
                'jaw_pose': np.expand_dims(smpl_data['jaw_pose'].astype(np.float32), axis=0),
                'betas': np.expand_dims(smpl_data['betas'].astype(np.float32), axis=0),
                'expression': np.expand_dims(smpl_data['expression'].astype(np.float32), axis=0),
                'leye_pose': np.expand_dims(smpl_data['leye_pose'].astype(np.float32), axis=0),
                'reye_pose': np.expand_dims(smpl_data['reye_pose'].astype(np.float32), axis=0),
                'left_hand_pose': np.expand_dims(smpl_data['left_hand_pose'].astype(np.float32), axis=0),
                'right_hand_pose': np.expand_dims(smpl_data['right_hand_pose'].astype(np.float32), axis=0),
                }

            smpl_param['R'] = np.eye(3).astype(np.float32)
            smpl_param['Th'] = smpl_param['transl'].astype(np.float32)

            smpl_param_tensor = {}
            for key in smpl_param.keys():
                smpl_param_tensor[key] = torch.from_numpy(smpl_param[key])

            body_model_output = smpl_model[gender](
                global_orient=smpl_param_tensor['global_orient'],
                betas=smpl_param_tensor['betas'],
                body_pose=smpl_param_tensor['body_pose'],
                jaw_pose=smpl_param_tensor['jaw_pose'],
                left_hand_pose=smpl_param_tensor['left_hand_pose'],
                right_hand_pose=smpl_param_tensor['right_hand_pose'],
                leye_pose=smpl_param_tensor['leye_pose'],
                reye_pose=smpl_param_tensor['reye_pose'],
                expression=smpl_param_tensor['expression'],
                transl=smpl_param_tensor['transl'],
                return_full_pose=True,
            )
            
            smpl_param['poses'] = body_model_output.full_pose.detach()
            smpl_param['shapes'] = np.concatenate([smpl_param['betas'], smpl_param['expression']], axis=-1)
            xyz = np.array(body_model_output.vertices.detach()).reshape(-1,3).astype(np.float32)
            np.savez(current_model, **smpl_param, xyz=xyz)

        
        else:
            loaded_data = np.load(current_model)
            smpl_param = {key: loaded_data[key] for key in loaded_data if key != 'xyz'}
            xyz = loaded_data['xyz']
        
        if args.smpl_type == 'simple_smplx':
            smpl_param['poses'] = smpl_param['poses'][:, :75]

        pair_t = []
        jump_note = []
        if render_novel_pose:
            jump_note = [-1 for i in range(len(pose_body_parts))]
            cur_pose_embedding = torch.tensor(smpl_param['poses'][:, :75]).reshape(1, -1, 3)   # no global no hand
            cur_mat = axis_angle_to_matrix(cur_pose_embedding.squeeze())
            for i in range(pose_step):
                lst_model_idx = max(0, pose_index-i-1)
                lst_loaded_data = np.load(model_path+f'/{str(lst_model_idx).zfill(6)}.npz')
                lst_smpl_param = {key: lst_loaded_data[key] for key in lst_loaded_data if key != 'xyz'}
                lst_pose_embedding = torch.tensor(lst_smpl_param['poses'][:, :75]).reshape(1, -1, 3).squeeze()
                lst_mat = axis_angle_to_matrix(lst_pose_embedding) #(B,3,3)
                # calculate delta pose
                cur_delta_mat = torch.bmm(cur_mat, torch.linalg.inv(lst_mat)) #(B,3,3)
                cur_delta_pose = matrix_to_axis_angle(cur_delta_mat)
                # update embedding
                cur_pose_embedding = torch.cat([cur_pose_embedding, cur_delta_pose.unsqueeze(0)], dim=0)
                cur_mat = lst_mat
            cur_pose_embedding = cur_pose_embedding.transpose(0, 1)
            cur_pose_embedding = cur_pose_embedding.reshape(cur_pose_embedding.shape[0], -1).unsqueeze(0)

            p_idx = 0
            for name in body_parts_smplx:
                cur_pose_part = cur_pose_embedding[:, body_parts_smplx[name], :]
                if sim_type == 'axis-angle':
                    pose_body_part = pose_body_parts[name]
                    maxsim_t, sim = find_most_similar(cur_pose_part, pose_body_part)
                elif sim_type == 'quater':
                    cur_pose_part = axis_angle_to_quaternion(cur_pose_part)
                    pose_body_part = axis_angle_to_quaternion(pose_body_parts[name])
                    maxsim_t, sim = find_most_similar(cur_pose_part, pose_body_part)
                elif sim_type == 'euclidean':
                    pose_body_part = pose_body_parts[name]
                    maxsim_t, sim = find_most_similar(cur_pose_part, pose_body_part, sim_type='euclidean')
                maxsim, maxt = maxsim_t     

                if len(pair_t_lst) == 0:      
                    selected_value = maxt[0]
                    pair_t.append(maxt[0])
                else:
                    t_lst = pair_t_lst[p_idx]
                    distances = torch.abs(maxt - t_lst)
                    neighbor_num = len(distances[distances<3])

                    if neighbor_num > 1:
                        w_ = maxsim[distances < 3][1]/(maxsim[distances < 3][0]+maxsim[distances < 3][1])
                        selected_value = w_ * maxt[distances < 3][0] + (1 - w_) * maxt[distances < 3][1]
                    elif neighbor_num == 1:
                        selected_value = maxt[distances < 3][0]
                    else:
                        # jitter
                        jump_note[p_idx] = t_lst
                        selected_value = maxt[0]

                    pair_t.append(selected_value)

                p_idx+=1

            pair_t_lst = pair_t

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bound = np.stack([min_xyz, max_xyz], axis=0)
    
        for view_index in output_view:
            
            # Load K, R, T
            cam_params = smc_annots_reader.get_Calibration(view_index)
            K = cam_params['K']
            D = cam_params['D'] # k1, k2, p1, p2, k3
            RT = cam_params['RT']
            R = RT[:3, :3]
            T = RT[:3, 3]

            dsr = int(1/image_scaling)
            # Load image, mask
            img_path = os.path.join(path, 'images_'+str(dsr)) + f'/{str(view_index).zfill(2)}'
            bkgd_mask_path = os.path.join(path, 'masks_'+str(dsr)) + f'/{str(view_index).zfill(2)}'
            os.makedirs(img_path, exist_ok=True)
            os.makedirs(bkgd_mask_path, exist_ok=True)
            image_name = os.path.join(img_path, str(pose_index).zfill(6)+'.jpg')
            bkgd_mask_name = os.path.join(bkgd_mask_path, str(pose_index).zfill(6)+'.png')

            if not os.path.exists(image_name):
                print('creating view ', str(view_index))
                if int(view_index)<48:
                    image_original = smc_reader.get_img('Camera_5mp', int(view_index), Image_type='color', Frame_id=int(pose_index))
                else:
                    image_original = smc_reader.get_img('Camera_12mp', int(view_index), Image_type='color', Frame_id=int(pose_index))
                
                msk = smc_annots_reader.get_mask(view_index, Frame_id=pose_index)
                msk[msk!=0] = 255
                msk = np.array(msk) / 255.
                image = np.array(image_original) 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.
                image = cv2.undistort(image, K, D)
                msk = cv2.undistort(msk, K, D)
                
                image[msk == 0] = 1 if white_background else 0

                # Reduce the image resolution by ratio, then remove the back ground
                ratio = image_scaling
                if ratio != 1.:
                    H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                    K[:2] = K[:2] * ratio

                image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")
                image.save(image_name)
                
                bkgd_mask = Image.fromarray(np.array(msk*255.0, dtype=np.byte))
                bkgd_mask.save(bkgd_mask_name)
            
            else:
                image = Image.open(image_name)
                bkgd_mask = Image.open(bkgd_mask_name)
                ratio = image_scaling
                
                if ratio != 1.:
                    K[:2] = K[:2] * ratio

            c2w = np.eye(4)
            c2w[:3,:3] = R
            c2w[:3,3:4] = T.reshape(-1, 1)

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            # if split != 'train':
            #     print(w2c)
            #     np.savetxt('render360/'+str(view_index)+'.txt', w2c, fmt='%.6f')
            # if mono_test:
            #     w2c = np.loadtxt(f'render360_dense/{pose_index:03d}.txt')
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            focalX = K[0,0]
            focalY = K[1,1]
            FovX = focal2fov(focalX, image.size[0])
            FovY = focal2fov(focalY, image.size[1])
            
            # get bounding mask and bcakground mask
            bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])
            bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.byte))

            cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=img_path, image_name=image_name, bkgd_mask=bkgd_mask, 
                            bound_mask=bound_mask, width=image.size[0], height=image.size[1], 
                            smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound, 
                            big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz, 
                            big_pose_world_bound=big_pose_world_bound, pair_t=pair_t, jump_note=jump_note, dataset_type='dna'))

            idx += 1

    return cam_infos, pose_interval*pose_num


def readDNARenderingInfo(path, white_background, output_path, eval, args, mono_test=False, render_novel_pose=False, sim_type=None, pose_step=-1):

    train_view = [i for i in range(0, 48, 2)]
    test_view = [i for i in range(48, 60, 2)]
    if mono_test:
        print('U R doing monocular test!')
        test_view = [51]

    print("Reading Training Transforms")
    train_cam_infos, frame_len = readCamerasDNARendering(path, train_view, white_background, image_scaling=0.25, split='train', args=args, mono_test=mono_test, render_novel_pose=render_novel_pose, sim_type=sim_type, pose_step=pose_step)
    print("Reading Test Transforms")
    test_cam_infos, frame_len = readCamerasDNARendering(path, test_view, white_background, \
        image_scaling=0.25, split='test', args=args, mono_test=mono_test, render_novel_pose=render_novel_pose, sim_type=sim_type, pose_step=pose_step)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    # ply_path = os.path.join(path, "points3d.ply")
    ply_path = os.path.join('output', output_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10475
        print(f"Generating big pose point cloud ({num_pts})...")
        
        # # We create random points inside the bounds of the synthetic Blender scenes
        xyz = train_cam_infos[0].big_pose_world_vertex
        
        # num_pts = 10_000

        # print(f"Generating random point cloud ({num_pts})...")
        
        # # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((num_pts, 3)) * (bigpose_bounds[1] - bigpose_bounds[0]) + bigpose_bounds[0]
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
                           ply_path=ply_path,
                           frame_len=frame_len)
    return scene_info

##################################   HiFi4G   ##################################

def readHiFi4GInfo(path, white_background, output_path, eval, args, mono_test=False, render_novel_pose=False, sim_type=None, pose_step=-1):

    train_view = [str(i) for i in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 36]]
    test_view = [str(i) for i in [2, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35]]
    if mono_test:
        train_view = ["2"]
        test_view = ["2"]


    print("Reading Training Transforms")
    train_cam_infos, frame_len = readCamerasHiFi4G(path, train_view, white_background, image_scaling=0.25, split='train', mono_test=mono_test, render_novel_pose=render_novel_pose)
    print("Reading Test Transforms")
    test_cam_infos, frame_len = readCamerasHiFi4G(path, test_view, white_background, image_scaling=0.25, split='test', args=args, mono_test=mono_test, render_novel_pose=render_novel_pose, sim_type=sim_type, pose_step=pose_step)

    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    ply_path = os.path.join('output', output_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 6890 #100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = train_cam_infos[0].big_pose_world_vertex

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
                           ply_path=ply_path,
                           frame_len=frame_len)
    return scene_info


def readCamerasHiFi4G(path, output_view, white_background, image_scaling=0.5, split='train', args=None, mono_test=False, render_novel_pose=False, sim_type=None, pose_step=-1):
    cam_infos = []

    if mono_test and render_novel_pose:
        if split == 'train':
            pose_start = 0
            pose_interval = 200
            pose_num = 1
        elif split == 'test':
            pose_start = 150
            pose_interval = 1
            pose_num = 50
    elif mono_test:
        if split == 'train':
            pose_start = 0
            pose_interval = 200
            pose_num = 1
        elif split == 'test':
            pose_start = 0
            pose_interval = 1
            pose_num = 140
    elif render_novel_pose:
        if split == 'train':
            pose_start = 0
            pose_interval = 200
            pose_num = 1
        elif split == 'test':
            pose_start = 150
            pose_interval = 10
            pose_num = 5
    else:
        if split == 'train':
            pose_start = 0
            pose_interval = 2
            pose_num = 70
        elif split == 'test':
            pose_start = 0
            pose_interval = 20
            pose_num = 7


    img_names = os.listdir(os.path.join(path, 'images', output_view[0]))
    img_names.sort()

    ims = np.array([
        np.array([
            os.path.join('images', view, fname) for view in output_view
        ]) for fname in img_names
    ])

    xyz_bound_flex = 0.5

    smpl_model = SMPL(sex='neutral', model_dir='./assets/SMPL_NEUTRAL.pkl')

    # SMPL in canonical space
    big_pose_smpl_param = {}
    big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
    big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['shapes'] = np.zeros((1,10)).astype(np.float32)
    big_pose_smpl_param['poses'] = np.zeros((1,72)).astype(np.float32)
    big_pose_smpl_param['poses'][0, 5] = 45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 8] = -45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 23] = -30/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 26] = 30/180*np.array(np.pi)

    big_pose_xyz, _ = smpl_model(big_pose_smpl_param['poses'], big_pose_smpl_param['shapes'].reshape(-1))
    big_pose_xyz = (np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose()) + big_pose_smpl_param['Th']).astype(np.float32)

    # obtain the original bounds for point sampling
    big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
    big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
    big_pose_min_xyz -= xyz_bound_flex
    big_pose_max_xyz += xyz_bound_flex
    big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)
    
    cam_path = os.path.join(path, 'camera.json')
    with open(cam_path, 'r') as f:
        cam_ls = json.load(f)
    
    extrinsics = {}

    for item in cam_ls:
        camera_id = item['camera_id']
        extrinsics[camera_id] = item
    
    if render_novel_pose:
        assert pose_step != -1, "please set pose step!"
        pose_embedding = []
        for pose_index in range(0, 140, 1): # training pose

            i_n = os.path.basename(ims[pose_index][0])[:-4].zfill(6)

            smpl_param_path = os.path.join(path, "smpl/smpl", i_n+'.json')
            with open(smpl_param_path, 'r', encoding='utf-8') as file:
                smpl_param = json.load(file)[0]
            smpl_param['poses'] = np.array(smpl_param['poses']).astype(np.float32)

            cur_pose_embedding = torch.tensor(smpl_param['poses']).reshape(1, -1, 3)
            cur_mat = axis_angle_to_matrix(cur_pose_embedding.squeeze())
            for i in range(pose_step):
                lst_model_idx = max(0, pose_index-i-1)
                i_n_lst = os.path.basename(ims[lst_model_idx][0])[:-4].zfill(6)
                lst_smpl_param_path = os.path.join(path, "smpl/smpl", i_n_lst+'.json')
                with open(lst_smpl_param_path, 'r', encoding='utf-8') as file:
                    lst_smpl_param = json.load(file)[0]
                lst_smpl_param['poses'] = np.array(lst_smpl_param['poses']).astype(np.float32)
                lst_pose_embedding = torch.tensor(lst_smpl_param['poses']).reshape(1, -1, 3).squeeze()
                lst_mat = axis_angle_to_matrix(lst_pose_embedding) #(B,3,3)
                # calculate delta pose
                cur_delta_mat = torch.bmm(cur_mat, torch.linalg.inv(lst_mat)) #(B,3,3)
                cur_delta_pose = matrix_to_axis_angle(cur_delta_mat)
                # update embedding
                cur_pose_embedding = torch.cat([cur_pose_embedding, cur_delta_pose.unsqueeze(0)], dim=0)
                cur_mat = lst_mat
            cur_pose_embedding = cur_pose_embedding.transpose(0, 1)
            cur_pose_embedding = cur_pose_embedding.reshape(cur_pose_embedding.shape[0], -1)
            pose_embedding.append(cur_pose_embedding.unsqueeze(0))
        pose_embedding = torch.cat(pose_embedding, dim=0)  # [100, 25, 3*(1+step)]
        
        pose_body_parts = {}
        for name in body_parts_smpl:
            pose_part = pose_embedding[:, body_parts_smpl[name], :]
            pose_body_parts[name] = pose_part

    idx = 0
    pair_t_lst = []
    for pose_index in tqdm(range(pose_start, pose_start+pose_num*pose_interval, pose_interval)):

        i = os.path.basename(ims[pose_index][0])[:-4].zfill(6)

        vertices_path = os.path.join(path, 'smpl_vertices', i+'.json')
        with open(vertices_path, 'r', encoding='utf-8') as file:
            xyz = np.array(json.load(file)[0]['vertices']).astype(np.float32)

        smpl_param_path = os.path.join(path, "smpl/smpl", i+'.json')
        with open(smpl_param_path, 'r', encoding='utf-8') as file:
            smpl_param = json.load(file)[0]

        smpl_param['Rh'] = np.array(smpl_param['Rh'])
        smpl_param['R'] = cv2.Rodrigues(smpl_param['Rh'])[0].astype(np.float32)
        smpl_param['Th'] = np.array(smpl_param['Th']).astype(np.float32)
        smpl_param['shapes'] = np.array(smpl_param['shapes']).astype(np.float32)
        smpl_param['poses'] = np.array(smpl_param['poses']).astype(np.float32)

        pair_t = []
        jump_note = []

        if render_novel_pose:
            cur_pose_embedding = torch.tensor(smpl_param['poses']).reshape(1, -1, 3)   # no global no hand
            cur_mat = axis_angle_to_matrix(cur_pose_embedding.squeeze())
            for i in range(pose_step):
                lst_model_idx = max(0, pose_index-i-1)
                i_n_lst = os.path.basename(ims[lst_model_idx][0])[:-4].zfill(6)
                lst_smpl_param_path = os.path.join(path, "smpl/smpl", i_n_lst+'.json')
                with open(lst_smpl_param_path, 'r', encoding='utf-8') as file:
                    lst_smpl_param = json.load(file)[0]
                lst_smpl_param['poses'] = np.array(lst_smpl_param['poses']).astype(np.float32)
                lst_pose_embedding = torch.tensor(lst_smpl_param['poses']).reshape(1, -1, 3).squeeze()
                lst_mat = axis_angle_to_matrix(lst_pose_embedding) #(B,3,3)
                # calculate delta pose
                cur_delta_mat = torch.bmm(cur_mat, torch.linalg.inv(lst_mat)) #(B,3,3)
                cur_delta_pose = matrix_to_axis_angle(cur_delta_mat)
                # update embedding
                cur_pose_embedding = torch.cat([cur_pose_embedding, cur_delta_pose.unsqueeze(0)], dim=0)
                cur_mat = lst_mat
            cur_pose_embedding = cur_pose_embedding.transpose(0, 1)
            cur_pose_embedding = cur_pose_embedding.reshape(cur_pose_embedding.shape[0], -1).unsqueeze(0)

            p_idx = 0
            for name in body_parts_smpl:
                cur_pose_part = cur_pose_embedding[:, body_parts_smpl[name], :]
                if sim_type == 'axis-angle':
                    pose_body_part = pose_body_parts[name]
                    maxsim_t, sim = find_most_similar(cur_pose_part, pose_body_part)
                elif sim_type == 'quater':
                    cur_pose_part = axis_angle_to_quaternion(cur_pose_part)
                    pose_body_part = axis_angle_to_quaternion(pose_body_parts[name])
                    maxsim_t, sim = find_most_similar(cur_pose_part, pose_body_part)
                elif sim_type == 'euclidean':
                    pose_body_part = pose_body_parts[name]
                    maxsim_t, sim = find_most_similar(cur_pose_part, pose_body_part, sim_type='euclidean')
                maxsim, maxt = maxsim_t     

                if len(pair_t_lst) == 0:      
                    selected_value = maxt[0]           
                    pair_t.append(maxt[0])
                else:
                    t_lst = pair_t_lst[p_idx]
                    distances = torch.abs(maxt - t_lst)
                    if (distances < 3).any():
                        _, distances_min = distances[distances<3].min(0)
                        selected_value = maxt[distances < 3][0]
                    else:
                        selected_value = maxt[0]

                    pair_t.append(selected_value)

                p_idx+=1

            pair_t_lst = pair_t

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= xyz_bound_flex
        max_xyz += xyz_bound_flex
        world_bound = np.stack([min_xyz, max_xyz], axis=0)

        for view_index in output_view:

            K = np.array(extrinsics[view_index]['K']).reshape(3,3)
            D = np.array(extrinsics[view_index]['dist'])
            R = np.array(extrinsics[view_index]['R'])
            T = np.array(extrinsics[view_index]['T'])

            dsr = int(1/image_scaling)
            # Load image, mask
            image_path = os.path.join(path, 'images_'+str(dsr)) + f'/{str(view_index)}'
            os.makedirs(image_path, exist_ok=True)
            image_name = os.path.join(image_path, str(pose_index).zfill(3)+'.png')

            if not os.path.exists(image_name):
                print('creating view ', str(view_index))
                # Load image, mask, K, D, R, T
                image = np.array(imageio.imread(image_name.replace('_'+str(dsr), '')).astype(np.float32)/255.)
                image = cv2.undistort(image, K, D)

                # Reduce the image resolution by ratio, then remove the back ground
                ratio = image_scaling
                if ratio != 1.:
                    H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                    K[:2] = K[:2] * ratio

                image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")
                image.save(image_name)
            
            else:
                image = Image.open(image_name)
                ratio = image_scaling
                if ratio != 1.:
                    K[:2] = K[:2] * ratio

            w2c = np.linalg.inv(convert_to_camera_to_world(R, T))

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]


            focalX = K[0,0]
            focalY = K[1,1]
            FovX = focal2fov(focalX, image.size[0])
            FovY = focal2fov(focalY, image.size[1])


            # get bounding mask and bcakground mask
            bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])
            bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.byte))

            cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, 
                                        R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, bkgd_mask=None, 
                            bound_mask=bound_mask, width=image.size[0], height=image.size[1], 
                            smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound, 
                            big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz, 
                            big_pose_world_bound=big_pose_world_bound, pair_t=pair_t, dataset_type='hifi4g', jump_note=jump_note))

            idx += 1
            
    return cam_infos, pose_interval*pose_num

##################################   MVHumanNet   ##################################

def readMVHumanNetInfo(path, white_background, output_path, eval, args, mono_test=False, render_novel_pose=False, sim_type=None, pose_step=-1):

    train_view = ['CC32871A'+str(i).zfill(3) for i in [4, 5, 6, 9, 10, 12, 13, 14, 17, 20, 21, 27, 32, 33, 36, 40, 41, 42, 48, 49, 58, 60]]
    test_view = ['CC32871A'+str(i).zfill(3) for i in [16, 22, 27, 29, 34, 38, 44, 55, 57]]
    if mono_test:
        train_view = ['CC32871A016']
        test_view = ['CC32871A016']


    print("Reading Training Transforms")
    train_cam_infos, frame_len = readCamerasMVHumanNet(path, train_view, white_background, image_scaling=0.25, split='train', args=args, mono_test=mono_test, render_novel_pose=render_novel_pose, sim_type=sim_type, pose_step=pose_step)
    print("Reading Test Transforms")
    test_cam_infos, frame_len = readCamerasMVHumanNet(path, test_view, white_background, \
        image_scaling=0.25, split='test', args=args, mono_test=mono_test, render_novel_pose=render_novel_pose, sim_type=sim_type, pose_step=pose_step)

    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    ply_path = os.path.join('output', output_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10475
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = train_cam_infos[0].big_pose_world_vertex


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
                           ply_path=ply_path,
                           frame_len=frame_len)
    return scene_info


def readCamerasMVHumanNet(path, output_view, white_background, image_scaling=0.25, \
                            split='train', args=None, mono_test=False, render_novel_pose=False, sim_type=None, pose_step=-1):
    cam_infos = []

    if mono_test and render_novel_pose:
        if split == 'train':
            pose_start = 0
            pose_interval = 150
            pose_num = 1
        elif split == 'test':
            pose_start = 200
            pose_interval = 1
            pose_num = 100
    elif mono_test:
        if split == 'train':
            pose_start = 0
            pose_interval = 150
            pose_num = 1
        elif split == 'test':
            pose_start = 0
            pose_interval = 1
            pose_num = 150
    elif render_novel_pose:
        if split == 'train':
            pose_start = 0
            pose_interval = 150
            pose_num = 1
        elif split == 'test':
            pose_start = 200
            pose_interval = 10
            pose_num = 10
    else:
        if split == 'train':
            pose_start = 0
            pose_interval = 1
            pose_num = 150
        elif split == 'test':
            pose_start = 0
            pose_interval = 3
            pose_num = 50

    img_names = os.listdir(os.path.join(path, 'images_lr', output_view[0]))
    img_names.sort()

    
    ims = np.array([
        np.array([
            os.path.join('images_lr', view, fname) for view in output_view
        ]) for fname in img_names
    ])

    xyz_bound_flex = 0.5

    gender = 'neutral'
    smpl_model = {}
    smpl_model[gender] = SMPLX('./assets/models/smplx/', smpl_type='smplx',
                                gender=gender, use_face_contour=True, flat_hand_mean=False, use_pca=True, 
                                num_pca_comps=6, num_betas=10,
                                num_expression_coeffs=10,
                                ext='npz')

    # SMPL in canonical space
    big_pose_smpl_param = {}
    big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
    big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['global_orient'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['betas'] = np.zeros((1,10)).astype(np.float32)
    big_pose_smpl_param['body_pose'] = np.zeros((1,63)).astype(np.float32)
    big_pose_smpl_param['jaw_pose'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['left_hand_pose'] = np.zeros((1,6)).astype(np.float32)
    big_pose_smpl_param['right_hand_pose'] = np.zeros((1,6)).astype(np.float32)
    big_pose_smpl_param['leye_pose'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['reye_pose'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['expression'] = np.zeros((1,10)).astype(np.float32)
    big_pose_smpl_param['transl'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['body_pose'][0, 2] = 45/180*np.array(np.pi)
    big_pose_smpl_param['body_pose'][0, 5] = -45/180*np.array(np.pi)
    big_pose_smpl_param['body_pose'][0, 20] = -30/180*np.array(np.pi)
    big_pose_smpl_param['body_pose'][0, 23] = 30/180*np.array(np.pi)

    big_pose_smpl_param_tensor= {}
    for key in big_pose_smpl_param.keys():
        big_pose_smpl_param_tensor[key] = torch.from_numpy(big_pose_smpl_param[key])

    body_model_output = smpl_model[gender](
        global_orient=big_pose_smpl_param_tensor['global_orient'],
        betas=big_pose_smpl_param_tensor['betas'],
        body_pose=big_pose_smpl_param_tensor['body_pose'],
        jaw_pose=big_pose_smpl_param_tensor['jaw_pose'],
        left_hand_pose=big_pose_smpl_param_tensor['left_hand_pose'],
        right_hand_pose=big_pose_smpl_param_tensor['right_hand_pose'],
        leye_pose=big_pose_smpl_param_tensor['leye_pose'],
        reye_pose=big_pose_smpl_param_tensor['reye_pose'],
        expression=big_pose_smpl_param_tensor['expression'],
        transl=big_pose_smpl_param_tensor['transl'],
        return_full_pose=True,
    )

    big_pose_smpl_param['poses'] = body_model_output.full_pose.detach()
    big_pose_smpl_param['shapes'] = np.concatenate([big_pose_smpl_param['betas'], big_pose_smpl_param['expression']], axis=-1)
    big_pose_xyz = np.array(body_model_output.vertices.detach()).reshape(-1,3).astype(np.float32)

    # obtain the original bounds for point sampling
    big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
    big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
    big_pose_min_xyz -= 0.05
    big_pose_max_xyz += 0.05
    big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)
    
    cam_extri = os.path.join(path, 'camera_extrinsics.json')
    with open(cam_extri, 'r') as f:
        cam_ls = json.load(f)
    extrinsics = {}
    for item in cam_ls:
        extrinsics[item[2:13]] = cam_ls[item]

    cam_intri = os.path.join(path, 'camera_intrinsics.json')
    with open(cam_intri, 'r') as f:
        intrinsics = json.load(f)

    cam_scales_path = os.path.join(path, 'camera_scale.pkl')
    with open(cam_scales_path, 'rb') as f:
        cam_scale = pickle.load(f)

    if render_novel_pose:
        assert pose_step != -1, "please set pose step!"
        pose_embedding = []
        for pose_index in range(0, 150, 1):
            model_path = os.path.join(path, 'model')
            i_p = str(int(int(os.path.basename(ims[pose_index][0])[:4])/5-1)).zfill(6)
            current_model = model_path+f'/{str(i_p).zfill(6)}.npz'
            loaded_data = np.load(current_model)
            smpl_param = {key: loaded_data[key] for key in loaded_data if key != 'xyz'}
            cur_pose_embedding = torch.tensor(smpl_param['poses'][:, 0:75]).reshape(1, -1, 3)   # no global no hand
            cur_mat = axis_angle_to_matrix(cur_pose_embedding.squeeze())
            for i in range(pose_step):
                lst_model_idx = max(0, pose_index-i-1)
                lst_loaded_data = np.load(model_path+f'/{str(lst_model_idx).zfill(6)}.npz')
                lst_smpl_param = {key: lst_loaded_data[key] for key in lst_loaded_data if key != 'xyz'}
                lst_pose_embedding = torch.tensor(lst_smpl_param['poses'][:, 0:75]).reshape(1, -1, 3).squeeze()
                lst_mat = axis_angle_to_matrix(lst_pose_embedding) #(B,3,3)
                # calculate delta pose
                cur_delta_mat = torch.bmm(cur_mat, torch.linalg.inv(lst_mat)) #(B,3,3)
                cur_delta_pose = matrix_to_axis_angle(cur_delta_mat)
                # update embedding
                cur_pose_embedding = torch.cat([cur_pose_embedding, cur_delta_pose.unsqueeze(0)], dim=0)
                cur_mat = lst_mat
            cur_pose_embedding = cur_pose_embedding.transpose(0, 1)
            cur_pose_embedding = cur_pose_embedding.reshape(cur_pose_embedding.shape[0], -1)
            pose_embedding.append(cur_pose_embedding.unsqueeze(0))
        pose_embedding = torch.cat(pose_embedding, dim=0)  # [100, 25, 3*(1+step)]
        
        pose_body_parts = {}
        for name in body_parts_smplx:
            pose_part = pose_embedding[:, body_parts_smplx[name], :]
            pose_body_parts[name] = pose_part

    idx = 0
    pair_t_lst = []

    for pose_index in tqdm(range(pose_start, pose_start+pose_num*pose_interval, pose_interval)):

        model_path = os.path.join(path, 'model')    
        os.makedirs(model_path, exist_ok=True)
        i_p = str(int(int(os.path.basename(ims[pose_index][0])[:4])/5-1)).zfill(6)
        current_model = model_path+f'/{str(i_p).zfill(6)}.npz'

        if not os.path.exists(current_model):
            print('creating model '+f'{str(i_p).zfill(6)}.npz')

            vertices_path = os.path.join(path, 'smplx/smplx_mesh', i_p+'.obj')
            mesh = trimesh.load(vertices_path)
            xyz = mesh.vertices.astype(np.float32)

            smpl_param_path = os.path.join(path, "smplx/smpl", i_p+'.json')
            with open(smpl_param_path, 'r', encoding='utf-8') as file:
                smpl_param = json.load(file)[0]
            smpl_param['poses'] = np.array(smpl_param['poses']).squeeze()

            smpl_data = {}
            smpl_data['global_orient'] = np.array(smpl_param['poses'][0:3]).reshape(-1)
            smpl_data['body_pose'] = np.array(smpl_param['poses'][3:66]).reshape(-1)
            smpl_data['jaw_pose'] = np.array(smpl_param['poses'][66:69]).reshape(-1)
            smpl_data['leye_pose'] = np.array(smpl_param['poses'][69:72]).reshape(-1)
            smpl_data['reye_pose'] = np.array(smpl_param['poses'][72:75]).reshape(-1)
            smpl_data['left_hand_pose'] = np.array(smpl_param['poses'][75:81]).reshape(-1)
            smpl_data['right_hand_pose'] = np.array(smpl_param['poses'][81:87]).reshape(-1)
            smpl_data['transl'] = np.zeros((1,3)).astype(np.float32)
            smpl_data['betas'] = np.array(smpl_param['shapes']).reshape(-1)
            smpl_data['expression'] = np.array(smpl_param['expression']).reshape(-1)

            # load smpl data
            smpl_param_new = {
                'global_orient': np.expand_dims(smpl_data['global_orient'].astype(np.float32), axis=0),
                'transl': smpl_data['transl'].astype(np.float32),
                'body_pose': np.expand_dims(smpl_data['body_pose'].astype(np.float32), axis=0),
                'jaw_pose': np.expand_dims(smpl_data['jaw_pose'].astype(np.float32), axis=0),
                'betas': np.expand_dims(smpl_data['betas'].astype(np.float32), axis=0),
                'expression': np.expand_dims(smpl_data['expression'].astype(np.float32), axis=0),
                'leye_pose': np.expand_dims(smpl_data['leye_pose'].astype(np.float32), axis=0),
                'reye_pose': np.expand_dims(smpl_data['reye_pose'].astype(np.float32), axis=0),
                'left_hand_pose': np.expand_dims(smpl_data['left_hand_pose'].astype(np.float32), axis=0),
                'right_hand_pose': np.expand_dims(smpl_data['right_hand_pose'].astype(np.float32), axis=0),
                }

            smpl_param_new['R'] = cv2.Rodrigues(np.array(smpl_param['Rh']))[0].astype(np.float32)

            smpl_param_new['Th'] = np.array(smpl_param['Th']).astype(np.float32)
            smpl_param_tensor = {}
            for key in smpl_param_new.keys():
                smpl_param_tensor[key] = torch.from_numpy(smpl_param_new[key])

            body_model_output = smpl_model[gender](
                global_orient=smpl_param_tensor['global_orient'],
                betas=smpl_param_tensor['betas'],
                body_pose=smpl_param_tensor['body_pose'],
                jaw_pose=smpl_param_tensor['jaw_pose'],
                left_hand_pose=smpl_param_tensor['left_hand_pose'],
                right_hand_pose=smpl_param_tensor['right_hand_pose'],
                leye_pose=smpl_param_tensor['leye_pose'],
                reye_pose=smpl_param_tensor['reye_pose'],
                expression=smpl_param_tensor['expression'],
                transl=smpl_param_tensor['transl'],
                return_full_pose=True,
            )
            
            smpl_param_new['poses'] = body_model_output.full_pose.detach()
            smpl_param_new['shapes'] = np.concatenate([smpl_param_new['betas'], smpl_param_new['expression']], axis=-1)
            xyz = xyz.reshape(-1,3).astype(np.float32)
            np.savez(current_model, **smpl_param_new, xyz=xyz)

            smpl_param = smpl_param_new
        
        else:
            loaded_data = np.load(current_model)
            smpl_param = {key: loaded_data[key] for key in loaded_data if key != 'xyz'}
            xyz = loaded_data['xyz']

        pair_t = []
        jump_note = []
        if render_novel_pose:
            jump_note = [-1 for _ in range(len(pose_body_parts))]
            cur_pose_embedding = torch.tensor(smpl_param['poses'][:, :75]).reshape(1, -1, 3)   # no global no hand
            cur_mat = axis_angle_to_matrix(cur_pose_embedding.squeeze())
            for i in range(pose_step):
                lst_model_idx = max(0, pose_index-i-1)
                lst_loaded_data = np.load(model_path+f'/{str(lst_model_idx).zfill(6)}.npz')
                lst_smpl_param = {key: lst_loaded_data[key] for key in lst_loaded_data if key != 'xyz'}
                lst_pose_embedding = torch.tensor(lst_smpl_param['poses'][:, :75]).reshape(1, -1, 3).squeeze()
                lst_mat = axis_angle_to_matrix(lst_pose_embedding) #(B,3,3)
                # calculate delta pose
                cur_delta_mat = torch.bmm(cur_mat, torch.linalg.inv(lst_mat)) #(B,3,3)
                cur_delta_pose = matrix_to_axis_angle(cur_delta_mat)
                # update embedding
                cur_pose_embedding = torch.cat([cur_pose_embedding, cur_delta_pose.unsqueeze(0)], dim=0)
                cur_mat = lst_mat
            cur_pose_embedding = cur_pose_embedding.transpose(0, 1)
            cur_pose_embedding = cur_pose_embedding.reshape(cur_pose_embedding.shape[0], -1).unsqueeze(0)

            p_idx = 0
            for name in body_parts_smplx:
                cur_pose_part = cur_pose_embedding[:, body_parts_smplx[name], :]
                if sim_type == 'axis-angle':
                    pose_body_part = pose_body_parts[name]
                    maxsim_t, sim = find_most_similar(cur_pose_part, pose_body_part)
                elif sim_type == 'quater':
                    cur_pose_part = axis_angle_to_quaternion(cur_pose_part)
                    pose_body_part = axis_angle_to_quaternion(pose_body_parts[name])
                    maxsim_t, sim = find_most_similar(cur_pose_part, pose_body_part)
                elif sim_type == 'euclidean':
                    pose_body_part = pose_body_parts[name]
                    maxsim_t, sim = find_most_similar(cur_pose_part, pose_body_part, sim_type='euclidean')
                maxsim, maxt = maxsim_t     

                if len(pair_t_lst) == 0:      
                    selected_value = maxt[0]
                    pair_t.append(maxt[0])
                else:
                    t_lst = pair_t_lst[p_idx]
                    distances = torch.abs(maxt - t_lst)
                    neighbor_num = len(distances[distances<3])
                    print(t_lst)
                    print(maxt[distances<3])
                    print(neighbor_num)

                    if neighbor_num > 1:
                        w_ = maxsim[distances < 3][1]/(maxsim[distances < 3][0]+maxsim[distances < 3][1])
                        selected_value = w_ * maxt[distances < 3][0] + (1 - w_) * maxt[distances < 3][1]

                    elif neighbor_num == 1:
                        selected_value = maxt[distances < 3][0]
                    else:
                        # jitter
                        jump_note[p_idx] = t_lst
                        selected_value = maxt[0]

                    pair_t.append(selected_value)

                p_idx+=1

            pair_t_lst = pair_t

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bound = np.stack([min_xyz, max_xyz], axis=0)
    
        for view_index in output_view:
            
            K = np.array(intrinsics['intrinsics'])
            K[:2] = K[:2] * 0.5  # original images are half resolution
            w2c_R = extrinsics[view_index]['rotation']
            w2c_t = np.array(extrinsics[view_index]['translation'])*cam_scale/1000
            w2c = np.eye(4)
            w2c[:3, :3] = w2c_R
            w2c[:3, 3] = w2c_t

            dsr = int(1/image_scaling)
            # Load image, mask
            img_path = os.path.join(path, 'images_'+str(dsr)) + f'/{str(view_index)}'
            bkgd_mask_path = os.path.join(path, 'masks_'+str(dsr)) + f'/{str(view_index)}'
            os.makedirs(img_path, exist_ok=True)
            os.makedirs(bkgd_mask_path, exist_ok=True)
            image_name = os.path.join(img_path, os.path.basename(ims[pose_index][0]))
            bkgd_mask_name = os.path.join(bkgd_mask_path, os.path.basename(ims[pose_index][0])[:-4]+'_fmask.png')

            if not os.path.exists(image_name):
                print('creating view ', str(view_index))
                image = np.array(imageio.imread(image_name.replace('images_'+str(dsr), 'images_lr')).astype(np.float32)/255.)

                msk = imageio.imread(bkgd_mask_name.replace('masks_'+str(dsr), 'fmask_lr'))
                msk = (msk != 0).astype(np.uint8)
                msk[msk!=0] = 255
                msk = np.array(msk) / 255.
                
                image[msk == 0] = 1 if white_background else 0

                # Reduce the image resolution by ratio, then remove the back ground
                ratio = image_scaling
                if ratio != 1.:
                    H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                    K[:2] = K[:2] * ratio


                image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")
                image.save(image_name)
                
                bkgd_mask = Image.fromarray(np.array(msk*255.0, dtype=np.byte))
                bkgd_mask.save(bkgd_mask_name)
            
            else:
                image = Image.open(image_name)
                bkgd_mask = Image.open(bkgd_mask_name)
                ratio = image_scaling
                
                if ratio != 1.:
                    K[:2] = K[:2] * ratio

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            focalX = K[0,0]
            focalY = K[1,1]
            FovX = focal2fov(focalX, image.size[0])
            FovY = focal2fov(focalY, image.size[1])
            
            # get bounding mask and bcakground mask
            bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])

            bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.byte))

            cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=img_path, image_name=image_name, bkgd_mask=bkgd_mask, 
                            bound_mask=bound_mask, width=image.size[0], height=image.size[1], 
                            smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound, 
                            big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz, 
                            big_pose_world_bound=big_pose_world_bound, pair_t=pair_t, jump_note=jump_note, dataset_type='mvhuman'))

            idx += 1
            
    return cam_infos, pose_interval*pose_num


def prepare_smpl_params(smpl_path, pose_index):
    params_ori = dict(np.load(smpl_path, allow_pickle=True))['smpl'].item()
    params = {}
    params['shapes'] = np.array(params_ori['betas']).astype(np.float32)
    params['poses'] = np.zeros((1,72)).astype(np.float32)
    params['poses'][:, :3] = np.array(params_ori['global_orient'][pose_index]).astype(np.float32)
    params['poses'][:, 3:] = np.array(params_ori['body_pose'][pose_index]).astype(np.float32)
    params['R'] = np.eye(3).astype(np.float32)
    params['Th'] = np.array(params_ori['transl'][pose_index:pose_index+1]).astype(np.float32)
    return params

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

def get_mask(path, index, view_index, ims):
    msk_path = os.path.join(path, 'mask_cihp',
                            ims[index][view_index])[:-4] + '.png'
    msk_cihp = imageio.imread(msk_path)
    msk_cihp = (msk_cihp != 0).astype(np.uint8)
    msk = msk_cihp.copy()

    return msk, msk_cihp

sceneLoadTypeCallbacks = {
    "ZJU_MoCap_refine" : readZJUMoCapRefineInfo,
    "dna_rendering": readDNARenderingInfo,
    "HiFi4G" : readHiFi4GInfo,
    "MVHumanNet" : readMVHumanNetInfo,
}