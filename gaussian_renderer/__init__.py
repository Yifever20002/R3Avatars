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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from torch import nn
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable
import itertools
import torch.nn.functional as F
import os
from utils.system_utils import mkdir_p
import numpy as np
from plyfile import PlyData, PlyElement

body_parts_smplx = {'l_leg': [1, 4, 7, 10],
              'r_leg': [2, 5, 8, 11],
              'center_body': [0, 3, 6, 9, 12, 15, 22, 23, 24],
              'l_arm': [13, 16, 18, 20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
              'r_arm': [14, 17, 19, 21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
                }

body_parts_smpl = {'l_leg': [1, 4, 7, 10],
              'r_leg': [2, 5, 8, 11],
              'center_body': [0, 3, 6, 9, 12, 15],
              'l_arm': [13, 16, 18, 20, 22],
              'r_arm': [14, 17, 19, 21, 23],
                }

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

embed_fn, xyzt_input_ch = get_embedder(5, 4)

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]

    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp

def save_ply(path, pc, xyz, shs):
    mkdir_p(os.path.dirname(path))

    xyz = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    shs = shs.detach()
    f_dc, f_rest = torch.split(shs, [1, 15], dim=-1)
    f_dc = f_dc.flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = f_rest.flatten(start_dim=1).contiguous().cpu().numpy()
    # f_dc = pc._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    # f_rest = pc._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    opacities = pc._opacity.detach().cpu().numpy()
    scale = pc._scaling.detach().cpu().numpy()
    rotation = pc._rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in pc.construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, return_smpl_rot=False, transforms=None, translation=None, save_pth=None, set_t=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D_origin = pc.get_xyz

    t = viewpoint_camera.pose_id

    if viewpoint_camera.dataset_type == 'dna':
        t = torch.tensor(t, device=means3D_origin.device).view(1, 1).\
                expand(means3D_origin.shape[0], 1).float() / 50 -1
    elif viewpoint_camera.dataset_type == 'zju':
        t = torch.tensor(t, device=means3D_origin.device).view(1, 1).\
                expand(means3D_origin.shape[0], 1).float() / 150 -1  
    elif viewpoint_camera.dataset_type == 'hifi4g':
        t = torch.tensor(t, device=means3D_origin.device).view(1, 1).\
                expand(means3D_origin.shape[0], 1).float() / 75 -1
    elif viewpoint_camera.dataset_type == 'mvhuman':
        t = (torch.tensor(t, device=means3D_origin.device).view(1, 1).\
                expand(means3D_origin.shape[0], 1).float()-0) / 75 -1

    inputs = torch.cat((normalize_aabb(means3D_origin, pc.aabb), t), dim=-1)  # [N, 4]
    features = interpolate_ms_features(
            inputs, ms_grids=pc.grids,  # noqa
            grid_dimensions=pc.grid_config[0]["grid_dimensions"],
            concat_features=True, num_levels=None)
    dx, opacity, scales, rotations = pc.gaussian_decoder(features)

    means3D = means3D_origin + dx

    if not pc.motion_offset_flag:
        _, means3D, _, transforms, _ = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
            viewpoint_camera.big_pose_smpl_param,
            viewpoint_camera.big_pose_world_vertex[None])
    else:
        if transforms is None:
            # pose offset
            dst_posevec = viewpoint_camera.smpl_param['poses'][:, 3:]
            pose_out = pc.pose_decoder(dst_posevec)
            correct_Rs = pose_out['Rs']

            # SMPL lbs weights
            lbs_weights = pc.lweight_offset_decoder(means3D_origin[None].detach())
            lbs_weights = lbs_weights.permute(0,2,1)

            # transform points
            _, means3D, _, transforms, translation = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
                viewpoint_camera.big_pose_smpl_param,
                viewpoint_camera.big_pose_world_vertex[None], lbs_weights=lbs_weights, correct_Rs=correct_Rs, return_transl=return_smpl_rot)
        else:
            correct_Rs = None
            means3D = torch.matmul(transforms, means3D[..., None]).squeeze(-1) + translation


    means3D = means3D.squeeze()
    means2D = screenspace_points

    # for right pruning
    
    pc._scaling = pc.scaling_inverse_activation(scales)
    pc._rotation = rotations
    pc._opacity = opacity

    opacity = pc.opacity_activation(opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scales, rotations, scaling_modifier, transforms.squeeze())
        scales = None
        rotations = None
    else:
        scales = scales
        rotations = pc.rotation_activation(rotations)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:

            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if save_pth:
        torch.save([means3D, means2D, colors_precomp, opacity, cov3D_precomp], os.path.join(save_pth, str(int(viewpoint_camera.pose_id)).zfill(3)+'.pth'))

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    alpha, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
    alpha = alpha[:1]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_depth": None,
            "render_alpha": alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "transforms": transforms,
            "translation": translation,
            "correct_Rs": correct_Rs,}

def render_nops(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, return_smpl_rot=False, transforms=None, translation=None, save_pth=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D_origin = pc.get_xyz
    
    pair_t = viewpoint_camera.pair_t

    # pair_t.fill_(pair_t[torch.randint(0, len(pair_t), (1,))][0])  # ab body parts
    jump_note = viewpoint_camera.jump_note
    dx_batch = []
    opacity_batch = []
    scales_batch = []
    rotations_batch = []
    for t_idx in range(len(pair_t)):
        t = pair_t[t_idx]
        t_lst = jump_note[t_idx]
        if viewpoint_camera.dataset_type == 'dna':
            t = torch.tensor(t, device=means3D_origin.device).view(1, 1).\
                    expand(means3D_origin.shape[0], 1).float() / 50 -1
            body_parts = body_parts_smplx
        elif viewpoint_camera.dataset_type == 'zju':
            t = torch.tensor(t, device=means3D_origin.device).view(1, 1).\
                    expand(means3D_origin.shape[0], 1).float() / 150 -1     
            body_parts = body_parts_smpl
        elif viewpoint_camera.dataset_type == 'hifi4g':
            t = torch.tensor(t, device=means3D_origin.device).view(1, 1).\
                    expand(means3D_origin.shape[0], 1).float() / 75 -1
            body_parts = body_parts_smpl
        elif viewpoint_camera.dataset_type == 'mvhuman':
            t = (torch.tensor(t, device=means3D_origin.device).view(1, 1).\
                    expand(means3D_origin.shape[0], 1).float()-0) / 75 -1
            body_parts = body_parts_smplx

        inputs = torch.cat((normalize_aabb(means3D_origin, pc.aabb), t), dim=-1)  # [N, 4]
        features = interpolate_ms_features(
                inputs, ms_grids=pc.grids,  # noqa
                grid_dimensions=pc.grid_config[0]["grid_dimensions"],
                concat_features=True, num_levels=None)

        dx, opacity, scales, rotations = pc.gaussian_decoder(features)

        if t_lst != -1:
            print("need for dealing jump")
            t = t_lst
            if viewpoint_camera.dataset_type == 'dna':
                t = torch.tensor(t, device=means3D_origin.device).view(1, 1).\
                        expand(means3D_origin.shape[0], 1).float() / 50 -1
                body_parts = body_parts_smplx
            elif viewpoint_camera.dataset_type == 'zju':
                t = torch.tensor(t, device=means3D_origin.device).view(1, 1).\
                        expand(means3D_origin.shape[0], 1).float() / 150 -1     
                body_parts = body_parts_smpl
            elif viewpoint_camera.dataset_type == 'hifi4g':
                t = torch.tensor(t, device=means3D_origin.device).view(1, 1).\
                        expand(means3D_origin.shape[0], 1).float() / 75 -1
                body_parts = body_parts_smpl
            elif viewpoint_camera.dataset_type == 'mvhuman':
                t = (torch.tensor(t, device=means3D_origin.device).view(1, 1).\
                        expand(means3D_origin.shape[0], 1).float()-0) / 75 -1
                body_parts = body_parts_smplx

            inputs = torch.cat((normalize_aabb(means3D_origin, pc.aabb), t), dim=-1)  # [N, 4]
            features = interpolate_ms_features(
                    inputs, ms_grids=pc.grids,  # noqa
                    grid_dimensions=pc.grid_config[0]["grid_dimensions"],
                    concat_features=True, num_levels=None)
            dx_lst, opacity_lst, scales_lst, rotations_lst = pc.gaussian_decoder(features)
            dx = (2*dx + dx_lst)/3
            opacity = (2*opacity + opacity_lst)/3
            scales = (2*scales + scales_lst)/3
            rotations = (2*rotations + rotations_lst)/3

        dx_batch.append(dx)
        opacity_batch.append(opacity)
        scales_batch.append(scales)
        rotations_batch.append(rotations)

    dx_batch = torch.stack(dx_batch, dim=0).permute(1, 0, 2)
    opacity_batch = torch.stack(opacity_batch, dim=0).permute(1, 0, 2)
    scales_batch = torch.stack(scales_batch, dim=0).permute(1, 0, 2)
    rotations_batch = torch.stack(rotations_batch, dim=0).permute(1, 0, 2)
    
    if not pc.motion_offset_flag:
        _, means3D, _, transforms, _ = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
            viewpoint_camera.big_pose_smpl_param,
            viewpoint_camera.big_pose_world_vertex[None])
    else:
        if transforms is None:

            means3D = means3D_origin + dx
            # pose offset
            dst_posevec = viewpoint_camera.smpl_param['poses'][:, 3:]
            pose_out = pc.pose_decoder(dst_posevec)
            correct_Rs = pose_out['Rs']

            # SMPL lbs weights
            lbs_weights = pc.lweight_offset_decoder(means3D_origin[None].detach())
            lbs_weights = lbs_weights.permute(0,2,1)

            pre_lbs_weight = pc.get_bweights(means3D[None], viewpoint_camera.big_pose_world_vertex[None], lbs_weights)

            lbs_parts = []
            for name in body_parts:
                cur_lbs = pre_lbs_weight[:,:,body_parts[name]].sum(-1)
                lbs_parts.append(cur_lbs)
            lbs_parts = torch.stack(lbs_parts, dim=-1).transpose(0, 1)
            dx = torch.bmm(lbs_parts, dx_batch).squeeze(1)   
            opacity = torch.bmm(lbs_parts, opacity_batch).squeeze(1)
            scales = torch.bmm(lbs_parts, scales_batch).squeeze(1)
            rotations = torch.bmm(lbs_parts, rotations_batch).squeeze(1)
            means3D = means3D_origin + dx

            # transform points
            _, means3D, _, transforms, translation = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
                viewpoint_camera.big_pose_smpl_param,
                viewpoint_camera.big_pose_world_vertex[None], lbs_weights=lbs_weights, correct_Rs=correct_Rs, return_transl=return_smpl_rot)
        else:
            correct_Rs = None
            means3D = torch.matmul(transforms, means3D[..., None]).squeeze(-1) + translation


    means3D = means3D.squeeze()
    means2D = screenspace_points

    # for right pruning
    
    pc._scaling = pc.scaling_inverse_activation(scales)
    pc._rotation = rotations
    pc._opacity = opacity

    opacity = pc.opacity_activation(opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scales, rotations, scaling_modifier, transforms.squeeze())
        scales = None
        rotations = None
    else:
        scales = scales
        rotations = pc.rotation_activation(rotations)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:

            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 

    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    alpha, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
    alpha = alpha[:1]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_depth": None,
            "render_alpha": alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "transforms": transforms,
            "translation": translation,
            "correct_Rs": correct_Rs,
            "pair_t": pair_t}

    """
    让 [N,3] 的点云绕自身中心旋转一定角度
    
    参数:
        points (torch.Tensor): [N,3] 形状的点云张量
        angle (float): 旋转角度（弧度）
        axis (str): 旋转轴，可选 'x'，'y'，'z'，默认 'z'
        
    返回:
        torch.Tensor: 旋转后的点云
    """
    # 计算点云的中心
    center = points.mean(dim=0, keepdim=True)
    
    # 平移到原点
    points_centered = points - center

    angle = torch.tensor(angle * torch.pi / 180)
    
    # 生成旋转矩阵
    c, s = torch.cos(angle), torch.sin(angle)
    
    if axis == 'x':
        R = torch.tensor([[1, 0, 0],
                          [0, c, -s],
                          [0, s, c]], dtype=points.dtype, device=points.device)
    elif axis == 'y':
        R = torch.tensor([[c, 0, s],
                          [0, 1, 0],
                          [-s, 0, c]], dtype=points.dtype, device=points.device)
    elif axis == 'z':
        R = torch.tensor([[c, -s, 0],
                          [s, c, 0],
                          [0, 0, 1]], dtype=points.dtype, device=points.device)
    else:
        raise ValueError("axis 必须是 'x', 'y', 或 'z'")
    
    # 应用旋转
    rotated_points = points_centered @ R.T

    # scales
    rotated_points = rotated_points*1.5
    
    # 平移回原位置
    rotated_points += center
    
    return rotated_points