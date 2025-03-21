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
from scene import Scene
import os
import time
import pickle
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_nops
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, ext=''):

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders"+ext)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    pcd_path = None
    if name=='test_mono' or 'mono_novel_pose':
        pcd_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pth"+ext)
        makedirs(pcd_path, exist_ok=True)
    
    for i in range(len(gaussians.grids)):
        cur_res_grids = gaussians.grids[i]
        for plane_idx in range(len(cur_res_grids)):
            cur_grid = cur_res_grids[plane_idx].permute(0, 2, 3, 1).squeeze()
            cur_grid = torch.norm(cur_grid, p=2, dim=-1)
            cur_grid_norm = (cur_grid - cur_grid.min()) / (cur_grid.max() - cur_grid.min())
            image_np = cur_grid_norm.detach().cpu().numpy() * 255
            image_np = image_np.astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            image_pil.save(os.path.join(\
                model_path, name, "ours_{}".format(iteration), 'feature_'+str(i)+'_'+str(plane_idx)+'.png'))


    rgbs = []
    rgbs_gt = []
    pair_ts = []
    elapsed_time = 0

    for tt, view in enumerate(tqdm(views, desc="Rendering progress")):

        gt = view.original_image[0:3, :, :].cuda()
        bound_mask = view.bound_mask
        # not use bound mask
        bound_mask = bound_mask.fill_(1)

        # Start timer
        start_time = time.time()
        if name == 'novel_pose' or name == 'mono_novel_pose':
            render_output = render_nops(view, gaussians, pipeline, background)
            pair_ts.append(render_output['pair_t'])
        else:
            render_output = render(view, gaussians, pipeline, background, save_pth=pcd_path)
        rendering = render_output["render"]
        
        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += end_time - start_time

        rendering.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1

        rgbs.append(rendering)
        rgbs_gt.append(gt)
    
    if len(pair_ts):
        pair_ts = torch.stack(pair_ts, dim=0).cpu()
        labels = ['l_leg', 'r_leg', 'center_body', 'l_arm', 'r_arm']
        x = torch.arange(views[0].pose_id, views[0].pose_id+pair_ts.shape[0])

        plt.figure(figsize=(10, 6))
        for pt_l in range(pair_ts.shape[1]):
            plt.plot(x.numpy(), pair_ts[:, pt_l].numpy(), label=labels[pt_l])

        plt.legend()
        plt.xlabel('novel_pose')
        plt.ylabel('t')
        plt.title('t-novel_pose '+model_path.split('/')[-1])
        plt.grid(True)
        plt.savefig(os.path.join(\
                model_path, name, "ours_{}".format(iteration), 't-novel_pose'+ext+'.png'))
        plt.close()

    print("Elapsed time: ", elapsed_time, " FPS: ", len(views)/elapsed_time) 

    psnrs = 0.0
    ssims = 0.0
    lpipss = 0.0
    gt_frames = []
    render_frames = []
    for id in range(len(views)):
        rendering = rgbs[id]
        gt = rgbs_gt[id]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(id) + ".png"))
        gt_frames.append(imageio.imread(os.path.join(gts_path, '{0:05d}'.format(id) + ".png")))
        render_frames.append(imageio.imread(os.path.join(render_path, '{0:05d}'.format(id) + ".png")))
        
        # metrics
        psnrs += psnr(rendering, gt).mean().double()
        ssims += ssim(rendering, gt).mean().double()
        lpipss += loss_fn_vgg(rendering, gt).mean().double()

    if name=='test_mono' or name == 'mono_novel_pose':
        imageio.mimwrite(os.path.join(gts_path, 'video.mp4'), gt_frames, fps=15)
        imageio.mimwrite(os.path.join(render_path, 'video.mp4'), render_frames, fps=15)
    
    psnrs /= len(views)   
    ssims /= len(views)
    lpipss /= len(views)  

    # evalution metrics
    print("\n[ITER {}] Evaluating {} #{}: PSNR {} SSIM {} LPIPS {}".format(iteration, name, len(views), psnrs, ssims, lpipss))

    save_metrics_path = os.path.join(model_path, name, "ours_{}".format(iteration), "metrics"+ext+".txt")

    with open(save_metrics_path, 'a') as f:
        f.write("\n[ITER {}] Evaluating {} #{}: PSNR {} SSIM {} LPIPS {}\n".format(iteration, name, len(views), psnrs, ssims, lpipss))



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, mono_test : bool, render_novel_pose : bool, sim_type=None, pose_step=-1):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)
        scene = Scene(dataset, gaussians, mono_test=mono_test, shuffle=False, render_novel_pose=render_novel_pose, sim_type=sim_type, pose_step=pose_step)
        if iteration:
            checkpoint = os.path.join('output', args.exp_name, 'chkpnt'+str(iteration)+'.pth')
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore_test(model_params)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        ext = '_'+sim_type+'_'+str(pose_step)+'iccv25ood'

        if not skip_train:
             render_set(dataset.model_path, "train", iteration, scene.getTrainCameras(), gaussians, pipeline, background)


        if not skip_test:
            if mono_test and render_novel_pose:
                render_set(dataset.model_path, "mono_novel_pose", iteration, \
                    scene.getTestCameras(), gaussians, pipeline, background, ext)
            elif mono_test:
                render_set(dataset.model_path, "test_mono", iteration, \
                    scene.getTestCameras(), gaussians, pipeline, background, ext)
            elif render_novel_pose:
                render_set(dataset.model_path, "novel_pose", iteration, \
                    scene.getTestCameras(), gaussians, pipeline, background, ext)                
            else:
                render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipeline, background, ext)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--mono_test", action="store_true")
    parser.add_argument("--render_novel_pose", action="store_true")
    parser.add_argument("--sim_type", type=str, default = None)
    parser.add_argument("--pose_step", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mono_test, args.render_novel_pose, args.sim_type, args.pose_step)