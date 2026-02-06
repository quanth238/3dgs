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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
try:
    from diff_gaussian_rasterization import compute_fw_score, compute_tile_moments
    FW_SCORE_AVAILABLE = True
except:
    FW_SCORE_AVAILABLE = False


def _log_iter_stats(progress_bar, log_path, msg):
    progress_bar.write(msg)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _effective_percentile(opt, iteration):
    start = getattr(opt, "densify_grad_percentile_start", 0.0)
    end = getattr(opt, "densify_grad_percentile_end", 0.0)
    ramp = getattr(opt, "densify_grad_percentile_ramp", 0)
    if start > 0.0 and end > 0.0 and ramp > 0:
        t = (iteration - opt.densify_from_iter) / float(ramp)
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        pct = start + (end - start) * t
    else:
        pct = getattr(opt, "densify_grad_percentile", 0.0)
    if pct > 1.0:
        pct = pct / 100.0
    return pct


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    iter_log_path = os.path.join(scene.model_path, "iter_stats.log")
    try:
        with open(iter_log_path, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    last_eff_threshold_clone = opt.densify_grad_threshold
    last_eff_threshold_split = opt.densify_grad_threshold
    last_eff_pct_clone = _effective_percentile(opt, first_iter)
    last_eff_pct_split = last_eff_pct_clone
    last_eff_topk_clone = int(getattr(opt, "densify_topk", 0) or 0)
    last_eff_topk_split = last_eff_topk_clone
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        should_densify = (iteration < opt.densify_until_iter and
                          iteration > opt.densify_from_iter and
                          iteration % opt.densification_interval == 0)
        need_fw_stats = opt.fw_densify and should_densify
        if opt.fw_densify and not FW_SCORE_AVAILABLE:
            raise RuntimeError("fw_densify is enabled but compute_fw_score is not available. Rebuild the rasterizer.")

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            return_aux=need_fw_stats
        )
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask
        if need_fw_stats:
            image.retain_grad()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            fw_mean = None
            fw_var = None
            if need_fw_stats:
                residual_img = image.grad.detach()
                tile_residual, tile_energy = compute_tile_moments(residual_img)
                _, H, W = residual_img.shape
                tiles_x = (W + 16 - 1) // 16
                fw_mean = compute_fw_score(
                    tile_residual,
                    tile_energy,
                    tiles_x,
                    radii,
                    render_pkg["geomBuffer"],
                    render_pkg["binningBuffer"],
                    3,
                )
                fw_var = compute_fw_score(
                    tile_residual,
                    tile_energy,
                    tiles_x,
                    radii,
                    render_pkg["geomBuffer"],
                    render_pkg["binningBuffer"],
                    5,
                )

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if opt.fw_densify:
                    if need_fw_stats:
                        gaussians.add_fw_stats(fw_mean, fw_var, visibility_filter)
                else:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if opt.fw_densify:
                    denom_raw = gaussians.fw_denom
                    denom = denom_raw.clamp_min(1.0)
                    mean_scores = gaussians.fw_mean_accum / denom
                    var_scores = gaussians.fw_var_accum / denom
                else:
                    denom_raw = gaussians.denom
                    denom = denom_raw.clamp_min(1.0)
                    mean_scores = gaussians.xyz_gradient_accum / denom
                    var_scores = mean_scores

                mean_scores = mean_scores.squeeze()
                var_scores = var_scores.squeeze()
                mean_scores[mean_scores.isnan()] = 0.0
                var_scores[var_scores.isnan()] = 0.0
                if mean_scores.dim() == 1:
                    mean_norm = mean_scores.abs()
                else:
                    mean_norm = torch.norm(mean_scores, dim=-1)
                if var_scores.dim() == 1:
                    var_norm = var_scores.abs()
                else:
                    var_norm = torch.norm(var_scores, dim=-1)
                denom_valid = denom_raw.squeeze() > 0
                scale_max = gaussians.get_scaling.max(dim=1).values
                clone_mask = scale_max <= gaussians.percent_dense * scene.cameras_extent
                split_mask = scale_max > gaussians.percent_dense * scene.cameras_extent
                valid_clone = torch.logical_and(denom_valid, torch.logical_and(mean_norm > 0, clone_mask))
                valid_split = torch.logical_and(denom_valid, torch.logical_and(var_norm > 0, split_mask))
                valid_clone_norm = mean_norm[valid_clone]
                valid_split_norm = var_norm[valid_split]

                if iteration % 200 == 0:
                    num_pts = int(gaussians.get_xyz.shape[0])
                    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                    if valid_clone_norm.numel() > 0 and valid_split_norm.numel() > 0:
                        valid_all = torch.cat([valid_clone_norm, valid_split_norm], dim=0)
                    elif valid_clone_norm.numel() > 0:
                        valid_all = valid_clone_norm
                    else:
                        valid_all = valid_split_norm
                    p90 = float(torch.quantile(valid_all, 0.90).item()) if valid_all.numel() > 0 else 0.0
                    p95 = float(torch.quantile(valid_all, 0.95).item()) if valid_all.numel() > 0 else 0.0
                    p99 = float(torch.quantile(valid_all, 0.99).item()) if valid_all.numel() > 0 else 0.0
                    sel_clone = torch.logical_and(
                        mean_norm >= last_eff_threshold_clone,
                        clone_mask
                    )
                    sel_split = torch.logical_and(
                        var_norm >= last_eff_threshold_split,
                        split_mask
                    )
                    sel_count = int((sel_clone | sel_split).sum().item())
                    sel_ratio = sel_count / max(1, num_pts)
                    valid_clone_count = int(valid_clone_norm.numel())
                    valid_split_count = int(valid_split_norm.numel())
                    valid_count = valid_clone_count + valid_split_count
                    valid_ratio = valid_count / max(1, num_pts)
                    pct_clone_msg = f" pct_c={last_eff_pct_clone:.3f}" if last_eff_pct_clone > 0 else ""
                    pct_split_msg = f" pct_s={last_eff_pct_split:.3f}" if last_eff_pct_split > 0 else ""
                    topk_clone_msg = f" topk_c={last_eff_topk_clone}" if last_eff_topk_clone > 0 else ""
                    topk_split_msg = f" topk_s={last_eff_topk_split}" if last_eff_topk_split > 0 else ""
                    msg = (
                        f"[ITER {iteration}] points={num_pts} "
                        f"sel={sel_count} ({sel_ratio:.4f}) "
                        f"valid={valid_count} ({valid_ratio:.4f}) "
                        f"clone={int(sel_clone.sum().item())} split={int(sel_split.sum().item())} "
                        f"thr_c={last_eff_threshold_clone:.6f}{pct_clone_msg}{topk_clone_msg} "
                        f"thr_s={last_eff_threshold_split:.6f}{pct_split_msg}{topk_split_msg} "
                        f"p90={p90:.6f} p95={p95:.6f} p99={p99:.6f} "
                        f"mem_alloc={mem_alloc:.2f}G mem_reserved={mem_reserved:.2f}G"
                    )
                    _log_iter_stats(progress_bar, iter_log_path, msg)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    base_threshold = opt.densify_grad_threshold
                    eff_pct = _effective_percentile(opt, iteration)
                    eff_topk = int(getattr(opt, "densify_topk", 0) or 0)
                    eff_topk_ratio = float(getattr(opt, "densify_topk_ratio", 0.0) or 0.0)

                    def _compute_threshold(values, topk, pct, default_thr):
                        if values.numel() == 0:
                            return float("inf"), 0
                        if topk > 0:
                            if values.numel() > topk:
                                thr = float(torch.topk(values, topk, largest=True, sorted=True).values[-1].item())
                            else:
                                thr = float("-inf")
                            return thr, topk
                        if pct > 0.0:
                            return float(torch.quantile(values, pct).item()), 0
                        return default_thr, 0

                    clone_values = valid_clone_norm
                    split_values = valid_split_norm
                    topk_clone = eff_topk
                    topk_split = eff_topk
                    if eff_topk_ratio > 0.0:
                        topk_clone = int(clone_values.numel() * eff_topk_ratio)
                        topk_split = int(split_values.numel() * eff_topk_ratio)
                        if clone_values.numel() > 0 and topk_clone == 0:
                            topk_clone = 1
                        if split_values.numel() > 0 and topk_split == 0:
                            topk_split = 1

                    thr_clone, used_topk_clone = _compute_threshold(clone_values, topk_clone, eff_pct, base_threshold)
                    thr_split, used_topk_split = _compute_threshold(split_values, topk_split, eff_pct, base_threshold)

                    last_eff_threshold_clone = thr_clone
                    last_eff_threshold_split = thr_split
                    last_eff_pct_clone = eff_pct if used_topk_clone == 0 else 0.0
                    last_eff_pct_split = eff_pct if used_topk_split == 0 else 0.0
                    last_eff_topk_clone = used_topk_clone
                    last_eff_topk_split = used_topk_split
                    if opt.fw_densify:
                        fw_mean_grads = gaussians.fw_mean_accum / gaussians.fw_denom.clamp_min(1.0)
                        fw_var_grads = gaussians.fw_var_accum / gaussians.fw_denom.clamp_min(1.0)
                        gaussians.densify_and_prune(thr_clone, 0.005, scene.cameras_extent, size_threshold, radii, grads_override=fw_mean_grads, grads_override_split=fw_var_grads, max_grad_split=thr_split)
                    else:
                        gaussians.densify_and_prune(base_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
