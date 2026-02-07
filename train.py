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
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, create_window
from gaussian_renderer import render, render_aux, network_gui
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


def _dssim_map(image, gt_image, window_size=11):
    x = image.detach().unsqueeze(0)
    y = gt_image.detach().unsqueeze(0)
    channel = x.size(1)
    window = create_window(window_size, channel).to(x.device).type_as(x)
    mu1 = F.conv2d(x, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(y, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(x * x, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    dssim = (1.0 - ssim_map) * 0.5
    return dssim.mean(dim=1, keepdim=True).squeeze(0)


def _adjoint_phi(image, gt_image, error_type):
    if error_type == "dssim":
        return _dssim_map(image, gt_image)
    if error_type == "l1":
        diff = (image.detach() - gt_image).abs()
        return diff.sum(dim=0, keepdim=True)
    # default: grad-based
    if image.grad is not None:
        grad = image.grad.detach().abs()
        return grad.sum(dim=0, keepdim=True)
    diff = (image.detach() - gt_image).abs()
    return diff.sum(dim=0, keepdim=True)


def _adjoint_scores(viewpoint_cam, gaussians, pipe, image, gt_image, use_trained_exp=False, error_type="grad"):
    # Scalar error map (detached) used for attribution.
    phi = _adjoint_phi(image, gt_image, error_type)
    w0 = torch.ones_like(phi)
    w1 = phi

    with torch.enable_grad():
        aux = torch.ones((gaussians.get_xyz.shape[0], 3), device="cuda", requires_grad=True)
        aux_pkg = render_aux(
            viewpoint_cam,
            gaussians,
            pipe,
            override_color=aux,
            use_trained_exp=use_trained_exp,
        )
        aux_img = aux_pkg["render"]
        if viewpoint_cam.alpha_mask is not None:
            aux_img = aux_img * viewpoint_cam.alpha_mask.cuda()

        loss = (aux_img[0] * w0 + aux_img[1] * w1).sum()
        grad = torch.autograd.grad(loss, aux, retain_graph=False, create_graph=False, allow_unused=False)[0]

    Z = grad[:, 0]
    M = grad[:, 1]
    return M, Z, aux_pkg["visibility_filter"]


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
    last_l1_for_log = None

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
        collect_every = opt.awsrm_collect_every if opt.awsrm_collect_every > 0 else max(1, min(32, opt.densification_interval // 4))
        should_collect = (opt.fw_densify and
                          iteration < opt.densify_until_iter and
                          iteration > opt.densify_from_iter and
                          iteration % collect_every == 0)
        need_fw_stats = should_collect
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            return_aux=False
        )
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if viewpoint_cam.alpha_mask is not None:
            gt_image = gt_image * alpha_mask
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        try:
            image.retain_grad()
        except Exception:
            pass

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
            fw_denom = None
            fw_vis_filter = None
            if need_fw_stats:
                fw_mean, fw_denom, fw_vis_filter = _adjoint_scores(
                    viewpoint_cam, gaussians, pipe, image, gt_image,
                    use_trained_exp=False, error_type=opt.awsrm_error_type
                )
                # Use max-over-views aggregation on normalized moments (AW-SRM++).
                eps = float(getattr(opt, "awsrm_eps", 1e-6))
                mu_view = fw_mean / (fw_denom + eps)
                if viewspace_point_tensor.grad is not None:
                    split_view = viewspace_point_tensor.grad[:, :2].abs().sum(dim=-1, keepdim=True)
                else:
                    split_view = torch.zeros_like(mu_view)
                update_filter = fw_vis_filter if fw_vis_filter is not None else visibility_filter
                gaussians.add_fw_stats(mu_view, split_view, fw_denom, update_filter, mode="max")

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
                    # fw stats are already updated inside need_fw_stats block (max aggregation).
                    pass
                else:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                awsrm_use_moments = bool(getattr(opt, "awsrm_use_moments", 1))
                if opt.fw_densify:
                    Z = gaussians.fw_denom
                    mean_scores = gaussians.fw_mean_accum
                    var_scores = gaussians.fw_var_accum
                    # If moments disabled, fall back to mean scores.
                    if not awsrm_use_moments:
                        var_scores = mean_scores
                    # Optional size factor for split score.
                    var_scores = var_scores * gaussians.max_radii2D.clamp_min(1.0).unsqueeze(-1)
                else:
                    denom_raw = gaussians.denom
                    denom = denom_raw.clamp_min(1.0)
                    mean_scores = gaussians.xyz_gradient_accum / denom
                    var_scores = mean_scores

                mean_scores = mean_scores.squeeze()
                var_scores = var_scores.squeeze()
                if opt.fw_densify:
                    severity_eta = 0.5
                    denom_scale = gaussians.fw_denom.squeeze().clamp_min(float(getattr(opt, "awsrm_eps", 1e-6)))
                    mean_scores = mean_scores * denom_scale.pow(1.0 - severity_eta)
                mean_scores[mean_scores.isnan()] = 0.0
                var_scores[var_scores.isnan()] = 0.0
                if mean_scores.dim() == 1:
                    mean_norm = mean_scores.abs()
                    mean_grads = mean_scores.unsqueeze(-1)
                else:
                    mean_norm = torch.norm(mean_scores, dim=-1)
                    mean_grads = mean_scores
                if var_scores.dim() == 1:
                    var_norm = var_scores.abs()
                    var_grads = var_scores.unsqueeze(-1)
                else:
                    var_norm = torch.norm(var_scores, dim=-1)
                    var_grads = var_scores
                if opt.fw_densify:
                    z_min = float(getattr(opt, "awsrm_eps", 1e-6))
                    denom_valid = gaussians.fw_denom.squeeze() > z_min
                else:
                    denom_valid = denom_raw.squeeze() > 0
                scale_max = gaussians.get_scaling.max(dim=1).values
                clone_mask = scale_max <= gaussians.percent_dense * scene.cameras_extent
                split_mask = scale_max > gaussians.percent_dense * scene.cameras_extent
                valid_clone = torch.logical_and(denom_valid, torch.logical_and(mean_norm > 0, clone_mask))
                valid_split = torch.logical_and(denom_valid, torch.logical_and(var_norm > 0, split_mask))
                valid_clone_norm = mean_norm[valid_clone]
                valid_split_norm = var_norm[valid_split]

                if iteration % 100 == 0:
                    num_pts = int(gaussians.get_xyz.shape[0])
                    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
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
                    cur_l1 = float(Ll1.item())
                    if last_l1_for_log is None:
                        delta_l1 = 0.0
                    else:
                        delta_l1 = cur_l1 - last_l1_for_log
                    last_l1_for_log = cur_l1
                    msg = (
                        f"[ITER {iteration}] points={num_pts} "
                        f"sel={sel_count} ({sel_ratio:.4f}) "
                        f"valid={valid_count} ({valid_ratio:.4f}) "
                        f"clone={int(sel_clone.sum().item())} split={int(sel_split.sum().item())} "
                        f"thr_c={last_eff_threshold_clone:.6f} "
                        f"thr_s={last_eff_threshold_split:.6f} "
                        f"L1={cur_l1:.6f} dL1={delta_l1:+.6f} "
                        f"mem_alloc={mem_alloc:.2f}G mem_reserved={mem_reserved:.2f}G"
                    )
                    _log_iter_stats(progress_bar, iter_log_path, msg)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    num_pts = int(mean_norm.numel())
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    base_threshold = opt.densify_grad_threshold
                    eff_pct = _effective_percentile(opt, iteration)
                    eff_topk = int(getattr(opt, "densify_topk", 0) or 0)
                    eff_topk_ratio = float(getattr(opt, "densify_topk_ratio", 0.0) or 0.0)
                    awsrm_k_clone = int(getattr(opt, "awsrm_K_clone", 0) or 0)
                    awsrm_k_split = int(getattr(opt, "awsrm_K_split", 0) or 0)
                    awsrm_clone_frac = float(getattr(opt, "awsrm_clone_frac", 0.0) or 0.0)
                    awsrm_split_frac = float(getattr(opt, "awsrm_split_frac", 0.0) or 0.0)
                    use_budget = opt.fw_densify and getattr(opt, "awsrm_max_primitives", 0) > 0
                    budget_total = 0
                    if use_budget:
                        budget_total = max(0, int(opt.awsrm_max_primitives) - num_pts)

                    def _compute_threshold(values, topk, pct, default_thr):
                        if values.numel() == 0:
                            return float("inf"), 0
                        if topk > 0:
                            k = min(topk, values.numel())
                            if k <= 0:
                                return float("inf"), 0
                            thr = float(torch.topk(values, k, largest=True, sorted=True).values[-1].item())
                            return thr, k
                        if pct > 0.0:
                            return float(torch.quantile(values, pct).item()), 0
                        return default_thr, 0

                    clone_values = valid_clone_norm
                    split_values = valid_split_norm

                    # Determine per-step offspring budget (LMO-style).
                    k_new = 0
                    if eff_topk > 0:
                        k_new = eff_topk
                    elif eff_topk_ratio > 0.0:
                        k_new = int(num_pts * eff_topk_ratio)

                    # If explicit K_clone/K_split are provided, use them.
                    topk_clone = awsrm_k_clone
                    topk_split = awsrm_k_split

                    # Enforce budgeted LMO if max_primitives is set.
                    pct_for_selection = 0.0 if opt.fw_densify else eff_pct
                    force_no_densify = False
                    if use_budget:
                        pct_for_selection = 0.0
                        if budget_total <= 0:
                            k_new = 0
                        elif k_new > 0:
                            k_new = min(k_new, budget_total)

                    # If no explicit K_* provided, split k_new by fraction or fixed 50/50.
                    if (topk_clone + topk_split) <= 0 and k_new > 0:
                        if awsrm_clone_frac > 0.0 or awsrm_split_frac > 0.0:
                            denom = awsrm_clone_frac + awsrm_split_frac
                            clone_share = awsrm_clone_frac / denom if denom > 0 else 0.5
                        else:
                            clone_share = 0.5
                        topk_clone = int(k_new * clone_share)
                        topk_split = max(0, k_new - topk_clone)

                    # If explicit K_* are provided but exceed k_new, scale down.
                    if k_new > 0 and (topk_clone + topk_split) > k_new:
                        total_req = topk_clone + topk_split
                        scale = k_new / float(total_req)
                        topk_clone = int(topk_clone * scale)
                        topk_split = max(0, k_new - topk_clone)

                    # If explicit K_* are provided, still respect global budget.
                    if use_budget and (topk_clone + topk_split) > 0:
                        if budget_total <= 0:
                            topk_clone = 0
                            topk_split = 0
                        elif (topk_clone + topk_split) > budget_total:
                            total_req = topk_clone + topk_split
                            scale = budget_total / float(total_req)
                            topk_clone = int(topk_clone * scale)
                            topk_split = max(0, budget_total - topk_clone)

                    # Cap by available valid candidates.
                    if topk_clone > clone_values.numel():
                        topk_clone = clone_values.numel()
                    if topk_split > split_values.numel():
                        topk_split = split_values.numel()

                    if topk_clone == 0 and topk_split == 0 and pct_for_selection == 0.0:
                        force_no_densify = True

                    thr_clone, used_topk_clone = _compute_threshold(clone_values, topk_clone, pct_for_selection, base_threshold)
                    thr_split, used_topk_split = _compute_threshold(split_values, topk_split, pct_for_selection, base_threshold)

                    sel_mask_clone = None
                    sel_mask_split = None
                    if force_no_densify:
                        sel_mask_clone = torch.zeros(num_pts, device=mean_norm.device, dtype=torch.bool)
                        sel_mask_split = torch.zeros(num_pts, device=var_norm.device, dtype=torch.bool)
                    if used_topk_clone > 0 and clone_values.numel() > 0:
                        valid_clone_idx = torch.nonzero(valid_clone, as_tuple=False).squeeze(-1)
                        if valid_clone_idx.numel() > 0:
                            clone_vals = mean_norm[valid_clone_idx]
                            k_clone = min(used_topk_clone, clone_vals.numel())
                            if k_clone > 0:
                                topk_sub_idx = torch.topk(clone_vals, k_clone, largest=True, sorted=True).indices
                                topk_idx = valid_clone_idx[topk_sub_idx]
                                sel_mask_clone = torch.zeros(num_pts, device=mean_norm.device, dtype=torch.bool)
                                sel_mask_clone[topk_idx] = True
                    if used_topk_split > 0 and split_values.numel() > 0:
                        valid_split_idx = torch.nonzero(valid_split, as_tuple=False).squeeze(-1)
                        if valid_split_idx.numel() > 0:
                            split_vals = var_norm[valid_split_idx]
                            k_split = min(used_topk_split, split_vals.numel())
                            if k_split > 0:
                                topk_sub_idx = torch.topk(split_vals, k_split, largest=True, sorted=True).indices
                                topk_idx = valid_split_idx[topk_sub_idx]
                                sel_mask_split = torch.zeros(num_pts, device=var_norm.device, dtype=torch.bool)
                                sel_mask_split[topk_idx] = True

                    last_eff_threshold_clone = thr_clone
                    last_eff_threshold_split = thr_split
                    last_eff_pct_clone = eff_pct if used_topk_clone == 0 else 0.0
                    last_eff_pct_split = eff_pct if used_topk_split == 0 else 0.0
                    last_eff_topk_clone = used_topk_clone
                    last_eff_topk_split = used_topk_split
                    if opt.fw_densify:
                        fw_mean_grads = mean_grads
                        fw_var_grads = var_grads
                        stats = gaussians.densify_and_prune(
                            thr_clone,
                            0.005,
                            scene.cameras_extent,
                            size_threshold,
                            radii,
                            grads_override=fw_mean_grads,
                            grads_override_split=fw_var_grads,
                            max_grad_split=thr_split,
                            selected_mask_clone=sel_mask_clone,
                            selected_mask_split=sel_mask_split,
                            opacity_correction=True,
                            max_primitives=opt.awsrm_max_primitives,
                        )
                    else:
                        stats = gaussians.densify_and_prune(base_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                    if stats is not None:
                        net = stats["after"] - stats["before"]
                        msg = (
                            f"[ITER {iteration}] densify added_clone={stats['added_clone']} "
                            f"added_split={stats['added_split']} pruned={stats['pruned']} "
                            f"net={net} points={stats['after']}"
                        )
                        _log_iter_stats(progress_bar, iter_log_path, msg)
                    if opt.fw_densify:
                        gaussians.fw_mean_accum.zero_()
                        gaussians.fw_var_accum.zero_()
                        gaussians.fw_denom.zero_()
                
                if not opt.fw_densify:
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
