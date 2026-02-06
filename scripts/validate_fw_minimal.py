#!/usr/bin/env python3
import json
import math
import os
import random
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene
from gaussian_renderer import render, render_aux, GaussianModel
from utils.graphics_utils import geom_transform_points
from utils.loss_utils import l1_loss, ssim, create_window
from utils.general_utils import safe_state


def ndc_to_pix(v, s):
    return ((v + 1.0) * s - 1.0) * 0.5


def _dssim_map(image, gt_image, window_size=11):
    x = image.detach().unsqueeze(0)
    y = gt_image.detach().unsqueeze(0)
    channel = x.size(1)
    window = create_window(window_size, channel).to(x.device).type_as(x)
    mu1 = torch.nn.functional.conv2d(x, window, padding=window_size // 2, groups=channel)
    mu2 = torch.nn.functional.conv2d(y, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = torch.nn.functional.conv2d(x * x, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(y * y, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(x * y, window, padding=window_size // 2, groups=channel) - mu1_mu2
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
    if image.grad is not None:
        grad = image.grad.detach().abs()
        return grad.sum(dim=0, keepdim=True)
    diff = (image.detach() - gt_image).abs()
    return diff.sum(dim=0, keepdim=True)


def _adjoint_scores(view, gaussians, pipe, background, image, gt_image, use_trained_exp=False, error_type="grad"):
    phi = _adjoint_phi(image, gt_image, error_type)
    w0 = torch.ones_like(phi)
    w1 = phi
    w2 = phi * phi

    with torch.enable_grad():
        aux = torch.ones((gaussians.get_xyz.shape[0], 3), device="cuda", requires_grad=True)
        aux_pkg = render_aux(view, gaussians, pipe, override_color=aux, use_trained_exp=use_trained_exp)
        aux_img = aux_pkg["render"]
        if view.alpha_mask is not None:
            aux_img = aux_img * view.alpha_mask.cuda()
        loss = (aux_img[0] * w0 + aux_img[1] * w1 + aux_img[2] * w2).sum()
        grad = torch.autograd.grad(loss, aux, retain_graph=False, create_graph=False, allow_unused=False)[0]

    Z = grad[:, 0].clamp_min(0.0)
    M = grad[:, 1].clamp_min(0.0)
    Q = grad[:, 2].clamp_min(0.0)
    return M, Q, Z


def compute_adjoint_score_for_view(view, gaussians, pipe, opt, background):
    use_trained_exp = getattr(opt, "train_test_exp", False)
    error_type = getattr(opt, "awsrm_error_type", "grad")
    render_pkg = render(view, gaussians, pipe, background, separate_sh=False, use_trained_exp=use_trained_exp, return_aux=False)
    image = render_pkg["render"]
    if view.alpha_mask is not None:
        image = image * view.alpha_mask.cuda()

    gt_image = view.original_image.cuda()
    if view.alpha_mask is not None:
        gt_image = gt_image * view.alpha_mask.cuda()
    Ll1 = l1_loss(image, gt_image)
    ssim_value = ssim(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
    try:
        image.retain_grad()
    except Exception:
        pass
    loss.backward()

    residual_img = (image.detach() - gt_image).abs()
    M, Q, Z = _adjoint_scores(view, gaussians, pipe, background, image, gt_image, use_trained_exp=use_trained_exp, error_type=error_type)

    return render_pkg, image.detach(), residual_img, M, Q, Z


def _default_args():
    default_parser = ArgumentParser(add_help=False)
    ModelParams(default_parser)
    PipelineParams(default_parser)
    OptimizationParams(default_parser)
    return default_parser.parse_args([])


def _safe_get_args(parser: ArgumentParser):
    args = parser.parse_args()
    cfg_path = None
    if hasattr(args, "model_path") and args.model_path:
        cfg_path = os.path.join(args.model_path, "cfg_args")
    if cfg_path and os.path.isfile(cfg_path):
        return get_combined_args(parser)
    defaults = _default_args()
    for key, value in vars(defaults).items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)
    return args


def _ensure_depths_available(dataset):
    depths = getattr(dataset, "depths", "")
    if depths is None:
        dataset.depths = ""
        return
    if depths != "":
        depth_params = os.path.join(dataset.source_path, "sparse/0/depth_params.json")
        if not os.path.isfile(depth_params):
            print(f"Warning: depth_params.json not found at '{depth_params}'. Disabling depths.")
            dataset.depths = ""


def score_heatmap_sanity(scene, pipe, opt, background, topk, out_dir):
    view = random.choice(scene.getTrainCameras())
    render_pkg, image, residual, M, Q, Z = compute_adjoint_score_for_view(view, scene.gaussians, pipe, opt, background)

    scale_max = scene.gaussians.get_scaling.max(dim=1).values
    clone_mask = scale_max <= scene.gaussians.percent_dense * scene.cameras_extent
    split_mask = scale_max > scene.gaussians.percent_dense * scene.cameras_extent
    eps = 1e-8
    clone_score = M
    split_score = (Q - (M * M) / (Z.clamp_min(eps))).clamp_min(0.0)
    split_score = split_score * render_pkg["radii"].clamp_min(1.0)
    combined_score = torch.where(clone_mask, clone_score, split_score)

    topk = min(topk, combined_score.numel())
    _, idxs = torch.topk(combined_score, k=topk, largest=True)

    # Project to pixel coordinates
    xyz = scene.gaussians.get_xyz.detach()
    ndc = geom_transform_points(xyz, view.full_proj_transform)
    px = ndc_to_pix(ndc[:, 0], view.image_width)
    py = ndc_to_pix(ndc[:, 1], view.image_height)

    overlay = image.clone()
    h, w = view.image_height, view.image_width
    for i in idxs.tolist():
        x = int(px[i].item())
        y = int(py[i].item())
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                xx = x + dx
                yy = y + dy
                if 0 <= xx < w and 0 <= yy < h:
                    overlay[0, yy, xx] = 1.0
                    overlay[1, yy, xx] = 0.0
                    overlay[2, yy, xx] = 0.0

    res_norm = torch.norm(residual, dim=0, keepdim=True)
    res_norm = res_norm / (res_norm.max() + 1e-8)

    torchvision.utils.save_image(image, f"{out_dir}/sanity_image.png")
    torchvision.utils.save_image(overlay, f"{out_dir}/sanity_topk_overlay.png")
    torchvision.utils.save_image(res_norm, f"{out_dir}/sanity_residual_norm.png")

    return {
        "topk": topk,
        "fw_score_mean": float(combined_score.mean().item()),
        "fw_score_max": float(combined_score.max().item()),
    }


def topk_stability(scene, pipe, opt, background, topk):
    views = scene.getTrainCameras()
    if len(views) < 2:
        return {"topk_overlap": 0.0}
    v1, v2 = random.sample(views, 2)
    render1, _, _, M1, Q1, Z1 = compute_adjoint_score_for_view(v1, scene.gaussians, pipe, opt, background)
    render2, _, _, M2, Q2, Z2 = compute_adjoint_score_for_view(v2, scene.gaussians, pipe, opt, background)
    scale_max = scene.gaussians.get_scaling.max(dim=1).values
    clone_mask = scale_max <= scene.gaussians.percent_dense * scene.cameras_extent
    split_mask = scale_max > scene.gaussians.percent_dense * scene.cameras_extent
    eps = 1e-8
    score1 = torch.where(clone_mask, M1, (Q1 - (M1 * M1) / (Z1.clamp_min(eps))).clamp_min(0.0) * render1["radii"].clamp_min(1.0))
    score2 = torch.where(clone_mask, M2, (Q2 - (M2 * M2) / (Z2.clamp_min(eps))).clamp_min(0.0) * render2["radii"].clamp_min(1.0))
    topk = min(topk, score1.numel(), score2.numel())
    idx1 = set(torch.topk(score1, k=topk).indices.tolist())
    idx2 = set(torch.topk(score2, k=topk).indices.tolist())
    inter = len(idx1.intersection(idx2))
    union = len(idx1.union(idx2))
    return {
        "topk": topk,
        "intersection": inter,
        "union": union,
        "jaccard": inter / max(1, union),
        "overlap_ratio": inter / max(1, topk),
    }


def densify_effect_test(dataset, pipe, opt, background, steps, use_fw):
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, shuffle=True)
    gaussians.training_setup(opt)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    losses = []
    for it in range(1, steps + 1):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = random.randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        need_fw = use_fw and it == 1
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            use_trained_exp=False,
            separate_sh=False,
            return_aux=False
        )
        image = render_pkg["render"]
        if viewpoint_cam.alpha_mask is not None:
            image = image * viewpoint_cam.alpha_mask.cuda()
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        if it == 1:
            gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(
                gaussians.max_radii2D[render_pkg["visibility_filter"]],
                render_pkg["radii"][render_pkg["visibility_filter"]]
            )
            if use_fw:
                fw_M, fw_Q, fw_Z = _adjoint_scores(viewpoint_cam, gaussians, pipe, background, image, gt_image)
                gaussians.add_fw_stats(fw_M, fw_Q, fw_Z, render_pkg["visibility_filter"])
                M = gaussians.fw_mean_accum
                Q = gaussians.fw_var_accum
                Z = gaussians.fw_denom
                mean_scores = M.squeeze().abs()
                var_scores = (Q - (M * M) / (Z.clamp_min(1e-8))).squeeze().clamp_min(0.0)
                var_scores = var_scores * gaussians.max_radii2D.clamp_min(1.0)
                fw_mean_grads = mean_scores.unsqueeze(-1)
                fw_var_grads = var_scores.unsqueeze(-1)
                fw_mean_grads = mean_scores
                fw_var_grads = var_scores
                scale_max = gaussians.get_scaling.max(dim=1).values
                clone_mask = scale_max <= gaussians.percent_dense * scene.cameras_extent
                split_mask = scale_max > gaussians.percent_dense * scene.cameras_extent
                mean_valid = mean_scores[clone_mask]
                var_valid = var_scores[split_mask]
                topk = int(getattr(opt, "densify_topk", 0) or 0)
                topk_ratio = float(getattr(opt, "densify_topk_ratio", 0.0) or 0.0)
                if topk_ratio > 0.0:
                    topk_clone = max(1, int(mean_valid.numel() * topk_ratio)) if mean_valid.numel() > 0 else 0
                    topk_split = max(1, int(var_valid.numel() * topk_ratio)) if var_valid.numel() > 0 else 0
                else:
                    topk_clone = topk
                    topk_split = topk
                thr_clone = opt.densify_grad_threshold
                thr_split = opt.densify_grad_threshold
                if topk_clone > 0 and mean_valid.numel() > 0:
                    if mean_valid.numel() > topk_clone:
                        thr_clone = float(torch.topk(mean_valid, topk_clone, largest=True, sorted=True).values[-1].item())
                    else:
                        thr_clone = float("-inf")
                if topk_split > 0 and var_valid.numel() > 0:
                    if var_valid.numel() > topk_split:
                        thr_split = float(torch.topk(var_valid, topk_split, largest=True, sorted=True).values[-1].item())
                    else:
                        thr_split = float("-inf")
                gaussians.densify_and_prune(
                    thr_clone,
                    0.005,
                    scene.cameras_extent,
                    None,
                    render_pkg["radii"],
                    grads_override=fw_mean_grads,
                    grads_override_split=fw_var_grads,
                    max_grad_split=thr_split
                )
            else:
                gaussians.add_densification_stats(render_pkg["viewspace_points"], render_pkg["visibility_filter"])
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, None, render_pkg["radii"])

        losses.append(float(loss.item()))

        gaussians.exposure_optimizer.step()
        gaussians.exposure_optimizer.zero_grad(set_to_none=True)
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

    if len(losses) > 1:
        slope = (losses[0] - losses[-1]) / (len(losses) - 1)
    else:
        slope = 0.0
    return {"losses": losses, "slope": slope}


def main():
    parser = ArgumentParser(description="Minimal FW validation A/B/C")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="./output/fw_minimal")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--quiet", action="store_true")
    args = _safe_get_args(parser)

    safe_state(args.quiet)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = model.extract(args)
    _ensure_depths_available(dataset)
    pipe = pipeline.extract(args)
    opt_args = opt.extract(args)

    # Make densify happen on step 1 for the test
    opt_args.densify_from_iter = 0
    opt_args.densification_interval = 1
    opt_args.densify_until_iter = 1
    opt_args.densify_grad_threshold = 0.0

    os.makedirs(args.out_dir, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    load_iter = args.iteration if args.iteration is not None and args.iteration >= 0 else None
    scene = Scene(dataset, gaussians, load_iteration=load_iter, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    results = {}
    results["score_heatmap_sanity"] = score_heatmap_sanity(scene, pipe, opt_args, background, args.topk, args.out_dir)
    results["topk_stability"] = topk_stability(scene, pipe, opt_args, background, args.topk)
    results["densify_effect_baseline"] = densify_effect_test(dataset, pipe, opt_args, background, args.steps, use_fw=False)
    results["densify_effect_fw"] = densify_effect_test(dataset, pipe, opt_args, background, args.steps, use_fw=True)

    with open(f"{args.out_dir}/fw_minimal_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
