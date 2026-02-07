#!/usr/bin/env python3
import json
import os
import random
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene
from gaussian_renderer import render, render_aux, GaussianModel
from utils.loss_utils import l1_loss, ssim, create_window
from utils.general_utils import safe_state


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
    if getattr(dataset, "depths", ""):
        depth_params = os.path.join(dataset.source_path, "sparse/0/depth_params.json")
        if not os.path.isfile(depth_params):
            print(f"Warning: depth_params.json not found at '{depth_params}'. Disabling depths.")
            dataset.depths = ""


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    _, idx = torch.sort(x)
    ranks = torch.empty_like(idx, dtype=torch.float32)
    ranks[idx] = torch.arange(0, x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def _ensure_exposure(gaussians, scene):
    if hasattr(gaussians, "_exposure"):
        return
    cams = scene.getTrainCameras() + scene.getTestCameras()
    if len(cams) == 0:
        return
    gaussians.exposure_mapping = {cam.image_name: idx for idx, cam in enumerate(cams)}
    gaussians.pretrained_exposures = None
    exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cams), 1, 1)
    gaussians._exposure = nn.Parameter(exposure.requires_grad_(True))


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten()
    y = y.flatten()
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-8)
    ry = (ry - ry.mean()) / (ry.std() + 1e-8)
    return float((rx * ry).mean().item())


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

    with torch.enable_grad():
        aux = torch.ones((gaussians.get_xyz.shape[0], 3), device="cuda", requires_grad=True)
        aux_pkg = render_aux(view, gaussians, pipe, override_color=aux, use_trained_exp=use_trained_exp)
        aux_img = aux_pkg["render"]
        if view.alpha_mask is not None:
            aux_img = aux_img * view.alpha_mask.cuda()
        loss = (aux_img[0] * w0 + aux_img[1] * w1).sum()
        grad = torch.autograd.grad(loss, aux, retain_graph=False, create_graph=False, allow_unused=False)[0]

    Z = grad[:, 0]
    M = grad[:, 1]
    return M, Z


def main():
    parser = ArgumentParser(description="Ablation tests A1/A2/A3/A4")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--num_candidates", type=int, default=200)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = _safe_get_args(parser)

    safe_state(args.quiet)
    random.seed(0)
    torch.manual_seed(0)

    dataset = model.extract(args)
    _ensure_depths_available(dataset)
    pipe = pipeline.extract(args)
    opt_args = opt.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, opt_args.optimizer_type)
    load_iter = args.iteration if args.iteration is not None and args.iteration >= 0 else None
    scene = Scene(dataset, gaussians, load_iteration=load_iter, shuffle=False)
    _ensure_exposure(gaussians, scene)
    gaussians.training_setup(opt_args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    view = random.choice(scene.getTrainCameras())
    use_trained_exp = getattr(opt_args, "train_test_exp", False)
    error_type = getattr(opt_args, "awsrm_error_type", "grad")
    render_pkg = render(view, gaussians, pipe, background, separate_sh=False, use_trained_exp=use_trained_exp, return_aux=False)
    image = render_pkg["render"]
    if view.alpha_mask is not None:
        image = image * view.alpha_mask.cuda()

    gt_image = view.original_image.cuda()
    if view.alpha_mask is not None:
        gt_image = gt_image * view.alpha_mask.cuda()
    Ll1 = l1_loss(image, gt_image)
    ssim_value = ssim(image, gt_image)
    loss = (1.0 - opt_args.lambda_dssim) * Ll1 + opt_args.lambda_dssim * (1.0 - ssim_value)
    try:
        image.retain_grad()
    except Exception:
        pass
    loss.backward()

    M, Z = _adjoint_scores(view, gaussians, pipe, background, image, gt_image, use_trained_exp=False, error_type=error_type)
    eps = float(getattr(opt_args, "awsrm_eps", 1e-6))
    mu_raw = M / (Z + eps)
    severity_eta = 0.5
    mu = mu_raw * (Z + eps).pow(1.0 - severity_eta)
    clone_score = mu
    split_score = render_pkg["viewspace_points"].grad[:, :2].abs().sum(dim=-1) * render_pkg["radii"].clamp_min(1.0)

    mean2d_score = torch.norm(render_pkg["viewspace_points"].grad[:, :2], dim=-1)
    feat_dc = gaussians._features_dc.grad
    feat_rest = gaussians._features_rest.grad
    feat_grad = torch.sqrt((feat_dc ** 2).sum(dim=(1, 2)) + (feat_rest ** 2).sum(dim=(1, 2)) + 1e-8)

    visible = render_pkg["radii"] > 0
    visible_idx = torch.where(visible)[0]
    num_candidates = min(args.num_candidates, visible_idx.numel())
    cand_idx = visible_idx[torch.randperm(visible_idx.numel())[:num_candidates]]

    results = {
        "corr_clone_featgrad": _spearman(clone_score[cand_idx], feat_grad[cand_idx]),
        "corr_split_featgrad": _spearman(split_score[cand_idx], feat_grad[cand_idx]),
        "corr_mean2d_featgrad": _spearman(mean2d_score[cand_idx], feat_grad[cand_idx]),
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
