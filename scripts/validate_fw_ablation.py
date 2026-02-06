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
from utils.loss_utils import l1_loss, ssim
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


def _adjoint_phi(image, gt_image):
    diff = (image.detach() - gt_image).abs()
    return diff.sum(dim=0, keepdim=True)


def _adjoint_scores(view, gaussians, pipe, background, image, gt_image):
    phi_scalar = _adjoint_phi(image, gt_image)
    phi = phi_scalar.repeat(3, 1, 1)
    ones = torch.ones_like(phi)

    def _grad_for(signal):
        aux = torch.ones((gaussians.get_xyz.shape[0], 3), device="cuda", requires_grad=True)
        aux_img = render_aux(view, gaussians, pipe, override_color=aux)["render"]
        if view.alpha_mask is not None:
            aux_img = aux_img * view.alpha_mask.cuda()
        loss = (aux_img * signal).sum()
        grad = torch.autograd.grad(loss, aux, retain_graph=False, create_graph=False, allow_unused=False)[0]
        return grad.sum(dim=-1)

    with torch.enable_grad():
        M = _grad_for(phi)
        Q = _grad_for(phi * phi)
        Z = _grad_for(ones)

    M = M.clamp_min(0.0)
    Q = Q.clamp_min(0.0)
    Z = Z.clamp_min(0.0)
    return M, Q, Z


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
    render_pkg = render(view, gaussians, pipe, background, separate_sh=False, use_trained_exp=False, return_aux=False)
    image = render_pkg["render"]
    if view.alpha_mask is not None:
        image = image * view.alpha_mask.cuda()

    gt_image = view.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    ssim_value = ssim(image, gt_image)
    loss = (1.0 - opt_args.lambda_dssim) * Ll1 + opt_args.lambda_dssim * (1.0 - ssim_value)
    loss.backward()

    M, Q, Z = _adjoint_scores(view, gaussians, pipe, background, image, gt_image)
    clone_score = M
    split_score = (Q - (M * M) / (Z.clamp_min(1e-8))).clamp_min(0.0) * render_pkg["radii"].clamp_min(1.0)

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
