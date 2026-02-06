#!/usr/bin/env python3
import json
import os
import random
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import compute_tile_moments, compute_fw_score
except Exception as exc:
    raise RuntimeError("compute_tile_moments/compute_fw_score not available. Rebuild rasterizer.") from exc


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
    # Simple rankdata (no tie handling). Good enough for sanity checks.
    _, idx = torch.sort(x)
    ranks = torch.empty_like(idx, dtype=torch.float32)
    ranks[idx] = torch.arange(0, x.numel(), device=x.device, dtype=torch.float32)
    return ranks

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

def _zero_grads(gaussians):
    for p in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
              gaussians._opacity, gaussians._scaling, gaussians._rotation]:
        if p.grad is not None:
            p.grad.zero_()

def main():
    parser = ArgumentParser(description="FW sanity/theory validation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = _safe_get_args(parser)

    safe_state(args.quiet)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = model.extract(args)
    _ensure_depths_available(dataset)
    pipe = pipeline.extract(args)
    opt_args = opt.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras()
    sel_views = random.sample(views, min(args.num_views, len(views)))

    results = {
        "spearman_fw_vs_featgrad": [],
        "spearman_mean2d_vs_featgrad": [],
        "fw_score_mean": [],
        "featgrad_mean": [],
        "mean2dgrad_mean": [],
    }

    for view in sel_views:
        _zero_grads(gaussians)

        render_pkg = render(view, gaussians, pipe, background, separate_sh=False, use_trained_exp=dataset.train_test_exp, return_aux=True)
        image = render_pkg["render"]
        if view.alpha_mask is not None:
            image = image * view.alpha_mask.cuda()
        image.retain_grad()

        gt_image = view.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt_args.lambda_dssim) * Ll1 + opt_args.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        residual_img = image.grad.detach()
        tile_residual, tile_energy = compute_tile_moments(residual_img)
        _, H, W = residual_img.shape
        tiles_x = (W + 16 - 1) // 16
        fw_score = compute_fw_score(tile_residual, tile_energy, tiles_x, render_pkg["radii"], render_pkg["geomBuffer"], render_pkg["binningBuffer"], opt.fw_norm_mode)

        feat_dc = gaussians._features_dc.grad
        feat_rest = gaussians._features_rest.grad
        feat_grad = torch.sqrt((feat_dc ** 2).sum(dim=(1, 2)) + (feat_rest ** 2).sum(dim=(1, 2)) + 1e-8)

        mean2d_grad = torch.norm(render_pkg["viewspace_points"].grad[:, :2], dim=-1)

        results["spearman_fw_vs_featgrad"].append(_spearman(fw_score, feat_grad))
        results["spearman_mean2d_vs_featgrad"].append(_spearman(mean2d_grad, feat_grad))
        results["fw_score_mean"].append(float(fw_score.mean().item()))
        results["featgrad_mean"].append(float(feat_grad.mean().item()))
        results["mean2dgrad_mean"].append(float(mean2d_grad.mean().item()))

    out = args.out
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
