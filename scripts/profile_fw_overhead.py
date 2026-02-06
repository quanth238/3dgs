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
from gaussian_renderer import render, GaussianModel
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


def main():
    parser = ArgumentParser(description="Profile FW overhead")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = _safe_get_args(parser)
    if not hasattr(args, "out"):
        args.out = None

    safe_state(args.quiet)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = model.extract(args)
    _ensure_depths_available(dataset)
    pipe = pipeline.extract(args)
    opt_args = opt.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, opt_args.optimizer_type)
    load_iter = args.iteration if args.iteration is not None and args.iteration >= 0 else None
    scene = Scene(dataset, gaussians, load_iteration=load_iter, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras()
    total_fw = 0.0
    total_tile = 0.0
    total_fw_mean = 0.0
    total_fw_var = 0.0
    total_core = 0.0

    # Warmup
    view = random.choice(views)
    render_pkg = render(view, gaussians, pipe, background, separate_sh=False, use_trained_exp=False, return_aux=True)
    image = render_pkg["render"]
    if view.alpha_mask is not None:
        image = image * view.alpha_mask.cuda()
    image.retain_grad()
    gt_image = view.original_image.cuda()
    loss = (1.0 - opt_args.lambda_dssim) * l1_loss(image, gt_image) + opt_args.lambda_dssim * (1.0 - ssim(image, gt_image))
    loss.backward()
    residual_img = image.grad.detach()
    tile_residual, tile_energy = compute_tile_moments(residual_img)
    _, H, W = residual_img.shape
    tiles_x = (W + 16 - 1) // 16
    _ = compute_fw_score(tile_residual, tile_energy, tiles_x, render_pkg["radii"], render_pkg["geomBuffer"], render_pkg["binningBuffer"], 3)
    _ = compute_fw_score(tile_residual, tile_energy, tiles_x, render_pkg["radii"], render_pkg["geomBuffer"], render_pkg["binningBuffer"], 5)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        view = random.choice(views)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        render_pkg = render(view, gaussians, pipe, background, separate_sh=False, use_trained_exp=False, return_aux=True)
        image = render_pkg["render"]
        if view.alpha_mask is not None:
            image = image * view.alpha_mask.cuda()
        image.retain_grad()
        gt_image = view.original_image.cuda()
        loss = (1.0 - opt_args.lambda_dssim) * l1_loss(image, gt_image) + opt_args.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        total_core += start.elapsed_time(end)

        # Tile moments
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        residual_img = image.grad.detach()
        tile_residual, tile_energy = compute_tile_moments(residual_img)
        end.record()
        torch.cuda.synchronize()
        total_tile += start.elapsed_time(end)

        _, H, W = residual_img.shape
        tiles_x = (W + 16 - 1) // 16

        # Mean score (clone)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = compute_fw_score(tile_residual, tile_energy, tiles_x, render_pkg["radii"], render_pkg["geomBuffer"], render_pkg["binningBuffer"], 3)
        end.record()
        torch.cuda.synchronize()
        total_fw_mean += start.elapsed_time(end)

        # Variance score (split)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = compute_fw_score(tile_residual, tile_energy, tiles_x, render_pkg["radii"], render_pkg["geomBuffer"], render_pkg["binningBuffer"], 5)
        end.record()
        torch.cuda.synchronize()
        total_fw_var += start.elapsed_time(end)

        # total_fw computed from components after loop

    avg_core = total_core / args.iters
    avg_tile = total_tile / args.iters
    avg_fw_mean = total_fw_mean / args.iters
    avg_fw_var = total_fw_var / args.iters
    avg_fw = avg_tile + avg_fw_mean + avg_fw_var
    ratio = avg_fw / max(1e-6, avg_core)

    results = {
        "iters": args.iters,
        "avg_core_ms": avg_core,
        "avg_tile_moments_ms": avg_tile,
        "avg_fw_mean_ms": avg_fw_mean,
        "avg_fw_var_ms": avg_fw_var,
        "avg_fw_ms": avg_fw,
        "fw_overhead_ratio": ratio,
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
