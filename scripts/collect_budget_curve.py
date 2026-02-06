#!/usr/bin/env python3
import json
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
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

def _mean_scalar(values):
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        if torch.is_tensor(v):
            total += float(v.detach().mean().item())
        else:
            total += float(v)
    return total / max(1, len(values))


def eval_iteration(dataset, pipe, iteration, max_views=None):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTestCameras()
    if max_views is not None:
        views = views[:max_views]

    ssims = []
    psnrs = []
    lpipss = []
    with torch.no_grad():
        for view in views:
            rendering = render(view, gaussians, pipe, background, separate_sh=False, use_trained_exp=dataset.train_test_exp)["render"]
            gt = view.original_image.cuda()
            ssims.append(ssim(rendering, gt))
            psnrs.append(psnr(rendering, gt))
            lpipss.append(lpips(rendering, gt, net_type='vgg'))

    return {
        "iteration": iteration,
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "psnr": _mean_scalar(psnrs),
        "ssim": _mean_scalar(ssims),
        "lpips": _mean_scalar(lpipss),
    }


def main():
    parser = ArgumentParser(description="Collect budget-quality curve")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = _safe_get_args(parser)
    if not hasattr(args, "max_views"):
        args.max_views = None

    safe_state(args.quiet)
    dataset = model.extract(args)
    _ensure_depths_available(dataset)
    pipe = pipeline.extract(args)

    results = []
    for it in args.iterations:
        results.append(eval_iteration(dataset, pipe, it, args.max_views))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
