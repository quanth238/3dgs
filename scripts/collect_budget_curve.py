#!/usr/bin/env python3
import json
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from utils.general_utils import safe_state


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
    for view in views:
        rendering = render(view, gaussians, pipe, background, separate_sh=False, use_trained_exp=dataset.train_test_exp)["render"]
        gt = view.original_image.cuda()
        ssims.append(ssim(rendering, gt))
        psnrs.append(psnr(rendering, gt))
        lpipss.append(lpips(rendering, gt, net_type='vgg'))

    return {
        "iteration": iteration,
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "psnr": float(torch.tensor(psnrs).mean().item()),
        "ssim": float(torch.tensor(ssims).mean().item()),
        "lpips": float(torch.tensor(lpipss).mean().item()),
    }


def main():
    parser = ArgumentParser(description="Collect budget-quality curve")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = get_combined_args(parser)

    safe_state(True)
    dataset = model.extract(args)
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
