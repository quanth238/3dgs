#!/usr/bin/env python3
import json
import random
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import compute_tile_residual, compute_fw_score
except Exception as exc:
    raise RuntimeError("compute_tile_residual/compute_fw_score not available. Rebuild rasterizer.") from exc


def _rankdata(x: torch.Tensor) -> torch.Tensor:
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


def compute_scores(view, gaussians, pipe, opt, background):
    _zero_grads(gaussians)
    render_pkg = render(view, gaussians, pipe, background, separate_sh=False, use_trained_exp=False, return_aux=True)
    image = render_pkg["render"]
    if view.alpha_mask is not None:
        image = image * view.alpha_mask.cuda()
    image.retain_grad()

    gt_image = view.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    ssim_value = ssim(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
    loss.backward()

    residual_grad = image.grad.detach()
    residual_abs = (image.detach() - gt_image).abs()

    tile_grad = compute_tile_residual(residual_grad)
    tile_abs = compute_tile_residual(residual_abs)

    fw_score = compute_fw_score(tile_grad, render_pkg["radii"], render_pkg["geomBuffer"], render_pkg["binningBuffer"], 0)
    abs_score = compute_fw_score(tile_abs, render_pkg["radii"], render_pkg["geomBuffer"], render_pkg["binningBuffer"], 0)
    mean2d_score = torch.norm(render_pkg["viewspace_points"].grad[:, :2], dim=-1)

    feat_dc = gaussians._features_dc.grad
    feat_rest = gaussians._features_rest.grad
    feat_grad = torch.sqrt((feat_dc ** 2).sum(dim=(1, 2)) + (feat_rest ** 2).sum(dim=(1, 2)) + 1e-8)

    return {
        "render_pkg": render_pkg,
        "loss": float(loss.item()),
        "fw_score": fw_score,
        "abs_score": abs_score,
        "mean2d_score": mean2d_score,
        "feat_grad": feat_grad,
    }


def clone_one(gaussians, idx, radii):
    gaussians.tmp_radii = radii
    new_xyz = gaussians._xyz[idx:idx+1].detach().clone()
    new_features_dc = gaussians._features_dc[idx:idx+1].detach().clone()
    new_features_rest = gaussians._features_rest[idx:idx+1].detach().clone()
    new_opacities = gaussians._opacity[idx:idx+1].detach().clone()
    new_scaling = gaussians._scaling[idx:idx+1].detach().clone()
    new_rotation = gaussians._rotation[idx:idx+1].detach().clone()
    new_tmp_radii = radii[idx:idx+1].detach().clone()
    gaussians.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)


def gain_for_index(gaussians, scene, view, pipe, opt, background, idx, base_loss, base_radii, steps):
    clone_one(gaussians, idx, base_radii)
    loss_after = base_loss
    for _ in range(steps):
        _zero_grads(gaussians)
        render_pkg = render(view, gaussians, pipe, background, separate_sh=False, use_trained_exp=False, return_aux=False)
        image = render_pkg["render"]
        if view.alpha_mask is not None:
            image = image * view.alpha_mask.cuda()
        gt_image = view.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        gaussians.exposure_optimizer.step()
        gaussians.exposure_optimizer.zero_grad(set_to_none=True)
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        loss_after = float(loss.item())
    return base_loss - loss_after


def main():
    parser = ArgumentParser(description="Oracle quality tests (E4/E5)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=100)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--inner_steps", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    args = get_combined_args(parser)

    safe_state(True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    opt_args = opt.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, opt_args.optimizer_type)
    load_iter = args.iteration if args.iteration is not None and args.iteration >= 0 else None
    scene = Scene(dataset, gaussians, load_iteration=load_iter, shuffle=False)
    gaussians.training_setup(opt_args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    view = random.choice(scene.getTrainCameras())
    scores = compute_scores(view, gaussians, pipe, opt_args, background)
    base_loss = scores["loss"]
    base_radii = scores["render_pkg"]["radii"].detach()
    visible = base_radii > 0
    visible_idx = torch.where(visible)[0]

    if visible_idx.numel() == 0:
        raise RuntimeError("No visible Gaussians for oracle test.")

    num_candidates = min(args.num_candidates, visible_idx.numel())
    cand_idx = visible_idx[torch.randperm(visible_idx.numel())[:num_candidates]]

    method_scores = {
        "fw": scores["fw_score"][cand_idx],
        "mean2d": scores["mean2d_score"][cand_idx],
        "abs": scores["abs_score"][cand_idx],
    }

    results = {
        "base_loss": base_loss,
        "num_candidates": int(num_candidates),
        "topk": int(args.topk),
        "inner_steps": int(args.inner_steps),
        "corr_score_featgrad": {
            "fw": _spearman(scores["fw_score"][cand_idx], scores["feat_grad"][cand_idx]),
            "mean2d": _spearman(scores["mean2d_score"][cand_idx], scores["feat_grad"][cand_idx]),
            "abs": _spearman(scores["abs_score"][cand_idx], scores["feat_grad"][cand_idx]),
        },
        "mean_gain_topk": {},
        "corr_score_gain": {},
    }

    snapshot = gaussians.capture()

    for method, ms in method_scores.items():
        topk = min(args.topk, ms.numel())
        _, local_top_idx = torch.topk(ms, k=topk, largest=True)
        chosen = cand_idx[local_top_idx]

        gains = []
        for idx in chosen.tolist():
            gaussians.restore(snapshot, opt_args)
            gain = gain_for_index(gaussians, scene, view, pipe, opt_args, background, idx, base_loss, base_radii, args.inner_steps)
            gains.append(gain)
        if len(gains) > 0:
            results["mean_gain_topk"][method] = float(torch.tensor(gains).mean().item())
        else:
            results["mean_gain_topk"][method] = float("nan")

        # Correlation score-gain on chosen set
        if len(gains) > 1:
            results["corr_score_gain"][method] = _spearman(ms[local_top_idx], torch.tensor(gains, device=ms.device))
        else:
            results["corr_score_gain"][method] = float("nan")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
