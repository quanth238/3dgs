#!/usr/bin/env python3
import json
import math
import random
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.graphics_utils import geom_transform_points
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


def ndc_to_pix(v, s):
    return ((v + 1.0) * s - 1.0) * 0.5


def weighted_score_subset(tile_residual, px, py, radii, idxs, tile_size, w, h, mode="norm_sum", use_weight=True):
    tiles_x = (w + tile_size - 1) // tile_size
    tiles_y = (h + tile_size - 1) // tile_size
    scores = torch.zeros((idxs.numel(),), device=tile_residual.device)
    for out_i, idx in enumerate(idxs.tolist()):
        x = px[idx].item()
        y = py[idx].item()
        r = radii[idx].item()
        if r <= 0:
            continue
        tx0 = int(max(0, math.floor((x - r) / tile_size)))
        tx1 = int(min(tiles_x - 1, math.floor((x + r) / tile_size)))
        ty0 = int(max(0, math.floor((y - r) / tile_size)))
        ty1 = int(min(tiles_y - 1, math.floor((y + r) / tile_size)))
        sum_vec = torch.zeros((3,), device=tile_residual.device)
        sum_norm = 0.0
        sigma = max(1.0, r / 3.0)
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile_id = ty * tiles_x + tx
                v = tile_residual[tile_id]
                if use_weight:
                    cx = tx * tile_size + tile_size * 0.5
                    cy = ty * tile_size + tile_size * 0.5
                    dx = cx - x
                    dy = cy - y
                    wgt = math.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
                    v = v * wgt
                if mode == "sum_norm":
                    sum_norm += torch.norm(v).item()
                else:
                    sum_vec += v
        if mode == "sum_norm":
            scores[out_i] = sum_norm
        else:
            scores[out_i] = torch.norm(sum_vec)
    return scores


def main():
    parser = ArgumentParser(description="Ablation tests A1/A2/A3/A4")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--num_candidates", type=int, default=200)
    parser.add_argument("--out", type=str, default=None)
    args = get_combined_args(parser)

    safe_state(True)
    random.seed(0)
    torch.manual_seed(0)

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
    render_pkg = render(view, gaussians, pipe, background, separate_sh=False, use_trained_exp=False, return_aux=True)
    image = render_pkg["render"]
    if view.alpha_mask is not None:
        image = image * view.alpha_mask.cuda()
    image.retain_grad()

    gt_image = view.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    ssim_value = ssim(image, gt_image)
    loss = (1.0 - opt_args.lambda_dssim) * Ll1 + opt_args.lambda_dssim * (1.0 - ssim_value)
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

    visible = render_pkg["radii"] > 0
    visible_idx = torch.where(visible)[0]
    num_candidates = min(args.num_candidates, visible_idx.numel())
    cand_idx = visible_idx[torch.randperm(visible_idx.numel())[:num_candidates]]

    xyz = gaussians.get_xyz.detach()
    ndc = geom_transform_points(xyz, view.full_proj_transform)
    px = ndc_to_pix(ndc[:, 0], view.image_width)
    py = ndc_to_pix(ndc[:, 1], view.image_height)

    w = view.image_width
    h = view.image_height
    radii = render_pkg["radii"].detach()
    tile_size = 16

    score_weight_norm = weighted_score_subset(tile_grad, px, py, radii, cand_idx, tile_size, w, h, mode="norm_sum", use_weight=True)
    score_unweight_norm = weighted_score_subset(tile_grad, px, py, radii, cand_idx, tile_size, w, h, mode="norm_sum", use_weight=False)
    score_weight_sum = weighted_score_subset(tile_grad, px, py, radii, cand_idx, tile_size, w, h, mode="sum_norm", use_weight=True)

    gated_fw = fw_score[cand_idx] * (mean2d_score[cand_idx] / (mean2d_score[cand_idx].mean() + 1e-8))

    results = {
        "corr_fw_featgrad": _spearman(fw_score[cand_idx], feat_grad[cand_idx]),
        "corr_abs_featgrad": _spearman(abs_score[cand_idx], feat_grad[cand_idx]),
        "corr_mean2d_featgrad": _spearman(mean2d_score[cand_idx], feat_grad[cand_idx]),
        "corr_weight_norm_featgrad": _spearman(score_weight_norm, feat_grad[cand_idx]),
        "corr_unweight_norm_featgrad": _spearman(score_unweight_norm, feat_grad[cand_idx]),
        "corr_weight_sumnorm_featgrad": _spearman(score_weight_sum, feat_grad[cand_idx]),
        "corr_gated_fw_featgrad": _spearman(gated_fw, feat_grad[cand_idx]),
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
