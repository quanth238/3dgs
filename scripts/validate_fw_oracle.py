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


def _adjoint_scores(view, gaussians, pipe, background, image, gt_image, use_trained_exp=False, error_type="grad", depth_map=None, depth_weight_gamma=0.0):
    phi = _adjoint_phi(image, gt_image, error_type)
    if depth_map is not None and depth_weight_gamma != 0.0:
        depth = 1.0 / (depth_map.detach().clamp_min(1e-6))
        depth_valid = depth[depth > 0]
        if depth_valid.numel() > 0:
            depth_median = depth_valid.median()
            depth_weight = (depth / (depth_median + 1e-6)).pow(depth_weight_gamma).clamp(0.25, 4.0)
            phi = phi * depth_weight
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


def compute_scores(view, gaussians, pipe, opt, background):
    _zero_grads(gaussians)
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

    M, Z = _adjoint_scores(
        view, gaussians, pipe, background, image, gt_image,
        use_trained_exp=False, error_type=error_type,
        depth_map=render_pkg.get("depth"),
        depth_weight_gamma=float(getattr(opt, "awsrm_depth_weight_gamma", 0.0) or 0.0),
    )
    eps = float(getattr(opt, "awsrm_eps", 1e-6))
    mu_raw = M / (Z + eps)
    severity_eta = float(getattr(opt, "awsrm_severity_eta", 0.5))
    denom_scale = Z + eps
    if denom_scale.numel() > 0:
        median_denom = denom_scale[denom_scale > 0].median() if (denom_scale > 0).any() else denom_scale.median()
        lambda_denom = max(eps, float(median_denom) * 0.1)
        denom_scale = denom_scale + lambda_denom
    mu = mu_raw * denom_scale.pow(1.0 - severity_eta)
    mean2d_score = torch.norm(render_pkg["viewspace_points"].grad[:, :2], dim=-1)

    feat_dc = gaussians._features_dc.grad
    feat_rest = gaussians._features_rest.grad
    feat_grad = torch.sqrt((feat_dc ** 2).sum(dim=(1, 2)) + (feat_rest ** 2).sum(dim=(1, 2)) + 1e-8)

    return {
        "render_pkg": render_pkg,
        "loss": float(loss.item()),
        "adjoint_score": mu,
        "split_score": render_pkg["viewspace_points"].grad[:, :2].abs().sum(dim=-1) * render_pkg["radii"].clamp_min(1.0),
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
        use_trained_exp = getattr(opt, "train_test_exp", False)
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
    parser.add_argument("--quiet", action="store_true")
    args = _safe_get_args(parser)

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
    _ensure_exposure(gaussians, scene)
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
        "adjoint": scores["adjoint_score"][cand_idx],
        "mean2d": scores["mean2d_score"][cand_idx],
    }

    results = {
        "base_loss": base_loss,
        "num_candidates": int(num_candidates),
        "topk": int(args.topk),
        "inner_steps": int(args.inner_steps),
        "corr_score_featgrad": {
            "adjoint": _spearman(scores["adjoint_score"][cand_idx], scores["feat_grad"][cand_idx]),
            "mean2d": _spearman(scores["mean2d_score"][cand_idx], scores["feat_grad"][cand_idx]),
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
