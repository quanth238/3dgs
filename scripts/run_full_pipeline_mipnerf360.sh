#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/home/tri-dev/dev/namn_workspace/dataset/mipnerf360}"
OUT_ROOT="${2:-./output/mipnerf360_full}"
IMAGES="${IMAGES:-images_4}"
ITERATIONS="${ITERATIONS:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES
DATA_DEVICE="${DATA_DEVICE:-cuda}"
DENSIFY_GRAD_PERCENTILE="${DENSIFY_GRAD_PERCENTILE:-0.0}"
DENSIFY_TOPK="${DENSIFY_TOPK:-0}"
DENSIFY_TOPK_RATIO="${DENSIFY_TOPK_RATIO:-0.05}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-27000}"
DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-500}"
DENSIFY_INTERVAL="${DENSIFY_INTERVAL:-100}"
AWSRM_COLLECT_EVERY="${AWSRM_COLLECT_EVERY:-0}"
AWSRM_ERROR_TYPE="${AWSRM_ERROR_TYPE:-l1}"
AWSRM_MAX_PRIMITIVES="${AWSRM_MAX_PRIMITIVES:-6300000}"
AWSRM_USE_MOMENTS="${AWSRM_USE_MOMENTS:-1}"
AWSRM_K_CLONE="${AWSRM_K_CLONE:-0}"
AWSRM_K_SPLIT="${AWSRM_K_SPLIT:-0}"
AWSRM_CLONE_FRAC="${AWSRM_CLONE_FRAC:-0.0}"
AWSRM_SPLIT_FRAC="${AWSRM_SPLIT_FRAC:-0.0}"
FORCE_BASELINE="${FORCE_BASELINE:-0}"
RUN_VALIDATE_ONLY="${RUN_VALIDATE_ONLY:-0}"
RUN_MINIMAL="${RUN_MINIMAL:-1}"
MINIMAL_STEPS="${MINIMAL_STEPS:-60}"
RUN_ORACLE="${RUN_ORACLE:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"
RUN_BUDGET="${RUN_BUDGET:-1}"
ORACLE_CANDIDATES="${ORACLE_CANDIDATES:-100}"
ORACLE_TOPK="${ORACLE_TOPK:-20}"
ORACLE_STEPS="${ORACLE_STEPS:-10}"
BUDGET_ITERS="${BUDGET_ITERS:-7000 30000}"
BUDGET_MAX_VIEWS="${BUDGET_MAX_VIEWS:-10}"

RASTER_ORIG="submodules/diff-gaussian-rasterization-3dgs"

# SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai)
SCENES=(bicycle)
IDX=$((RANDOM % ${#SCENES[@]}))
SCENE="${SCENES[$IDX]}"
SRC="${DATASET_ROOT}/${SCENE}"

BASELINE_OUT="${OUT_ROOT}/${SCENE}_baseline"
FW_OUT="${OUT_ROOT}/${SCENE}_fw"
MINIMAL_OUT="${OUT_ROOT}/${SCENE}_minimal"
BASELINE_PLY="${BASELINE_OUT}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
BASELINE_RENDER_DIR="${BASELINE_OUT}/test/ours_${ITERATIONS}/renders"
FW_PLY="${FW_OUT}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
FW_RENDER_DIR="${FW_OUT}/test/ours_${ITERATIONS}/renders"

NEED_BASELINE_TRAIN=0
if [[ "${FORCE_BASELINE}" == "1" || ! -f "${BASELINE_PLY}" ]]; then
  NEED_BASELINE_TRAIN=1
fi
NEED_BASELINE_RENDER=0
if [[ ! -d "${BASELINE_RENDER_DIR}" ]]; then
  NEED_BASELINE_RENDER=1
fi
if [[ "${NEED_BASELINE_TRAIN}" == "1" ]]; then
  NEED_BASELINE_RENDER=1
fi
NEED_ORIG_INSTALL=0
if [[ "${NEED_BASELINE_TRAIN}" == "1" || "${NEED_BASELINE_RENDER}" == "1" || "${RUN_VALIDATE_ONLY}" == "1" ]]; then
  NEED_ORIG_INSTALL=1
fi

echo "Picked random scene: ${SCENE}"
echo "Source: ${SRC}"
echo "Images: ${IMAGES}"
echo "Output baseline: ${BASELINE_OUT}"
echo "Output fw: ${FW_OUT}"
echo "Output minimal: ${MINIMAL_OUT}"

mkdir -p "${BASELINE_OUT}" "${FW_OUT}" "${MINIMAL_OUT}"

if [[ "${NEED_ORIG_INSTALL}" == "1" ]]; then
  echo "==> Install ORIGINAL rasterizer (3DGS baseline)"
  pip uninstall -y diff-gaussian-rasterization >/dev/null 2>&1 || true
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" pip install -e "${RASTER_ORIG}" --no-build-isolation
fi

if [[ "${NEED_BASELINE_TRAIN}" == "1" ]]; then
  echo "==> Train baseline (original rasterizer)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" \
    --disable_viewer --quiet --eval --iterations "${ITERATIONS}" --data_device "${DATA_DEVICE}"
else
  echo "==> Skip baseline training (found ${BASELINE_PLY})"
fi

if [[ "${NEED_BASELINE_RENDER}" == "1" ]]; then
  echo "==> Render baseline"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python render.py --iteration "${ITERATIONS}" -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" --quiet --eval --skip_train --data_device "${DATA_DEVICE}"
else
  echo "==> Skip baseline render (found ${BASELINE_RENDER_DIR})"
fi

if [[ "${RUN_VALIDATE_ONLY}" == "1" && ! -f "${FW_PLY}" ]]; then
  echo "ERROR: RUN_VALIDATE_ONLY=1 but FW model not found at ${FW_PLY}"
  exit 1
fi

if [[ "${RUN_MINIMAL}" == "1" ]]; then
  echo "==> Run minimal A/B/C validation (adjoint score)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/validate_fw_minimal.py -s "${SRC}" -i "${IMAGES}" -m "${MINIMAL_OUT}" \
    --iteration -1 --steps "${MINIMAL_STEPS}" --topk 500 --out_dir "${MINIMAL_OUT}" --depths "" --data_device "${DATA_DEVICE}"
fi

if [[ "${RUN_VALIDATE_ONLY}" != "1" ]]; then
  echo "==> Train FW (adjoint score)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" \
    --disable_viewer --quiet --eval --iterations "${ITERATIONS}" --data_device "${DATA_DEVICE}" \
    --densify_from_iter "${DENSIFY_FROM_ITER}" --densification_interval "${DENSIFY_INTERVAL}" \
    --densify_grad_percentile "${DENSIFY_GRAD_PERCENTILE}" \
    --densify_topk "${DENSIFY_TOPK}" --densify_topk_ratio "${DENSIFY_TOPK_RATIO}" \
    --densify_until_iter "${DENSIFY_UNTIL_ITER}" \
    --fw_densify \
    --awsrm_collect_every "${AWSRM_COLLECT_EVERY}" \
    --awsrm_error_type "${AWSRM_ERROR_TYPE}" \
    --awsrm_max_primitives "${AWSRM_MAX_PRIMITIVES}" \
    --awsrm_use_moments "${AWSRM_USE_MOMENTS}" \
    --awsrm_K_clone "${AWSRM_K_CLONE}" \
    --awsrm_K_split "${AWSRM_K_SPLIT}" \
    --awsrm_clone_frac "${AWSRM_CLONE_FRAC}" \
    --awsrm_split_frac "${AWSRM_SPLIT_FRAC}"
else
  echo "==> Skip FW training (RUN_VALIDATE_ONLY=1)"
fi

if [[ "${RUN_VALIDATE_ONLY}" != "1" ]]; then
  echo "==> Render FW"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python render.py --iteration "${ITERATIONS}" -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --quiet --eval --skip_train --data_device "${DATA_DEVICE}"
else
  echo "==> Skip FW render (RUN_VALIDATE_ONLY=1)"
fi

if [[ "${RUN_VALIDATE_ONLY}" != "1" ]]; then
  echo "==> Metrics (baseline vs FW)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python metrics.py -m "${BASELINE_OUT}" "${FW_OUT}"
else
  echo "==> Skip metrics (RUN_VALIDATE_ONLY=1)"
fi

echo "==> Sanity correlation (baseline vs FW) using adjoint score"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/validate_fw_sanity.py -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" --iteration "${ITERATIONS}" \
  --out "${BASELINE_OUT}/fw_sanity.json" --depths "" --data_device "${DATA_DEVICE}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/validate_fw_sanity.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --iteration "${ITERATIONS}" \
  --out "${FW_OUT}/fw_sanity.json" --depths "" --data_device "${DATA_DEVICE}"

if [[ "${RUN_ORACLE}" == "1" ]]; then
  echo "==> Oracle quality tests (E4/E5) on FW model"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/validate_fw_oracle.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --iteration "${ITERATIONS}" \
    --num_candidates "${ORACLE_CANDIDATES}" --topk "${ORACLE_TOPK}" --inner_steps "${ORACLE_STEPS}" \
    --out "${FW_OUT}/fw_oracle.json" --depths "" --data_device "${DATA_DEVICE}"
fi

if [[ "${RUN_ABLATION}" == "1" ]]; then
  echo "==> Ablation tests (A1/A2/A3/A4) on FW model"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/validate_fw_ablation.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --iteration "${ITERATIONS}" \
    --num_candidates 200 --out "${FW_OUT}/fw_ablation.json" --depths "" --data_device "${DATA_DEVICE}"
fi

if [[ "${RUN_PROFILE}" == "1" ]]; then
  echo "==> Profiling FW overhead"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/profile_fw_overhead.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --iteration "${ITERATIONS}" \
    --iters 10 --out "${FW_OUT}/fw_overhead.json" --depths "" --data_device "${DATA_DEVICE}"
fi

if [[ "${RUN_BUDGET}" == "1" ]]; then
  echo "==> Budget-quality curves"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/collect_budget_curve.py -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" --iterations ${BUDGET_ITERS} \
    --max_views "${BUDGET_MAX_VIEWS}" --out "${BASELINE_OUT}/budget_curve.json" --depths "" --data_device "${DATA_DEVICE}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/collect_budget_curve.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --iterations ${BUDGET_ITERS} \
    --max_views "${BUDGET_MAX_VIEWS}" --out "${FW_OUT}/budget_curve.json" --depths "" --data_device "${DATA_DEVICE}"
fi

echo "Done."
