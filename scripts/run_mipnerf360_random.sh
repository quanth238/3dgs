#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/home/tri-dev/dev/namn_workspace/dataset/mipnerf360}"
OUT_ROOT="${2:-./output/mipnerf360_fw}"
IMAGES="${IMAGES:-images_4}"
ITERATIONS="${ITERATIONS:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES
DATA_DEVICE="${DATA_DEVICE:-cuda}"
DENSIFY_GRAD_PERCENTILE="${DENSIFY_GRAD_PERCENTILE:-0.0}"
DENSIFY_TOPK="${DENSIFY_TOPK:-0}"
DENSIFY_TOPK_RATIO="${DENSIFY_TOPK_RATIO:-0.05}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-20000}"
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
AWSRM_SEVERITY_ETA="${AWSRM_SEVERITY_ETA:-0.5}"
AWSRM_DEPTH_BINS="${AWSRM_DEPTH_BINS:-4}"
AWSRM_DEPTH_FAR_FRAC="${AWSRM_DEPTH_FAR_FRAC:-0.4}"
AWSRM_DEPTH_WEIGHT_GAMMA="${AWSRM_DEPTH_WEIGHT_GAMMA:-0.0}"
AWSRM_DEPTH_PROTECT_FRAC="${AWSRM_DEPTH_PROTECT_FRAC:-0.2}"
FORCE_BASELINE="${FORCE_BASELINE:-0}"
RUN_MINIMAL="${RUN_MINIMAL:-1}"
MINIMAL_STEPS="${MINIMAL_STEPS:-60}"

SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai)
IDX=$((RANDOM % ${#SCENES[@]}))
SCENE="${SCENES[$IDX]}"
SRC="${DATASET_ROOT}/${SCENE}"

BASELINE_OUT="${OUT_ROOT}/${SCENE}_baseline"
FW_OUT="${OUT_ROOT}/${SCENE}_fw"
MINIMAL_OUT="${OUT_ROOT}/${SCENE}_minimal"
BASELINE_PLY="${BASELINE_OUT}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
BASELINE_RENDER_DIR="${BASELINE_OUT}/test/ours_${ITERATIONS}/renders"

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

echo "Picked random scene: ${SCENE}"
echo "Source: ${SRC}"
echo "Images: ${IMAGES}"
echo "Output baseline: ${BASELINE_OUT}"
echo "Output fw: ${FW_OUT}"

mkdir -p "${BASELINE_OUT}" "${FW_OUT}"

if [[ "${RUN_MINIMAL}" == "1" ]]; then
  echo "Running minimal A/B/C validation (adjoint score) before full benchmark..."
  mkdir -p "${MINIMAL_OUT}"
  python scripts/validate_fw_minimal.py -s "${SRC}" -i "${IMAGES}" -m "${MINIMAL_OUT}" \
    --iteration -1 --steps "${MINIMAL_STEPS}" --topk 500 --out_dir "${MINIMAL_OUT}" --depths "" --data_device "${DATA_DEVICE}"
fi

if [[ "${NEED_BASELINE_TRAIN}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" \
    --disable_viewer --quiet --eval --iterations "${ITERATIONS}" --data_device "${DATA_DEVICE}"
else
  echo "==> Skip baseline training (found ${BASELINE_PLY})"
fi

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
  --awsrm_split_frac "${AWSRM_SPLIT_FRAC}" \
  --awsrm_severity_eta "${AWSRM_SEVERITY_ETA}" \
  --awsrm_depth_bins "${AWSRM_DEPTH_BINS}" \
  --awsrm_depth_far_frac "${AWSRM_DEPTH_FAR_FRAC}" \
  --awsrm_depth_weight_gamma "${AWSRM_DEPTH_WEIGHT_GAMMA}" \
  --awsrm_depth_protect_frac "${AWSRM_DEPTH_PROTECT_FRAC}"

if [[ "${NEED_BASELINE_RENDER}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python render.py --iteration "${ITERATIONS}" -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" --quiet --eval --skip_train --data_device "${DATA_DEVICE}"
else
  echo "==> Skip baseline render (found ${BASELINE_RENDER_DIR})"
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python render.py --iteration "${ITERATIONS}" -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --quiet --eval --skip_train --data_device "${DATA_DEVICE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python metrics.py -m "${BASELINE_OUT}" "${FW_OUT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/validate_fw_sanity.py -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" --iteration "${ITERATIONS}" \
  --out "${BASELINE_OUT}/fw_sanity.json" --depths "" --data_device "${DATA_DEVICE}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/validate_fw_sanity.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --iteration "${ITERATIONS}" \
  --out "${FW_OUT}/fw_sanity.json" --depths "" --data_device "${DATA_DEVICE}"
