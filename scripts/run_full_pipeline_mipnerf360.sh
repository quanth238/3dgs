#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/home/tri-dev/dev/namn_workspace/dataset/mipnerf360}"
OUT_ROOT="${2:-./output/mipnerf360_full}"
IMAGES="${IMAGES:-images_4}"
ITERATIONS="${ITERATIONS:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES
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
RASTER_FW="submodules/diff-gaussian-rasterization"

SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai)
IDX=$((RANDOM % ${#SCENES[@]}))
SCENE="${SCENES[$IDX]}"
SRC="${DATASET_ROOT}/${SCENE}"

BASELINE_OUT="${OUT_ROOT}/${SCENE}_baseline"
FW_OUT="${OUT_ROOT}/${SCENE}_fw"
MINIMAL_OUT="${OUT_ROOT}/${SCENE}_minimal"

echo "Picked random scene: ${SCENE}"
echo "Source: ${SRC}"
echo "Images: ${IMAGES}"
echo "Output baseline: ${BASELINE_OUT}"
echo "Output fw: ${FW_OUT}"
echo "Output minimal: ${MINIMAL_OUT}"

mkdir -p "${BASELINE_OUT}" "${FW_OUT}" "${MINIMAL_OUT}"

echo "==> Install ORIGINAL rasterizer (3DGS baseline)"
pip uninstall -y diff-gaussian-rasterization >/dev/null 2>&1 || true
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" pip install -e "${RASTER_ORIG}" --no-build-isolation

echo "==> Train baseline (original rasterizer)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" \
  --disable_viewer --quiet --eval --iterations "${ITERATIONS}"

echo "==> Render baseline"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python render.py --iteration "${ITERATIONS}" -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" --quiet --eval --skip_train

echo "==> Install FW rasterizer (modified)"
pip uninstall -y diff-gaussian-rasterization >/dev/null 2>&1 || true
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" pip install -e "${RASTER_FW}" --no-build-isolation

if [[ "${RUN_MINIMAL}" == "1" ]]; then
  echo "==> Run minimal A/B/C validation (uses FW rasterizer)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/validate_fw_minimal.py -s "${SRC}" -i "${IMAGES}" -m "${MINIMAL_OUT}" \
    --iteration -1 --steps "${MINIMAL_STEPS}" --topk 500 --out_dir "${MINIMAL_OUT}"
fi

echo "==> Train FW (modified rasterizer)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" \
  --disable_viewer --quiet --eval --iterations "${ITERATIONS}" \
  --fw_densify

echo "==> Render FW"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python render.py --iteration "${ITERATIONS}" -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --quiet --eval --skip_train

echo "==> Metrics (baseline vs FW)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python metrics.py -m "${BASELINE_OUT}" "${FW_OUT}"

echo "==> Sanity correlation (baseline vs FW) using FW rasterizer"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/validate_fw_sanity.py -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" --iteration "${ITERATIONS}" \
  --out "${BASELINE_OUT}/fw_sanity.json"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/validate_fw_sanity.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --iteration "${ITERATIONS}" \
  --out "${FW_OUT}/fw_sanity.json"

if [[ "${RUN_ORACLE}" == "1" ]]; then
  echo "==> Oracle quality tests (E4/E5) on FW model"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/validate_fw_oracle.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --iteration "${ITERATIONS}" \
    --num_candidates "${ORACLE_CANDIDATES}" --topk "${ORACLE_TOPK}" --inner_steps "${ORACLE_STEPS}" \
    --out "${FW_OUT}/fw_oracle.json"
fi

if [[ "${RUN_ABLATION}" == "1" ]]; then
  echo "==> Ablation tests (A1/A2/A3/A4) on FW model"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/validate_fw_ablation.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --iteration "${ITERATIONS}" \
    --num_candidates 200 --out "${FW_OUT}/fw_ablation.json"
fi

if [[ "${RUN_PROFILE}" == "1" ]]; then
  echo "==> Profiling FW overhead"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/profile_fw_overhead.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --iteration "${ITERATIONS}" \
    --iters 10 --out "${FW_OUT}/fw_overhead.json"
fi

if [[ "${RUN_BUDGET}" == "1" ]]; then
  echo "==> Budget-quality curves"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/collect_budget_curve.py -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" --iterations ${BUDGET_ITERS} \
    --max_views "${BUDGET_MAX_VIEWS}" --out "${BASELINE_OUT}/budget_curve.json"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/collect_budget_curve.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --iterations ${BUDGET_ITERS} \
    --max_views "${BUDGET_MAX_VIEWS}" --out "${FW_OUT}/budget_curve.json"
fi

echo "Done."
