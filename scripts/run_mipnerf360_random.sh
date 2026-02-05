#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/home/tri-dev/dev/namn_workspace/dataset/mipnerf360}"
OUT_ROOT="${2:-./output/mipnerf360_fw}"
IMAGES="${IMAGES:-images_4}"
ITERATIONS="${ITERATIONS:-30000}"
CUDA_INVISIBLE_DEVICES="${CUDA_INVISIBLE_DEVICES:-3}"
export CUDA_INVISIBLE_DEVICES
RUN_MINIMAL="${RUN_MINIMAL:-1}"
MINIMAL_STEPS="${MINIMAL_STEPS:-60}"

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

mkdir -p "${BASELINE_OUT}" "${FW_OUT}"

if [[ "${RUN_MINIMAL}" == "1" ]]; then
  echo "Running minimal A/B/C validation before full benchmark..."
  mkdir -p "${MINIMAL_OUT}"
  python scripts/validate_fw_minimal.py -s "${SRC}" -i "${IMAGES}" -m "${MINIMAL_OUT}" \
    --iteration -1 --steps "${MINIMAL_STEPS}" --topk 500 --out_dir "${MINIMAL_OUT}"
fi

CUDA_INVISIBLE_DEVICES="${CUDA_INVISIBLE_DEVICES}" python train.py -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" \
  --disable_viewer --quiet --eval --iterations "${ITERATIONS}"

CUDA_INVISIBLE_DEVICES="${CUDA_INVISIBLE_DEVICES}" python train.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" \
  --disable_viewer --quiet --eval --iterations "${ITERATIONS}" \
  --fw_densify

CUDA_INVISIBLE_DEVICES="${CUDA_INVISIBLE_DEVICES}" python render.py --iteration "${ITERATIONS}" -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" --quiet --eval --skip_train
CUDA_INVISIBLE_DEVICES="${CUDA_INVISIBLE_DEVICES}" python render.py --iteration "${ITERATIONS}" -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --quiet --eval --skip_train

CUDA_INVISIBLE_DEVICES="${CUDA_INVISIBLE_DEVICES}" python metrics.py -m "${BASELINE_OUT}" "${FW_OUT}"

CUDA_INVISIBLE_DEVICES="${CUDA_INVISIBLE_DEVICES}" python scripts/validate_fw_sanity.py -s "${SRC}" -i "${IMAGES}" -m "${BASELINE_OUT}" --iteration "${ITERATIONS}" \
  --out "${BASELINE_OUT}/fw_sanity.json"
CUDA_INVISIBLE_DEVICES="${CUDA_INVISIBLE_DEVICES}" python scripts/validate_fw_sanity.py -s "${SRC}" -i "${IMAGES}" -m "${FW_OUT}" --iteration "${ITERATIONS}" \
  --out "${FW_OUT}/fw_sanity.json"
