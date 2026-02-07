#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FILES=(
  "train.py"
  "scripts/validate_fw_minimal.py"
  "scripts/validate_fw_ablation.py"
  "scripts/validate_fw_oracle.py"
  "scripts/validate_fw_sanity.py"
)

fail=0

echo "[CHECK] Q usage (should be none)"
q_hits="$(rg -n "(^|[^A-Za-z0-9_])Q([^A-Za-z0-9_]|$)" "${FILES[@]}" || true)"
if [[ -n "$q_hits" ]]; then
  echo "FAIL: Found Q usage:"
  echo "$q_hits"
  fail=1
else
  echo "OK: No Q symbol found in target files."
fi

echo "[CHECK] _adjoint_scores return signature"
sig_hits="$(rg -n "return\\s+M\\s*,\\s*Z" "${FILES[@]}" || true)"
if [[ -z "$sig_hits" ]]; then
  echo "WARN: Did not find 'return M, Z' in target files."
  fail=1
else
  echo "OK: return M, Z found."
fi

echo "[CHECK] AbsGS split score (homodirectional grad abs-sum)"
abs_hits="$(rg -n "grad\\[:,\\s*:2\\]\\.abs\\(\\)\\.sum" "${FILES[@]}" || true)"
if [[ -z "$abs_hits" ]]; then
  echo "FAIL: Did not find abs-sum split score pattern."
  fail=1
else
  echo "OK: Found abs-sum split score pattern."
fi

if [[ "$fail" -ne 0 ]]; then
  echo "CHECK FAILED"
  exit 1
fi

echo "CHECK PASSED"
