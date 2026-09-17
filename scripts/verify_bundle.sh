#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

bash -n \
  scripts/run_capture.sh \
  scripts/run_right_arm_capture.sh \
  scripts/verify_bundle.sh \
  scripts/capture_realman_x5_force_aligned_app.sh \
  scripts/capture_realman_x5_force_app.sh \
  scripts/stop_capture_app.sh

PYTHON_COMMAND=(python)
if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV:-lerobot51}"; then
  PYTHON_COMMAND=(conda run --no-capture-output -n "${CONDA_ENV:-lerobot51}" python)
fi

PYTHONDONTWRITEBYTECODE=1 "${PYTHON_COMMAND[@]}" -m py_compile \
  scripts/preflight_capture.py \
  scripts/capture_realman_x5_force_aligned_app.py \
  scripts/capture_realman_x5_force_app.py \
  scripts/listen_foot_pedal.py \
  capture_support/tactile_frame_bridge.py
PYTHONDONTWRITEBYTECODE=1 "${PYTHON_COMMAND[@]}" -m unittest -v \
  tests.test_capture_partial_leader_fallback \
  tests.test_foot_pedal_two_pedal_stop

echo "[verify] static checks and hardware-free unit tests passed"
