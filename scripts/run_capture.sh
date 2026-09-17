#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CAPTURE_CONFIG:-${PROJECT_ROOT}/config/lab_5090.env}"
CHECK_ONLY=false
STOP_FIRST=true
RIGHT_ARM_ONLY=false
APP_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_capture.sh [options] [-- app arguments]

Options:
  --config PATH  Load a different shell-format environment profile.
  --right-arm    Force right-arm-only drag collection after loading the profile.
  --check        Validate configuration without stopping processes or touching hardware.
  --no-stop      Do not call stop_capture_app.sh before a real launch.
  -h, --help     Show this help.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ "$#" -lt 2 ]]; then
        echo "[capture-runner] ERROR: --config requires a path" >&2
        exit 2
      fi
      CONFIG_FILE="$2"
      shift 2
      ;;
    --check)
      CHECK_ONLY=true
      shift
      ;;
    --right-arm)
      RIGHT_ARM_ONLY=true
      shift
      ;;
    --no-stop)
      STOP_FIRST=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      APP_ARGS+=("$@")
      break
      ;;
    *)
      echo "[capture-runner] ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[capture-runner] ERROR: configuration file not found: $CONFIG_FILE" >&2
  echo "[capture-runner] Run: cp config/lab_5090.env.example config/lab_5090.env" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

if [[ "$RIGHT_ARM_ONLY" == "true" ]]; then
  export TASK_INDEX=right
  export CAPTURE_CONTROL_MODE="${RIGHT_ARM_CAPTURE_CONTROL_MODE:-drag}"
  # On this rig the physical right leader is wired to the driver's left FTDI
  # slot (DU0E1ZZP). Enable the cross-prefix mapping for right-arm capture.
  export SWAP_TELEOP_ACTIONS="${RIGHT_ARM_SWAP_TELEOP_ACTIONS:-true}"
  export RUN_NAME_PREFIX="${RIGHT_ARM_RUN_NAME_PREFIX:-kd_tacmae_force_right}"
  export CONNECT_LEFT_FORCE_SENSOR=false
  export CONNECT_RIGHT_FORCE_SENSOR=true
  export EPISODE_RETURN_TO_START_ENABLED="${RIGHT_ARM_EPISODE_RETURN_TO_START_ENABLED:-false}"
  export TASK_RIGHT="${TASK_RIGHT:-Pick up the three pens on the table and place them into the container.}"
  echo "[capture-runner] profile=right-arm-only control_mode=${CAPTURE_CONTROL_MODE} swap_teleop_actions=${SWAP_TELEOP_ACTIONS} task=${TASK_RIGHT}"
fi

if [[ -z "${LEROBOT_ROOT:-}" ]]; then
  echo "[capture-runner] ERROR: LEROBOT_ROOT is empty in $CONFIG_FILE" >&2
  exit 2
fi
if [[ ! -d "${LEROBOT_ROOT}/src/lerobot" || ! -f "${LEROBOT_ROOT}/tools/bi_x5_capture_app.py" ]]; then
  echo "[capture-runner] ERROR: LEROBOT_ROOT is not a compatible checkout: ${LEROBOT_ROOT}" >&2
  echo "[capture-runner] Expected src/lerobot/ and tools/bi_x5_capture_app.py" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
echo "[capture-runner] config=${CONFIG_FILE}"
echo "[capture-runner] lerobot=${LEROBOT_ROOT} conda_env=${CONDA_ENV:-lerobot51}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[capture-runner] ERROR: conda is not available on PATH" >&2
  exit 2
fi
CONDA_ENV_NAME="${CONDA_ENV:-lerobot51}"
if [[ -z "${CAPTURE_PYTHON_BIN:-}" ]]; then
  CONDA_ENV_PREFIX="$(
    conda env list | awk -v env_name="${CONDA_ENV_NAME}" '$1 == env_name {print $NF; exit}'
  )"
  if [[ -z "${CONDA_ENV_PREFIX}" ]]; then
    echo "[capture-runner] ERROR: conda environment not found: ${CONDA_ENV_NAME}" >&2
    exit 2
  fi
  CAPTURE_PYTHON_BIN="${CONDA_ENV_PREFIX}/bin/python"
fi
if [[ ! -x "${CAPTURE_PYTHON_BIN}" ]]; then
  echo "[capture-runner] ERROR: environment Python is not executable: ${CAPTURE_PYTHON_BIN}" >&2
  exit 2
fi
export CAPTURE_PYTHON_BIN
echo "[capture-runner] python=${CAPTURE_PYTHON_BIN}"
"${CAPTURE_PYTHON_BIN}" scripts/preflight_capture.py

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "[capture-runner] configuration check only; no process will be stopped and no hardware will be opened"
  DRY_RUN=true scripts/capture_realman_x5_force_aligned_app.sh "${APP_ARGS[@]}"
  exit 0
fi

if [[ "$STOP_FIRST" == "true" ]]; then
  scripts/stop_capture_app.sh || true
fi

exec scripts/capture_realman_x5_force_aligned_app.sh "${APP_ARGS[@]}"
