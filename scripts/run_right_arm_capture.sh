#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load the normal machine profile, then let run_capture.sh apply the reviewed
# right-arm-only overrides after that profile has been sourced.
exec "${PROJECT_ROOT}/scripts/run_capture.sh" --right-arm "$@"
