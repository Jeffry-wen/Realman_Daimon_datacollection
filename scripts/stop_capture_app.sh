#!/usr/bin/env bash
set -euo pipefail

PORT="${CAPTURE_UI_PORT:-8766}"
WAIT_S="${CAPTURE_STOP_WAIT_S:-3}"

echo "[stop-capture] port=${PORT}"

if command -v python >/dev/null 2>&1; then
  python - "$PORT" <<'PY' || true
import sys
import urllib.request

port = sys.argv[1]
try:
    urllib.request.urlopen(f"http://127.0.0.1:{port}/cmd?name=stop", timeout=2).read()
    print("[stop-capture] sent stop command to UI")
except Exception as exc:
    print(f"[stop-capture] UI stop command failed: {exc}")
PY
fi

sleep "$WAIT_S"

mapfile -t PIDS < <(
  {
    if command -v ss >/dev/null 2>&1; then
      ss -ltnp "sport = :${PORT}" 2>/dev/null \
        | sed -n 's/.*pid=\([0-9]\+\).*/\1/p'
    fi
    pgrep -f "conda run .*capture_realman_x5_force_app.py" || true
    pgrep -f "conda run .*capture_realman_x5_force_aligned_app.py" || true
    pgrep -f "scripts/capture_realman_x5_force_app.py" || true
    pgrep -f "scripts/capture_realman_x5_force_aligned_app.py" || true
    pgrep -f "scripts/capture_realman_x5_force_aligned_app.sh" || true
    pgrep -f "scripts/capture_realman_x5_force_raw_spool_app.sh" || true
    pgrep -f "scripts/test_x5_wrist_queued_pipeline.py" || true
    pgrep -f "lerobot.scripts.lerobot_dm_tactile_sidecar" || true
    pgrep -f "tools/bi_x5_capture_app.py" || true
    pgrep -f "scripts/record_x5_wrist_raw_streams.py" || true
    pgrep -f "scripts/preview_x5_wrist_udp_streams.py" || true
    pgrep -f "ffmpeg .*_wrist_raw/episode-[0-9]+/.*wrist_raw\\.(mkv|ts)" || true
  } | sort -n | uniq
)

collect_descendants() {
  local pid="$1"
  local child
  while read -r child; do
    [[ -n "$child" ]] || continue
    echo "$child"
    collect_descendants "$child"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
}

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  mapfile -t PIDS < <(
    {
      printf '%s\n' "${PIDS[@]}"
      for pid in "${PIDS[@]}"; do
        collect_descendants "$pid"
      done
    } | sort -n | uniq
  )
fi

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "[stop-capture] no capture process found"
else
  echo "[stop-capture] stopping pids: ${PIDS[*]}"
  kill -INT "${PIDS[@]}" 2>/dev/null || true
  sleep 2

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 2

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop-capture] force killing pid=${pid}"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
fi

if command -v ss >/dev/null 2>&1; then
  ss -ltnp "sport = :${PORT}" || true
fi

# The X5 Flux receivers bind UDP ports directly. A different project (for
# example tactile-correction capturedagger-mixed) may legitimately own them,
# so report those owners without killing them by port alone.
if command -v ss >/dev/null 2>&1; then
  X5_PORTS=(
    "${X5_TACTILE_FLUX_LEFT_LEFT_PC_PORT:-61000}"
    "${X5_TACTILE_FLUX_LEFT_RIGHT_PC_PORT:-61001}"
    "${X5_TACTILE_FLUX_RIGHT_LEFT_PC_PORT:-61002}"
    "${X5_TACTILE_FLUX_RIGHT_RIGHT_PC_PORT:-61003}"
  )
  for x5_port in "${X5_PORTS[@]}"; do
    x5_owner="$(ss -H -lunp 2>/dev/null | awk -v port=":${x5_port}" '$4 ~ (port "$") {print; exit}')"
    if [[ -n "${x5_owner}" ]]; then
      echo "[stop-capture] NOTE: X5 UDP port ${x5_port} remains occupied by an external process: ${x5_owner}"
    fi
  done
fi
