#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import logging
import os
import queue
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_LEROBOT_ROOT_VALUE = os.environ.get("LEROBOT_ROOT", "").strip()
LEROBOT_ROOT = (
    Path(_LEROBOT_ROOT_VALUE).expanduser().resolve()
    if _LEROBOT_ROOT_VALUE
    else Path("__LEROBOT_ROOT_NOT_SET__")
)
APP_PATH = LEROBOT_ROOT / "tools" / "bi_x5_capture_app.py"
_ACTIVE_ROBOTS: list[Any] = []
_RGB_PREVIEWER: Any | None = None
_TACTILE_READ_CACHES: list[Any] = []
_WRIST_PROCESSED_CACHES: list[Any] = []
_RAW_SPOOL_SAVE_PATCHED_DATASETS: set[int] = set()
_RAW_SPOOL_FINALIZE_PATCHED_DATASETS: set[int] = set()
_RAW_SPOOL_FINALIZE_QUEUE: queue.Queue[dict[str, Any] | None] | None = None
_RAW_SPOOL_FINALIZE_WORKER: threading.Thread | None = None
_RAW_SPOOL_FINALIZE_ERRORS: list[str] = []
_RAW_SPOOL_FINALIZE_LOCK = threading.Lock()
_RAW_SPOOL_NEXT_EPISODE_INDEX: int | None = None
_RAW_SPOOL_PENDING_FINISHED_EPISODES: list[int] = []
_TACTILE_SIDECAR_PROC: subprocess.Popen[Any] | None = None
_CAPTURE_TARGET_TIME_PERF = threading.local()


def set_capture_target_time_perf(value: float | None) -> None:
    _CAPTURE_TARGET_TIME_PERF.value = 0.0 if value is None else float(value)


def get_capture_target_time_perf() -> float:
    return float(getattr(_CAPTURE_TARGET_TIME_PERF, "value", 0.0) or 0.0)

SYNC_DIAGNOSTIC_FIELDS = (
    "sync_obs_read_dt_ms",
    "sync_age_ms_left_cam_left_wrist",
    "sync_age_ms_right_cam_right_wrist",
    "sync_age_ms_cam_d405_color",
    "sync_age_ms_tactile_left_left",
    "sync_age_ms_tactile_left_right",
    "sync_age_ms_tactile_right_left",
    "sync_age_ms_tactile_right_right",
    "sync_age_ms_visual_max",
    "sync_age_ms_visual_mean",
)

SYNC_SOURCE_TO_FIELD = {
    "left_cam_left_wrist": "sync_age_ms_left_cam_left_wrist",
    "right_cam_right_wrist": "sync_age_ms_right_cam_right_wrist",
    "cam_d405_color": "sync_age_ms_cam_d405_color",
    "depth_deformation.tactile_left_left": "sync_age_ms_tactile_left_left",
    "depth_deformation.tactile_left_right": "sync_age_ms_tactile_left_right",
    "depth_deformation.tactile_right_left": "sync_age_ms_tactile_right_left",
    "depth_deformation.tactile_right_right": "sync_age_ms_tactile_right_right",
}

# Optional no-file wrist calibration defaults. Fill this table with the measured
# X5 camera intrinsics keyed by arm x5_ip, for example:
# "192.168.1.10": {
#     "camera_matrix": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
#     "dist_coeffs": [k1, k2, k3, k4],
#     "image_size": [1920, 1080],
# }
DEFAULT_WRIST_UNDISTORT_CALIBRATIONS_BY_IP: dict[str, dict[str, Any]] = {
    "192.168.1.10": {
        "camera_matrix": [
            [640.5965928696197, 0.0, 961.8777675439231],
            [0.0, 645.1134249713024, 565.2584490745693],
            [0.0, 0.0, 1.0],
        ],
        "dist_coeffs": [
            -0.02458011521671438,
            -0.0040641137508396225,
            -0.0005996888623257521,
            0.00016229704144288647,
        ],
        "image_size": [1920, 1080],
        "camera_model": "pinhole",
        "distortion_model": "equidistant",
    },
    "192.168.1.11": {
        "camera_matrix": [
            [647.1996142622046, 0.0, 959.6477732146316],
            [0.0, 651.8967543490795, 535.123521951627],
            [0.0, 0.0, 1.0],
        ],
        "dist_coeffs": [
            -0.031098601154318915,
            -0.0017026869931588656,
            -0.0008363765213525148,
            0.00003650454432836755,
        ],
        "image_size": [1920, 1080],
        "camera_model": "pinhole",
        "distortion_model": "equidistant",
    },
}


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in {None, ""}:
        return default
    return float(value)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in {None, ""}:
        return default
    return int(value)


def env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}. Got: {value!r}")
    return value


def env_optional_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or None


def env_optional_first(*names: str) -> str | None:
    for name in names:
        value = env_optional_str(name)
        if value is not None:
            return value
    return None


def _parse_numeric_json_or_csv(value: str, *, name: str) -> Any:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} is empty")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return [float(item) for item in value.replace(";", ",").split(",") if item.strip()]


def env_cv2_backend(name: str, default: str) -> int:
    from lerobot.cameras import Cv2Backends

    value = os.environ.get(name, default).strip()
    if not value:
        value = default
    upper = value.upper()
    if upper in Cv2Backends.__members__:
        return int(Cv2Backends[upper])
    return int(value)


def configure_opencv_runtime() -> None:
    threads = env_int("OPENCV_NUM_THREADS", env_int("WRIST_UNDISTORT_OPENCV_THREADS", 1))
    if threads > 0:
        cv2.setNumThreads(threads)
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass
    opencl_enabled = env_bool("OPENCV_OPENCL", False)
    try:
        cv2.ocl.setUseOpenCL(opencl_enabled)
    except Exception:
        pass
    try:
        actual_threads = cv2.getNumThreads()
    except Exception:
        actual_threads = threads
    try:
        actual_opencl = bool(cv2.ocl.useOpenCL())
    except Exception:
        actual_opencl = opencl_enabled
    logging.info(
        "OpenCV runtime configured: threads=%s opencl=%s optimized=true",
        actual_threads,
        actual_opencl,
    )


def _tcp_port_open(host: str, port: int, timeout_s: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _recv_exact_from_socket(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _tactile_sidecar_status_available(host: str, port: int, timeout_s: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            sock.sendall(b'{"cmd":"status"}\n')
            header = _recv_exact_from_socket(sock, 4)
            if len(header) != 4:
                return False
            size = struct.unpack(">I", header)[0]
            return size > 0
    except OSError:
        return False


def _tactile_sidecar_sensor_specs() -> list[str]:
    left_ip = os.environ.get("X5_LEFT_IP", "192.168.1.10").strip() or "192.168.1.10"
    right_ip = os.environ.get("X5_RIGHT_IP", "192.168.1.11").strip() or "192.168.1.11"
    return [
        (
            "left_left="
            f"{left_ip}:{env_int('X5_TACTILE_FLUX_LEFT_LEFT_GRPC_PORT', 50051)}:"
            f"{env_int('X5_TACTILE_FLUX_LEFT_LEFT_DEV_ID', 0)}:"
            f"{env_int('X5_TACTILE_FLUX_LEFT_LEFT_PC_PORT', 61000)}"
        ),
        (
            "left_right="
            f"{left_ip}:{env_int('X5_TACTILE_FLUX_LEFT_RIGHT_GRPC_PORT', 50052)}:"
            f"{env_int('X5_TACTILE_FLUX_LEFT_RIGHT_DEV_ID', 2)}:"
            f"{env_int('X5_TACTILE_FLUX_LEFT_RIGHT_PC_PORT', 61001)}"
        ),
        (
            "right_left="
            f"{right_ip}:{env_int('X5_TACTILE_FLUX_RIGHT_LEFT_GRPC_PORT', 50051)}:"
            f"{env_int('X5_TACTILE_FLUX_RIGHT_LEFT_DEV_ID', 0)}:"
            f"{env_int('X5_TACTILE_FLUX_RIGHT_LEFT_PC_PORT', 61002)}"
        ),
        (
            "right_right="
            f"{right_ip}:{env_int('X5_TACTILE_FLUX_RIGHT_RIGHT_GRPC_PORT', 50052)}:"
            f"{env_int('X5_TACTILE_FLUX_RIGHT_RIGHT_DEV_ID', 2)}:"
            f"{env_int('X5_TACTILE_FLUX_RIGHT_RIGHT_PC_PORT', 61003)}"
        ),
    ]


def start_tactile_sidecar_if_requested() -> None:
    global _TACTILE_SIDECAR_PROC
    if not env_bool("CONNECT_TACTILE_SIDECAR", False):
        return
    if not env_bool("TACTILE_SIDECAR_AUTOSTART", True):
        return
    if _TACTILE_SIDECAR_PROC is not None and _TACTILE_SIDECAR_PROC.poll() is None:
        return

    host = os.environ.get("TACTILE_SIDECAR_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = env_int("TACTILE_SIDECAR_PORT", 61300)
    if _tactile_sidecar_status_available(host, port, timeout_s=0.2):
        logging.info("Tactile sidecar already listening on %s:%s; using existing process", host, port)
        return

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_dm_tactile_sidecar",
        "--bind-host",
        host,
        "--bind-port",
        str(port),
        "--pc-host",
        os.environ.get("X5_TACTILE_FLUX_PC_HOST", "192.168.1.100").strip() or "192.168.1.100",
        "--backend",
        os.environ.get("TACTILE_SIDECAR_BACKEND", "Flux").strip() or "Flux",
        "--mode",
        os.environ.get("X5_TACTILE_MODE", "standard").strip().lower() or "standard",
        "--max-fps",
        str(env_int("X5_TACTILE_MAX_FPS", 120)),
        "--wait-timeout-ms",
        str(env_int("TACTILE_SIDECAR_WAIT_TIMEOUT_MS", 500)),
        "--log-interval-s",
        str(env_float("TACTILE_SIDECAR_LOG_INTERVAL_S", 5.0)),
        "--no-raw",
        "--enable-depth",
        "--enable-deformation",
    ]
    sdk_root = os.environ.get("DMROBOTICS_SDK_PATH", "").strip()
    if sdk_root:
        cmd.extend(["--sdk-root", sdk_root])
    for spec in _tactile_sidecar_sensor_specs():
        cmd.extend(["--sensor", spec])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{LEROBOT_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    logging.info("Starting tactile sidecar: %s", " ".join(cmd))
    _TACTILE_SIDECAR_PROC = subprocess.Popen(cmd, env=env)

    deadline = time.perf_counter() + max(0.1, env_float("TACTILE_SIDECAR_START_TIMEOUT_S", 8.0))
    while time.perf_counter() < deadline:
        if _TACTILE_SIDECAR_PROC.poll() is not None:
            raise RuntimeError(
                f"Tactile sidecar exited before listening on {host}:{port} "
                f"(returncode={_TACTILE_SIDECAR_PROC.returncode})"
            )
        if _tactile_sidecar_status_available(host, port, timeout_s=0.2):
            logging.info("Tactile sidecar is listening on %s:%s", host, port)
            return
        time.sleep(0.05)
    raise TimeoutError(f"Tactile sidecar did not listen on {host}:{port} before timeout")


def stop_tactile_sidecar() -> None:
    global _TACTILE_SIDECAR_PROC
    proc = _TACTILE_SIDECAR_PROC
    _TACTILE_SIDECAR_PROC = None
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3.0)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception as exc:
            logging.debug("Failed stopping tactile sidecar: %s", exc)


def patch_d405_camera_async_read(camera: Any, camera_name: str) -> None:
    if getattr(camera, "_kd_tacmae_d405_latest_patch", False):
        return

    original_async_read = camera.async_read
    read_latest = getattr(camera, "read_latest", None)
    timeout_ms_default = env_float("D405_CAMERA_ASYNC_TIMEOUT_MS", 500.0)
    max_age_ms = env_int("D405_CAMERA_MAX_AGE_MS", 1000)
    reuse_last = env_bool("D405_CAMERA_REUSE_LAST_ON_TIMEOUT", True)
    wait_for_new_on_stale = env_bool("D405_CAMERA_WAIT_FOR_NEW_ON_STALE", False)
    warn_every = env_int("D405_CAMERA_REUSE_WARN_EVERY", 30)
    last_good: dict[str, Any] = {"frame": None, "reuse_count": 0}

    def reuse_last_frame(latest_error: Exception, async_error: Exception | None = None) -> Any:
        last_good["reuse_count"] = int(last_good["reuse_count"]) + 1
        count = int(last_good["reuse_count"])
        if warn_every > 0 and (count == 1 or count % warn_every == 0):
            if async_error is None:
                logging.warning(
                    "D405 camera %s reused last good frame without blocking "
                    "(reuse_count=%s, latest_error=%s)",
                    camera_name,
                    count,
                    latest_error,
                )
            else:
                logging.warning(
                    "D405 camera %s reused last good frame after read timeout "
                    "(reuse_count=%s, latest_error=%s, async_error=%s)",
                    camera_name,
                    count,
                    latest_error,
                    async_error,
                )
        return last_good["frame"]

    def async_read_latest(timeout_ms: float | None = None) -> Any:
        timeout_ms_value = timeout_ms_default if timeout_ms is None else float(timeout_ms)

        if callable(read_latest):
            try:
                frame = read_latest(max_age_ms=max_age_ms)
                last_good["frame"] = frame
                last_good["reuse_count"] = 0
                return frame
            except Exception as latest_exc:
                latest_error = latest_exc
            else:
                latest_error = None
        else:
            latest_error = RuntimeError(f"{camera_name} has no read_latest()")

        if reuse_last and last_good["frame"] is not None and not wait_for_new_on_stale:
            return reuse_last_frame(latest_error)

        try:
            frame = original_async_read(timeout_ms=timeout_ms_value)
            last_good["frame"] = frame
            last_good["reuse_count"] = 0
            return frame
        except Exception as async_exc:
            if reuse_last and last_good["frame"] is not None:
                return reuse_last_frame(latest_error, async_exc)
            raise

    camera.async_read = async_read_latest
    camera._kd_tacmae_d405_latest_patch = True
    logging.info(
        "D405 camera %s async_read patched: latest max_age=%sms, fallback timeout=%sms, "
        "reuse_last=%s, wait_for_new_on_stale=%s",
        camera_name,
        max_age_ms,
        timeout_ms_default,
        reuse_last,
        wait_for_new_on_stale,
    )


def patch_d405_shared_camera(robot: Any) -> None:
    if not env_bool("CONNECT_D405_CAMERA", False):
        return
    camera_name = os.environ.get("D405_CAMERA_NAME", "cam_d405_color").strip() or "cam_d405_color"
    shared_cameras = getattr(robot, "shared_cameras", None)
    if not isinstance(shared_cameras, dict):
        logging.warning("D405 camera patch skipped: robot.shared_cameras is not a dict")
        return
    camera = shared_cameras.get(camera_name)
    if camera is None:
        logging.warning("D405 camera patch skipped: shared camera %s not found", camera_name)
        return
    patch_d405_camera_async_read(camera, camera_name)


def cleanup_d405_shared_cameras(robot: Any) -> None:
    """Best-effort D405 cleanup for shutdown paths that skip robot.disconnect()."""
    if not env_bool("CONNECT_D405_CAMERA", False):
        return
    camera_name = os.environ.get("D405_CAMERA_NAME", "cam_d405_color").strip() or "cam_d405_color"
    shared_cameras = getattr(robot, "shared_cameras", None)
    if not isinstance(shared_cameras, dict):
        return
    camera = shared_cameras.get(camera_name)
    if camera is None:
        return
    try:
        thread = getattr(camera, "thread", None)
        is_connected = bool(getattr(camera, "is_connected", False))
        if is_connected or thread is not None:
            camera.disconnect()
            logging.info("D405 shared camera %s disconnected by kd-tacmae cleanup", camera_name)
    except Exception as exc:
        logging.warning("D405 shared camera %s cleanup failed: %s", camera_name, exc)


def cleanup_active_d405_cameras() -> None:
    for robot in list(_ACTIVE_ROBOTS):
        cleanup_d405_shared_cameras(robot)


def _enabled_tactile_key(key: str, *, left_enabled: bool, right_enabled: bool) -> bool:
    left_tokens = ("left_left", "left_right", "tactile_left_left", "tactile_left_right")
    right_tokens = ("right_left", "right_right", "tactile_right_left", "tactile_right_right")
    if any(token in key for token in left_tokens):
        return left_enabled
    if any(token in key for token in right_tokens):
        return right_enabled
    return True


TACTILE_SENSOR_TOKENS: dict[str, tuple[str, ...]] = {
    "left_left": ("left_left", "tactile_left_left"),
    "left_right": ("left_right", "tactile_left_right"),
    "right_left": ("right_left", "tactile_right_left"),
    "right_right": ("right_right", "tactile_right_right"),
}


def _tactile_sensor_for_key(key: str) -> str | None:
    for name, tokens in TACTILE_SENSOR_TOKENS.items():
        if any(token in key for token in tokens):
            return name
    return None


def _has_tactile_sensor(images: dict[str, Any], sensor_name: str) -> bool:
    tokens = TACTILE_SENSOR_TOKENS.get(sensor_name)
    if tokens is None:
        return False
    return any(any(token in key for token in tokens) for key in images)


def _requested_tactile_sensors(*, left_enabled: bool, right_enabled: bool) -> list[str]:
    names: list[str] = []
    if left_enabled:
        names.extend(["left_left", "left_right"])
    if right_enabled:
        names.extend(["right_left", "right_right"])
    return names


def _has_tactile_side(images: dict[str, Any], side: str) -> bool:
    if side == "left":
        tokens = ("left_left", "left_right", "tactile_left_left", "tactile_left_right")
    elif side == "right":
        tokens = ("right_left", "right_right", "tactile_right_left", "tactile_right_right")
    else:
        return False
    return any(any(token in key for token in tokens) for key in images)


class X5TactileReadCache:
    """Preassemble X5 tactile observation arrays off the 30Hz capture loop."""

    def __init__(self, receiver: Any, *, label: str = "X5 tactile") -> None:
        self.receiver = receiver
        self.label = label
        self.original_read_images = receiver.read_images
        self.original_get_last_update_times_perf = getattr(receiver, "get_last_update_times_perf", None)
        self.fps = max(1.0, env_float("X5_TACTILE_ASYNC_CACHE_FPS", 60.0))
        self.warn_ms = max(0.0, env_float("X5_TACTILE_ASYNC_CACHE_WARN_MS", 12.0))
        self.return_copy = env_bool("X5_TACTILE_ASYNC_CACHE_RETURN_COPY", False)
        self.main_thread_warmup = env_bool("X5_TACTILE_ASYNC_CACHE_MAIN_THREAD_WARMUP", False)
        self.history_seconds = max(1.0, env_float("X5_TACTILE_ASYNC_CACHE_HISTORY_SECONDS", 12.0))
        self.mode = env_choice(
            "X5_TACTILE_ASYNC_CACHE_MODE",
            "per_sensor_thread",
            {"full", "staggered_arm", "staggered_sensor", "per_sensor", "per_sensor_thread"},
        )
        self._period_s = 1.0 / self.fps
        self._sensor_order = ["left_left", "left_right", "right_left", "right_right"]
        self._sensor_fps = max(
            1.0,
            env_float("X5_TACTILE_ASYNC_CACHE_SENSOR_FPS", self.fps / float(len(self._sensor_order))),
        )
        self._sensor_period_s = 1.0 / self._sensor_fps
        if self.mode in {"per_sensor", "per_sensor_thread", "staggered_sensor"}:
            history_fps = self._sensor_fps
        elif self.mode == "staggered_arm":
            history_fps = max(1.0, self.fps / 2.0)
        else:
            history_fps = self.fps
        self.history_maxlen_per_key = max(4, int(history_fps * self.history_seconds))
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._sensor_threads: list[threading.Thread] = []
        self._latest_images: dict[str, np.ndarray] = {}
        self._latest_times: dict[str, float] = {}
        self._history: dict[str, deque[tuple[int, float, np.ndarray]]] = {}
        self._history_seq: dict[str, int] = {}
        self._latest_refresh_perf = 0.0
        self._stagger_next_left = True
        self._stagger_sensor_index = 0
        self._last_not_ready_warn_t = 0.0

    def start(self) -> None:
        if self.mode in {"per_sensor", "per_sensor_thread"}:
            if any(thread.is_alive() for thread in self._sensor_threads):
                return
        elif self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        if self.mode in {"per_sensor", "per_sensor_thread"}:
            sensor_names = self._available_sensor_names()
            if sensor_names:
                self._sensor_threads = []
                for sensor_name in sensor_names:
                    thread = threading.Thread(
                        target=self._sensor_loop,
                        args=(sensor_name,),
                        name=f"X5TactileReadCache-{sensor_name}",
                        daemon=True,
                    )
                    self._sensor_threads.append(thread)
                    thread.start()
            else:
                logging.warning(
                    "%s per-sensor async cache requested, but receiver exposes no sensor objects; falling back to full mode",
                    self.label,
                )
                self.mode = "full"
                self._thread = threading.Thread(target=self._loop, name="X5TactileReadCache", daemon=True)
                self._thread.start()
        else:
            self._thread = threading.Thread(target=self._loop, name="X5TactileReadCache", daemon=True)
            self._thread.start()
        logging.info(
            "%s async read cache enabled: fps=%.1f mode=%s sensor_fps=%.1f history_frames_per_key=%s sensors=%s return_copy=%s main_thread_warmup=%s warn_ms=%.1f",
            self.label,
            self.fps,
            self.mode,
            self._sensor_fps,
            self.history_maxlen_per_key,
            self._available_sensor_names() if self.mode in {"per_sensor", "per_sensor_thread"} else [],
            self.return_copy,
            self.main_thread_warmup,
            self.warn_ms,
        )

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        for sensor_thread in list(self._sensor_threads):
            if sensor_thread.is_alive():
                sensor_thread.join(timeout=0.5)
        self._sensor_threads = []

    def _available_sensor_names(self) -> list[str]:
        sensors = getattr(self.receiver, "_sensors", None)
        active_names = set(getattr(self.receiver, "_active_names", []) or [])
        if not isinstance(sensors, dict):
            return []
        return [
            name
            for name in self._sensor_order
            if name in sensors and (not active_names or name in active_names)
        ]

    def _refresh_once(
        self,
        *,
        left_enabled: bool | None = None,
        right_enabled: bool | None = None,
        label: str | None = None,
        sensor_names: list[str] | None = None,
    ) -> None:
        with self._refresh_lock:
            if left_enabled is None or right_enabled is None:
                if self.mode == "staggered_sensor":
                    sensor_name = self._sensor_order[self._stagger_sensor_index % len(self._sensor_order)]
                    self._stagger_sensor_index += 1
                    sensor_names = [sensor_name]
                    left_enabled = sensor_name.startswith("left_")
                    right_enabled = sensor_name.startswith("right_")
                    label = sensor_name
                elif self.mode == "staggered_arm":
                    left_enabled = self._stagger_next_left
                    right_enabled = not self._stagger_next_left
                    self._stagger_next_left = not self._stagger_next_left
                    label = "left" if left_enabled else "right"
                else:
                    left_enabled = True
                    right_enabled = True
                    label = "full"
            label = label or (
                "left" if left_enabled and not right_enabled else "right" if right_enabled and not left_enabled else "full"
            )
            t0 = time.perf_counter()
            images = self._read_images_for_refresh(
                left_enabled=bool(left_enabled),
                right_enabled=bool(right_enabled),
                sensor_names=sensor_names,
            )
            if callable(self.original_get_last_update_times_perf):
                times = self.original_get_last_update_times_perf()
            else:
                times = {}
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if self.warn_ms > 0.0 and dt_ms >= self.warn_ms:
                logging.warning("%s async cache refresh %s %.1fms", self.label, label, dt_ms)
            refresh_t = time.perf_counter()
            self._publish_images(
                images,
                refresh_t=refresh_t,
                times=times,
                replace=bool(left_enabled) and bool(right_enabled) and not sensor_names,
                left_enabled=True,
                right_enabled=True,
            )

    def _read_images_for_refresh(
        self,
        *,
        left_enabled: bool,
        right_enabled: bool,
        sensor_names: list[str] | None,
    ) -> dict[str, np.ndarray]:
        if sensor_names:
            sensors = getattr(self.receiver, "_sensors", None)
            active_names = set(getattr(self.receiver, "_active_names", []) or [])
            if isinstance(sensors, dict):
                images: dict[str, np.ndarray] = {}
                for name in sensor_names:
                    sensor = sensors.get(name)
                    if sensor is None or (active_names and name not in active_names):
                        continue
                    images.update(sensor.read_images())
                if images:
                    return images
        return self.original_read_images(left_enabled=left_enabled, right_enabled=right_enabled)

    def _publish_images(
        self,
        images: dict[str, np.ndarray],
        *,
        refresh_t: float,
        times: dict[str, float] | None,
        replace: bool,
        left_enabled: bool,
        right_enabled: bool,
    ) -> None:
        with self._lock:
            if replace:
                self._latest_images = dict(images)
            else:
                self._latest_images.update(dict(images))
            for key, image in images.items():
                if not isinstance(image, np.ndarray):
                    continue
                seq = int(self._history_seq.get(key, 0)) + 1
                self._history_seq[key] = seq
                history = self._history.get(key)
                if history is None:
                    history = deque(maxlen=self.history_maxlen_per_key)
                    self._history[key] = history
                history.append((seq, refresh_t, image))
            if times:
                self._latest_times.update(dict(times))
            else:
                for key, image in images.items():
                    if isinstance(image, np.ndarray):
                        self._latest_times[key] = refresh_t
            self._latest_refresh_perf = refresh_t
            if self._has_all_requested_locked(left_enabled=left_enabled, right_enabled=right_enabled):
                self._ready.set()

    def _has_all_requested_locked(self, *, left_enabled: bool, right_enabled: bool) -> bool:
        return all(
            _has_tactile_sensor(self._latest_images, name)
            for name in _requested_tactile_sensors(left_enabled=left_enabled, right_enabled=right_enabled)
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            loop_t = time.perf_counter()
            try:
                self._refresh_once()
            except Exception as exc:
                logging.debug("X5 tactile async cache refresh failed: %s", exc)
            elapsed = time.perf_counter() - loop_t
            self._stop.wait(max(0.0, self._period_s - elapsed))

    def _sensor_loop(self, sensor_name: str) -> None:
        left_enabled = sensor_name.startswith("left_")
        right_enabled = sensor_name.startswith("right_")
        while not self._stop.is_set():
            loop_t = time.perf_counter()
            try:
                t0 = time.perf_counter()
                images = self._read_images_for_refresh(
                    left_enabled=left_enabled,
                    right_enabled=right_enabled,
                    sensor_names=[sensor_name],
                )
                refresh_t = time.perf_counter()
                dt_ms = (refresh_t - t0) * 1000.0
                if self.warn_ms > 0.0 and dt_ms >= self.warn_ms:
                    logging.warning("%s async cache refresh %s %.1fms", self.label, sensor_name, dt_ms)
                if images:
                    self._publish_images(
                        images,
                        refresh_t=refresh_t,
                        times=None,
                        replace=False,
                        left_enabled=True,
                        right_enabled=True,
                    )
            except Exception as exc:
                logging.debug("X5 tactile async cache refresh %s failed: %s", sensor_name, exc)
            elapsed = time.perf_counter() - loop_t
            self._stop.wait(max(0.0, self._sensor_period_s - elapsed))

    @staticmethod
    def _copy_images(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in images.items()}

    def read_images(self, *, left_enabled: bool = True, right_enabled: bool = True) -> dict[str, np.ndarray]:
        self.start()
        with self._lock:
            has_requested_sensors = self._has_all_requested_locked(
                left_enabled=left_enabled,
                right_enabled=right_enabled,
            )
        needs_refresh = not self._ready.is_set() or not has_requested_sensors
        in_capture_loop = get_capture_target_time_perf() > 0.0
        if (
            needs_refresh
            and self.mode not in {"per_sensor", "per_sensor_thread"}
            and (self.main_thread_warmup or not in_capture_loop)
        ):
            self._refresh_once(left_enabled=True, right_enabled=True, label="warmup")
        elif needs_refresh and self.warn_ms > 0:
            now = time.perf_counter()
            if now - self._last_not_ready_warn_t >= 1.0:
                with self._lock:
                    cached_keys = sorted(self._latest_images)
                logging.warning(
                    "X5 tactile async cache not ready in capture loop; returning cached keys=%s",
                    cached_keys,
                )
                self._last_not_ready_warn_t = now
        with self._lock:
            images = {
                key: value
                for key, value in self._latest_images.items()
                if _enabled_tactile_key(key, left_enabled=left_enabled, right_enabled=right_enabled)
            }
        if self.return_copy:
            return self._copy_images(images)
        return dict(images)

    def get_last_update_times_perf(self) -> dict[str, float]:
        self.start()
        with self._lock:
            if self._latest_times:
                return dict(self._latest_times)
        if callable(self.original_get_last_update_times_perf):
            return dict(self.original_get_last_update_times_perf())
        return {}

    def startup_snapshot(self) -> dict[str, Any]:
        self.start()
        with self._lock:
            return {
                "keys": sorted(self._latest_images),
                "ready": self._ready.is_set(),
                "latest_refresh_perf": self._latest_refresh_perf,
                "has_all": self._has_all_requested_locked(left_enabled=True, right_enabled=True),
            }

    def get_frames_after(
        self,
        last_seq_by_key: dict[str, int],
        *,
        max_frames: int = 512,
    ) -> list[tuple[str, int, float, np.ndarray]]:
        self.start()
        frames: list[tuple[str, int, float, np.ndarray]] = []
        with self._lock:
            for key, history in self._history.items():
                last_seq = int(last_seq_by_key.get(key, 0))
                for seq, source_t, image in history:
                    if seq > last_seq:
                        frames.append((key, int(seq), float(source_t), image))
                        if len(frames) >= max_frames:
                            break
                if len(frames) >= max_frames:
                    break
        frames.sort(key=lambda item: (item[2], item[1], item[0]))
        return frames


def patch_x5_tactile_async_cache(robot: Any) -> None:
    if not env_bool("X5_TACTILE_ASYNC_CACHE", False):
        return
    receiver = getattr(robot, "x5_tactile", None)
    label = "X5 tactile"
    if receiver is None:
        receiver = getattr(robot, "tactile_sidecar", None)
        label = "dm tactile sidecar"
    if receiver is None or getattr(receiver, "_kd_tacmae_async_cache_patch", False):
        return
    if not hasattr(receiver, "read_images"):
        logging.warning("%s async cache skipped: receiver has no read_images", label)
        return
    cache = X5TactileReadCache(receiver, label=label)
    original_connect = getattr(receiver, "connect", None)
    original_disconnect = getattr(receiver, "disconnect", None)

    if callable(original_connect):
        def connect_with_async_cache(*args: Any, **kwargs: Any) -> Any:
            result = original_connect(*args, **kwargs)
            cache.start()
            return result

        receiver.connect = connect_with_async_cache

    if callable(original_disconnect):
        def disconnect_with_async_cache(*args: Any, **kwargs: Any) -> Any:
            cache.close()
            return original_disconnect(*args, **kwargs)

        receiver.disconnect = disconnect_with_async_cache

    receiver.read_images = cache.read_images
    if hasattr(receiver, "get_last_update_times_perf"):
        receiver.get_last_update_times_perf = cache.get_last_update_times_perf
    receiver._kd_tacmae_async_cache_patch = True
    receiver._kd_tacmae_async_cache = cache
    _TACTILE_READ_CACHES.append(cache)


def cleanup_tactile_read_caches() -> None:
    for cache in list(_TACTILE_READ_CACHES):
        try:
            cache.close()
        except Exception as exc:
            logging.debug("X5 tactile async cache cleanup failed: %s", exc)


class WristUndistortCenterCropper:
    """Fisheye undistort full-resolution wrist RGB frames, then crop around the new camera center."""

    def __init__(
        self,
        *,
        side: str,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        image_size: tuple[int, int],
        calibration_source: str,
        balance: float,
        crop_size: int,
    ) -> None:
        self.side = side
        self.calibration_source = calibration_source
        self.balance = float(balance)
        self.crop_size = int(crop_size)
        if self.crop_size <= 0:
            raise ValueError(f"WRIST_UNDISTORT_CROP_SIZE must be positive, got {self.crop_size}")

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.calib_image_size = image_size
        self._maps_by_size: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int, int]] = {}

    @staticmethod
    def _coerce_calibration(
        data: dict[str, Any],
        *,
        source: str,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        camera_matrix = np.asarray(
            data.get("camera_matrix", data.get("K")),
            dtype=np.float64,
        )
        if camera_matrix.shape == (9,):
            camera_matrix = camera_matrix.reshape(3, 3)
        dist_coeffs = np.asarray(
            data.get("dist_coeffs", data.get("D")),
            dtype=np.float64,
        ).reshape(-1, 1)
        image_size_raw = data.get("image_size", data.get("resolution"))

        if camera_matrix.shape != (3, 3):
            raise ValueError(f"{source}: camera_matrix/K must be 3x3, got {camera_matrix.shape}")
        if dist_coeffs.size < 4:
            raise ValueError(f"{source}: dist_coeffs/D must contain at least 4 fisheye coefficients")
        if image_size_raw is None or len(image_size_raw) != 2:
            raise ValueError(f"{source}: image_size/resolution must be [width, height]")

        image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError(f"{source}: invalid image_size {image_size}")

        return camera_matrix, dist_coeffs[:4].reshape(4, 1), image_size

    @classmethod
    def load_calibration_from_file(cls, calib_file: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        path = Path(calib_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"wrist undistort calibration file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._coerce_calibration(data, source=str(path))

    def _scaled_intrinsics(self, actual_size: tuple[int, int]) -> np.ndarray:
        actual_w, actual_h = actual_size
        calib_w, calib_h = self.calib_image_size
        if (actual_w, actual_h) == (calib_w, calib_h):
            return self.camera_matrix.copy()

        scale_x = actual_w / calib_w
        scale_y = actual_h / calib_h
        k = self.camera_matrix.copy()
        k[0, 0] *= scale_x
        k[0, 2] *= scale_x
        k[1, 1] *= scale_y
        k[1, 2] *= scale_y
        return k

    @staticmethod
    def _crop_origin_from_center(center_x: float, center_y: float, width: int, height: int, crop: int) -> tuple[int, int]:
        x0 = int(round(center_x - crop / 2))
        y0 = int(round(center_y - crop / 2))
        x0 = min(max(0, x0), width - crop)
        y0 = min(max(0, y0), height - crop)
        return x0, y0

    def _maps_for_size(self, actual_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, int, int]:
        maps = self._maps_by_size.get(actual_size)
        if maps is not None:
            return maps

        actual_w, actual_h = actual_size
        if self.crop_size > min(actual_w, actual_h):
            raise ValueError(
                f"WRIST_UNDISTORT_CROP_SIZE={self.crop_size} is larger than wrist frame "
                f"{actual_w}x{actual_h}"
            )

        k = self._scaled_intrinsics(actual_size)
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            k,
            self.dist_coeffs,
            actual_size,
            np.eye(3),
            balance=self.balance,
            new_size=actual_size,
        )
        maps = cv2.fisheye.initUndistortRectifyMap(
            k,
            self.dist_coeffs,
            np.eye(3),
            new_k,
            actual_size,
            cv2.CV_16SC2,
        )
        x0, y0 = self._crop_origin_from_center(
            float(new_k[0, 2]),
            float(new_k[1, 2]),
            actual_w,
            actual_h,
            self.crop_size,
        )
        crop = self.crop_size
        # Only remap the final crop ROI. Remapping the full 1920x1080 frame and
        # cropping afterwards is much more expensive and can push 30 Hz capture
        # over budget when tactile streams also spike.
        roi_map1 = np.ascontiguousarray(maps[0][y0 : y0 + crop, x0 : x0 + crop])
        roi_map2 = np.ascontiguousarray(maps[1][y0 : y0 + crop, x0 : x0 + crop])
        maps_and_crop = (roi_map1, roi_map2, x0, y0)
        self._maps_by_size[actual_size] = maps_and_crop
        logging.info(
            "%s wrist undistort maps ready: calib=%s calib_size=%sx%s frame=%sx%s "
            "balance=%.3f new_principal=(%.1f,%.1f) crop=%sx%s crop_origin=(%d,%d) roi_remap=true",
            self.side,
            self.calibration_source,
            self.calib_image_size[0],
            self.calib_image_size[1],
            actual_w,
            actual_h,
            self.balance,
            float(new_k[0, 2]),
            float(new_k[1, 2]),
            self.crop_size,
            self.crop_size,
            x0,
            y0,
        )
        return maps_and_crop

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.ndim < 2:
            return frame

        h, w = frame.shape[:2]
        map1, map2, _x0, _y0 = self._maps_for_size((w, h))
        cropped = cv2.remap(
            frame,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        return np.ascontiguousarray(cropped)


class WristProcessedFrameCache:
    """Background wrist frame undistort/crop cache.

    The fish-camera receiver decodes full-resolution BGR frames in its own
    thread. This cache consumes those raw frames, performs the expensive wrist
    crop/remap outside the 30 Hz capture loop, and exposes the latest processed
    BGR frame through a receiver-compatible async_read().
    """

    def __init__(
        self,
        *,
        receiver: Any,
        cropper: WristUndistortCenterCropper,
        label: str,
    ) -> None:
        self.receiver = receiver
        self.cropper = cropper
        self.label = label
        self.original_async_read = receiver.async_read
        self.original_get_latest_frame_time_perf = getattr(receiver, "get_latest_frame_time_perf", None)
        self.return_copy = env_bool("WRIST_PROCESSED_ASYNC_CACHE_RETURN_COPY", False)
        self.raw_no_copy = env_bool("WRIST_PROCESSED_ASYNC_CACHE_RAW_NO_COPY", True)
        self.nonblocking = env_bool("WRIST_PROCESSED_ASYNC_CACHE_NONBLOCKING", True)
        self.align_to_capture_time = env_bool("WRIST_PROCESSED_ASYNC_CACHE_ALIGN_TO_CAPTURE_TIME", False)
        self.align_offset_s = env_float("WRIST_PROCESSED_ASYNC_CACHE_ALIGN_OFFSET_S", 0.0)
        self.align_max_age_ms = env_float("WRIST_PROCESSED_ASYNC_CACHE_ALIGN_MAX_AGE_MS", 250.0)
        self.align_buffer_frames = max(1, env_int("WRIST_PROCESSED_ASYNC_CACHE_ALIGN_BUFFER_FRAMES", 45))
        self.warn_age_ms = env_float("WRIST_PROCESSED_ASYNC_CACHE_WARN_AGE_MS", 120.0)
        self.warn_ms = env_float("WRIST_PROCESSED_ASYNC_CACHE_WARN_MS", 8.0)
        self.read_timeout_s = max(0.001, env_float("WRIST_PROCESSED_ASYNC_CACHE_READ_TIMEOUT_S", 0.2))
        self.workers = max(1, env_int("WRIST_PROCESSED_ASYNC_CACHE_WORKERS", 1))
        self.stats_every_s = max(0.0, env_float("WRIST_PROCESSED_ASYNC_CACHE_STATS_EVERY_S", 1.0))
        self._raw_last_read_id = 0
        self._raw_read_lock = threading.Lock()
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest_frame: np.ndarray | None = None
        self._latest_source_time_perf = 0.0
        self._latest_process_time_perf = 0.0
        self._last_returned_source_time_perf = 0.0
        self._history: deque[tuple[float, int, np.ndarray]] = deque(maxlen=self.align_buffer_frames)
        self._frame_id = 0
        self._last_read_id = 0
        self._last_warn_t = 0.0
        self._last_age_warn_t = 0.0
        self._last_stats_t = time.perf_counter()
        self._processed_count = 0
        self._dropped_out_of_order = 0
        self._raw_gap_count = 0
        self._last_processed_raw_id = 0
        self._threads = [
            threading.Thread(
                target=lambda idx=idx: self._loop(idx),
                name=f"kd_tacmae_wrist_cache_{label}_{idx}",
                daemon=True,
            )
            for idx in range(self.workers)
        ]
        for thread in self._threads:
            thread.start()

    def _read_raw_latest_no_copy(
        self,
        timeout_s: float,
        *,
        require_new: bool,
    ) -> tuple[np.ndarray, float, int] | None:
        frame_lock = getattr(self.receiver, "_frame_lock", None)
        frame_ready = getattr(self.receiver, "_frame_ready", None)
        if frame_lock is None or frame_ready is None:
            return None

        deadline = time.perf_counter() + max(0.0, timeout_s)
        while not self._stop.is_set():
            with frame_lock:
                frame = getattr(self.receiver, "_latest_frame", None)
                frame_id = int(getattr(self.receiver, "_frame_id", 0))
                source_time = float(getattr(self.receiver, "_latest_frame_time_perf", 0.0) or 0.0)
                if frame is not None and (not require_new or frame_id != self._raw_last_read_id):
                    self._raw_last_read_id = frame_id
                    return frame, source_time, frame_id

            is_connected = getattr(self.receiver, "is_connected", True)
            if not is_connected:
                return None

            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return None
            with frame_ready:
                frame_ready.wait(timeout=remaining)
        return None

    def _read_raw_frame(self) -> tuple[np.ndarray, float, int] | None:
        with self._raw_read_lock:
            if self.raw_no_copy:
                try:
                    frame = self._read_raw_latest_no_copy(self.read_timeout_s, require_new=True)
                    if frame is not None:
                        return frame
                except Exception as exc:
                    logging.debug("Wrist processed cache %s raw no-copy read failed: %s", self.label, exc)

            try:
                frame = self.original_async_read(timeout_s=self.read_timeout_s, require_new=True)
            except TypeError:
                try:
                    frame = self.original_async_read(require_new=True)
                except TypeError:
                    frame = self.original_async_read()
            if frame is None:
                return None
            source_time = self._source_time_perf() or time.perf_counter()
            raw_frame_id = int(getattr(self.receiver, "_frame_id", 0) or 0)
            return frame, source_time, raw_frame_id

    def _source_time_perf(self) -> float:
        if callable(self.original_get_latest_frame_time_perf):
            try:
                return float(self.original_get_latest_frame_time_perf() or 0.0)
            except Exception:
                return 0.0
        return 0.0

    def _select_aligned_frame_locked(self, target_time_perf: float) -> tuple[float, int, np.ndarray] | None:
        if not self._history:
            return None

        selected = self._history[0]
        for item in self._history:
            source_time, _frame_id, _frame = item
            if source_time <= target_time_perf:
                selected = item
            else:
                break

        source_time, frame_id, _frame = selected
        age_ms = (target_time_perf - source_time) * 1000.0
        now = time.perf_counter()
        if self.align_max_age_ms > 0 and age_ms >= self.align_max_age_ms and now - self._last_age_warn_t >= 1.0:
            logging.warning(
                "Wrist processed cache %s aligned stale frame target_age=%.1fms frame_id=%s latest_id=%s",
                self.label,
                age_ms,
                frame_id,
                self._frame_id,
            )
            self._last_age_warn_t = now
        return selected

    def _publish_processed_frame(self, processed_bgr: np.ndarray, source_time: float, raw_frame_id: int) -> None:
        if source_time <= 0:
            source_time = time.perf_counter()
        with self._condition:
            if source_time <= self._latest_source_time_perf:
                self._dropped_out_of_order += 1
                return
            if self._last_processed_raw_id > 0 and raw_frame_id > self._last_processed_raw_id + 1:
                self._raw_gap_count += raw_frame_id - self._last_processed_raw_id - 1
            self._last_processed_raw_id = max(self._last_processed_raw_id, raw_frame_id)
            self._latest_frame = np.ascontiguousarray(processed_bgr)
            self._latest_source_time_perf = source_time
            self._latest_process_time_perf = time.perf_counter()
            self._frame_id += 1
            self._processed_count += 1
            self._history.append((source_time, self._frame_id, self._latest_frame))
            self._condition.notify_all()

        now = time.perf_counter()
        if self.stats_every_s > 0 and now - self._last_stats_t >= self.stats_every_s:
            elapsed = max(1e-6, now - self._last_stats_t)
            processed_fps = self._processed_count / elapsed
            logging.info(
                "Wrist processed cache %s stats processed_fps=%.1f published_id=%s raw_id=%s raw_gap=%s out_of_order_drop=%s workers=%s",
                self.label,
                processed_fps,
                self._frame_id,
                raw_frame_id,
                self._raw_gap_count,
                self._dropped_out_of_order,
                self.workers,
            )
            self._processed_count = 0
            self._raw_gap_count = 0
            self._dropped_out_of_order = 0
            self._last_stats_t = now

    def _maybe_return_aligned_frame_locked(self) -> np.ndarray | None:
        if not self.align_to_capture_time:
            return None
        target_time_perf = get_capture_target_time_perf()
        if target_time_perf <= 0.0:
            return None

        selected = self._select_aligned_frame_locked(target_time_perf + self.align_offset_s)
        if selected is None:
            return None

        _source_time, frame_id, frame = selected
        self._last_read_id = frame_id
        self._last_returned_source_time_perf = _source_time
        return frame.copy() if self.return_copy else frame

    def _loop(self, worker_idx: int) -> None:
        while not self._stop.is_set():
            try:
                raw = self._read_raw_frame()
            except Exception as exc:
                if not self._stop.is_set():
                    logging.warning("Wrist processed cache %s raw read failed: %s", self.label, exc)
                time.sleep(0.002)
                continue

            if raw is None:
                continue
            frame_bgr, source_time, raw_frame_id = raw

            t0 = time.perf_counter()
            try:
                processed_bgr = self.cropper(frame_bgr)
            except Exception as exc:
                logging.warning("Wrist processed cache %s crop failed: %s", self.label, exc)
                continue
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if dt_ms >= self.warn_ms:
                now = time.perf_counter()
                if now - self._last_warn_t >= 0.25:
                    logging.warning(
                        "Wrist processed cache %s worker=%s crop %.1fms raw_id=%s",
                        self.label,
                        worker_idx,
                        dt_ms,
                        raw_frame_id,
                    )
                    self._last_warn_t = now

            self._publish_processed_frame(processed_bgr, source_time, raw_frame_id)

    def async_read(self, timeout_s: float | None = None, *, require_new: bool = False) -> np.ndarray | None:
        timeout = self.read_timeout_s if timeout_s is None else max(0.0, float(timeout_s))
        deadline = time.perf_counter() + timeout
        while True:
            with self._condition:
                if self._latest_frame is not None:
                    aligned_frame = self._maybe_return_aligned_frame_locked()
                    if aligned_frame is not None:
                        return aligned_frame
                    if self.nonblocking:
                        now = time.perf_counter()
                        age_ms = (now - self._latest_process_time_perf) * 1000.0 if self._latest_process_time_perf > 0 else 0.0
                        if self.warn_age_ms > 0 and age_ms >= self.warn_age_ms and now - self._last_age_warn_t >= 1.0:
                            logging.warning(
                                "Wrist processed cache %s returned stale latest frame age=%.1fms frame_id=%s",
                                self.label,
                                age_ms,
                                self._frame_id,
                            )
                            self._last_age_warn_t = now
                        self._last_read_id = self._frame_id
                        self._last_returned_source_time_perf = self._latest_source_time_perf
                        return self._latest_frame.copy() if self.return_copy else self._latest_frame
                    if not require_new or self._frame_id != self._last_read_id:
                        self._last_read_id = self._frame_id
                        self._last_returned_source_time_perf = self._latest_source_time_perf
                        return self._latest_frame.copy() if self.return_copy else self._latest_frame
                if self._stop.is_set():
                    return None
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)

    def get_latest_frame_time_perf(self) -> float:
        with self._condition:
            return self._last_returned_source_time_perf or self._latest_source_time_perf

    def get_processed_frames_after(self, frame_id: int) -> list[tuple[float, int, np.ndarray]]:
        with self._condition:
            return [
                (source_time, published_id, frame.copy() if self.return_copy else frame)
                for source_time, published_id, frame in self._history
                if published_id > frame_id
            ]

    def startup_snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "label": self.label,
                "frame_id": self._frame_id,
                "latest_source_time_perf": self._latest_source_time_perf,
                "latest_process_time_perf": self._latest_process_time_perf,
                "has_frame": self._latest_frame is not None,
            }

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=1.0)


@dataclass(frozen=True)
class WristRawFrame:
    frame_id: int
    source_time_perf: float
    frame_bgr: np.ndarray


@dataclass(frozen=True)
class WristProcessedFrame:
    frame_id: int
    source_time_perf: float
    frame_bgr: np.ndarray


class WristQueuedPipeline:
    """Queued wrist undistort/crop pipeline that processes every decoded frame."""

    def __init__(
        self,
        *,
        receiver: Any,
        cropper: WristUndistortCenterCropper,
        label: str,
    ) -> None:
        self.receiver = receiver
        self.cropper = cropper
        self.label = label
        self.original_async_read = receiver.async_read
        self.original_get_latest_frame_time_perf = getattr(receiver, "get_latest_frame_time_perf", None)
        self.return_copy = env_bool("WRIST_PROCESSED_ASYNC_CACHE_RETURN_COPY", False)
        self.workers = max(1, env_int("WRIST_UNDISTORT_WORKERS_PER_CAMERA", env_int("WRIST_PROCESSED_ASYNC_CACHE_WORKERS", 4)))
        self.raw_queue_frames = max(1, env_int("WRIST_RAW_QUEUE_FRAMES", 120))
        self.processed_queue_frames = max(1, env_int("WRIST_PROCESSED_QUEUE_FRAMES", 120))
        self.overflow_action = env_choice("WRIST_PIPELINE_OVERFLOW_ACTION", "abort", {"abort", "drop_oldest"})
        self.timestamp_mode = env_choice("WRIST_TIMESTAMP_MODE", "frame_clock", {"arrival", "frame_clock"})
        receiver_fps = float(getattr(receiver, "fps", 0.0) or 0.0)
        self.frame_clock_fps = max(1e-6, env_float("WRIST_FRAME_CLOCK_FPS", receiver_fps if receiver_fps > 0 else 30.0))
        side = "LEFT" if "left" in label.lower() else "RIGHT" if "right" in label.lower() else ""
        default_latency_frames = env_float("WRIST_SOURCE_LATENCY_FRAMES", 0.0)
        default_latency_s = env_float("WRIST_SOURCE_LATENCY_S", 0.0)
        if side:
            latency_frames = env_float(f"WRIST_{side}_SOURCE_LATENCY_FRAMES", default_latency_frames)
            latency_s = env_float(f"WRIST_{side}_SOURCE_LATENCY_S", default_latency_s)
        else:
            latency_frames = default_latency_frames
            latency_s = default_latency_s
        self.source_latency_s = max(0.0, latency_s + latency_frames / self.frame_clock_fps)
        self._frame_clock_base_raw_id: int | None = None
        self._frame_clock_base_time_perf = 0.0
        self.warn_ms = env_float("WRIST_PROCESSED_ASYNC_CACHE_WARN_MS", 8.0)
        self.warn_age_ms = env_float("WRIST_PROCESSED_ASYNC_CACHE_WARN_AGE_MS", 120.0)
        self.read_timeout_s = max(0.001, env_float("WRIST_PROCESSED_ASYNC_CACHE_READ_TIMEOUT_S", 0.2))
        self.stats_every_s = max(0.0, env_float("WRIST_PROCESSED_ASYNC_CACHE_STATS_EVERY_S", 1.0))
        self._raw_queue: queue.Queue[WristRawFrame | None] = queue.Queue(maxsize=self.raw_queue_frames)
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest_frame: np.ndarray | None = None
        self._latest_source_time_perf = 0.0
        self._latest_process_time_perf = 0.0
        self._last_returned_source_time_perf = 0.0
        self._history: deque[tuple[float, int, np.ndarray]] = deque(maxlen=self.processed_queue_frames)
        self._published_frame_id = 0
        self._last_read_id = 0
        self._pending_processed: dict[int, WristProcessedFrame] = {}
        self._processed_callback_lock = threading.Lock()
        self._processed_callbacks: list[Any] = []
        self._next_publish_raw_id: int | None = None
        self._last_callback_raw_id = 0
        self._raw_gap_count = 0
        self._raw_count = 0
        self._processed_count = 0
        self._dropped_count = 0
        self._last_warn_t = 0.0
        self._last_age_warn_t = 0.0
        self._last_stats_t = time.perf_counter()
        self._error: BaseException | None = None
        self._callback_registered = False
        add_callback = getattr(receiver, "add_frame_callback", None)
        if not callable(add_callback):
            raise RuntimeError(
                f"{label}: WRIST_PIPELINE_MODE=queued requires FishCameraReceiver.add_frame_callback; "
                "restart after updating fish_camera_receiver.py"
            )
        add_callback(self._on_raw_frame)
        self._callback_registered = True
        self._threads = [
            threading.Thread(
                target=lambda idx=idx: self._worker_loop(idx),
                name=f"kd_tacmae_wrist_queued_{label}_{idx}",
                daemon=True,
            )
            for idx in range(self.workers)
        ]
        for thread in self._threads:
            thread.start()

    def _set_error(self, exc: BaseException) -> None:
        with self._condition:
            if self._error is None:
                self._error = exc
                logging.error("Wrist queued pipeline %s failed: %s", self.label, exc)
            self._condition.notify_all()

    def _raise_if_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Wrist queued pipeline {self.label} failed: {self._error}") from self._error

    def _on_raw_frame(self, frame_bgr: np.ndarray, frame_id: int, frame_time_perf: float) -> None:
        if self._stop.is_set():
            return
        if frame_time_perf <= 0:
            frame_time_perf = time.perf_counter()
        frame_id = int(frame_id)
        source_time_perf = float(frame_time_perf)
        if self.timestamp_mode == "frame_clock":
            if self._frame_clock_base_raw_id is None:
                self._frame_clock_base_raw_id = frame_id
                self._frame_clock_base_time_perf = source_time_perf
                logging.info(
                    "Wrist queued pipeline %s timestamp_mode=frame_clock fps=%.3f source_latency_s=%.3f base_raw_id=%s",
                    self.label,
                    self.frame_clock_fps,
                    self.source_latency_s,
                    frame_id,
                )
            source_time_perf = self._frame_clock_base_time_perf + (
                (frame_id - self._frame_clock_base_raw_id) / self.frame_clock_fps
            )
        if self.source_latency_s > 0.0:
            source_time_perf -= self.source_latency_s
        if self._last_callback_raw_id > 0 and frame_id > self._last_callback_raw_id + 1:
            self._raw_gap_count += frame_id - self._last_callback_raw_id - 1
        self._last_callback_raw_id = max(self._last_callback_raw_id, frame_id)
        raw = WristRawFrame(frame_id=frame_id, source_time_perf=source_time_perf, frame_bgr=frame_bgr)
        try:
            self._raw_queue.put_nowait(raw)
            self._raw_count += 1
        except queue.Full:
            if self.overflow_action == "drop_oldest":
                try:
                    dropped = self._raw_queue.get_nowait()
                    if dropped is not None:
                        self._raw_queue.task_done()
                    self._dropped_count += 1
                    self._raw_queue.put_nowait(raw)
                    self._raw_count += 1
                    return
                except queue.Empty:
                    pass
            self._set_error(
                RuntimeError(
                    f"raw queue overflow for {self.label}: size={self._raw_queue.qsize()} "
                    f"max={self.raw_queue_frames} latest_raw_id={frame_id}"
                )
            )

    def add_processed_callback(self, callback: Any) -> None:
        with self._processed_callback_lock:
            if callback not in self._processed_callbacks:
                self._processed_callbacks.append(callback)

    def remove_processed_callback(self, callback: Any) -> None:
        with self._processed_callback_lock:
            self._processed_callbacks = [item for item in self._processed_callbacks if item is not callback]

    def _emit_processed_callbacks(self, published: list[tuple[float, int, np.ndarray]]) -> None:
        if not published:
            return
        with self._processed_callback_lock:
            callbacks = tuple(self._processed_callbacks)
        if not callbacks:
            return
        for source_time, published_id, frame in published:
            for callback in callbacks:
                try:
                    callback(source_time, published_id, frame)
                except Exception as exc:
                    logging.warning("Wrist queued pipeline %s processed callback failed: %s", self.label, exc)

    def _publish_ready_locked(self) -> list[tuple[float, int, np.ndarray]]:
        published: list[tuple[float, int, np.ndarray]] = []
        if not self._pending_processed:
            return published
        if self._next_publish_raw_id is None:
            self._next_publish_raw_id = min(self._pending_processed)
        while self._next_publish_raw_id in self._pending_processed:
            processed = self._pending_processed.pop(self._next_publish_raw_id)
            self._latest_frame = np.ascontiguousarray(processed.frame_bgr)
            self._latest_source_time_perf = processed.source_time_perf
            self._latest_process_time_perf = time.perf_counter()
            self._published_frame_id += 1
            self._processed_count += 1
            self._history.append((processed.source_time_perf, self._published_frame_id, self._latest_frame))
            published.append((processed.source_time_perf, self._published_frame_id, self._latest_frame))
            self._next_publish_raw_id += 1
            self._condition.notify_all()
        return published

    def _log_stats(self) -> None:
        now = time.perf_counter()
        if self.stats_every_s <= 0 or now - self._last_stats_t < self.stats_every_s:
            return
        elapsed = max(1e-6, now - self._last_stats_t)
        logging.info(
            "Wrist queued pipeline %s stats raw_fps=%.1f processed_fps=%.1f raw_q=%s pending=%s raw_gap=%s dropped=%s published_id=%s raw_id=%s workers=%s",
            self.label,
            self._raw_count / elapsed,
            self._processed_count / elapsed,
            self._raw_queue.qsize(),
            len(self._pending_processed),
            self._raw_gap_count,
            self._dropped_count,
            self._published_frame_id,
            self._last_callback_raw_id,
            self.workers,
        )
        self._raw_count = 0
        self._processed_count = 0
        self._raw_gap_count = 0
        self._dropped_count = 0
        self._last_stats_t = now

    def _worker_loop(self, worker_idx: int) -> None:
        while not self._stop.is_set():
            with self._condition:
                if self._error is not None:
                    return
            try:
                raw = self._raw_queue.get(timeout=0.1)
            except queue.Empty:
                self._log_stats()
                continue
            if raw is None:
                self._raw_queue.task_done()
                return

            t0 = time.perf_counter()
            try:
                processed_bgr = self.cropper(raw.frame_bgr)
            except Exception as exc:
                self._raw_queue.task_done()
                self._set_error(RuntimeError(f"crop failed for raw_id={raw.frame_id}: {exc}"))
                return
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if dt_ms >= self.warn_ms:
                now = time.perf_counter()
                if now - self._last_warn_t >= 0.25:
                    logging.warning(
                        "Wrist queued pipeline %s worker=%s crop %.1fms raw_id=%s",
                        self.label,
                        worker_idx,
                        dt_ms,
                        raw.frame_id,
                    )
                    self._last_warn_t = now

            with self._condition:
                self._pending_processed[raw.frame_id] = WristProcessedFrame(
                    frame_id=raw.frame_id,
                    source_time_perf=raw.source_time_perf,
                    frame_bgr=processed_bgr,
                )
                published = self._publish_ready_locked()
            self._raw_queue.task_done()
            self._emit_processed_callbacks(published)
            self._log_stats()

    def async_read(self, timeout_s: float | None = None, *, require_new: bool = False) -> np.ndarray | None:
        timeout = self.read_timeout_s if timeout_s is None else max(0.0, float(timeout_s))
        deadline = time.perf_counter() + timeout
        while True:
            with self._condition:
                self._raise_if_error_locked()
                if self._latest_frame is not None and (not require_new or self._published_frame_id != self._last_read_id):
                    now = time.perf_counter()
                    age_ms = (now - self._latest_process_time_perf) * 1000.0 if self._latest_process_time_perf > 0 else 0.0
                    if self.warn_age_ms > 0 and age_ms >= self.warn_age_ms and now - self._last_age_warn_t >= 1.0:
                        logging.warning(
                            "Wrist queued pipeline %s returned stale latest frame age=%.1fms published_id=%s raw_id=%s raw_q=%s pending=%s",
                            self.label,
                            age_ms,
                            self._published_frame_id,
                            self._last_callback_raw_id,
                            self._raw_queue.qsize(),
                            len(self._pending_processed),
                        )
                        self._last_age_warn_t = now
                    self._last_read_id = self._published_frame_id
                    self._last_returned_source_time_perf = self._latest_source_time_perf
                    return self._latest_frame.copy() if self.return_copy else self._latest_frame
                if self._stop.is_set():
                    return None
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)

    def get_latest_frame_time_perf(self) -> float:
        with self._condition:
            return self._last_returned_source_time_perf or self._latest_source_time_perf

    def get_processed_frames_after(self, frame_id: int) -> list[tuple[float, int, np.ndarray]]:
        with self._condition:
            self._raise_if_error_locked()
            if self._history and frame_id > 0:
                oldest_id = self._history[0][1]
                if frame_id < oldest_id:
                    raise RuntimeError(
                        f"Wrist processed history overflow for {self.label}: "
                        f"requested_after={frame_id} oldest_available={oldest_id} "
                        f"latest={self._published_frame_id} max={self.processed_queue_frames}"
                    )
            return [
                (source_time, published_id, frame.copy() if self.return_copy else frame)
                for source_time, published_id, frame in self._history
                if published_id > frame_id
            ]

    def startup_snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "label": self.label,
                "frame_id": self._published_frame_id,
                "latest_source_time_perf": self._latest_source_time_perf,
                "latest_process_time_perf": self._latest_process_time_perf,
                "has_frame": self._latest_frame is not None,
                "raw_queue": self._raw_queue.qsize(),
                "pending": len(self._pending_processed),
            }

    def close(self) -> None:
        self._stop.set()
        remove_callback = getattr(self.receiver, "remove_frame_callback", None)
        if self._callback_registered and callable(remove_callback):
            try:
                remove_callback(self._on_raw_frame)
            except Exception:
                pass
        for _ in self._threads:
            try:
                self._raw_queue.put_nowait(None)
            except queue.Full:
                pass
        with self._condition:
            self._condition.notify_all()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=1.0)


def _patch_wrist_receiver_processed_cache(receiver: Any, cropper: WristUndistortCenterCropper, label: str) -> None:
    if receiver is None or not hasattr(receiver, "async_read"):
        return
    if getattr(receiver, "_kd_tacmae_processed_cache_patch", False):
        return
    pipeline_mode = env_choice("WRIST_PIPELINE_MODE", "queued", {"queued", "latest"})
    if pipeline_mode == "queued":
        cache = WristQueuedPipeline(receiver=receiver, cropper=cropper, label=label)
    else:
        cache = WristProcessedFrameCache(receiver=receiver, cropper=cropper, label=label)
    receiver.async_read = cache.async_read
    receiver.get_latest_frame_time_perf = cache.get_latest_frame_time_perf
    if hasattr(cache, "get_processed_frames_after"):
        receiver.get_processed_frames_after = cache.get_processed_frames_after
    receiver._kd_tacmae_processed_cache_patch = True
    receiver._kd_tacmae_processed_cache = cache
    _WRIST_PROCESSED_CACHES.append(cache)
    logging.info(
        "Wrist processed pipeline enabled for %s: mode=%s workers=%s return_copy=%s warn_ms=%.1f warn_age_ms=%.1f read_timeout_s=%.3f",
        label,
        pipeline_mode,
        cache.workers,
        cache.return_copy,
        cache.warn_ms,
        cache.warn_age_ms,
        cache.read_timeout_s,
    )


def _patch_arm_wrist_processed_cache(
    arm: Any,
    side: str,
    cropper: WristUndistortCenterCropper,
    wrist_key: str,
) -> None:
    if getattr(arm, "_kd_tacmae_wrist_processed_cache_connect_patch", False):
        receivers = getattr(arm, "_tcp_receivers", None)
        if isinstance(receivers, dict):
            receiver = receivers.get(wrist_key)
            _patch_wrist_receiver_processed_cache(receiver, cropper, f"{side}_{wrist_key}")
        return

    original_connect = arm.connect

    def connect_with_wrist_processed_cache(*args: Any, **kwargs: Any) -> Any:
        result = original_connect(*args, **kwargs)
        receivers = getattr(arm, "_tcp_receivers", None)
        if isinstance(receivers, dict):
            receiver = receivers.get(wrist_key)
            _patch_wrist_receiver_processed_cache(receiver, cropper, f"{side}_{wrist_key}")
        return result

    arm.connect = connect_with_wrist_processed_cache
    arm._kd_tacmae_wrist_processed_cache_connect_patch = True
    receivers = getattr(arm, "_tcp_receivers", None)
    if isinstance(receivers, dict):
        receiver = receivers.get(wrist_key)
        _patch_wrist_receiver_processed_cache(receiver, cropper, f"{side}_{wrist_key}")


def cleanup_wrist_processed_caches() -> None:
    while _WRIST_PROCESSED_CACHES:
        cache = _WRIST_PROCESSED_CACHES.pop()
        try:
            cache.close()
        except Exception as exc:
            logging.debug("Wrist processed cache cleanup failed: %s", exc)


def wait_async_observation_caches_ready() -> None:
    wrist_frames = max(0, env_int("WRIST_PROCESSED_ASYNC_CACHE_STARTUP_FRAMES", 3))
    wrist_timeout_s = max(0.0, env_float("WRIST_PROCESSED_ASYNC_CACHE_STARTUP_TIMEOUT_S", 3.0))
    wrist_max_age_ms = env_float("WRIST_PROCESSED_ASYNC_CACHE_STARTUP_MAX_AGE_MS", 120.0)
    tactile_timeout_s = max(0.0, env_float("X5_TACTILE_ASYNC_CACHE_STARTUP_TIMEOUT_S", 2.0))
    tactile_max_age_ms = env_float("X5_TACTILE_ASYNC_CACHE_STARTUP_MAX_AGE_MS", 120.0)

    if wrist_frames > 0 and _WRIST_PROCESSED_CACHES:
        initial_ids = {
            cache: int(cache.startup_snapshot()["frame_id"])
            for cache in list(_WRIST_PROCESSED_CACHES)
        }
        deadline = time.perf_counter() + wrist_timeout_s
        last_status = ""
        while True:
            now = time.perf_counter()
            statuses: list[str] = []
            ready = True
            for cache in list(_WRIST_PROCESSED_CACHES):
                snap = cache.startup_snapshot()
                frame_id = int(snap["frame_id"])
                start_id = initial_ids.get(cache, frame_id)
                process_time = float(snap["latest_process_time_perf"] or 0.0)
                age_ms = (now - process_time) * 1000.0 if process_time > 0.0 else float("inf")
                cache_ready = (
                    bool(snap["has_frame"])
                    and frame_id >= start_id + wrist_frames
                    and (wrist_max_age_ms <= 0.0 or age_ms <= wrist_max_age_ms)
                )
                ready = ready and cache_ready
                statuses.append(
                    f"{snap['label']}:frame={frame_id} new={frame_id - start_id} age={age_ms:.1f}ms"
                )
            last_status = "; ".join(statuses)
            if ready:
                logging.info("Wrist processed cache startup ready: %s", last_status)
                break
            if now >= deadline:
                logging.warning(
                    "Wrist processed cache startup wait timed out after %.1fs: %s",
                    wrist_timeout_s,
                    last_status,
                )
                break
            time.sleep(0.01)

    if tactile_timeout_s > 0.0 and _TACTILE_READ_CACHES:
        deadline = time.perf_counter() + tactile_timeout_s
        last_status = ""
        while True:
            now = time.perf_counter()
            statuses: list[str] = []
            ready = True
            for cache in list(_TACTILE_READ_CACHES):
                snap = cache.startup_snapshot()
                refresh_time = float(snap["latest_refresh_perf"] or 0.0)
                age_ms = (now - refresh_time) * 1000.0 if refresh_time > 0.0 else float("inf")
                cache_ready = (
                    bool(snap["ready"])
                    and bool(snap["has_all"])
                    and (tactile_max_age_ms <= 0.0 or age_ms <= tactile_max_age_ms)
                )
                ready = ready and cache_ready
                statuses.append(
                    f"keys={len(snap['keys'])} ready={snap['ready']} has_all={snap['has_all']} age={age_ms:.1f}ms"
                )
            last_status = "; ".join(statuses)
            if ready:
                logging.info("X5 tactile async cache startup ready: %s", last_status)
                break
            if now >= deadline:
                logging.warning(
                    "X5 tactile async cache startup wait timed out after %.1fs: %s",
                    tactile_timeout_s,
                    last_status,
                )
                break
            time.sleep(0.01)


def _wrist_calib_file_for_side(side: str) -> str | None:
    side_upper = side.upper()
    return env_optional_first(
        f"WRIST_UNDISTORT_{side_upper}_CALIB_FILE",
        f"{side_upper}_WRIST_UNDISTORT_CALIB_FILE",
        f"{side_upper}_WRIST_CALIB_FILE",
        "WRIST_UNDISTORT_CALIB_FILE",
    )


def _wrist_calibration_from_env(side: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int], str] | None:
    side_upper = side.upper()
    matrix_raw = env_optional_first(
        f"WRIST_UNDISTORT_{side_upper}_K",
        f"WRIST_UNDISTORT_{side_upper}_CAMERA_MATRIX",
        f"{side_upper}_WRIST_UNDISTORT_K",
        f"{side_upper}_WRIST_UNDISTORT_CAMERA_MATRIX",
        "WRIST_UNDISTORT_K",
        "WRIST_UNDISTORT_CAMERA_MATRIX",
    )
    dist_raw = env_optional_first(
        f"WRIST_UNDISTORT_{side_upper}_D",
        f"WRIST_UNDISTORT_{side_upper}_DIST_COEFFS",
        f"{side_upper}_WRIST_UNDISTORT_D",
        f"{side_upper}_WRIST_UNDISTORT_DIST_COEFFS",
        "WRIST_UNDISTORT_D",
        "WRIST_UNDISTORT_DIST_COEFFS",
    )
    size_raw = env_optional_first(
        f"WRIST_UNDISTORT_{side_upper}_IMAGE_SIZE",
        f"{side_upper}_WRIST_UNDISTORT_IMAGE_SIZE",
        "WRIST_UNDISTORT_IMAGE_SIZE",
    )
    if matrix_raw is None and dist_raw is None and size_raw is None:
        return None
    if matrix_raw is None or dist_raw is None or size_raw is None:
        raise ValueError(
            f"{side} wrist env calibration is incomplete; provide K, D, and IMAGE_SIZE together."
        )

    source = f"environment:{side}"
    data = {
        "camera_matrix": _parse_numeric_json_or_csv(matrix_raw, name=f"{side}_K"),
        "dist_coeffs": _parse_numeric_json_or_csv(dist_raw, name=f"{side}_D"),
        "image_size": _parse_numeric_json_or_csv(size_raw, name=f"{side}_IMAGE_SIZE"),
    }
    camera_matrix, dist_coeffs, image_size = WristUndistortCenterCropper._coerce_calibration(
        data,
        source=source,
    )
    return camera_matrix, dist_coeffs, image_size, source


def _wrist_calibration_for_side(arm: Any, side: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int], str]:
    env_calibration = _wrist_calibration_from_env(side)
    if env_calibration is not None:
        return env_calibration

    cfg = getattr(arm, "config", None)
    x5_ip = str(getattr(cfg, "x5_ip", "") or "").strip()
    if x5_ip and x5_ip in DEFAULT_WRIST_UNDISTORT_CALIBRATIONS_BY_IP:
        source = f"default_by_ip:{x5_ip}"
        camera_matrix, dist_coeffs, image_size = WristUndistortCenterCropper._coerce_calibration(
            DEFAULT_WRIST_UNDISTORT_CALIBRATIONS_BY_IP[x5_ip],
            source=source,
        )
        return camera_matrix, dist_coeffs, image_size, source

    calib_file = _wrist_calib_file_for_side(side)
    if calib_file is not None:
        source = f"file:{calib_file}"
        camera_matrix, dist_coeffs, image_size = WristUndistortCenterCropper.load_calibration_from_file(
            calib_file
        )
        return camera_matrix, dist_coeffs, image_size, source

    known_ips = ", ".join(sorted(DEFAULT_WRIST_UNDISTORT_CALIBRATIONS_BY_IP)) or "<empty>"
    raise ValueError(
        "WRIST_UNDISTORT=true requires wrist intrinsics. Provide env K/D/IMAGE_SIZE, "
        f"or add x5_ip={x5_ip or '<unset>'} to DEFAULT_WRIST_UNDISTORT_CALIBRATIONS_BY_IP "
        f"(known IPs: {known_ips}), or set WRIST_UNDISTORT_{side.upper()}_CALIB_FILE."
    )


def _patch_arm_wrist_undistort_crop(arm: Any, side: str) -> bool:
    cfg = getattr(arm, "config", None)
    if arm is None or cfg is None:
        return False
    if not bool(getattr(cfg, "enabled", True)) or not bool(getattr(cfg, "connect_wrist_camera", True)):
        return False
    if getattr(arm, "_kd_tacmae_wrist_undistort_crop_patch", False):
        return True

    camera_matrix, dist_coeffs, image_size, calibration_source = _wrist_calibration_for_side(arm, side)
    crop_size = env_int("WRIST_UNDISTORT_CROP_SIZE", 896)
    cropper = WristUndistortCenterCropper(
        side=side,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        calibration_source=calibration_source,
        balance=env_float("WRIST_UNDISTORT_BALANCE", 0.0),
        crop_size=crop_size,
    )
    wrist_key = str(getattr(cfg, "wrist_cam_name", ""))
    if not wrist_key:
        raise ValueError(f"{side} wrist undistort patch cannot find config.wrist_cam_name")

    use_processed_cache = env_bool("WRIST_PROCESSED_ASYNC_CACHE", False)
    if use_processed_cache:
        _patch_arm_wrist_processed_cache(arm, side, cropper, wrist_key)
        arm._kd_tacmae_wrist_undistort_crop_patch = True
        arm._kd_tacmae_wrist_undistort_processed_cache = True
        arm._kd_tacmae_wrist_undistort_crop_size = crop_size
        arm._kd_tacmae_wrist_undistort_original_size = (
            int(getattr(cfg, "wrist_cam_width", 0)),
            int(getattr(cfg, "wrist_cam_height", 0)),
        )
        try:
            features = dict(getattr(arm, "observation_features"))
            features[wrist_key] = (crop_size, crop_size, 3)
            arm.__dict__["observation_features"] = features
        except Exception as exc:
            logging.warning("%s wrist processed cache could not patch arm observation_features: %s", side, exc)
        logging.info(
            "%s wrist processed async cache requested for %s: source_request=%sx%s output=%sx%s calib=%s",
            side,
            wrist_key,
            getattr(cfg, "wrist_cam_width", None),
            getattr(cfg, "wrist_cam_height", None),
            crop_size,
            crop_size,
            calibration_source,
        )
        return True

    original_get_observation = arm.get_observation

    def get_observation_with_wrist_undistort_crop() -> dict[str, Any]:
        profile = env_bool("CAPTURE_OBS_SOURCE_PROFILE", False)
        warn_ms = env_float("CAPTURE_OBS_SOURCE_PROFILE_WARN_MS", 8.0)
        t0 = time.perf_counter()
        obs = original_get_observation()
        raw_dt_ms = (time.perf_counter() - t0) * 1000.0
        if profile and raw_dt_ms >= warn_ms:
            logging.warning("Observation source profile %s_arm.raw_observation %.1fms", side, raw_dt_ms)
        frame = obs.get(wrist_key)
        if isinstance(frame, np.ndarray):
            t1 = time.perf_counter()
            obs[wrist_key] = cropper(frame)
            crop_dt_ms = (time.perf_counter() - t1) * 1000.0
            if profile and crop_dt_ms >= warn_ms:
                logging.warning("Observation source profile %s_arm.wrist_undistort_crop %.1fms", side, crop_dt_ms)
        return obs

    arm.get_observation = get_observation_with_wrist_undistort_crop
    arm._kd_tacmae_wrist_undistort_crop_patch = True
    arm._kd_tacmae_wrist_undistort_crop_size = crop_size
    arm._kd_tacmae_wrist_undistort_original_size = (
        int(getattr(cfg, "wrist_cam_width", 0)),
        int(getattr(cfg, "wrist_cam_height", 0)),
    )

    try:
        features = dict(getattr(arm, "observation_features"))
        features[wrist_key] = (crop_size, crop_size, 3)
        arm.__dict__["observation_features"] = features
    except Exception as exc:
        logging.warning("%s wrist undistort could not patch arm observation_features: %s", side, exc)

    logging.info(
        "%s wrist undistort/crop enabled for %s: source_request=%sx%s output=%sx%s calib=%s",
        side,
        wrist_key,
        getattr(cfg, "wrist_cam_width", None),
        getattr(cfg, "wrist_cam_height", None),
        crop_size,
        crop_size,
        calibration_source,
    )
    return True


def patch_wrist_undistort_crop(robot: Any) -> None:
    if not env_bool("WRIST_UNDISTORT", False):
        return

    patched = False
    robot_cfg = getattr(robot, "config", None)
    for side, attr in (("left", "left_arm"), ("right", "right_arm")):
        parent_arm_cfg = getattr(robot_cfg, f"{side}_arm_config", None)
        if parent_arm_cfg is not None and (
            not bool(getattr(parent_arm_cfg, "enabled", False))
            or not bool(getattr(parent_arm_cfg, "connect_wrist_camera", True))
        ):
            continue
        arm = getattr(robot, attr, None)
        patched = _patch_arm_wrist_undistort_crop(arm, side) or patched

    if not patched:
        raise RuntimeError("WRIST_UNDISTORT=true but no enabled wrist camera arm was patched")

    crop_size = env_int("WRIST_UNDISTORT_CROP_SIZE", 896)
    try:
        features = dict(getattr(robot, "observation_features"))
        for side, arm_attr in (("left", "left_arm"), ("right", "right_arm")):
            arm = getattr(robot, arm_attr, None)
            cfg = getattr(arm, "config", None)
            if arm is None or cfg is None:
                continue
            if not getattr(arm, "_kd_tacmae_wrist_undistort_crop_patch", False):
                continue
            features[f"{side}_{getattr(cfg, 'wrist_cam_name')}"] = (crop_size, crop_size, 3)
        robot.__dict__["observation_features"] = features
    except Exception as exc:
        logging.warning("Failed to patch bimanual wrist undistort observation features: %s", exc)


def _patch_wrist_receiver_require_new(receiver: Any, label: str) -> None:
    if receiver is None or not hasattr(receiver, "async_read"):
        return
    if getattr(receiver, "_kd_tacmae_require_new_patch", False):
        return

    original_async_read = receiver.async_read
    default_timeout_s = max(0.0, env_float("WRIST_CAMERA_REQUIRE_NEW_TIMEOUT_S", 0.006))
    mode = env_choice("WRIST_CAMERA_REQUIRE_NEW_MODE", "adaptive", {"adaptive", "always"})
    adaptive_min_age_ms = max(0.0, env_float("WRIST_CAMERA_REQUIRE_NEW_MIN_AGE_MS", 28.0))
    fallback_last = env_bool("WRIST_CAMERA_REQUIRE_NEW_FALLBACK_LAST", True)

    def async_read_require_new(timeout_s: float | None = None, *, require_new: bool = False) -> Any:
        effective_require_new = require_new or env_bool("WRIST_CAMERA_REQUIRE_NEW", False)
        if not effective_require_new:
            return original_async_read(timeout_s=timeout_s, require_new=require_new)

        effective_timeout_s = timeout_s if timeout_s is not None else default_timeout_s
        if mode == "adaptive" and not require_new:
            frame_lock = getattr(receiver, "_frame_lock", None)
            latest_frame = None
            frame_id = None
            last_read_id = None
            latest_time = 0.0
            if frame_lock is not None:
                try:
                    with frame_lock:
                        latest_frame = getattr(receiver, "_latest_frame", None)
                        frame_id = getattr(receiver, "_frame_id", None)
                        last_read_id = getattr(receiver, "_last_read_id", None)
                        latest_time = float(getattr(receiver, "_latest_frame_time_perf", 0.0) or 0.0)
                except Exception:
                    latest_frame = None

            if latest_frame is not None and frame_id is not None and last_read_id is not None:
                age_ms = (time.perf_counter() - latest_time) * 1000.0 if latest_time > 0.0 else 0.0
                same_as_last_read = frame_id == last_read_id
                if not same_as_last_read or age_ms < adaptive_min_age_ms:
                    return original_async_read(timeout_s=0.0, require_new=False)

        frame = original_async_read(timeout_s=effective_timeout_s, require_new=True)
        if frame is not None or not fallback_last:
            return frame
        return original_async_read(timeout_s=0.0, require_new=False)

    receiver.async_read = async_read_require_new
    receiver._kd_tacmae_require_new_patch = True
    logging.info(
        "Wrist camera %s require-new patch enabled: mode=%s timeout_s=%.4f "
        "adaptive_min_age_ms=%.1f fallback_last=%s",
        label,
        mode,
        default_timeout_s,
        adaptive_min_age_ms,
        fallback_last,
    )


def _patch_arm_wrist_receiver_require_new(arm: Any, side: str) -> None:
    receivers = getattr(arm, "_tcp_receivers", None)
    if not isinstance(receivers, dict):
        return
    for name, receiver in receivers.items():
        _patch_wrist_receiver_require_new(receiver, f"{side}_{name}")


def patch_wrist_camera_require_new(robot: Any) -> None:
    if not env_bool("WRIST_CAMERA_REQUIRE_NEW", False):
        return
    if getattr(robot, "_kd_tacmae_wrist_require_new_patch", False):
        return

    def patch_arm_connect(arm: Any, side: str) -> None:
        if arm is None or not hasattr(arm, "connect"):
            return
        if getattr(arm, "_kd_tacmae_wrist_require_new_connect_patch", False):
            _patch_arm_wrist_receiver_require_new(arm, side)
            return
        original_connect = arm.connect

        def connect_with_wrist_require_new(*args: Any, **kwargs: Any) -> Any:
            result = original_connect(*args, **kwargs)
            _patch_arm_wrist_receiver_require_new(arm, side)
            return result

        arm.connect = connect_with_wrist_require_new
        arm._kd_tacmae_wrist_require_new_connect_patch = True
        _patch_arm_wrist_receiver_require_new(arm, side)

    patch_arm_connect(getattr(robot, "left_arm", None), "left")
    patch_arm_connect(getattr(robot, "right_arm", None), "right")
    robot._kd_tacmae_wrist_require_new_patch = True


def _sync_diagnostic_values(timing: dict[str, Any]) -> dict[str, float]:
    values = {field: -1.0 for field in SYNC_DIAGNOSTIC_FIELDS}
    start_t = timing.get("obs_read_start_time")
    finish_t = timing.get("obs_read_finish_time")
    if isinstance(start_t, (int, float)) and isinstance(finish_t, (int, float)) and finish_t >= start_t:
        values["sync_obs_read_dt_ms"] = float((finish_t - start_t) * 1000.0)

    source_ages = timing.get("visual_source_ages_ms", {})
    valid_ages: list[float] = []
    if isinstance(source_ages, dict):
        for source_key, field in SYNC_SOURCE_TO_FIELD.items():
            age = source_ages.get(source_key)
            if isinstance(age, (int, float)) and age >= 0:
                value = float(age)
                values[field] = value
                valid_ages.append(value)

    if valid_ages:
        values["sync_age_ms_visual_max"] = float(max(valid_ages))
        values["sync_age_ms_visual_mean"] = float(sum(valid_ages) / len(valid_ages))
    return values


def patch_sync_diagnostics(robot: Any) -> None:
    if not env_bool("RECORD_SYNC_DIAGNOSTICS", True):
        return
    if getattr(robot, "_kd_tacmae_sync_diagnostics_patch", False):
        return

    try:
        features = dict(getattr(robot, "observation_features"))
        for field in SYNC_DIAGNOSTIC_FIELDS:
            features[field] = float
        # BiRealmanUGripperNotacNew.observation_features is a cached_property.
        # Writing the instance cache keeps the patch local to this recorder.
        robot.__dict__["observation_features"] = features
    except Exception as exc:
        logging.warning("Failed to add sync diagnostics to observation features: %s", exc)
        return

    original_get_observation = robot.get_observation
    warn_age_ms = env_float("SYNC_WARN_AGE_MS", 120.0)
    warn_every = env_int("SYNC_WARN_EVERY", 30)
    counter = {"frames": 0}

    def get_observation_with_sync_diagnostics() -> dict[str, Any]:
        obs = original_get_observation()
        getter = getattr(robot, "get_last_observation_timing", None)
        timing = getter() if callable(getter) else {}
        values = _sync_diagnostic_values(timing if isinstance(timing, dict) else {})
        obs.update(values)

        counter["frames"] += 1
        if warn_every > 0 and counter["frames"] % warn_every == 0:
            over = {
                field: round(value, 1)
                for field, value in values.items()
                if field.startswith("sync_age_ms_") and value >= warn_age_ms
            }
            if over:
                logging.warning(
                    "Capture sync diagnostics: stale visual source age over %.1f ms at frame %s: %s",
                    warn_age_ms,
                    counter["frames"],
                    over,
                )
        return obs

    robot.get_observation = get_observation_with_sync_diagnostics
    robot._kd_tacmae_sync_diagnostics_patch = True
    logging.info("Capture sync diagnostics enabled; fields appended to observation.state: %s", ", ".join(SYNC_DIAGNOSTIC_FIELDS))


def patch_raw_spool_wrist_placeholders(robot: Any) -> None:
    """Keep LeRobot frame construction happy while raw-spool writes real wrist videos later."""
    if not env_bool("RAW_SPOOL_PER_EPISODE", False):
        return
    if env_bool("CONNECT_WRIST_CAMERA", True):
        return
    if getattr(robot, "_kd_tacmae_raw_spool_wrist_placeholder_patch", False):
        return
    candidate_keys = (
        os.environ.get("RAW_SPOOL_LEFT_WRIST_KEY", "left_cam_left_wrist").strip() or "left_cam_left_wrist",
        os.environ.get("RAW_SPOOL_RIGHT_WRIST_KEY", "right_cam_right_wrist").strip() or "right_cam_right_wrist",
    )
    if not env_bool("RAW_SPOOL_ENCODE_PLACEHOLDER_WRIST", False):
        try:
            features = dict(getattr(robot, "observation_features"))
        except Exception as exc:
            logging.warning("Raw-spool wrist feature removal could not read observation_features: %s", exc)
            features = {}
        removed = []
        for key in candidate_keys:
            if key in features:
                features.pop(key, None)
                removed.append(key)
        try:
            robot.__dict__["observation_features"] = features
        except Exception as exc:
            logging.warning("Raw-spool wrist feature removal could not patch observation_features: %s", exc)
        if removed:
            logging.info(
                "Raw-spool wrist placeholder encoding disabled; removed wrist features from main writer: %s",
                ",".join(removed),
            )
        else:
            logging.info(
                "Raw-spool wrist placeholder encoding disabled; no wrist features were present in main writer"
            )
        logging.info(
            "Raw-spool wrist placeholders disabled; wrist videos will be added by raw-spool finalizer only"
        )
        robot._kd_tacmae_raw_spool_wrist_placeholder_patch = True
        return

    crop_size = env_int("WRIST_UNDISTORT_CROP_SIZE", 896)

    try:
        features = dict(getattr(robot, "observation_features"))
    except Exception as exc:
        logging.warning("Raw-spool wrist placeholder could not read observation_features: %s", exc)
        features = {}

    placeholders: dict[str, np.ndarray] = {}
    for key in candidate_keys:
        feature = features.get(key)
        height = width = crop_size
        channels = 3
        if isinstance(feature, (tuple, list)) and len(feature) >= 3:
            try:
                height, width, channels = int(feature[0]), int(feature[1]), int(feature[2])
            except Exception:
                height = width = crop_size
                channels = 3
        if channels != 3 or height <= 0 or width <= 0:
            height = width = crop_size
            channels = 3
        # Force the schema to match the undistorted wrist videos produced by
        # finalize_raw_spool_wrist_videos.py. These placeholder frames are
        # overwritten by the finalized videos after dataset.finalize().
        features[key] = (crop_size, crop_size, 3)
        placeholders[key] = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    try:
        robot.__dict__["observation_features"] = features
    except Exception as exc:
        logging.warning("Raw-spool wrist placeholder could not patch observation_features: %s", exc)

    original_get_observation = robot.get_observation

    def get_observation_with_raw_spool_wrist_placeholders() -> dict[str, Any]:
        obs = original_get_observation()
        for key, frame in placeholders.items():
            obs.setdefault(key, frame)
        return obs

    robot.get_observation = get_observation_with_raw_spool_wrist_placeholders
    robot._kd_tacmae_raw_spool_wrist_placeholder_patch = True
    logging.info(
        "Raw-spool wrist placeholders enabled for %s at %sx%s; finalized wrist videos will overwrite placeholders",
        ", ".join(placeholders),
        crop_size,
        crop_size,
    )


class RGBPreviewer:
    """Low-priority RGB/tactile preview window fed by a drop-oldest queue."""

    def __init__(self) -> None:
        self.rgb_enabled = env_bool("RGB_PREVIEW", False)
        self.tactile_enabled = env_bool("TACTILE_PREVIEW", False)
        self.enabled = self.rgb_enabled or self.tactile_enabled
        self.keys = tuple(
            key.strip()
            for key in os.environ.get(
                "RGB_PREVIEW_KEYS",
                "left_cam_left_wrist,cam_d405_color,right_cam_right_wrist",
            ).split(",")
            if key.strip()
        )
        self.tactile_keys = tuple(
            key.strip()
            for key in os.environ.get(
                "TACTILE_PREVIEW_KEYS",
                ",".join(
                    [
                        "depth_deformation.tactile_left_left",
                        "depth_deformation.tactile_left_right",
                        "depth_deformation.tactile_right_left",
                        "depth_deformation.tactile_right_right",
                    ]
                ),
            ).split(",")
            if key.strip()
        )
        self.fps = max(0.1, env_float("PREVIEW_FPS", env_float("RGB_PREVIEW_FPS", 5.0)))
        self.width = max(64, env_int("RGB_PREVIEW_WIDTH", 320))
        self.tactile_width = max(48, env_int("TACTILE_PREVIEW_WIDTH", 160))
        self.tactile_depth_weight = env_float("TACTILE_PREVIEW_DEPTH_WEIGHT", 1.0)
        self.tactile_shear_weight = env_float("TACTILE_PREVIEW_SHEAR_WEIGHT", 0.35)
        self.tactile_depth_mode = os.environ.get("TACTILE_PREVIEW_DEPTH_MODE", "abs").strip().lower()
        if self.tactile_depth_mode not in {"abs", "positive", "negative"}:
            self.tactile_depth_mode = "abs"
        self.tactile_percentile = min(100.0, max(50.0, env_float("TACTILE_PREVIEW_PERCENTILE", 99.5)))
        self.tactile_noise_sigma = max(0.0, env_float("TACTILE_PREVIEW_NOISE_SIGMA", 8.0))
        self.tactile_min_active_pixels = max(0, env_int("TACTILE_PREVIEW_MIN_ACTIVE_PIXELS", 96))
        self.tactile_min_signal = max(0.0, env_float("TACTILE_PREVIEW_MIN_SIGNAL", 0.02))
        self.tactile_display_floor = max(0.0, env_float("TACTILE_PREVIEW_DISPLAY_FLOOR", 0.05))
        self.tactile_median_blur = max(0, env_int("TACTILE_PREVIEW_MEDIAN_BLUR", 3))
        self.tactile_rgb_passthrough = env_bool("TACTILE_PREVIEW_RGB_PASSTHROUGH", False)
        self.tactile_baseline_frames = max(1, env_int("TACTILE_PREVIEW_BASELINE_FRAMES", 15))
        self.tactile_baseline_update_alpha = max(0.0, env_float("TACTILE_PREVIEW_BASELINE_UPDATE_ALPHA", 0.01))
        self.tactile_thresholds, self.tactile_calibrations = self._load_tactile_calibrations()
        self.tactile_threshold_scale = max(0.0, env_float("TACTILE_PREVIEW_THRESHOLD_SCALE", 1.0))
        self.tactile_use_startup_baseline = env_bool(
            "TACTILE_PREVIEW_USE_STARTUP_BASELINE",
            not bool(self.tactile_thresholds),
        )
        self.tactile_rotate_clockwise = env_bool("TACTILE_PREVIEW_ROTATE_CLOCKWISE", True)
        self.window_scale = max(0.2, env_float("PREVIEW_WINDOW_SCALE", 1.0))
        self.window_name = (
            os.environ.get("PREVIEW_WINDOW", os.environ.get("RGB_PREVIEW_WINDOW", "KD-TacMAE Preview")).strip()
            or "KD-TacMAE Preview"
        )
        self.udp_host = os.environ.get("PREVIEW_UDP_HOST", "127.0.0.1").strip() or "127.0.0.1"
        self.udp_port = max(0, env_int("PREVIEW_UDP_PORT", 0))
        self.udp_jpeg_quality = min(95, max(20, env_int("PREVIEW_JPEG_QUALITY", 72)))
        self.show_window = env_bool("PREVIEW_SHOW_WINDOW", self.udp_port <= 0)
        self._udp_socket: socket.socket | None = None
        if self.udp_port > 0:
            self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._interval_s = 1.0 / self.fps
        self._last_submit_s = 0.0
        self._queue: queue.Queue[dict[str, np.ndarray]] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._disabled_after_error = False
        self._window_initialized = False
        self._last_window_size: tuple[int, int] | None = None
        self._tactile_baselines: dict[str, np.ndarray] = {}
        self._tactile_baseline_counts: dict[str, int] = {}

        if self.enabled:
            self._thread = threading.Thread(target=self._loop, name="kd_tacmae_preview", daemon=True)
            self._thread.start()
            logging.info(
                "Preview enabled: rgb=%s tactile=%s rgb_keys=%s tactile_keys=%s fps=%.1f rgb_width=%s tactile_width=%s window_scale=%.2f show_window=%s udp=%s:%s queue=drop-oldest",
                self.rgb_enabled,
                self.tactile_enabled,
                ",".join(self.keys),
                ",".join(self.tactile_keys),
                self.fps,
                self.width,
                self.tactile_width,
                self.window_scale,
                self.show_window,
                self.udp_host if self.udp_port > 0 else "",
                self.udp_port if self.udp_port > 0 else "",
            )
            if self.tactile_thresholds:
                logging.info("Tactile preview fixed thresholds loaded for %s stream(s)", len(self.tactile_thresholds))

    @staticmethod
    def _as_hwc(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3:
            raise ValueError(f"expected image ndim 2/3, got shape={arr.shape}")
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]
        return arr

    @staticmethod
    def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
        arr = RGBPreviewer._as_hwc(frame)
        if arr.dtype == np.uint8:
            return np.ascontiguousarray(arr)
        arr = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr)
        if not finite.any():
            return np.zeros((*arr.shape[:2], 3), dtype=np.uint8)
        arr = np.where(finite, arr, 0.0)
        if float(np.nanmax(arr)) <= 1.5 and float(np.nanmin(arr)) >= -0.1:
            arr = arr * 255.0
        return np.ascontiguousarray(np.clip(arr, 0.0, 255.0).astype(np.uint8))

    def _make_small(self, frame: np.ndarray) -> np.ndarray:
        rgb = self._to_uint8_rgb(frame)
        h, w = rgb.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"invalid image shape={rgb.shape}")
        out_h = max(1, int(round(h * (self.width / float(w)))))
        return cv2.resize(rgb, (self.width, out_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _as_chw(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3 and arr.shape[0] in {1, 2, 3, 4} and arr.shape[-1] not in {1, 2, 3, 4}:
            pass
        elif arr.ndim == 3 and arr.shape[-1] in {1, 2, 3, 4}:
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"expected tactile ndim 2/3, got shape={arr.shape}")
        if arr.shape[0] < 1:
            raise ValueError(f"expected at least one tactile channel, got shape={arr.shape}")
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _normalize_tactile_key(key: str) -> str:
        return key.replace("observation.", "", 1)

    @staticmethod
    def _threshold_from_value(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for name in ("threshold", "q99", "p99", "value"):
                if name in value:
                    return float(value[name])
        return None

    @staticmethod
    def _as_float_vector(value: Any) -> np.ndarray | None:
        if not isinstance(value, (list, tuple)):
            return None
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
        if arr.ndim != 1 or arr.size == 0:
            return None
        return arr

    def _load_tactile_calibrations(self) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        path = env_optional_str("TACTILE_PREVIEW_THRESHOLD_FILE")
        inline = env_optional_str("TACTILE_PREVIEW_THRESHOLDS_JSON")
        if not path and not inline:
            return {}, {}
        try:
            data = json.loads(Path(path).read_text()) if path else json.loads(inline or "{}")
        except Exception as exc:
            logging.warning("Failed to load tactile preview thresholds: %s", exc)
            return {}, {}

        raw = data.get("thresholds", data) if isinstance(data, dict) else {}
        thresholds: dict[str, float] = {}
        calibrations: dict[str, dict[str, Any]] = {}
        if not isinstance(raw, dict):
            return thresholds, calibrations
        for key, value in raw.items():
            threshold = self._threshold_from_value(value)
            if threshold is None:
                continue
            normalized = self._normalize_tactile_key(str(key))
            thresholds[normalized] = float(threshold)
            thresholds[f"observation.{normalized}"] = float(threshold)

            if isinstance(value, dict):
                item: dict[str, Any] = {
                    "threshold_raw": float(value.get("threshold_raw", threshold)),
                    "threshold_decoded": float(value.get("threshold_decoded", float(threshold) / 1000.0)),
                }
                for field in (
                    "reference_raw",
                    "noise_scale_raw",
                    "reference_decoded",
                    "noise_scale_decoded",
                ):
                    vec = self._as_float_vector(value.get(field))
                    if vec is not None:
                        item[field] = vec
                calibrations[normalized] = item
                calibrations[f"observation.{normalized}"] = item
        return thresholds, calibrations

    def _lookup_tactile_calibration(self, key: str) -> dict[str, Any] | None:
        item = self.tactile_calibrations.get(key)
        if item is None:
            item = self.tactile_calibrations.get(self._normalize_tactile_key(key))
        return item

    @staticmethod
    def _looks_like_raw_tactile(chw: np.ndarray) -> bool:
        if chw.shape[0] >= 3:
            probe = chw[1]
        else:
            probe = chw[0]
        finite = np.isfinite(probe)
        if not finite.any():
            return False
        return abs(float(np.nanmedian(probe[finite]))) > 1000.0

    def _depth_response(self, delta: np.ndarray, gate: float) -> np.ndarray:
        gate = max(0.0, float(gate))
        if self.tactile_depth_mode == "positive":
            return np.maximum(delta - gate, 0.0)
        if self.tactile_depth_mode == "negative":
            return np.maximum(-delta - gate, 0.0)
        return np.maximum(np.abs(delta) - gate, 0.0)

    def _make_calibrated_tactile_intensity(self, chw: np.ndarray, key: str) -> tuple[np.ndarray, float] | None:
        calibration = self._lookup_tactile_calibration(key)
        if not calibration:
            return None
        raw_mode = self._looks_like_raw_tactile(chw)
        suffix = "raw" if raw_mode else "decoded"
        reference = calibration.get(f"reference_{suffix}")
        noise_scale = calibration.get(f"noise_scale_{suffix}")
        if not isinstance(reference, np.ndarray) or not isinstance(noise_scale, np.ndarray):
            return None
        if reference.size < chw.shape[0] or noise_scale.size < chw.shape[0]:
            return None

        depth = chw[0]
        depth_delta = depth - float(reference[0])
        depth_gate = self.tactile_noise_sigma * float(noise_scale[0])
        depth_residual = self._depth_response(depth_delta, depth_gate)
        if chw.shape[0] >= 3:
            dx = chw[1] - float(reference[1])
            dy = chw[2] - float(reference[2])
            dx_gate = self.tactile_noise_sigma * float(noise_scale[1])
            dy_gate = self.tactile_noise_sigma * float(noise_scale[2])
            dx = np.sign(dx) * np.maximum(np.abs(dx) - dx_gate, 0.0)
            dy = np.sign(dy) * np.maximum(np.abs(dy) - dy_gate, 0.0)
            shear = np.sqrt(dx * dx + dy * dy)
        elif chw.shape[0] >= 2:
            shear_delta = chw[1] - float(reference[1])
            shear_gate = self.tactile_noise_sigma * float(noise_scale[1])
            shear = np.maximum(np.abs(shear_delta) - shear_gate, 0.0)
        else:
            shear = np.zeros_like(depth_residual)
        threshold = float(calibration.get(f"threshold_{suffix}", calibration.get("threshold_raw", 0.0)))
        return self.tactile_depth_weight * depth_residual + self.tactile_shear_weight * shear, threshold

    @staticmethod
    def _robust_scale(centered: np.ndarray) -> float:
        finite = np.isfinite(centered)
        if not finite.any():
            return 0.0
        mad = float(np.nanmedian(np.abs(centered[finite])))
        return 1.4826 * mad

    def _baseline_residual(self, key: str, chw: np.ndarray) -> tuple[np.ndarray, bool]:
        baseline = self._tactile_baselines.get(key)
        count = self._tactile_baseline_counts.get(key, 0)
        if baseline is None or baseline.shape != chw.shape:
            self._tactile_baselines[key] = chw.copy()
            self._tactile_baseline_counts[key] = 1
            return np.zeros_like(chw), False
        if count < self.tactile_baseline_frames:
            next_count = count + 1
            baseline += (chw - baseline) / float(next_count)
            self._tactile_baseline_counts[key] = next_count
            return np.zeros_like(chw), False
        return chw - baseline, True

    def _make_tactile_intensity(self, frame: np.ndarray, key: str = "") -> np.ndarray:
        chw = self._as_chw(frame)
        residual, baseline_ready = (
            self._baseline_residual(key, chw) if key and self.tactile_use_startup_baseline else (chw, True)
        )
        depth = residual[0]
        if not baseline_ready:
            gray = np.zeros(chw.shape[1:], dtype=np.uint8)
            colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
            bgr = cv2.applyColorMap(gray, colormap)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.tactile_rotate_clockwise:
                rgb = np.rot90(rgb, k=-1)
            h, w = rgb.shape[:2]
            out_h = max(1, int(round(h * (self.tactile_width / float(w)))))
            return cv2.resize(rgb, (self.tactile_width, out_h), interpolation=cv2.INTER_NEAREST)

        calibrated = None if self.tactile_use_startup_baseline else self._make_calibrated_tactile_intensity(chw, key)
        threshold: float | None = None
        if calibrated is not None:
            intensity, threshold = calibrated
        else:
            use_baseline_delta = key and self.tactile_use_startup_baseline and baseline_ready
            depth_delta = depth if use_baseline_delta else depth - np.nanmedian(depth)
            depth_gate = self.tactile_noise_sigma * self._robust_scale(depth_delta - np.nanmedian(depth_delta))
            depth_residual = self._depth_response(depth_delta, depth_gate)
            if chw.shape[0] >= 3:
                dx = residual[1] if use_baseline_delta else residual[1] - np.nanmedian(residual[1])
                dy = residual[2] if use_baseline_delta else residual[2] - np.nanmedian(residual[2])
                dx_gate = self.tactile_noise_sigma * self._robust_scale(dx - np.nanmedian(dx))
                dy_gate = self.tactile_noise_sigma * self._robust_scale(dy - np.nanmedian(dy))
                dx = np.sign(dx) * np.maximum(np.abs(dx) - dx_gate, 0.0)
                dy = np.sign(dy) * np.maximum(np.abs(dy) - dy_gate, 0.0)
                shear = np.sqrt(dx * dx + dy * dy)
            elif chw.shape[0] >= 2:
                shear_delta = residual[1] if use_baseline_delta else residual[1] - np.nanmedian(residual[1])
                shear_gate = self.tactile_noise_sigma * self._robust_scale(shear_delta - np.nanmedian(shear_delta))
                shear = np.maximum(np.abs(shear_delta) - shear_gate, 0.0)
            else:
                shear = np.zeros_like(depth_residual)
            intensity = self.tactile_depth_weight * depth_residual + self.tactile_shear_weight * shear
        finite = np.isfinite(intensity)
        if not finite.any():
            gray = np.zeros(depth.shape, dtype=np.uint8)
        else:
            intensity = np.where(finite, intensity, 0.0)
            threshold = threshold if threshold is not None else self.tactile_thresholds.get(key)
            if threshold is None:
                threshold = self.tactile_thresholds.get(self._normalize_tactile_key(key))
            if threshold is not None:
                threshold = float(threshold) * self.tactile_threshold_scale
                intensity = np.where(intensity >= threshold, intensity - threshold, 0.0)
            if self.tactile_min_active_pixels > 0 and int(np.count_nonzero(intensity > 0.0)) < self.tactile_min_active_pixels:
                intensity = np.zeros_like(intensity)
            active = intensity[intensity > 0.0]
            active_ref = float(np.percentile(active, self.tactile_percentile)) if active.size else 0.0
            min_signal = self.tactile_min_signal
            if threshold is not None and threshold < 1.0:
                min_signal = min(min_signal, max(threshold * 0.25, 1e-4))
            if active_ref < min_signal:
                intensity = np.zeros_like(intensity)
                active_ref = 0.0
            if key and self.tactile_baseline_update_alpha > 0.0 and active_ref <= 0.0:
                baseline = self._tactile_baselines.get(key)
                if baseline is not None and baseline.shape == chw.shape:
                    baseline += self.tactile_baseline_update_alpha * (chw - baseline)
            vmax = max(active_ref, self.tactile_display_floor)
            if vmax <= 1e-6:
                gray = np.zeros(depth.shape, dtype=np.uint8)
            else:
                gray = np.clip(intensity / vmax * 255.0, 0.0, 255.0).astype(np.uint8)
                if self.tactile_median_blur >= 3:
                    ksize = self.tactile_median_blur | 1
                    gray = cv2.medianBlur(gray, ksize)

        colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        bgr = cv2.applyColorMap(gray, colormap)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.tactile_rotate_clockwise:
            rgb = np.rot90(rgb, k=-1)
        h, w = rgb.shape[:2]
        out_h = max(1, int(round(h * (self.tactile_width / float(w)))))
        return cv2.resize(rgb, (self.tactile_width, out_h), interpolation=cv2.INTER_NEAREST)

    def _make_tactile_rgb_passthrough(self, frame: np.ndarray) -> np.ndarray:
        rgb = self._to_uint8_rgb(frame)
        if self.tactile_rotate_clockwise:
            rgb = np.rot90(rgb, k=-1)
        h, w = rgb.shape[:2]
        out_h = max(1, int(round(h * (self.tactile_width / float(w)))))
        return cv2.resize(rgb, (self.tactile_width, out_h), interpolation=cv2.INTER_NEAREST)

    def submit(self, obs: dict[str, Any]) -> None:
        if not self.enabled or self._disabled_after_error or self._stop.is_set():
            return
        now = time.perf_counter()
        if now - self._last_submit_s < self._interval_s:
            return
        self._last_submit_s = now

        # Keep the capture thread cheap: only grab references to the latest
        # frames and drop them into the single-slot queue. All resizing,
        # pseudo-coloring and cv2.imshow work happens in the preview thread.
        item: dict[str, np.ndarray] = {}
        if self.rgb_enabled:
            for key in self.keys:
                frame = obs.get(key)
                if not isinstance(frame, np.ndarray):
                    continue
                item[key] = frame
        if self.tactile_enabled:
            for key in self.tactile_keys:
                frame = obs.get(key)
                if not isinstance(frame, np.ndarray):
                    continue
                item[key] = frame

        if not item:
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass

    def _prepare_item(self, raw_item: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        item: dict[str, np.ndarray] = {}
        if self.rgb_enabled:
            for key in self.keys:
                frame = raw_item.get(key)
                if not isinstance(frame, np.ndarray):
                    continue
                try:
                    item[key] = self._make_small(frame)
                except Exception as exc:
                    logging.debug("RGB preview skipped %s: %s", key, exc)
        if self.tactile_enabled:
            for key in self.tactile_keys:
                frame = raw_item.get(key)
                if not isinstance(frame, np.ndarray):
                    continue
                try:
                    if (
                        self.tactile_rgb_passthrough
                        and frame.dtype == np.uint8
                        and frame.ndim == 3
                        and frame.shape[-1] in {3, 4}
                    ):
                        item[key] = self._make_tactile_rgb_passthrough(frame)
                    else:
                        item[key] = self._make_tactile_intensity(frame, key)
                except Exception as exc:
                    logging.debug("Tactile preview skipped %s: %s", key, exc)
        return item

    @staticmethod
    def _display_label(key: str) -> str:
        labels = {
            "left_cam_left_wrist": "wrist-left",
            "cam_d405_color": "d405",
            "right_cam_right_wrist": "wrist-right",
            "depth_deformation.tactile_left_left": "left-left",
            "depth_deformation.tactile_left_right": "left-right",
            "depth_deformation.tactile_right_left": "right-left",
            "depth_deformation.tactile_right_right": "right-right",
        }
        return labels.get(key, key)

    @staticmethod
    def _pad_row_width(row: np.ndarray, width: int) -> np.ndarray:
        if row.shape[1] >= width:
            return row
        left = (width - row.shape[1]) // 2
        right = width - row.shape[1] - left
        return np.pad(row, ((0, 0), (left, right), (0, 0)), mode="constant")

    def _compose_row(self, keys: tuple[str, ...], item: dict[str, np.ndarray], default_width: int) -> np.ndarray:
        panels: list[np.ndarray] = []
        label_h = 24
        row_images = [item[key] for key in keys if key in item]
        max_h = max((img.shape[0] for img in row_images), default=1)
        for key in keys:
            img = item.get(key)
            if img is None:
                img = np.zeros((max_h, default_width, 3), dtype=np.uint8)
            if img.shape[0] < max_h:
                pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                img = np.concatenate([img, pad], axis=0)
            panel = np.zeros((max_h + label_h, img.shape[1], 3), dtype=np.uint8)
            panel[label_h:, :, :] = img
            cv2.putText(
                panel,
                self._display_label(key),
                (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)
        return np.concatenate(panels, axis=1) if panels else np.zeros((label_h + 1, default_width, 3), dtype=np.uint8)

    def _compose(self, item: dict[str, np.ndarray]) -> np.ndarray:
        rows: list[np.ndarray] = []
        if self.rgb_enabled:
            rows.append(self._compose_row(tuple(self.keys), item, self.width))
        if self.tactile_enabled:
            rows.append(self._compose_row(tuple(self.tactile_keys), item, self.tactile_width))
        if not rows:
            return np.zeros((25, self.width, 3), dtype=np.uint8)
        canvas_w = max(row.shape[1] for row in rows)
        rows = [self._pad_row_width(row, canvas_w) for row in rows]
        return np.concatenate(rows, axis=0)

    def _show_canvas(self, canvas_bgr: np.ndarray) -> None:
        if not self._window_initialized:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_initialized = True
        if self.window_scale > 0:
            h, w = canvas_bgr.shape[:2]
            target_size = (max(1, int(round(w * self.window_scale))), max(1, int(round(h * self.window_scale))))
            if target_size != self._last_window_size:
                cv2.resizeWindow(self.window_name, target_size[0], target_size[1])
                self._last_window_size = target_size
        cv2.imshow(self.window_name, canvas_bgr)

    def _send_udp_item(self, item: dict[str, np.ndarray]) -> None:
        if self._udp_socket is None or self.udp_port <= 0:
            return
        now = time.time()
        for key, frame in item.items():
            try:
                rgb = self._to_uint8_rgb(frame)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.udp_jpeg_quality)],
                )
                if not ok:
                    continue
                header = {
                    "key": key,
                    "timestamp": now,
                    "height": int(rgb.shape[0]),
                    "width": int(rgb.shape[1]),
                    "encoding": "jpg",
                }
                header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                payload = b"KDPV1\n" + header_bytes + b"\n" + encoded.tobytes()
                if len(payload) > 65000:
                    logging.debug("Preview UDP frame skipped for %s; payload too large: %s bytes", key, len(payload))
                    continue
                self._udp_socket.sendto(payload, (self.udp_host, self.udp_port))
            except Exception as exc:
                logging.debug("Preview UDP send skipped for %s: %s", key, exc)

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    raw_item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                item = self._prepare_item(raw_item)
                if not item:
                    continue
                if self.udp_port > 0:
                    self._send_udp_item(item)
                if self.show_window:
                    canvas_rgb = self._compose(item)
                    canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
                    self._show_canvas(canvas_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {27, ord("q")}:
                        self._stop.set()
                        self.enabled = False
        except Exception as exc:
            self._disabled_after_error = True
            self.enabled = False
            logging.warning("Preview disabled after OpenCV window error: %s", exc)
        finally:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()
        self.enabled = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._udp_socket is not None:
            try:
                self._udp_socket.close()
            except Exception:
                pass
            self._udp_socket = None
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


def get_rgb_previewer() -> RGBPreviewer | None:
    global _RGB_PREVIEWER
    if not (env_bool("RGB_PREVIEW", False) or env_bool("TACTILE_PREVIEW", False)):
        return None
    if _RGB_PREVIEWER is None:
        _RGB_PREVIEWER = RGBPreviewer()
    return _RGB_PREVIEWER


def close_rgb_preview() -> None:
    global _RGB_PREVIEWER
    if _RGB_PREVIEWER is not None:
        _RGB_PREVIEWER.close()
        _RGB_PREVIEWER = None


def patch_rgb_preview(robot: Any) -> None:
    previewer = get_rgb_previewer()
    if previewer is None:
        return
    if getattr(robot, "_kd_tacmae_rgb_preview_patch", False):
        return

    original_get_observation = robot.get_observation

    def get_observation_with_rgb_preview() -> dict[str, Any]:
        obs = original_get_observation()
        try:
            previewer.submit(obs)
        except Exception as exc:
            logging.debug("Preview submit failed: %s", exc)
        return obs

    robot.get_observation = get_observation_with_rgb_preview
    robot._kd_tacmae_rgb_preview_patch = True


def patch_observation_source_profile(robot: Any) -> None:
    if not env_bool("CAPTURE_OBS_SOURCE_PROFILE", False):
        return
    if getattr(robot, "_kd_tacmae_obs_source_profile_patch", False):
        return

    warn_ms = env_float("CAPTURE_OBS_SOURCE_PROFILE_WARN_MS", 8.0)

    def wrap_method(owner: Any, attr: str, label: str) -> None:
        if owner is None or not hasattr(owner, attr):
            return
        marker = f"_kd_tacmae_obs_source_profile_{attr}"
        if getattr(owner, marker, False):
            return
        original = getattr(owner, attr)

        def timed_method(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            result = original(*args, **kwargs)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if dt_ms >= warn_ms:
                logging.warning("Observation source profile %s %.1fms", label, dt_ms)
            return result

        setattr(owner, attr, timed_method)
        setattr(owner, marker, True)

    wrap_method(getattr(robot, "left_arm", None), "get_observation", "left_arm.get_observation")
    wrap_method(getattr(robot, "right_arm", None), "get_observation", "right_arm.get_observation")
    wrap_method(getattr(robot, "x5_tactile", None), "read_images", "x5_tactile.read_images")
    wrap_method(getattr(robot, "tactile_sidecar", None), "read_images", "tactile_sidecar.read_images")

    for cam_name, camera in getattr(robot, "shared_cameras", {}).items():
        wrap_method(camera, "async_read", f"shared_camera.{cam_name}.async_read")

    robot._kd_tacmae_obs_source_profile_patch = True


def load_capture_app() -> Any:
    if not _LEROBOT_ROOT_VALUE:
        raise RuntimeError(
            "LEROBOT_ROOT is required; point it at the compatible LeRobot checkout"
        )
    if not APP_PATH.exists():
        raise FileNotFoundError(f"missing LeRobot capture app: {APP_PATH}")
    src = LEROBOT_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("lerobot_bi_x5_capture_app_force", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load capture app from {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def patch_make_robot(app: Any) -> None:
    def describe_arm_config(label: str, arm_cfg: Any) -> None:
        print(
            "[capture-app] "
            f"{label}: enabled={getattr(arm_cfg, 'enabled', None)} "
            f"follower={getattr(arm_cfg, 'follower_ip', None)}:{getattr(arm_cfg, 'follower_tcp_port', None)} "
            f"x5={getattr(arm_cfg, 'x5_ip', None)} "
            f"gripper={getattr(arm_cfg, 'gripper_server', None)} "
            f"force={getattr(arm_cfg, 'connect_force_sensor', None)} "
            f"force_key={getattr(arm_cfg, 'force_sensor_data_key', None)}",
            flush=True,
        )

    def make_robot(settings: Any) -> Any:
        connect_tactile_sidecar = env_bool("CONNECT_TACTILE_SIDECAR", False)
        connect_x5_tactile = env_bool("CONNECT_X5_TACTILE", not connect_tactile_sidecar)
        cfg = app.BiRealmanUGripperNotacNewConfig(
            send_action_enabled=settings.send_action_enabled,
            send_gripper_action_enabled=settings.send_gripper_action_enabled,
            connect_x5_tactile=connect_x5_tactile and not connect_tactile_sidecar,
            connect_tactile_sidecar=connect_tactile_sidecar,
            tactile_sidecar_host=os.environ.get("TACTILE_SIDECAR_HOST", "127.0.0.1").strip() or "127.0.0.1",
            tactile_sidecar_port=env_int("TACTILE_SIDECAR_PORT", 61300),
            tactile_sidecar_timeout_s=env_float("TACTILE_SIDECAR_TIMEOUT_S", 0.05),
            tactile_sidecar_connect_timeout_s=env_float("TACTILE_SIDECAR_CONNECT_TIMEOUT_S", 8.0),
            tactile_sidecar_required=env_bool("TACTILE_SIDECAR_REQUIRED", True),
            left_left_tactile_name="tactile_left_left",
            left_right_tactile_name="tactile_left_right",
            right_left_tactile_name="tactile_right_left",
            right_right_tactile_name="tactile_right_right",
            x5_tactile_transport=settings.x5_tactile_transport,
            x5_tactile_modalities=settings.x5_tactile_modalities,
            x5_tactile_required=True,
            tactile_width=settings.tactile_width,
            tactile_height=settings.tactile_height,
            tactile_raw_width=getattr(settings, "tactile_raw_width", 640),
            tactile_raw_height=getattr(settings, "tactile_raw_height", 480),
            x5_tactile_flux_mode=getattr(settings, "x5_tactile_mode", "standard"),
            x5_tactile_flux_pc_host=settings.x5_tactile_pc_host,
            x5_tactile_flux_max_fps=settings.x5_tactile_max_fps,
            x5_tactile_depth_min=settings.x5_depth_min,
            x5_tactile_depth_max=settings.x5_depth_max,
            x5_tactile_deformation_min=settings.x5_deformation_min,
            x5_tactile_deformation_max=settings.x5_deformation_max,
            x5_tactile_shear_min=settings.x5_shear_min,
            x5_tactile_shear_max=settings.x5_shear_max,
            connect_shared_cameras=settings.connect_shared_cameras,
        )
        if env_bool("CONNECT_D405_CAMERA", False):
            from lerobot.cameras.opencv import OpenCVCameraConfig

            d405_name = os.environ.get("D405_CAMERA_NAME", "cam_d405_color").strip() or "cam_d405_color"
            d405_device = os.environ.get("D405_CAMERA_DEVICE", "/dev/video4").strip() or "/dev/video4"
            d405_camera = OpenCVCameraConfig(
                index_or_path=d405_device,
                fps=env_int("D405_CAMERA_FPS", 30),
                width=env_int("D405_CAMERA_WIDTH", 640),
                height=env_int("D405_CAMERA_HEIGHT", 480),
                fourcc=env_optional_str("D405_CAMERA_FOURCC", "YUYV"),
                backend=env_cv2_backend("D405_CAMERA_BACKEND", "V4L2"),
                warmup_s=env_int("D405_CAMERA_WARMUP_S", 0),
            )
            cfg.shared_cameras = {**({} if not settings.connect_shared_cameras else cfg.shared_cameras), d405_name: d405_camera}
            cfg.connect_shared_cameras = True
        cfg.left_arm_config.connect_wrist_camera = settings.connect_wrist_camera
        cfg.right_arm_config.connect_wrist_camera = settings.connect_wrist_camera
        cfg.left_arm_config.wrist_cam_width = env_int("WRIST_CAM_WIDTH", cfg.left_arm_config.wrist_cam_width)
        cfg.right_arm_config.wrist_cam_width = env_int("WRIST_CAM_WIDTH", cfg.right_arm_config.wrist_cam_width)
        cfg.left_arm_config.wrist_cam_height = env_int("WRIST_CAM_HEIGHT", cfg.left_arm_config.wrist_cam_height)
        cfg.right_arm_config.wrist_cam_height = env_int("WRIST_CAM_HEIGHT", cfg.right_arm_config.wrist_cam_height)
        cfg.left_arm_config.wrist_cam_fps = env_int("WRIST_CAM_FPS", cfg.left_arm_config.wrist_cam_fps)
        cfg.right_arm_config.wrist_cam_fps = env_int("WRIST_CAM_FPS", cfg.right_arm_config.wrist_cam_fps)
        cfg.left_arm_config.connect_local_cameras = settings.connect_local_cameras
        cfg.right_arm_config.connect_local_cameras = settings.connect_local_cameras
        cfg.left_arm_config.connect_gripper = settings.connect_gripper
        cfg.right_arm_config.connect_gripper = settings.connect_gripper
        cfg.left_arm_config.gripper_speed = settings.gripper_speed
        cfg.right_arm_config.gripper_speed = settings.gripper_speed

        if settings.task_index == 0:
            cfg.left_arm_config.enabled = False
            cfg.right_arm_config.enabled = True
        elif settings.task_index == 1:
            cfg.left_arm_config.enabled = True
            cfg.right_arm_config.enabled = False
        else:
            cfg.left_arm_config.enabled = True
            cfg.right_arm_config.enabled = True

        cfg.left_arm_config.connect_force_sensor = env_bool("CONNECT_LEFT_FORCE_SENSOR", True)
        cfg.right_arm_config.connect_force_sensor = env_bool("CONNECT_RIGHT_FORCE_SENSOR", True)
        cfg.left_arm_config.force_sensor_required = env_bool("FORCE_SENSOR_REQUIRED", True)
        cfg.right_arm_config.force_sensor_required = env_bool("FORCE_SENSOR_REQUIRED", True)
        cfg.left_arm_config.force_sensor_data_key = os.environ.get("FORCE_SENSOR_DATA_KEY", "zero_force_data")
        cfg.right_arm_config.force_sensor_data_key = os.environ.get("FORCE_SENSOR_DATA_KEY", "zero_force_data")
        cfg.left_arm_config.force_sensor_read_hz = env_float("FORCE_SENSOR_READ_HZ", 100.0)
        cfg.right_arm_config.force_sensor_read_hz = env_float("FORCE_SENSOR_READ_HZ", 100.0)
        cfg.left_arm_config.force_sensor_connect_timeout_s = env_float("FORCE_SENSOR_CONNECT_TIMEOUT_S", 3.0)
        cfg.right_arm_config.force_sensor_connect_timeout_s = env_float("FORCE_SENSOR_CONNECT_TIMEOUT_S", 3.0)
        cfg.left_arm_config.force_sensor_clear_on_connect = env_bool("FORCE_SENSOR_CLEAR_ON_CONNECT", False)
        cfg.right_arm_config.force_sensor_clear_on_connect = env_bool("FORCE_SENSOR_CLEAR_ON_CONNECT", False)

        cfg.x5_tactile_flux_left_left_grpc_port = env_int("X5_TACTILE_FLUX_LEFT_LEFT_GRPC_PORT", cfg.x5_tactile_flux_left_left_grpc_port)
        cfg.x5_tactile_flux_left_right_grpc_port = env_int("X5_TACTILE_FLUX_LEFT_RIGHT_GRPC_PORT", cfg.x5_tactile_flux_left_right_grpc_port)
        cfg.x5_tactile_flux_right_left_grpc_port = env_int("X5_TACTILE_FLUX_RIGHT_LEFT_GRPC_PORT", cfg.x5_tactile_flux_right_left_grpc_port)
        cfg.x5_tactile_flux_right_right_grpc_port = env_int("X5_TACTILE_FLUX_RIGHT_RIGHT_GRPC_PORT", cfg.x5_tactile_flux_right_right_grpc_port)
        cfg.x5_tactile_flux_left_left_dev_id = env_int("X5_TACTILE_FLUX_LEFT_LEFT_DEV_ID", cfg.x5_tactile_flux_left_left_dev_id)
        cfg.x5_tactile_flux_left_right_dev_id = env_int("X5_TACTILE_FLUX_LEFT_RIGHT_DEV_ID", cfg.x5_tactile_flux_left_right_dev_id)
        cfg.x5_tactile_flux_right_left_dev_id = env_int("X5_TACTILE_FLUX_RIGHT_LEFT_DEV_ID", cfg.x5_tactile_flux_right_left_dev_id)
        cfg.x5_tactile_flux_right_right_dev_id = env_int("X5_TACTILE_FLUX_RIGHT_RIGHT_DEV_ID", cfg.x5_tactile_flux_right_right_dev_id)
        cfg.x5_tactile_flux_left_left_pc_port = env_int("X5_TACTILE_FLUX_LEFT_LEFT_PC_PORT", cfg.x5_tactile_flux_left_left_pc_port)
        cfg.x5_tactile_flux_left_right_pc_port = env_int("X5_TACTILE_FLUX_LEFT_RIGHT_PC_PORT", cfg.x5_tactile_flux_left_right_pc_port)
        cfg.x5_tactile_flux_right_left_pc_port = env_int("X5_TACTILE_FLUX_RIGHT_LEFT_PC_PORT", cfg.x5_tactile_flux_right_left_pc_port)
        cfg.x5_tactile_flux_right_right_pc_port = env_int("X5_TACTILE_FLUX_RIGHT_RIGHT_PC_PORT", cfg.x5_tactile_flux_right_right_pc_port)

        print(
            "[capture-app] "
            f"tactile source={'sidecar' if connect_tactile_sidecar else 'x5_flux'} "
            f"connect_x5_tactile={cfg.connect_x5_tactile} "
            f"connect_tactile_sidecar={cfg.connect_tactile_sidecar} "
            f"sidecar={cfg.tactile_sidecar_host}:{cfg.tactile_sidecar_port}",
            flush=True,
        )
        describe_arm_config("left_arm", cfg.left_arm_config)
        describe_arm_config("right_arm", cfg.right_arm_config)

        robot = app.BiRealmanUGripperNotacNew(cfg)
        _ACTIVE_ROBOTS.append(robot)
        patch_d405_shared_camera(robot)
        patch_wrist_undistort_crop(robot)
        patch_wrist_camera_require_new(robot)
        patch_x5_tactile_async_cache(robot)
        patch_tactile_stale_guard(robot)
        patch_sync_diagnostics(robot)
        patch_observation_source_profile(robot)
        patch_raw_spool_wrist_placeholders(robot)
        patch_rgb_preview(robot)
        return robot

    app.make_robot = make_robot


def patch_run_d405_cleanup(app: Any) -> None:
    original_run = app.run

    def run_with_d405_cleanup() -> int:
        try:
            return int(original_run())
        finally:
            cleanup_wrist_processed_caches()
            cleanup_tactile_read_caches()
            close_rgb_preview()
            cleanup_active_d405_cameras()

    app.run = run_with_d405_cleanup


class TactileStaleGuard:
    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self.enabled = env_bool("TACTILE_STALE_GUARD", True)
        self.abort = env_bool("TACTILE_STALE_ABORT", True)
        self.max_repeats = env_int("TACTILE_STALE_MAX_REPEATS", 90)
        self.min_abort_age_ms = env_float("TACTILE_STALE_MIN_ABORT_AGE_MS", 500.0)
        self.warn_every = env_int("TACTILE_STALE_WARN_EVERY", 30)
        self.min_checked_keys = env_int("TACTILE_STALE_MIN_KEYS", 1)
        self._previous: dict[str, np.ndarray] = {}
        self._repeat_counts: dict[str, int] = {}
        self._previous_markers: dict[str, tuple[int | None, int | None]] = {}

    def _debug_status_obj(self) -> dict[str, Any]:
        x5 = getattr(self.robot, "x5_tactile", None)
        getter = getattr(x5, "get_debug_status", None)
        if getter is not None:
            try:
                status = getter()
            except Exception:
                status = {}
            if isinstance(status, dict) and status:
                return status

        sidecar = getattr(self.robot, "tactile_sidecar", None)
        time_getter = getattr(sidecar, "get_last_update_times_perf", None)
        if time_getter is None:
            return {}
        try:
            update_times = time_getter()
        except Exception:
            return {}
        if not isinstance(update_times, dict):
            return {}
        now = time.perf_counter()
        status: dict[str, Any] = {}
        for key, update_time in update_times.items():
            try:
                update_perf = float(update_time)
            except Exception:
                continue
            if update_perf <= 0.0:
                continue
            stream_name = self._stream_name_from_key(str(key))
            status[stream_name] = {"age_ms": max(0.0, (now - update_perf) * 1000.0)}
        return status

    @staticmethod
    def _debug_status_text(status: dict[str, Any]) -> str:
        return f"; debug={status}" if status else ""

    @staticmethod
    def _stream_name_from_key(key: str) -> str:
        return key.rsplit("tactile_", 1)[-1]

    @staticmethod
    def _stream_marker(status: dict[str, Any]) -> tuple[int | None, int | None] | None:
        try:
            frame_count_raw = status.get("frame_count")
            last_fid_raw = status.get("last_fid")
            frame_count = int(frame_count_raw) if frame_count_raw is not None else None
            last_fid = int(last_fid_raw) if last_fid_raw is not None else None
        except Exception:
            return None
        if frame_count is None and last_fid is None:
            return None
        return frame_count, last_fid

    @staticmethod
    def _stream_age_ms(status: dict[str, Any]) -> float | None:
        try:
            age = status.get("age_ms")
            return float(age) if age is not None else None
        except Exception:
            return None

    @staticmethod
    def _marker_advanced(
        previous: tuple[int | None, int | None] | None,
        current: tuple[int | None, int | None] | None,
    ) -> bool:
        if previous is None or current is None:
            return False
        prev_count, prev_fid = previous
        cur_count, cur_fid = current
        if prev_count is not None and cur_count is not None and cur_count > prev_count:
            return True
        if prev_fid is not None and cur_fid is not None and cur_fid != prev_fid:
            return True
        return False

    @staticmethod
    def _is_tactile_key(key: str) -> bool:
        return key.startswith("depth_deformation.tactile_")

    def check(self, obs: dict[str, Any]) -> None:
        if not self.enabled:
            return

        stale_keys: list[str] = []
        checked = 0
        debug_status = self._debug_status_obj()
        debug_text = self._debug_status_text(debug_status)
        for key, value in obs.items():
            if not self._is_tactile_key(key):
                continue
            if not isinstance(value, np.ndarray):
                continue
            checked += 1
            previous = self._previous.get(key)
            stream_status = debug_status.get(self._stream_name_from_key(key), {})
            marker = self._stream_marker(stream_status) if isinstance(stream_status, dict) else None
            age_ms = self._stream_age_ms(stream_status) if isinstance(stream_status, dict) else None
            previous_marker = self._previous_markers.get(key)
            stream_advanced = self._marker_advanced(previous_marker, marker)
            value_repeated = previous is not None and value.shape == previous.shape and np.array_equal(value, previous)
            if value_repeated and not stream_advanced:
                count = self._repeat_counts.get(key, 0) + 1
            else:
                count = 0
            self._repeat_counts[key] = count
            self._previous[key] = value.copy()
            if marker is not None:
                self._previous_markers[key] = marker

            if count > 0 and self.warn_every > 0 and count % self.warn_every == 0:
                logging.warning(
                    "X5 tactile stream %s repeated exactly for %s frames%s",
                    key,
                    count,
                    debug_text,
                )
            if count >= self.max_repeats and (age_ms is None or age_ms >= self.min_abort_age_ms):
                stale_keys.append(f"{key}({count})")
            elif count == self.max_repeats:
                logging.warning(
                    "X5 tactile stream %s repeated for %s capture frames but source age is only %.1fms; treating as transient%s",
                    key,
                    count,
                    -1.0 if age_ms is None else age_ms,
                    debug_text,
                )

        if checked < self.min_checked_keys:
            message = (
                f"X5 tactile stale guard saw only {checked} tactile streams; "
                f"expected at least {self.min_checked_keys}"
            )
            if self.abort:
                raise RuntimeError(message)
            logging.warning(message)

        if stale_keys:
            message = (
                "X5 tactile streams are stale/repeating: "
                + ", ".join(stale_keys)
                + ". Stop recording and restart the X5/Flux tactile stream before collecting data."
                + debug_text
            )
            if self.abort:
                raise RuntimeError(message)
            logging.warning(message)


def patch_tactile_stale_guard(robot: Any) -> None:
    guard = TactileStaleGuard(robot)
    if not guard.enabled:
        return
    original_get_observation = robot.get_observation

    def get_observation_with_tactile_guard() -> dict[str, Any]:
        obs = original_get_observation()
        guard.check(obs)
        return obs

    robot.get_observation = get_observation_with_tactile_guard


def patch_parse_args(app: Any) -> None:
    original_parse_args = app.parse_args

    def parse_args() -> Any:
        original_task_index = os.environ.get("TASK_INDEX")
        original_argv = sys.argv[:]

        task_choice = original_task_index
        for idx, item in enumerate(sys.argv):
            if item == "--task-index" and idx + 1 < len(sys.argv):
                task_choice = sys.argv[idx + 1]
            elif item.startswith("--task-index="):
                task_choice = item.split("=", 1)[1]

        task_choice = str(task_choice if task_choice is not None else "0").strip().lower()
        is_bimanual = task_choice in {"2", "both", "bimanual", "bi"}
        if is_bimanual:
            os.environ["TASK_INDEX"] = "2"
            patched_argv = []
            skip_next = False
            for item in sys.argv:
                if skip_next:
                    patched_argv.append("2")
                    skip_next = False
                    continue
                if item == "--task-index":
                    patched_argv.append(item)
                    skip_next = True
                elif item.startswith("--task-index="):
                    patched_argv.append("--task-index=2")
                else:
                    patched_argv.append(item)
            sys.argv[:] = patched_argv

        try:
            settings = original_parse_args()
        finally:
            sys.argv[:] = original_argv
            if original_task_index is None:
                os.environ.pop("TASK_INDEX", None)
            else:
                os.environ["TASK_INDEX"] = original_task_index

        if task_choice in {"2", "both", "bimanual", "bi"}:
            settings.task_index = 2
        settings.task_right = os.environ.get("TASK_RIGHT", settings.task_right)
        settings.task_left = os.environ.get("TASK_LEFT", settings.task_left)
        settings.task_reserved_2 = os.environ.get(
            "TASK_BOTH",
            "Bimanual contact-rich manipulation",
        )
        settings.x5_tactile_modalities = os.environ.get(
            "X5_TACTILE_MODALITIES",
            settings.x5_tactile_modalities,
        )
        settings.x5_tactile_mode = env_choice("X5_TACTILE_MODE", "standard", {"standard", "high"})
        if settings.x5_tactile_mode == "high":
            default_width, default_height = 384, 288
        else:
            default_width, default_height = settings.tactile_width, settings.tactile_height
        settings.tactile_width = env_int("TACTILE_WIDTH", default_width)
        settings.tactile_height = env_int("TACTILE_HEIGHT", default_height)
        settings.tactile_raw_width = env_int("TACTILE_RAW_WIDTH", 640)
        settings.tactile_raw_height = env_int("TACTILE_RAW_HEIGHT", 480)
        settings.x5_tactile_max_fps = env_int(
            "X5_TACTILE_MAX_FPS",
            int(getattr(settings, "x5_tactile_max_fps", 120)),
        )
        return settings

    app.parse_args = parse_args


def clear_force_sensor_for_arm(arm: Any, side: str, *, required: bool) -> bool:
    if arm is None:
        return True
    cfg = getattr(arm, "config", None)
    if not bool(getattr(cfg, "enabled", True)):
        return True
    if not bool(getattr(cfg, "connect_force_sensor", False)):
        return True

    follower_arm = getattr(arm, "_follower_arm", None)
    if follower_arm is None:
        message = f"{side} force clear skipped: follower arm is not connected"
        if required:
            raise RuntimeError(message)
        logging.warning(message)
        return False

    try:
        ret = follower_arm.rm_clear_force_data()
    except Exception as exc:
        message = f"{side} force clear failed: {exc}"
        if required:
            raise RuntimeError(message) from exc
        logging.warning(message)
        return False

    if ret != 0:
        message = f"{side} force clear returned error code {ret}"
        if required:
            raise RuntimeError(message)
        logging.warning(message)
        return False

    logging.info("%s force sensor cleared for this episode", side)
    return True


def clear_force_sensors_for_episode(robot: Any, state: Any | None = None) -> None:
    required = env_bool("FORCE_SENSOR_REQUIRED", True)
    if state is not None:
        state.update(status="CALIBRATING", message="Clearing force sensors before episode")

    clear_force_sensor_for_arm(getattr(robot, "left_arm", None), "left", required=required)
    clear_force_sensor_for_arm(getattr(robot, "right_arm", None), "right", required=required)

    settle_s = env_float("FORCE_SENSOR_CLEAR_SETTLE_S", 0.2)
    if settle_s > 0:
        time.sleep(settle_s)


def reconnect_x5_tactile_for_episode(robot: Any, state: Any | None = None) -> None:
    x5_tactile = getattr(robot, "x5_tactile", None)
    if x5_tactile is None:
        return

    if state is not None:
        state.update(status="CALIBRATING", message="Reconnecting X5 tactile streams before episode")

    left_cfg = getattr(getattr(robot, "config", None), "left_arm_config", None)
    right_cfg = getattr(getattr(robot, "config", None), "right_arm_config", None)
    left_enabled = bool(getattr(left_cfg, "enabled", True))
    right_enabled = bool(getattr(right_cfg, "enabled", True))

    logging.info("Reconnecting X5 tactile streams before episode")
    try:
        x5_tactile.disconnect()
    except Exception as exc:
        logging.warning("X5 tactile disconnect before episode failed: %s", exc)
    time.sleep(env_float("TACTILE_RECONNECT_SETTLE_S", 0.3))
    x5_tactile.connect(left_enabled=left_enabled, right_enabled=right_enabled)


def x5_tactile_streams_healthy(robot: Any) -> bool:
    x5_tactile = getattr(robot, "x5_tactile", None)
    getter = getattr(x5_tactile, "get_debug_status", None)
    if getter is None:
        return False
    try:
        status = getter()
    except Exception as exc:
        logging.warning("X5 tactile health check failed: %s", exc)
        return False

    max_age_ms = env_float("TACTILE_AUTO_RECONNECT_MAX_AGE_MS", 1000.0)
    min_frames = env_int("TACTILE_AUTO_RECONNECT_MIN_FRAMES", 10)
    min_fps = env_float("TACTILE_AUTO_RECONNECT_MIN_FPS", 1.0)

    robot_cfg = getattr(robot, "config", None)
    enabled_sides = {
        side
        for side in ("left", "right")
        if bool(
            getattr(
                getattr(robot_cfg, f"{side}_arm_config", None),
                "enabled",
                True,
            )
        )
    }

    unhealthy = {}
    for name, item in status.items():
        # The dual-arm tactile container retains placeholder status entries for
        # disabled sides. They must not force a reconnect in single-arm mode.
        side = str(name).split("_", 1)[0]
        if side in {"left", "right"} and side not in enabled_sides:
            continue
        if not item.get("connected", False) or not item.get("has_frame", False):
            unhealthy[name] = item
            continue
        if int(item.get("frame_count") or 0) < min_frames:
            unhealthy[name] = item
            continue
        age_ms = item.get("age_ms")
        if age_ms is None or float(age_ms) > max_age_ms:
            unhealthy[name] = item
            continue
        if float(item.get("fps") or 0.0) < min_fps:
            unhealthy[name] = item

    if unhealthy:
        logging.info("X5 tactile auto reconnect needed; unhealthy=%s", unhealthy)
        return False
    logging.info("X5 tactile streams healthy; skip episode-start reconnect")
    return True


def prewarm_gripper_for_episode(robot: Any, teleop: Any | None = None, state: Any | None = None) -> None:
    if state is not None:
        state.update(status="CALIBRATING", message="Prewarming gripper action workers")

    reset_rate_limit = getattr(robot, "reset_gripper_rate_limit", None)
    if reset_rate_limit is not None:
        reset_rate_limit()

    ensure_worker = getattr(robot, "_ensure_async_gripper_worker", None)
    if ensure_worker is not None and env_bool("ASYNC_GRIPPER_ACTION_THREAD", True):
        for side in ("left_arm", "right_arm"):
            arm = getattr(robot, side, None)
            cfg = getattr(arm, "config", None)
            if arm is not None and bool(getattr(cfg, "enabled", True)):
                ensure_worker(arm)

    if not env_bool("GRIPPER_PREWARM_SEND_CURRENT_ACTION", False):
        return
    if teleop is None:
        return

    try:
        action = teleop.get_action()
        robot.send_action(action)
        logging.info("Sent one current gripper action before episode start")
    except Exception as exc:
        logging.warning("Prewarm gripper action send failed: %s", exc)


def write_raw_spool_sync_event(event: dict[str, Any], path: Path | None = None) -> None:
    path_text = str(path) if path is not None else os.environ.get("RAW_SPOOL_SYNC_PATH", "").strip()
    if not path_text:
        return
    try:
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "wall_time": time.time(),
            "perf_counter": time.perf_counter(),
            **event,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        logging.warning("Failed to write raw-spool sync event: %s", exc)


def raw_spool_base_dir() -> Path:
    default_root = Path(os.environ.get("DATASET_ROOT", "dataset/raw_spool_unknown")).with_name(
        Path(os.environ.get("DATASET_ROOT", "dataset/raw_spool_unknown")).name + "_wrist_raw"
    )
    return Path(os.environ.get("RAW_WRIST_OUTPUT_DIR", str(default_root))).resolve()


def raw_spool_episode_dir(episode_index: int) -> Path:
    return raw_spool_base_dir() / f"episode-{episode_index:06d}"


def start_raw_spool_episode_recorder(episode_index: int) -> tuple[subprocess.Popen[Any] | None, Path | None]:
    if not env_bool("RAW_SPOOL_PER_EPISODE", False):
        return None, None
    output_dir = raw_spool_episode_dir(episode_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    sync_path = output_dir / "main_capture_sync.jsonl"
    log_path = output_dir / "raw_wrist_recording.log"
    cmd = [
        sys.executable,
        "scripts/record_x5_wrist_raw_streams.py",
        "--output-dir",
        str(output_dir),
        "--left-ip",
        os.environ.get("RAW_WRIST_LEFT_IP", os.environ.get("LEFT_X5_IP", "192.168.1.10")),
        "--right-ip",
        os.environ.get("RAW_WRIST_RIGHT_IP", os.environ.get("RIGHT_X5_IP", "192.168.1.11")),
        "--grpc-port",
        os.environ.get("RAW_WRIST_GRPC_PORT", os.environ.get("FISH_CAMERA_GRPC_PORT", "50088")),
        "--left-udp-port",
        os.environ.get("RAW_WRIST_LEFT_UDP_PORT", "56210"),
        "--right-udp-port",
        os.environ.get("RAW_WRIST_RIGHT_UDP_PORT", "56220"),
        "--width",
        os.environ.get("RAW_WRIST_WIDTH", os.environ.get("WRIST_CAM_WIDTH", "1920")),
        "--height",
        os.environ.get("RAW_WRIST_HEIGHT", os.environ.get("WRIST_CAM_HEIGHT", "1080")),
        "--fps",
        os.environ.get("RAW_WRIST_FPS", os.environ.get("WRIST_CAM_FPS", "30")),
        "--codec",
        os.environ.get("RAW_WRIST_CODEC", os.environ.get("WRIST_CAM_CODEC", "HEVC")),
        "--device",
        os.environ.get("RAW_WRIST_DEVICE", os.environ.get("WRIST_CAM_DEVICE", "/dev/video4")),
        "--preview-host",
        os.environ.get("RAW_WRIST_PREVIEW_HOST", "127.0.0.1"),
        "--left-preview-udp-port",
        os.environ.get("RAW_WRIST_LEFT_PREVIEW_UDP_PORT", "0"),
        "--right-preview-udp-port",
        os.environ.get("RAW_WRIST_RIGHT_PREVIEW_UDP_PORT", "0"),
    ]
    cmd.append("--save-raw" if env_bool("RAW_WRIST_SAVE_RAW", True) else "--no-save-raw")
    if env_bool("RAW_WRIST_STREAM_PROCESSED", False):
        cmd.extend(
            [
                "--processed-host",
                os.environ.get("RAW_WRIST_PROCESSED_HOST", os.environ.get("RAW_WRIST_PREVIEW_HOST", "127.0.0.1")),
                "--left-processed-udp-port",
                os.environ.get("RAW_WRIST_PROCESSED_LEFT_UDP_PORT", "56510"),
                "--right-processed-udp-port",
                os.environ.get("RAW_WRIST_PROCESSED_RIGHT_UDP_PORT", "56520"),
            ]
        )
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    setattr(proc, "_kd_tacmae_log_file", log_file)
    wait_s = env_float("RAW_WRIST_EPISODE_START_WAIT_S", 3.0)
    time.sleep(max(0.0, wait_s))
    if proc.poll() is not None:
        try:
            log_file.close()
        except Exception:
            pass
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
        except Exception:
            pass
        raise RuntimeError(f"Raw wrist episode recorder failed during startup; log={log_path}\n{tail}")
    logging.info("Raw wrist episode recorder started: episode=%s dir=%s pid=%s", episode_index, output_dir, proc.pid)
    return proc, sync_path


def stop_raw_spool_episode_recorder(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        stop_timeout_s = env_float("RAW_WRIST_EPISODE_STOP_TIMEOUT_S", 15.0)
        try:
            proc.wait(timeout=stop_timeout_s)
        except subprocess.TimeoutExpired:
            logging.warning(
                "Raw wrist episode recorder did not stop within %.1fs; sending SIGKILL pid=%s",
                stop_timeout_s,
                proc.pid,
            )
            proc.kill()
            proc.wait(timeout=5.0)
    if proc.returncode not in (0, None):
        log_file_obj = getattr(proc, "_kd_tacmae_log_file", None)
        log_path = getattr(log_file_obj, "name", "")
        tail = ""
        if log_path:
            try:
                tail = "\n".join(Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
            except Exception:
                tail = ""
        logging.warning(
            "Raw wrist episode recorder exited with non-zero returncode=%s pid=%s log=%s%s",
            proc.returncode,
            proc.pid,
            log_path or "<unknown>",
            f"\n{tail}" if tail else "",
        )
    log_file = getattr(proc, "_kd_tacmae_log_file", None)
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass
    logging.info("Raw wrist episode recorder stopped: pid=%s returncode=%s", proc.pid, proc.returncode)


def raw_spool_data_parquet_ready(dataset_root: Path, *, timeout_s: float, poll_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_error = ""
    while True:
        paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
        if paths:
            try:
                import pyarrow.parquet as pq  # type: ignore[import-not-found]

                for path in paths:
                    pq.ParquetFile(path).metadata
                return True
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        else:
            last_error = "no data parquet found yet"

        if time.monotonic() >= deadline:
            logging.info("Raw-spool save finalize deferred; data parquet is not ready (%s)", last_error)
            return False
        time.sleep(max(0.01, poll_s))


def _raw_spool_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _raw_spool_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_raw_spool_jsonable(v) for v in value]
    return value


def _raw_spool_scalar(value: Any) -> Any:
    value = _raw_spool_jsonable(value)
    if isinstance(value, list) and value:
        return _raw_spool_scalar(value[0])
    return value


def raw_spool_episode_hint_path(episode_index: int) -> Path:
    return raw_spool_episode_dir(episode_index) / "episode_hint.json"


def raw_spool_finalize_result_path(episode_index: int) -> Path:
    return raw_spool_episode_dir(episode_index) / "finalize_result.json"


def write_raw_spool_episode_hint(dataset: Any, episode_index: int) -> Path | None:
    raw_dir = raw_spool_episode_dir(episode_index)
    if not raw_dir.exists():
        return None
    latest_meta = getattr(getattr(dataset, "meta", None), "latest_episode", None)
    latest_writer = getattr(getattr(dataset, "writer", None), "_latest_episode", None)
    row: dict[str, Any] = {}
    if isinstance(latest_writer, dict):
        row.update(_raw_spool_jsonable(latest_writer))
    if isinstance(latest_meta, dict):
        row.update(_raw_spool_jsonable(latest_meta))
    if not row:
        return None
    row["episode_index"] = int(episode_index)
    if "length" not in row and "index" in row:
        try:
            row["length"] = len(row["index"])
        except Exception:
            pass
    hint_path = raw_spool_episode_hint_path(episode_index)
    hint_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return hint_path


def raw_spool_raw_outputs_valid(raw_dir: Path) -> tuple[bool, str]:
    min_bytes = env_int("RAW_WRIST_MIN_RAW_BYTES", 65536)
    missing_or_small: list[str] = []
    for side in ("left", "right"):
        candidates = [raw_dir / f"{side}_wrist_raw.ts", raw_dir / f"{side}_wrist_raw.mkv"]
        existing = [path for path in candidates if path.exists()]
        if not existing:
            missing_or_small.append(f"{side}:missing")
            continue
        size = max(path.stat().st_size for path in existing)
        if size < min_bytes:
            missing_or_small.append(f"{side}:{size}B<min{min_bytes}B")
    if missing_or_small:
        return False, ", ".join(missing_or_small)
    return True, "ok"


def raw_spool_processed_result_valid(raw_dir: Path) -> tuple[bool, str]:
    result_path = raw_dir / "finalize_result.json"
    if not result_path.exists():
        return False, f"missing {result_path.name}"
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        side_results = payload.get("side_results", {})
        missing: list[str] = []
        for side in ("left", "right"):
            results = side_results.get(side) or {}
            if not results:
                missing.append(f"{side}:missing_result")
                continue
            first = next(iter(results.values()))
            output_path = Path(str(first.get("output_path", "")))
            frames = int(first.get("frames", 0))
            if frames <= 0:
                missing.append(f"{side}:frames={frames}")
            if not output_path.exists() or output_path.stat().st_size <= 0:
                missing.append(f"{side}:missing_video={output_path}")
        if missing:
            return False, ", ".join(missing)
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def wait_raw_spool_processed_result_ready(episode_index: int) -> bool:
    raw_dir = raw_spool_episode_dir(episode_index)
    timeout_s = env_float("RAW_WRIST_STREAM_PROCESSED_RESULT_TIMEOUT_S", 8.0)
    poll_s = env_float("RAW_WRIST_STREAM_PROCESSED_RESULT_POLL_S", 0.05)
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_message = ""
    while True:
        valid, message = raw_spool_processed_result_valid(raw_dir)
        if valid:
            logging.info("Processed wrist result ready for episode %s: %s", episode_index, raw_dir / "finalize_result.json")
            return True
        last_message = message
        if time.monotonic() >= deadline:
            logging.warning(
                "Processed wrist result not ready for episode %s after %.1fs: %s",
                episode_index,
                timeout_s,
                last_message,
            )
            return False
        time.sleep(max(0.01, poll_s))


def prepare_raw_spool_finalize_command(
    dataset: Any,
    episode_index: int,
    *,
    during_dataset_finalize: bool = False,
    skip_metadata: bool = False,
) -> tuple[list[str], dict[str, str], Path, bool] | None:
    if not env_bool("RAW_SPOOL_PER_EPISODE", False):
        return None
    if not env_bool("RAW_SPOOL_FINALIZE_ON_SAVE", True):
        return None
    dataset_root_value = getattr(getattr(dataset, "meta", None), "root", None) or os.environ.get("DATASET_ROOT", "")
    dataset_root = Path(str(dataset_root_value)) if dataset_root_value else Path()
    if not str(dataset_root):
        raise RuntimeError("Cannot infer DATASET_ROOT for raw-spool finalize")
    episode_meta_ready = bool(list((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")))
    if not episode_meta_ready and env_bool("RAW_SPOOL_REQUIRE_EPISODE_META_FOR_SAVE_FINALIZE", True):
        logging.info(
            "Raw-spool episode %s finalize deferred until dataset.finalize(); episode metadata is not ready",
            episode_index,
        )
        return None
    raw_dir = raw_spool_episode_dir(episode_index)
    if not raw_dir.exists():
        logging.info("Raw-spool episode %s finalize skipped; raw dir does not exist: %s", episode_index, raw_dir)
        return None
    result_path = raw_spool_finalize_result_path(episode_index)
    result_valid, result_message = raw_spool_processed_result_valid(raw_dir)
    if result_path.exists() and result_valid:
        cmd = [
            sys.executable,
            "scripts/finalize_raw_spool_wrist_videos.py",
            "--dataset-root",
            str(dataset_root),
            "--raw-wrist-dir",
            str(raw_dir),
            "--episode-index",
            str(episode_index),
            "--result-json",
            str(result_path),
            "--metadata-only",
        ]
        env = os.environ.copy()
        logging.info("Raw-spool episode %s updating wrist metadata from cached processed result: %s", episode_index, result_path)
        return cmd, env, raw_dir, episode_meta_ready
    if result_path.exists() and not result_valid:
        logging.warning("Raw-spool episode %s cached processed result is not usable yet: %s", episode_index, result_message)
    if not env_bool("RAW_WRIST_SAVE_RAW", True):
        raise RuntimeError(
            f"Raw-spool wrist raw saving is disabled and processed result is not usable for episode {episode_index}: "
            f"{result_message}; raw_dir={raw_dir}"
        )
    valid, valid_message = raw_spool_raw_outputs_valid(raw_dir)
    if not valid:
        raise RuntimeError(f"Raw-spool wrist raw output is invalid for episode {episode_index}: {valid_message}; raw_dir={raw_dir}")
    hint_path = raw_spool_episode_hint_path(episode_index)
    if not during_dataset_finalize and not episode_meta_ready:
        if not hint_path.exists():
            ready = raw_spool_data_parquet_ready(
                dataset_root,
                timeout_s=env_float("RAW_SPOOL_SAVE_PARQUET_READY_TIMEOUT_S", 1.0),
                poll_s=env_float("RAW_SPOOL_SAVE_PARQUET_READY_POLL_S", 0.1),
            )
            if not ready:
                return
        else:
            logging.info(
                "Raw-spool episode %s using save-time episode hint for wrist finalization: %s",
                episode_index,
                hint_path,
            )
    if not episode_meta_ready and not hint_path.exists():
        ready = raw_spool_data_parquet_ready(
            dataset_root,
            timeout_s=env_float("RAW_SPOOL_SAVE_PARQUET_READY_TIMEOUT_S", 1.0),
            poll_s=env_float("RAW_SPOOL_SAVE_PARQUET_READY_POLL_S", 0.1),
        )
        if not ready:
            return None
    cmd = [
        sys.executable,
        "scripts/finalize_raw_spool_wrist_videos.py",
        "--dataset-root",
        str(dataset_root),
        "--raw-wrist-dir",
        str(raw_dir),
        "--episode-index",
        str(episode_index),
        "--fps",
        os.environ.get("FPS", "30"),
        "--crop-size",
        os.environ.get("WRIST_UNDISTORT_CROP_SIZE", "896"),
        "--output-size",
        os.environ.get("WRIST_UNDISTORT_CROP_SIZE", "896"),
        "--balance",
        os.environ.get("WRIST_UNDISTORT_BALANCE", "0.0"),
        "--crf",
        os.environ.get("RAW_SPOOL_FINALIZE_CRF", "20"),
        "--vcodec",
        os.environ.get("RAW_SPOOL_FINALIZE_VCODEC", "libx264"),
        "--preset",
        os.environ.get("RAW_SPOOL_FINALIZE_PRESET", "veryfast"),
        "--cq",
        os.environ.get("RAW_SPOOL_FINALIZE_CQ", "23"),
        "--pix-fmt",
        os.environ.get("RAW_SPOOL_FINALIZE_PIX_FMT", "yuv420p"),
    ]
    if hint_path.exists():
        cmd.extend(["--episode-hint", str(hint_path)])
    cmd.extend(["--result-json", str(result_path)])
    if skip_metadata:
        cmd.append("--skip-metadata")
    logging.info("Finalizing raw wrist episode %s into dataset %s", episode_index, dataset_root)
    if not episode_meta_ready:
        logging.info(
            "Raw-spool episode %s metadata is not ready; wrist videos will be generated now, "
            "episode span will be inferred from data parquet",
            episode_index,
        )
    env = os.environ.copy()
    if not during_dataset_finalize:
        env["RAW_SPOOL_PARQUET_READY_TIMEOUT_S"] = os.environ.get("RAW_SPOOL_SAVE_PARQUET_READY_TIMEOUT_S", "1.0")
        env["RAW_SPOOL_PARQUET_READY_POLL_S"] = os.environ.get("RAW_SPOOL_SAVE_PARQUET_READY_POLL_S", "0.1")
    return cmd, env, raw_dir, episode_meta_ready


def run_raw_spool_finalize_command(
    *,
    cmd: list[str],
    env: dict[str, str],
    raw_dir: Path,
    episode_index: int,
    episode_meta_ready: bool,
    delete_raw_after_finalize: bool,
) -> None:
    nice_value = env_int("RAW_SPOOL_ASYNC_NICE", 10)
    preexec_fn = None
    if nice_value != 0 and hasattr(os, "nice"):
        def _nice_child() -> None:
            try:
                os.nice(nice_value)
            except Exception:
                pass

        preexec_fn = _nice_child
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True, env=env, preexec_fn=preexec_fn)
    if delete_raw_after_finalize:
        if not episode_meta_ready:
            logging.info(
                "Raw-spool episode %s finalized before episode metadata was ready; "
                "keeping wrist cache so dataset.finalize() can finish metadata later",
                episode_index,
            )
        else:
            shutil.rmtree(raw_dir, ignore_errors=True)


def finalize_raw_spool_episode(
    dataset: Any,
    episode_index: int,
    *,
    during_dataset_finalize: bool = False,
    skip_metadata: bool = False,
) -> None:
    prepared = prepare_raw_spool_finalize_command(
        dataset,
        episode_index,
        during_dataset_finalize=during_dataset_finalize,
        skip_metadata=skip_metadata,
    )
    if prepared is None:
        return
    cmd, env, raw_dir, episode_meta_ready = prepared
    run_raw_spool_finalize_command(
        cmd=cmd,
        env=env,
        raw_dir=raw_dir,
        episode_index=episode_index,
        episode_meta_ready=episode_meta_ready,
        delete_raw_after_finalize=env_bool("RAW_SPOOL_DELETE_RAW_AFTER_FINALIZE", False),
    )


def _raw_spool_finalize_worker_loop() -> None:
    assert _RAW_SPOOL_FINALIZE_QUEUE is not None
    while True:
        job = _RAW_SPOOL_FINALIZE_QUEUE.get()
        try:
            if job is None:
                return
            episode_index = int(job["episode_index"])
            logging.info("Raw-spool async wrist finalize started: episode=%s", episode_index)
            run_raw_spool_finalize_command(
                cmd=job["cmd"],
                env=job["env"],
                raw_dir=job["raw_dir"],
                episode_index=episode_index,
                episode_meta_ready=bool(job["episode_meta_ready"]),
                delete_raw_after_finalize=bool(job["delete_raw_after_finalize"]),
            )
            logging.info("Raw-spool async wrist finalize finished: episode=%s", episode_index)
        except Exception as exc:
            message = f"Raw-spool async wrist finalize failed: {exc}"
            logging.exception(message)
            _RAW_SPOOL_FINALIZE_ERRORS.append(message)
        finally:
            _RAW_SPOOL_FINALIZE_QUEUE.task_done()


def ensure_raw_spool_finalize_worker() -> queue.Queue[dict[str, Any] | None]:
    global _RAW_SPOOL_FINALIZE_QUEUE, _RAW_SPOOL_FINALIZE_WORKER
    with _RAW_SPOOL_FINALIZE_LOCK:
        if _RAW_SPOOL_FINALIZE_QUEUE is None:
            _RAW_SPOOL_FINALIZE_QUEUE = queue.Queue()
        if _RAW_SPOOL_FINALIZE_WORKER is None or not _RAW_SPOOL_FINALIZE_WORKER.is_alive():
            _RAW_SPOOL_FINALIZE_WORKER = threading.Thread(
                target=_raw_spool_finalize_worker_loop,
                name="raw_spool_finalize_worker",
                daemon=True,
            )
            _RAW_SPOOL_FINALIZE_WORKER.start()
        return _RAW_SPOOL_FINALIZE_QUEUE


def enqueue_raw_spool_finalize_episode(dataset: Any, episode_index: int) -> None:
    prepared = prepare_raw_spool_finalize_command(
        dataset,
        episode_index,
        during_dataset_finalize=False,
        skip_metadata=env_bool("RAW_SPOOL_ASYNC_SKIP_METADATA", True),
    )
    if prepared is None:
        return
    cmd, env, raw_dir, episode_meta_ready = prepared
    q = ensure_raw_spool_finalize_worker()
    q.put(
        {
            "cmd": cmd,
            "env": env,
            "raw_dir": raw_dir,
            "episode_index": int(episode_index),
            "episode_meta_ready": bool(episode_meta_ready),
            "delete_raw_after_finalize": (
                env_bool("RAW_SPOOL_DELETE_RAW_AFTER_FINALIZE", False)
                and not env_bool("RAW_SPOOL_ASYNC_SKIP_METADATA", True)
            ),
        }
    )
    logging.info("Raw-spool async wrist finalize queued: episode=%s raw_dir=%s", episode_index, raw_dir)


def wait_raw_spool_async_finalizers() -> None:
    q = _RAW_SPOOL_FINALIZE_QUEUE
    if q is None:
        return
    pending = q.qsize()
    if pending > 0:
        logging.info("Waiting for %s pending raw-spool async wrist finalize job(s)", pending)
    q.join()
    if _RAW_SPOOL_FINALIZE_ERRORS and env_bool("RAW_SPOOL_ASYNC_FAIL_ON_ERROR", True):
        raise RuntimeError("; ".join(_RAW_SPOOL_FINALIZE_ERRORS[-5:]))


def finalize_all_raw_spool_episodes(dataset: Any) -> None:
    if not env_bool("RAW_SPOOL_PER_EPISODE", False):
        return
    if not env_bool("RAW_SPOOL_FINALIZE_ON_SAVE", True):
        return
    base_dir = raw_spool_base_dir()
    if not base_dir.exists():
        return
    episode_dirs = sorted(p for p in base_dir.glob("episode-*") if p.is_dir())
    if not episode_dirs:
        if env_bool("RAW_SPOOL_DELETE_RAW_AFTER_FINALIZE", False):
            logging.info("Raw-spool no pending episode raw dirs; deleting raw wrist root: %s", base_dir)
            shutil.rmtree(base_dir, ignore_errors=True)
        return
    logging.info("Raw-spool dataset finalize hook: found %s pending wrist episode cache(s)", len(episode_dirs))
    failed_episodes: list[int] = []
    for raw_dir in episode_dirs:
        try:
            episode_index = int(raw_dir.name.split("-")[-1])
        except ValueError:
            logging.warning("Raw-spool finalize skipped unexpected directory: %s", raw_dir)
            failed_episodes.append(-1)
            continue
        try:
            finalize_raw_spool_episode(dataset, episode_index, during_dataset_finalize=True)
        except Exception as exc:
            failed_episodes.append(episode_index)
            logging.exception("Raw-spool episode %s finalize failed; raw cache kept for retry: %s", episode_index, exc)
    if env_bool("RAW_SPOOL_DELETE_RAW_AFTER_FINALIZE", False):
        remaining_episode_dirs = [p for p in base_dir.glob("episode-*") if p.is_dir()]
        if not failed_episodes and not remaining_episode_dirs:
            logging.info("Raw-spool finalize succeeded; deleting raw wrist root: %s", base_dir)
            shutil.rmtree(base_dir, ignore_errors=True)
        else:
            logging.info(
                "Raw-spool raw root kept for retry/debug: failed=%s remaining_episode_dirs=%s root=%s",
                failed_episodes,
                [p.name for p in remaining_episode_dirs],
                base_dir,
            )


def patch_dataset_save_episode_for_raw_spool(dataset: Any) -> None:
    if not env_bool("RAW_SPOOL_PER_EPISODE", False):
        return
    dataset_id = id(dataset)
    if dataset_id in _RAW_SPOOL_SAVE_PATCHED_DATASETS:
        return
    original_save_episode = getattr(dataset, "save_episode", None)
    if not callable(original_save_episode):
        logging.warning("Raw-spool save hook skipped: dataset has no save_episode")
        return

    def save_episode_with_raw_spool_finalize(*args: Any, **kwargs: Any) -> Any:
        result = original_save_episode(*args, **kwargs)
        try:
            if _RAW_SPOOL_PENDING_FINISHED_EPISODES:
                episode_index = _RAW_SPOOL_PENDING_FINISHED_EPISODES.pop(0)
            else:
                total_episodes = int(getattr(getattr(dataset, "meta", None), "total_episodes", 0) or 0)
                episode_index = total_episodes - 1
            if episode_index >= 0:
                write_raw_spool_episode_hint(dataset, episode_index)
                if env_bool("RAW_SPOOL_FINALIZE_ASYNC", True):
                    enqueue_raw_spool_finalize_episode(dataset, episode_index)
                else:
                    finalize_raw_spool_episode(dataset, episode_index)
        except subprocess.CalledProcessError as exc:
            logging.warning("Raw-spool episode finalize deferred; raw cache kept for retry: %s", exc)
        except Exception as exc:
            logging.exception("Raw-spool episode finalize failed; raw cache kept for retry: %s", exc)
        return result

    dataset.save_episode = save_episode_with_raw_spool_finalize
    _RAW_SPOOL_SAVE_PATCHED_DATASETS.add(dataset_id)
    logging.info("Raw-spool save_episode hook enabled")


def patch_dataset_finalize_for_raw_spool(dataset: Any) -> None:
    if not env_bool("RAW_SPOOL_PER_EPISODE", False):
        return
    dataset_id = id(dataset)
    if dataset_id in _RAW_SPOOL_FINALIZE_PATCHED_DATASETS:
        return
    original_finalize = getattr(dataset, "finalize", None)
    if not callable(original_finalize):
        logging.warning("Raw-spool finalize hook skipped: dataset has no finalize")
        return

    def finalize_with_raw_spool(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_finalize(*args, **kwargs)
        original_dt = time.perf_counter() - t0
        logging.info("LeRobot dataset.finalize() finished in %.2fs before raw-spool exit handling", original_dt)
        if not env_bool("RAW_SPOOL_FINALIZE_ON_EXIT", True):
            dataset_root_value = getattr(getattr(dataset, "meta", None), "root", None) or os.environ.get("DATASET_ROOT", "")
            dataset_root = Path(str(dataset_root_value)) if dataset_root_value else Path()
            base_dir = raw_spool_base_dir()
            pending = 0
            if _RAW_SPOOL_FINALIZE_QUEUE is not None:
                pending = _RAW_SPOOL_FINALIZE_QUEUE.qsize()
            logging.info(
                "Raw-spool finalize-on-exit disabled; skipped async wait/pending wrist finalization "
                "(pending_async_jobs=%s, raw_root=%s). Finalize later with: "
                "conda run --no-capture-output -n lerobot51 python scripts/finalize_raw_spool_dataset.py "
                "--dataset-root %s --raw-wrist-root %s --delete-raw",
                pending,
                base_dir,
                dataset_root,
                base_dir,
            )
            return result
        try:
            t1 = time.perf_counter()
            wait_raw_spool_async_finalizers()
            wait_dt = time.perf_counter() - t1
            if wait_dt > 0.1:
                logging.info("Raw-spool async wrist finalize wait finished in %.2fs", wait_dt)
            t2 = time.perf_counter()
            finalize_all_raw_spool_episodes(dataset)
            raw_dt = time.perf_counter() - t2
            if raw_dt > 0.1:
                logging.info("Raw-spool pending wrist finalize/metadata hook finished in %.2fs", raw_dt)
        except Exception as exc:
            logging.exception("Raw-spool dataset finalize hook failed; raw cache kept for retry: %s", exc)
        return result

    dataset.finalize = finalize_with_raw_spool
    _RAW_SPOOL_FINALIZE_PATCHED_DATASETS.add(dataset_id)
    logging.info("Raw-spool dataset finalize hook enabled")


def patch_capture_episode(app: Any) -> None:
    def capture_episode_realtime_guarded(**kwargs: Any) -> str:
        global _RAW_SPOOL_NEXT_EPISODE_INDEX
        robot = kwargs["robot"]
        teleop = kwargs["teleop"]
        dataset = kwargs["dataset"]
        settings = kwargs["settings"]
        state = kwargs["state"]
        commands = kwargs["commands"]

        control_interval = 1.0 / settings.fps
        max_dt_ratio = env_float("CAPTURE_LOOP_MAX_DT_RATIO", 1.25)
        max_consecutive_overruns = env_int("CAPTURE_LOOP_MAX_OVERRUNS", 5)
        grace_frames = env_int("CAPTURE_LOOP_GUARD_GRACE_FRAMES", 10)
        abort_on_overrun = env_bool("CAPTURE_LOOP_ABORT_ON_OVERRUN", True)
        guard_enabled = env_bool("CAPTURE_LOOP_REALTIME_GUARD", True)
        profile_enabled = env_bool("CAPTURE_LOOP_PROFILE", False)
        profile_warn_ratio = env_float("CAPTURE_LOOP_PROFILE_WARN_RATIO", max_dt_ratio)
        patch_dataset_save_episode_for_raw_spool(dataset)
        patch_dataset_finalize_for_raw_spool(dataset)

        warmup_frames = env_int("CAPTURE_OBS_WARMUP_FRAMES", 0)
        if warmup_frames > 0:
            state.update(status="CALIBRATING", is_recording=False, message="Warming up observations")
            for warmup_idx in range(warmup_frames):
                for cmd in app.drain_commands(commands):
                    if cmd in {"finish", "discard", "stop"}:
                        state.update(is_recording=False)
                        return cmd
                warmup_t = time.perf_counter()
                robot.get_observation()
                warmup_dt_ms = (time.perf_counter() - warmup_t) * 1000.0
                if profile_enabled:
                    logging.info(
                        "Capture observation warmup frame=%s/%s get_observation=%.1fms",
                        warmup_idx + 1,
                        warmup_frames,
                        warmup_dt_ms,
                    )

        meta_episode_index = int(getattr(getattr(dataset, "meta", None), "total_episodes", 0) or 0)
        if _RAW_SPOOL_NEXT_EPISODE_INDEX is None:
            _RAW_SPOOL_NEXT_EPISODE_INDEX = meta_episode_index
        episode_index = max(meta_episode_index, int(_RAW_SPOOL_NEXT_EPISODE_INDEX))
        raw_spool_proc: subprocess.Popen[Any] | None = None
        raw_spool_sync_path: Path | None = None
        if env_bool("RAW_SPOOL_PER_EPISODE", False):
            state.update(status="CALIBRATING", is_recording=False, message="Starting raw wrist recorder")
            raw_spool_proc, raw_spool_sync_path = start_raw_spool_episode_recorder(episode_index)

        state.update(status="CALIBRATING", is_recording=False, message="Warming up async observation caches")
        wait_async_observation_caches_ready()

        start_t = time.perf_counter()
        frame_count = 0
        consecutive_overruns = 0
        raw_spool_result = "unknown"
        write_raw_spool_sync_event(
            {
                "event": "episode_start",
                "episode_index": episode_index,
                "dataset_root": str(settings.resolved_dataset_root),
                "repo_id": settings.resolved_repo_id,
                "fps": float(settings.fps),
            },
            path=raw_spool_sync_path,
        )
        state.update(status="RECORDING", is_recording=True, current_frame=0, message="Recording")
        app.drain_commands(commands)

        try:
            while time.perf_counter() - start_t < settings.episode_time_s:
                loop_t = time.perf_counter()
                for cmd in app.drain_commands(commands):
                    if cmd in {"finish", "discard", "stop"}:
                        raw_spool_result = cmd
                        state.update(is_recording=False)
                        return cmd

                target_obs_time_perf = start_t + frame_count * control_interval
                stage_t = time.perf_counter()
                set_capture_target_time_perf(target_obs_time_perf)
                try:
                    obs = robot.get_observation()
                finally:
                    set_capture_target_time_perf(None)
                obs_dt = time.perf_counter() - stage_t

                stage_t = time.perf_counter()
                action = teleop.get_action()
                action_dt = time.perf_counter() - stage_t
                if settings.swap_teleop_actions:
                    action = app.swap_left_right_action(action)

                stage_t = time.perf_counter()
                robot.send_action(action)
                send_dt = time.perf_counter() - stage_t

                stage_t = time.perf_counter()
                recorded_action = app.map_gripper_actions_for_dataset(action, robot, settings)
                observation_frame = app.build_dataset_frame(dataset.features, obs, prefix=app.OBS_STR)
                action_frame = app.build_dataset_frame(dataset.features, recorded_action, prefix=app.ACTION)
                build_dt = time.perf_counter() - stage_t

                stage_t = time.perf_counter()
                dataset.add_frame({**observation_frame, **action_frame, "task": settings.single_task})
                add_frame_dt = time.perf_counter() - stage_t

                frame_count += 1
                loop_dt = time.perf_counter() - loop_t
                target_hz = settings.fps
                actual_hz = 1.0 / loop_dt if loop_dt > 0 else 0.0
                overrun = loop_dt > control_interval * max_dt_ratio
                if guard_enabled and frame_count > grace_frames and overrun:
                    consecutive_overruns += 1
                else:
                    consecutive_overruns = 0

                if profile_enabled and loop_dt > control_interval * profile_warn_ratio:
                    logging.warning(
                        "Capture loop overrun profile frame=%s loop_dt=%.1fms target=%.1fms "
                        "get_observation=%.1fms get_action=%.1fms send_action=%.1fms "
                        "build_frame=%.1fms add_frame=%.1fms consecutive_overruns=%s",
                        frame_count,
                        loop_dt * 1000.0,
                        control_interval * 1000.0,
                        obs_dt * 1000.0,
                        action_dt * 1000.0,
                        send_dt * 1000.0,
                        build_dt * 1000.0,
                        add_frame_dt * 1000.0,
                        consecutive_overruns,
                    )

                if guard_enabled and consecutive_overruns >= max_consecutive_overruns:
                    message = (
                        "Capture loop cannot sustain requested FPS: "
                        f"loop_dt={loop_dt * 1000:.1f}ms target={1000.0 / target_hz:.1f}ms "
                        f"actual_hz={actual_hz:.1f} target_hz={target_hz} "
                        f"consecutive_overruns={consecutive_overruns}. "
                        "Current episode is not safe to save; reduce FPS/resolution/modalities or disable extra cameras."
                    )
                    state.update(status="ERROR", is_recording=False, message=message)
                    if abort_on_overrun:
                        raw_spool_result = "error"
                        raise RuntimeError(message)
                    raw_spool_result = "discard"
                    return "discard"

                state.update(
                    current_frame=frame_count,
                    last_loop_hz=actual_hz,
                    message=f"Recording frame {frame_count}",
                )
                app.precise_sleep(max(0.0, control_interval - loop_dt))

            state.update(is_recording=False)
            raw_spool_result = "finish"
            return "finish"
        finally:
            write_raw_spool_sync_event(
                {
                    "event": "episode_end",
                    "episode_index": episode_index,
                    "dataset_root": str(settings.resolved_dataset_root),
                    "repo_id": settings.resolved_repo_id,
                    "fps": float(settings.fps),
                    "frame_count": int(frame_count),
                    "relative_duration_s": float(time.perf_counter() - start_t),
                    "result": raw_spool_result,
                },
                path=raw_spool_sync_path,
            )
            if raw_spool_result == "finish" and raw_spool_proc is not None:
                postroll_s = env_float("RAW_WRIST_EPISODE_POSTROLL_S", 0.0)
                if postroll_s > 0:
                    logging.info("Raw wrist episode post-roll %.2fs before stopping recorder", postroll_s)
                    time.sleep(postroll_s)
            stop_raw_spool_episode_recorder(raw_spool_proc)
            if raw_spool_result == "finish" and env_bool("RAW_WRIST_STREAM_PROCESSED", False):
                wait_raw_spool_processed_result_ready(episode_index)
            if raw_spool_result == "finish" and env_bool("RAW_SPOOL_REQUIRE_VALID_RAW_ON_FINISH", False):
                raw_dir = raw_spool_episode_dir(episode_index)
                if env_bool("RAW_WRIST_STREAM_PROCESSED", False):
                    valid, valid_message = raw_spool_processed_result_valid(raw_dir)
                    validation_label = "processed wrist videos"
                else:
                    valid, valid_message = raw_spool_raw_outputs_valid(raw_dir)
                    validation_label = "raw wrist output"
                if not valid:
                    message = (
                        f"Raw wrist episode recorder produced unusable {validation_label} for episode {episode_index}: "
                        f"{valid_message}. Refusing to save a dataset episode without valid wrist videos. "
                        f"raw_dir={raw_dir}"
                    )
                    invalid_action = os.environ.get("RAW_SPOOL_INVALID_RAW_ACTION", "discard").strip().lower()
                    if invalid_action in {"discard", "drop", "skip"}:
                        logging.error("%s Current episode will be discarded and capture can continue.", message)
                        state.update(status="READY", is_recording=False, message="Discarded: invalid raw wrist output")
                        raw_spool_result = "discard"
                        if env_bool("RAW_SPOOL_DELETE_RAW_ON_DISCARD", True):
                            shutil.rmtree(raw_dir, ignore_errors=True)
                        return "discard"
                    state.update(status="ERROR", is_recording=False, message=message)
                    raise RuntimeError(message)
            if raw_spool_result == "finish":
                _RAW_SPOOL_PENDING_FINISHED_EPISODES.append(episode_index)
                _RAW_SPOOL_NEXT_EPISODE_INDEX = episode_index + 1
            if raw_spool_result != "finish" and env_bool("RAW_SPOOL_DELETE_RAW_ON_DISCARD", True):
                shutil.rmtree(raw_spool_episode_dir(episode_index), ignore_errors=True)

    def capture_episode(**kwargs: Any) -> str:
        reconnect_mode = os.environ.get("TACTILE_RECONNECT_ON_EPISODE_START", "auto").strip().lower()
        should_reconnect = reconnect_mode in {"1", "true", "yes", "on"}
        if reconnect_mode in {"auto", ""}:
            should_reconnect = not x5_tactile_streams_healthy(kwargs["robot"])
        if should_reconnect:
            reconnect_x5_tactile_for_episode(kwargs["robot"], kwargs.get("state"))
        if env_bool("FORCE_SENSOR_CLEAR_ON_EPISODE_START", False):
            clear_force_sensors_for_episode(kwargs["robot"], kwargs.get("state"))
        if env_bool("GRIPPER_PREWARM_ON_EPISODE_START", True):
            prewarm_gripper_for_episode(
                kwargs["robot"],
                kwargs.get("teleop"),
                kwargs.get("state"),
            )
        return capture_episode_realtime_guarded(**kwargs)

    app.capture_episode = capture_episode


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s:%(message)s")
    configure_opencv_runtime()
    for logger_name in (
        "lerobot",
        "lerobot.robots.realman_ugripper_notac.realman_ugripper_notac",
        "lerobot.robots.bi_realman_ugripper_notac_new.bi_realman_ugripper_notac_new",
    ):
        logging.getLogger(logger_name).setLevel(logging.INFO)

    app = load_capture_app()
    patch_make_robot(app)
    patch_parse_args(app)
    patch_capture_episode(app)
    patch_run_d405_cleanup(app)
    start_tactile_sidecar_if_requested()
    try:
        return int(app.run())
    finally:
        stop_tactile_sidecar()


if __name__ == "__main__":
    raise SystemExit(main())
