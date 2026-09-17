#!/usr/bin/env python3
"""Validate capture configuration without opening cameras or robot hardware."""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    lerobot_value = os.environ.get("LEROBOT_ROOT", "").strip()
    if not lerobot_value:
        errors.append("LEROBOT_ROOT is not set")
        lerobot_root = Path("__missing_lerobot_root__")
    else:
        lerobot_root = Path(lerobot_value).expanduser().resolve()

    required_paths = (
        lerobot_root / "tools" / "bi_x5_capture_app.py",
        lerobot_root / "src" / "lerobot" / "robots" / "bi_realman_ugripper_notac_new" / "__init__.py",
        lerobot_root / "src" / "lerobot" / "robots" / "realman_ugripper_notac_new" / "__init__.py",
        lerobot_root / "src" / "lerobot" / "teleoperators" / "bi_realman_rm75b_leader" / "__init__.py",
        lerobot_root / "src" / "lerobot" / "teleoperators" / "realman_rm75b_leader" / "__init__.py",
    )
    for path in required_paths:
        if not path.is_file():
            errors.append(f"missing compatible LeRobot file: {path}")

    lerobot_src = lerobot_root / "src"
    if lerobot_src.is_dir():
        sys.path.insert(0, str(lerobot_src))
        for module_name in (
            "cv2",
            "numpy",
            "lerobot.datasets.lerobot_dataset",
            "lerobot.robots.bi_realman_ugripper_notac_new",
            "lerobot.teleoperators.bi_realman_rm75b_leader",
        ):
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 - report complete dependency failures
                errors.append(f"cannot import {module_name}: {type(exc).__name__}: {exc}")

    threshold_value = os.environ.get("TACTILE_PREVIEW_THRESHOLD_FILE", "").strip()
    if env_bool("TACTILE_PREVIEW", True) and threshold_value:
        threshold_path = Path(threshold_value).expanduser()
        if not threshold_path.is_absolute():
            threshold_path = PROJECT_ROOT / threshold_path
        if not threshold_path.is_file():
            errors.append(f"missing tactile preview threshold file: {threshold_path}")

    if env_bool("FOOT_PEDAL_CONTROL", False):
        pedal_path = Path(os.environ.get("FOOT_PEDAL_DEVICE", ""))
        if not pedal_path.exists():
            errors.append(f"foot pedal device does not exist: {pedal_path}")
        elif not os.access(pedal_path, os.R_OK):
            errors.append(f"foot pedal device is not readable: {pedal_path}")
        if env_bool("FOOT_PEDAL_TWO_PEDAL_STOP_ENABLED", False):
            pedal_keys = [
                os.environ.get("FOOT_PEDAL_START_KEY", "KEY_ESC").strip().upper(),
                os.environ.get("FOOT_PEDAL_FINISH_KEY", "KEY_LEFT").strip().upper(),
                os.environ.get("FOOT_PEDAL_DISCARD_KEY", "KEY_RIGHT").strip().upper(),
            ]
            if any(not key for key in pedal_keys):
                errors.append("two-pedal stop requires three non-empty pedal keys")
            if len(set(pedal_keys)) != 3:
                errors.append("foot-pedal start/finish/discard keys must be distinct")
            try:
                chord_window_s = float(
                    os.environ.get("FOOT_PEDAL_TWO_PEDAL_WINDOW_S", "0.12")
                )
            except ValueError:
                errors.append("FOOT_PEDAL_TWO_PEDAL_WINDOW_S must be a number")
            else:
                if not 0 <= chord_window_s <= 1.0:
                    errors.append(
                        "FOOT_PEDAL_TWO_PEDAL_WINDOW_S must be within 0..1 seconds"
                    )

    if env_bool("CONNECT_D405_CAMERA", False):
        d405_path = Path(os.environ.get("D405_CAMERA_DEVICE", ""))
        if not d405_path.exists():
            errors.append(f"D405 device does not exist: {d405_path}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        errors.append("ffmpeg is not available on PATH")
    elif os.environ.get("VCODEC", "auto").strip() == "h264_nvenc":
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
        )
        if "h264_nvenc" not in result.stdout:
            errors.append("ffmpeg does not advertise the h264_nvenc encoder")

    tactile_ports: list[int] = []
    for name in (
        "X5_TACTILE_FLUX_LEFT_LEFT_PC_PORT",
        "X5_TACTILE_FLUX_LEFT_RIGHT_PC_PORT",
        "X5_TACTILE_FLUX_RIGHT_LEFT_PC_PORT",
        "X5_TACTILE_FLUX_RIGHT_RIGHT_PC_PORT",
    ):
        try:
            value = int(os.environ.get(name, ""))
        except ValueError:
            errors.append(f"{name} must be an integer")
            continue
        if not 1 <= value <= 65535:
            errors.append(f"{name} must be in 1..65535")
        tactile_ports.append(value)
    if len(tactile_ports) != len(set(tactile_ports)):
        errors.append("the four X5 tactile UDP ports must be distinct")

    control_mode = os.environ.get("CAPTURE_CONTROL_MODE", "").strip().lower()
    if control_mode not in {"leader", "program", "drag"}:
        errors.append("CAPTURE_CONTROL_MODE must be leader, program or drag")

    try:
        gripper_min = int(os.environ.get("GRIPPER_MIN_POSITION", "0"))
        gripper_max = int(os.environ.get("GRIPPER_MAX_POSITION", "1000"))
        gripper_torque = int(os.environ.get("GRIPPER_TORQUE_LIMIT", "90"))
        if not 0 <= gripper_min <= gripper_max <= 1000:
            errors.append("gripper positions must satisfy 0 <= min <= max <= 1000")
        if not 10 <= gripper_torque <= 100:
            errors.append("GRIPPER_TORQUE_LIMIT must be in 10..100")
    except ValueError:
        errors.append("gripper position and torque limits must be integers")

    if env_bool("EPISODE_RIGHT_ARM_RESET_ENABLED", False):
        raw_target = os.environ.get("EPISODE_RIGHT_ARM_RESET_JOINTS_DEG", "")
        try:
            target = [float(part.strip()) for part in raw_target.split(",")]
        except ValueError:
            target = []
        if len(target) != 7 or not all(math.isfinite(value) for value in target):
            errors.append("enabled right-arm reset requires exactly 7 finite joint angles")
        warnings.append("automatic right-arm motion is ENABLED; confirm a clear workspace and emergency stop")

    return_to_start_enabled = env_bool("EPISODE_RETURN_TO_START_ENABLED", False)
    if return_to_start_enabled:
        if control_mode != "drag":
            errors.append("dynamic return-to-start requires CAPTURE_CONTROL_MODE=drag")
        if os.environ.get("TASK_INDEX", "both").strip().lower() not in {"right", "0"}:
            errors.append("dynamic return-to-start is only allowed for TASK_INDEX=right")
        if env_bool("EPISODE_RIGHT_ARM_RESET_ENABLED", False):
            errors.append("fixed pre-episode reset and dynamic return-to-start cannot both be enabled")
        try:
            return_speed = int(os.environ.get("EPISODE_RETURN_TO_START_SPEED", "5"))
            return_max_delta = float(
                os.environ.get("EPISODE_RETURN_TO_START_MAX_DELTA_DEG", "60")
            )
            if not 1 <= return_speed <= 100:
                errors.append("EPISODE_RETURN_TO_START_SPEED must be in 1..100")
            if not 0.0 < return_max_delta <= 360.0 or not math.isfinite(return_max_delta):
                errors.append("EPISODE_RETURN_TO_START_MAX_DELTA_DEG must be finite and in (0, 360]")
            gripper_speed = int(
                os.environ.get("EPISODE_RETURN_TO_START_GRIPPER_SPEED", "30")
            )
            gripper_tolerance = int(
                os.environ.get("EPISODE_RETURN_TO_START_GRIPPER_TOLERANCE", "25")
            )
            if not 10 <= gripper_speed <= 100:
                errors.append("EPISODE_RETURN_TO_START_GRIPPER_SPEED must be in 10..100")
            if not 0 <= gripper_tolerance <= 200:
                errors.append("EPISODE_RETURN_TO_START_GRIPPER_TOLERANCE must be in 0..200")
        except ValueError:
            errors.append("dynamic return-to-start speed/max delta values are invalid")
        warnings.append(
            "dynamic right-arm return is ENABLED; finish/discard will command real motion"
        )

    try:
        ready_gripper_hz = float(os.environ.get("READY_GRIPPER_FOLLOW_HZ", "30"))
        ready_takeover_tolerance = int(
            os.environ.get("READY_GRIPPER_TAKEOVER_TOLERANCE", "75")
        )
        if not math.isfinite(ready_gripper_hz) or not 1.0 <= ready_gripper_hz <= 100.0:
            errors.append("READY_GRIPPER_FOLLOW_HZ must be finite and in 1..100")
        if not 0 <= ready_takeover_tolerance <= 300:
            errors.append("READY_GRIPPER_TAKEOVER_TOLERANCE must be in 0..300")
    except ValueError:
        errors.append("READY gripper follow values are invalid")

    ui_host = os.environ.get("CAPTURE_UI_HOST", "127.0.0.1").strip()
    if ui_host not in {"127.0.0.1", "localhost", "::1"}:
        warnings.append(f"capture UI is bound beyond loopback: {ui_host}")

    for warning in warnings:
        print(f"[preflight] WARNING: {warning}")
    for error in errors:
        print(f"[preflight] ERROR: {error}", file=sys.stderr)
    if errors:
        print(f"[preflight] failed with {len(errors)} error(s)", file=sys.stderr)
        return 1

    print(
        "[preflight] passed: configuration, local devices, encoder and "
        "compatible LeRobot imports are available"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
