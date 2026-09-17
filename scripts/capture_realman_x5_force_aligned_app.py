#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import queue
import select
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import capture_realman_x5_force_app as compat
import listen_foot_pedal
from capture_support.tactile_frame_bridge import (
    TactileFrameBridgeServer,
    TactileFramePublisher,
)


@dataclass(frozen=True)
class GripperSafetyLimits:
    min_position: int
    max_position: int
    torque_limit: int

    def __post_init__(self) -> None:
        if not 0 <= int(self.min_position) <= int(self.max_position) <= 1000:
            raise ValueError(
                "夹爪位置限制必须满足 "
                "0 <= GRIPPER_MIN_POSITION <= GRIPPER_MAX_POSITION <= 1000"
            )
        if not 10 <= int(self.torque_limit) <= 100:
            raise ValueError("GRIPPER_TORQUE_LIMIT 必须在 10..100 范围内")

    def clamp_position(self, position: int | float) -> int:
        return max(
            int(self.min_position),
            min(int(self.max_position), int(round(float(position)))),
        )

    @staticmethod
    def _mapping(arm: Any) -> tuple[float, float, float]:
        cfg = getattr(arm, "config", None)
        leader_min = float(getattr(cfg, "leader_gripper_min", 0.066))
        leader_max = float(getattr(cfg, "leader_gripper_max", 0.971))
        gain = float(getattr(cfg, "gripper_gain", 1.1))
        if abs(leader_max - leader_min) < 1e-9:
            raise RuntimeError("leader_gripper_min/max 不能相同")
        if abs(gain) < 1e-9:
            raise RuntimeError("gripper_gain 不能为 0")
        return leader_min, leader_max, gain

    @classmethod
    def leader_value_to_position(cls, value: float, arm: Any) -> float:
        leader_min, leader_max, gain = cls._mapping(arm)
        normalized = (float(value) - leader_min) / (leader_max - leader_min)
        normalized = 0.5 + (normalized - 0.5) * gain
        return float(np.clip(normalized, 0.0, 1.0) * 1000.0)

    @classmethod
    def position_to_leader_value(cls, position: int, arm: Any) -> float:
        leader_min, leader_max, gain = cls._mapping(arm)
        normalized_target = float(np.clip(position, 0, 1000)) / 1000.0
        normalized_leader = 0.5 + (normalized_target - 0.5) / gain
        return leader_min + normalized_leader * (leader_max - leader_min)

    def clamp_action(
        self,
        action: dict[str, Any],
        robot: Any,
    ) -> dict[str, Any]:
        limited = dict(action)
        robot_cfg = getattr(robot, "config", None)
        for side, arm in _enabled_arm_items(robot):
            parent_arm_cfg = getattr(robot_cfg, f"{side}_arm_config", None)
            arm_cfg = parent_arm_cfg or getattr(arm, "config", None)
            if not bool(getattr(arm_cfg, "connect_gripper", True)):
                continue
            key = f"{side}_{arm.GRIPPER_NAME}"
            if key not in limited:
                continue
            position = self.leader_value_to_position(float(limited[key]), arm)
            safe_position = self.clamp_position(position)
            if abs(float(safe_position) - position) > 0.5:
                limited[key] = self.position_to_leader_value(
                    safe_position,
                    arm,
                )
        return limited

    def apply_torque_limit(self, robot: Any) -> None:
        robot_cfg = getattr(robot, "config", None)
        for side, arm in _enabled_arm_items(robot):
            parent_arm_cfg = getattr(robot_cfg, f"{side}_arm_config", None)
            arm_cfg = parent_arm_cfg or getattr(arm, "config", None)
            if not bool(getattr(arm_cfg, "connect_gripper", True)):
                continue
            gripper = getattr(arm, "_gripper", None)
            raw_gripper = getattr(gripper, "_gripper", None)
            if raw_gripper is None:
                raise RuntimeError(f"{side} 夹爪未连接，无法设置力矩限制")
            if not bool(raw_gripper.set_torque_limit(self.torque_limit)):
                raise RuntimeError(
                    f"{side} 夹爪设置力矩限制 {self.torque_limit} 失败"
                )


@dataclass(frozen=True)
class EpisodeRightArmResetConfig:
    """Fail-closed right-arm reset performed before an aligned episode."""

    enabled: bool
    joints_deg: tuple[float, ...] = ()
    speed: int = 10
    tolerance_deg: float = 0.5
    max_start_delta_deg: float = 45.0
    timeout_s: float = 45.0
    poll_hz: float = 20.0
    stable_samples: int = 3
    settle_s: float = 0.5
    state_max_age_s: float = 0.25
    query_controller_limits: bool = False
    stop_drag_first: bool = True
    pause_readers: bool = True

    @classmethod
    def from_env(cls) -> "EpisodeRightArmResetConfig":
        enabled = compat.env_bool("EPISODE_RIGHT_ARM_RESET_ENABLED", False)
        raw_joints = os.environ.get("EPISODE_RIGHT_ARM_RESET_JOINTS_DEG", "").strip()
        if not enabled:
            return cls(enabled=False)
        if not raw_joints:
            raise ValueError(
                "EPISODE_RIGHT_ARM_RESET_ENABLED=true 时必须设置 "
                "EPISODE_RIGHT_ARM_RESET_JOINTS_DEG（7 个逗号分隔的角度值）"
            )
        try:
            joints = tuple(float(item.strip()) for item in raw_joints.split(","))
        except ValueError as exc:
            raise ValueError(
                "EPISODE_RIGHT_ARM_RESET_JOINTS_DEG 必须是 7 个逗号分隔的有限角度值"
            ) from exc
        if len(joints) != 7 or not all(np.isfinite(joints)):
            raise ValueError(
                "EPISODE_RIGHT_ARM_RESET_JOINTS_DEG 必须正好包含 7 个有限角度值"
            )
        if any(abs(angle) > 360.0 for angle in joints):
            raise ValueError("右臂复位关节角超出通用保护范围 [-360, 360]°")

        config = cls(
            enabled=True,
            joints_deg=joints,
            speed=compat.env_int("EPISODE_RIGHT_ARM_RESET_SPEED", 10),
            tolerance_deg=compat.env_float("EPISODE_RIGHT_ARM_RESET_TOLERANCE_DEG", 0.5),
            max_start_delta_deg=compat.env_float(
                "EPISODE_RIGHT_ARM_RESET_MAX_START_DELTA_DEG",
                45.0,
            ),
            timeout_s=compat.env_float("EPISODE_RIGHT_ARM_RESET_TIMEOUT_S", 45.0),
            poll_hz=compat.env_float("EPISODE_RIGHT_ARM_RESET_POLL_HZ", 20.0),
            stable_samples=compat.env_int("EPISODE_RIGHT_ARM_RESET_STABLE_SAMPLES", 3),
            settle_s=compat.env_float("EPISODE_RIGHT_ARM_RESET_SETTLE_S", 0.5),
            state_max_age_s=compat.env_float(
                "EPISODE_RIGHT_ARM_RESET_STATE_MAX_AGE_S",
                0.25,
            ),
            query_controller_limits=compat.env_bool(
                "EPISODE_RIGHT_ARM_RESET_QUERY_CONTROLLER_LIMITS",
                False,
            ),
            stop_drag_first=compat.env_bool(
                "EPISODE_RIGHT_ARM_RESET_STOP_DRAG_FIRST",
                True,
            ),
            pause_readers=compat.env_bool(
                "EPISODE_RIGHT_ARM_RESET_PAUSE_READERS",
                True,
            ),
        )
        if not 1 <= config.speed <= 100:
            raise ValueError("EPISODE_RIGHT_ARM_RESET_SPEED 必须在 1..100 范围内")
        if not 0.0 < config.tolerance_deg <= 5.0:
            raise ValueError("EPISODE_RIGHT_ARM_RESET_TOLERANCE_DEG 必须在 (0, 5]° 范围内")
        if not 0.0 < config.max_start_delta_deg <= 360.0:
            raise ValueError(
                "EPISODE_RIGHT_ARM_RESET_MAX_START_DELTA_DEG 必须在 (0, 360]° 范围内"
            )
        if not 0.0 < config.timeout_s <= 300.0:
            raise ValueError("EPISODE_RIGHT_ARM_RESET_TIMEOUT_S 必须在 (0, 300] 秒范围内")
        if not 1.0 <= config.poll_hz <= 100.0:
            raise ValueError("EPISODE_RIGHT_ARM_RESET_POLL_HZ 必须在 1..100 Hz 范围内")
        if not 1 <= config.stable_samples <= 100:
            raise ValueError("EPISODE_RIGHT_ARM_RESET_STABLE_SAMPLES 必须在 1..100 范围内")
        if not 0.0 <= config.settle_s <= 10.0:
            raise ValueError("EPISODE_RIGHT_ARM_RESET_SETTLE_S 必须在 0..10 秒范围内")
        if not 0.0 < config.state_max_age_s <= 5.0:
            raise ValueError(
                "EPISODE_RIGHT_ARM_RESET_STATE_MAX_AGE_S 必须在 (0, 5] 秒范围内"
            )
        return config


@dataclass(frozen=True)
class EpisodeReturnToStartConfig:
    """Return the right follower to the pose captured when start is accepted."""

    enabled: bool
    speed: int = 5
    tolerance_deg: float = 0.5
    max_start_delta_deg: float = 60.0
    timeout_s: float = 60.0
    poll_hz: float = 20.0
    stable_samples: int = 3
    settle_s: float = 0.5
    query_controller_limits: bool = False
    pause_readers: bool = True
    restore_gripper: bool = False
    gripper_speed: int = 30
    gripper_tolerance: int = 25
    gripper_timeout_s: float = 10.0

    @classmethod
    def from_env(cls) -> "EpisodeReturnToStartConfig":
        config = cls(
            enabled=compat.env_bool("EPISODE_RETURN_TO_START_ENABLED", False),
            speed=compat.env_int("EPISODE_RETURN_TO_START_SPEED", 5),
            tolerance_deg=compat.env_float(
                "EPISODE_RETURN_TO_START_TOLERANCE_DEG",
                0.5,
            ),
            max_start_delta_deg=compat.env_float(
                "EPISODE_RETURN_TO_START_MAX_DELTA_DEG",
                60.0,
            ),
            timeout_s=compat.env_float("EPISODE_RETURN_TO_START_TIMEOUT_S", 60.0),
            poll_hz=compat.env_float("EPISODE_RETURN_TO_START_POLL_HZ", 20.0),
            stable_samples=compat.env_int(
                "EPISODE_RETURN_TO_START_STABLE_SAMPLES",
                3,
            ),
            settle_s=compat.env_float("EPISODE_RETURN_TO_START_SETTLE_S", 0.5),
            query_controller_limits=compat.env_bool(
                "EPISODE_RETURN_TO_START_QUERY_CONTROLLER_LIMITS",
                False,
            ),
            pause_readers=compat.env_bool(
                "EPISODE_RETURN_TO_START_PAUSE_READERS",
                True,
            ),
            restore_gripper=compat.env_bool(
                "EPISODE_RETURN_TO_START_RESTORE_GRIPPER",
                False,
            ),
            gripper_speed=compat.env_int(
                "EPISODE_RETURN_TO_START_GRIPPER_SPEED",
                30,
            ),
            gripper_tolerance=compat.env_int(
                "EPISODE_RETURN_TO_START_GRIPPER_TOLERANCE",
                25,
            ),
            gripper_timeout_s=compat.env_float(
                "EPISODE_RETURN_TO_START_GRIPPER_TIMEOUT_S",
                10.0,
            ),
        )
        if not 1 <= config.speed <= 100:
            raise ValueError("EPISODE_RETURN_TO_START_SPEED 必须在 1..100 范围内")
        if not 0.0 < config.tolerance_deg <= 5.0:
            raise ValueError(
                "EPISODE_RETURN_TO_START_TOLERANCE_DEG 必须在 (0, 5]° 范围内"
            )
        if not 0.0 < config.max_start_delta_deg <= 360.0:
            raise ValueError(
                "EPISODE_RETURN_TO_START_MAX_DELTA_DEG 必须在 (0, 360]° 范围内"
            )
        if not 0.0 < config.timeout_s <= 300.0:
            raise ValueError("EPISODE_RETURN_TO_START_TIMEOUT_S 必须在 (0, 300] 秒范围内")
        if not 1.0 <= config.poll_hz <= 100.0:
            raise ValueError("EPISODE_RETURN_TO_START_POLL_HZ 必须在 1..100 Hz 范围内")
        if not 1 <= config.stable_samples <= 100:
            raise ValueError(
                "EPISODE_RETURN_TO_START_STABLE_SAMPLES 必须在 1..100 范围内"
            )
        if not 0.0 <= config.settle_s <= 10.0:
            raise ValueError("EPISODE_RETURN_TO_START_SETTLE_S 必须在 0..10 秒范围内")
        if not 10 <= config.gripper_speed <= 100:
            raise ValueError(
                "EPISODE_RETURN_TO_START_GRIPPER_SPEED 必须在 10..100 范围内"
            )
        if not 0 <= config.gripper_tolerance <= 200:
            raise ValueError(
                "EPISODE_RETURN_TO_START_GRIPPER_TOLERANCE 必须在 0..200 范围内"
            )
        if not 0.0 < config.gripper_timeout_s <= 60.0:
            raise ValueError(
                "EPISODE_RETURN_TO_START_GRIPPER_TIMEOUT_S 必须在 (0, 60] 秒范围内"
            )
        return config

    def reset_config(self, joints_deg: tuple[float, ...]) -> EpisodeRightArmResetConfig:
        return EpisodeRightArmResetConfig(
            enabled=self.enabled,
            joints_deg=joints_deg,
            speed=self.speed,
            tolerance_deg=self.tolerance_deg,
            max_start_delta_deg=self.max_start_delta_deg,
            timeout_s=self.timeout_s,
            poll_hz=self.poll_hz,
            stable_samples=self.stable_samples,
            settle_s=self.settle_s,
            query_controller_limits=self.query_controller_limits,
            stop_drag_first=False,
            pause_readers=self.pause_readers,
        )


@dataclass(frozen=True)
class EpisodeStartPose:
    episode_index: int
    task: str
    right_joint_positions_deg: tuple[float, ...]
    right_gripper_position: int | None
    captured_at_utc: str


def resolve_capture_control_mode(default: str = "leader") -> str:
    """Resolve the capture mode while retaining the old gripper-source switch."""

    raw = os.environ.get("CAPTURE_CONTROL_MODE")
    if raw is None or not raw.strip():
        raw = os.environ.get("GRIPPER_COMMAND_SOURCE", default)
    mode = raw.strip().lower()
    if mode not in {"leader", "program", "drag"}:
        raise ValueError("CAPTURE_CONTROL_MODE 必须是 leader、program 或 drag")
    return mode


def _enabled_arm_items(robot: Any) -> list[tuple[str, Any]]:
    enabled: list[tuple[str, Any]] = []
    robot_cfg = getattr(robot, "config", None)
    for side in ("left", "right"):
        arm_cfg = getattr(robot_cfg, f"{side}_arm_config", None)
        if arm_cfg is not None and not bool(getattr(arm_cfg, "enabled", False)):
            continue
        arm = getattr(robot, f"{side}_arm", None)
        if arm is not None:
            enabled.append((side, arm))
    return enabled


def connect_leaders_with_missing_port_fallback(
    teleop: Any,
    *,
    allow_partial: bool,
) -> tuple[str, ...]:
    """Disable a physically absent leader USB side, then connect the remainder."""

    if not allow_partial:
        teleop.connect()
        return ()

    missing: list[str] = []
    enabled: list[str] = []
    for side in ("left", "right"):
        arm_cfg = getattr(teleop.config, f"{side}_arm_config")
        if not bool(getattr(arm_cfg, "enabled", False)):
            continue
        port = str(getattr(arm_cfg, "port", ""))
        if not port or not Path(port).exists():
            arm_cfg.enabled = False
            missing.append(side)
            logging.warning(
                "%s 主臂 USB 不存在 (%s)，该主臂对应 action 将使用从臂 state",
                side,
                port or "未配置",
            )
        else:
            enabled.append(side)
    if not enabled:
        raise ConnectionError("左右主臂 USB 都不可用；leader 模式至少需要一侧主臂")

    teleop.connect()
    missing_sides = tuple(missing)
    setattr(teleop, "_capture_missing_leader_sides", missing_sides)
    return missing_sides


def configure_teleop_for_capture_task(config: Any, settings: Any) -> tuple[str, ...]:
    """Enable only the physical leader sides needed by the selected followers."""

    task_index = int(settings.task_index)
    if task_index not in {0, 1}:
        enabled = tuple(
            side
            for side in ("left", "right")
            if bool(getattr(getattr(config, f"{side}_arm_config"), "enabled", False))
        )
        return enabled

    follower_side = "right" if task_index == 0 else "left"
    if bool(settings.swap_teleop_actions):
        physical_leader_side = "left" if follower_side == "right" else "right"
    else:
        physical_leader_side = follower_side
    for side in ("left", "right"):
        getattr(config, f"{side}_arm_config").enabled = side == physical_leader_side
    logging.info(
        "Single-arm capture: follower=%s physical_gripper_leader=%s swap_teleop_actions=%s",
        follower_side,
        physical_leader_side,
        bool(settings.swap_teleop_actions),
    )
    return (physical_leader_side,)


def leader_fallback_action_sides(
    missing_leader_sides: tuple[str, ...],
    *,
    swap_teleop_actions: bool,
) -> tuple[str, ...]:
    """Map missing physical leader sides to their follower action prefixes."""

    if not swap_teleop_actions:
        return missing_leader_sides
    opposite = {"left": "right", "right": "left"}
    return tuple(opposite[side] for side in missing_leader_sides)


def _append_send_skip_sides(sides: tuple[str, ...]) -> tuple[str, ...]:
    prefixes = [
        value.strip()
        for value in os.environ.get("BI_REALMAN_SKIP_SEND_PREFIXES", "").split(",")
        if value.strip()
    ]
    for side in sides:
        prefix = f"{side}_"
        if prefix not in prefixes:
            prefixes.append(prefix)
    os.environ["BI_REALMAN_SKIP_SEND_PREFIXES"] = ",".join(prefixes)
    return tuple(prefixes)


def replace_arm_actions_with_follower_state(
    action: dict[str, Any],
    follower_state: dict[str, Any],
    robot: Any,
    sides: tuple[str, ...],
    *,
    dataset_gripper_coordinates: bool = False,
) -> dict[str, Any]:
    """Replace selected sides' seven joints and gripper with follower state."""

    replaced = dict(action)
    missing: list[str] = []
    invalid: list[str] = []
    for side in sides:
        arm = getattr(robot, f"{side}_arm")
        names = tuple(getattr(arm, "JOINT_NAMES", ())) + (
            getattr(arm, "GRIPPER_NAME", "main_gripper"),
        )
        for name in names:
            key = f"{side}_{name}"
            if key not in follower_state:
                missing.append(key)
                continue
            value = float(follower_state[key])
            if not np.isfinite(value):
                invalid.append(key)
                continue
            if name == getattr(arm, "GRIPPER_NAME", "main_gripper") and not dataset_gripper_coordinates:
                value = GripperSafetyLimits.position_to_leader_value(value * 1000.0, arm)
            replaced[key] = value
    if missing:
        raise RuntimeError("leader fallback 缺少从臂 state 字段: " + ", ".join(missing))
    if invalid:
        raise RuntimeError("leader fallback 从臂 state 含 NaN/Inf: " + ", ".join(invalid))
    return replaced


def replace_joint_actions_with_follower_state(
    action: dict[str, Any],
    follower_state: dict[str, Any],
    robot: Any,
) -> dict[str, Any]:
    """Keep leader gripper commands but record follower joint state as action."""

    replaced = dict(action)
    missing: list[str] = []
    invalid: list[str] = []
    for side, arm in _enabled_arm_items(robot):
        for joint_name in getattr(arm, "JOINT_NAMES", ()):
            key = f"{side}_{joint_name}"
            if key not in follower_state:
                missing.append(key)
                continue
            value = float(follower_state[key])
            if not np.isfinite(value):
                invalid.append(key)
                continue
            replaced[key] = value
    if missing:
        raise RuntimeError(
            "drag 模式缺少从臂 state 关节字段: " + ", ".join(missing)
        )
    if invalid:
        raise RuntimeError(
            "drag 模式从臂 state 含 NaN/Inf: " + ", ".join(invalid)
        )
    return replaced


class DragTeachSession:
    """Start and reliably stop precise 6-axis force drag teaching."""

    _SIX_AXIS_FORCE_TYPES = {"6F", "6FB", "6FB-V", 2, 3, 5}

    def __init__(
        self,
        robot: Any,
        *,
        precise: bool = True,
        singular_wall: bool = True,
    ) -> None:
        self.robot = robot
        self.precise = bool(precise)
        self.singular_wall = bool(singular_wall)
        self._active: list[tuple[str, Any]] = []

    @property
    def is_active(self) -> bool:
        return bool(self._active)

    def _arm_for_side(self, side: str) -> Any:
        for enabled_side, arm in _enabled_arm_items(self.robot):
            if enabled_side == side:
                return arm
        raise RuntimeError(f"{side} 从臂未启用，无法切换拖动模式")

    def _start_side(self, side: str, arm: Any) -> None:
        if any(active_side == side for active_side, _ in self._active):
            return
        follower = getattr(arm, "_follower_arm", None)
        get_info = getattr(follower, "rm_get_robot_info", None)
        set_force_drag_mode = getattr(follower, "rm_set_force_drag_mode", None)
        start_drag = getattr(follower, "rm_start_multi_drag_teach", None)
        if not callable(get_info):
            raise RuntimeError(f"{side} 从臂 SDK 不支持 rm_get_robot_info")
        if not callable(set_force_drag_mode):
            raise RuntimeError(f"{side} 从臂 SDK 不支持 rm_set_force_drag_mode")
        if not callable(start_drag):
            raise RuntimeError(f"{side} 从臂 SDK 不支持 rm_start_multi_drag_teach")

        info_code, info = get_info()
        if int(info_code) != 0:
            raise RuntimeError(f"{side} 从臂读取硬件信息失败，错误码 {info_code}")
        force_type = info.get("force_type") if isinstance(info, dict) else None
        if force_type not in self._SIX_AXIS_FORCE_TYPES:
            raise RuntimeError(f"{side} 从臂不是六维力版本，force_type={force_type!r}")

        precision_mode = 1 if self.precise else 0
        code = int(set_force_drag_mode(precision_mode))
        if code != 0:
            raise RuntimeError(
                f"{side} 从臂设置六维力"
                f"{'精准' if self.precise else '快速'}拖动失败，错误码 {code}"
            )

        # mode=3: end-effector 6-axis force controls both position and
        # orientation. The controller does not record a teach trajectory.
        code = int(start_drag(3, int(self.singular_wall)))
        if code != 0:
            raise RuntimeError(f"{side} 从臂开启六维力位姿拖动失败，错误码 {code}")
        self._active.append((side, follower))
        logging.info(
            "%s follower precise 6-axis force drag enabled "
            "(position+orientation, singular_wall=%s, force_type=%s)",
            side,
            self.singular_wall,
            force_type,
        )

    def start(self) -> None:
        if self._active:
            return
        try:
            for side, arm in _enabled_arm_items(self.robot):
                self._start_side(side, arm)
        except BaseException:
            self.stop()
            raise
        if not self._active:
            raise RuntimeError("drag 模式没有找到已启用的六维力从臂")

    def pause_side(self, side: str) -> None:
        for index, (active_side, follower) in enumerate(self._active):
            if active_side != side:
                continue
            stop_drag = getattr(follower, "rm_stop_drag_teach", None)
            if not callable(stop_drag):
                raise RuntimeError(f"{side} 从臂 SDK 不支持 rm_stop_drag_teach")
            code = int(stop_drag())
            if code != 0:
                raise RuntimeError(f"{side} 从臂停止拖动失败，错误码 {code}")
            del self._active[index]
            logging.info("%s follower drag teaching paused", side)
            return
        raise RuntimeError(f"{side} 从臂拖动模式当前未启用")

    def resume_side(self, side: str) -> None:
        self._start_side(side, self._arm_for_side(side))

    def stop(self, *, strict: bool = False) -> None:
        active, self._active = self._active, []
        if not active:
            return

        failed: list[tuple[str, Any, str]] = []
        paused_readers: list[_PausedArmSdkReaders] = []
        current_by_side: dict[str, tuple[float, ...]] = {}
        try:
            # RealMan's state/force readers share the vendor SDK in this process.
            # Stop them while changing drag state, and capture the live pose before
            # stopping drag so neither arm can spring back to an older target.
            paused_readers = _pause_enabled_arm_sdk_readers(self.robot)
            for side, follower in active:
                current_by_side[side] = _read_follower_joints_deg(follower, side)

            for side, follower in reversed(active):
                try:
                    code = int(follower.rm_stop_drag_teach())
                    if code != 0:
                        failed.append((side, follower, f"错误码 {code}"))
                        logging.error(
                            "%s follower failed to stop drag teaching, code=%s",
                            side,
                            code,
                        )
                        continue

                    hold = getattr(follower, "rm_movej_canfd", None)
                    if not callable(hold):
                        failed.append((side, follower, "SDK 不支持 rm_movej_canfd 当前位姿锁定"))
                        continue
                    hold_code = int(hold(list(current_by_side[side]), False, 0))
                    if hold_code != 0:
                        failed.append((side, follower, f"当前位姿锁定错误码 {hold_code}"))
                        logging.error(
                            "%s follower failed to latch current pose after drag stop, code=%s",
                            side,
                            hold_code,
                        )
                        continue
                    logging.info(
                        "%s follower drag teaching disabled and current pose latched",
                        side,
                    )
                except Exception as exc:
                    failed.append((side, follower, str(exc)))
                    logging.exception("%s follower failed to stop drag teaching", side)
        except BaseException:
            self._active = active
            raise
        finally:
            if paused_readers:
                _resume_arm_sdk_readers(paused_readers)
        if failed:
            failed_sides = {side for side, _follower, _reason in failed}
            self._active = [item for item in active if item[0] in failed_sides]
            if strict:
                detail = "; ".join(f"{side}: {reason}" for side, _follower, reason in failed)
                raise RuntimeError(f"回合结束时关闭拖动失败，拒绝进入下一回合：{detail}")

    def mark_controller_emergency_stopped(self) -> None:
        """Prevent any post-stop hold command after a controller hard stop."""

        if self._active:
            logging.critical(
                "Controller hard stop acknowledged; suppressing drag-stop hold commands for %s",
                [side for side, _follower in self._active],
            )
        self._active = []


def _read_follower_joints_deg(follower: Any, side: str) -> tuple[float, ...]:
    read_joints = getattr(follower, "rm_get_joint_degree", None)
    if not callable(read_joints):
        raise RuntimeError(f"{side} 从臂 SDK 不支持 rm_get_joint_degree")
    result = read_joints()
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(f"{side} 从臂关节读取返回格式无效: {result!r}")
    code, raw_joints = result[0], result[1]
    if int(code) != 0:
        raise RuntimeError(f"{side} 从臂读取关节角失败，错误码 {code}")
    joints = tuple(float(value) for value in raw_joints)
    if len(joints) != 7 or not all(np.isfinite(joints)):
        raise RuntimeError(f"{side} 从臂关节角必须是 7 个有限值，收到 {joints!r}")
    return joints


def _read_arm_joints_deg(
    arm: Any,
    follower: Any,
    side: str,
    *,
    max_cache_age_s: float,
) -> tuple[tuple[float, ...], str, float | None]:
    """Read joints in degrees, preferring the driver's nonblocking state cache."""

    reader = getattr(arm, "_follower_state_reader", None)
    if reader is None:
        return _read_follower_joints_deg(follower, side), "sdk-direct", None

    get_state = getattr(reader, "get_state", None)
    get_state_age = getattr(reader, "get_state_age", None)
    if not callable(get_state) or not callable(get_state_age):
        raise RuntimeError(f"{side} 从臂异步状态缓存接口不完整")
    age_s = float(get_state_age())
    if not np.isfinite(age_s) or age_s > max_cache_age_s:
        raise RuntimeError(
            f"{side} 从臂异步关节状态过旧：age={age_s:.3f}s > {max_cache_age_s:.3f}s"
        )

    cached = tuple(float(value) for value in get_state())
    if len(cached) != 7 or not all(np.isfinite(cached)):
        raise RuntimeError(f"{side} 从臂异步关节状态必须是 7 个有限值，收到 {cached!r}")
    use_degrees = bool(getattr(getattr(arm, "config", None), "use_degrees", False))
    joints_deg = cached if use_degrees else tuple(float(np.degrees(value)) for value in cached)
    return joints_deg, "async-cache", age_s


class _PausedArmSdkReaders:
    """Temporarily stop RealMan readers that share one vendor SDK handle."""

    _READER_SPECS = (
        ("_follower_state_reader", "_start_async_state_reader", "state"),
        ("_force_sensor_reader", "_start_async_force_reader", "force"),
    )

    def __init__(self, arm: Any, side: str) -> None:
        self.arm = arm
        self.side = side
        self._paused: list[tuple[str, str, str]] = []
        self._resumed = False

    def pause(self) -> None:
        readers: list[tuple[str, str, str, Any]] = []
        try:
            for attr, restart_name, label in self._READER_SPECS:
                reader = getattr(self.arm, attr, None)
                if reader is None:
                    continue
                stop = getattr(reader, "stop", None)
                join = getattr(reader, "join", None)
                if not callable(stop) or not callable(join):
                    raise RuntimeError(
                        f"{self.side} 从臂 {label} 读取器不支持安全暂停，拒绝与复位命令并发访问 SDK"
                    )
                stop()
                readers.append((attr, restart_name, label, reader))
                self._paused.append((attr, restart_name, label))

            timeout_s = compat.env_float("EPISODE_RIGHT_ARM_RESET_READER_STOP_TIMEOUT_S", 2.0)
            for _attr, _restart_name, label, reader in readers:
                reader.join(timeout=timeout_s)
                is_alive = getattr(reader, "is_alive", None)
                if callable(is_alive) and is_alive():
                    raise RuntimeError(
                        f"{self.side} 从臂 {label} 读取器未在 {timeout_s:.1f}s 内停止，拒绝执行复位"
                    )
            if self._paused:
                logging.info(
                    "%s follower SDK readers paused for reset: %s",
                    self.side,
                    [label for _attr, _restart, label in self._paused],
                )
        except BaseException:
            self.resume()
            raise

    def resume(self) -> None:
        if self._resumed:
            return
        self._resumed = True
        for _attr, restart_name, label in self._paused:
            restart = getattr(self.arm, restart_name, None)
            if not callable(restart):
                raise RuntimeError(
                    f"{self.side} 从臂缺少 {restart_name}，无法在复位后恢复 {label} 读取器"
                )
            restart()
            logging.info("%s follower %s reader resumed after reset", self.side, label)


def _pause_enabled_arm_sdk_readers(robot: Any) -> list[_PausedArmSdkReaders]:
    paused: list[_PausedArmSdkReaders] = []
    try:
        for side, arm in _enabled_arm_items(robot):
            item = _PausedArmSdkReaders(arm, side)
            item.pause()
            paused.append(item)
    except BaseException:
        for item in reversed(paused):
            try:
                item.resume()
            except Exception:
                logging.exception("Failed restoring %s arm readers after pause error", item.side)
        raise
    return paused


def _resume_arm_sdk_readers(paused: list[_PausedArmSdkReaders]) -> None:
    failures: list[str] = []
    for item in paused:
        try:
            item.resume()
        except Exception as exc:
            failures.append(f"{item.side}: {exc}")
            logging.exception("Failed restoring %s arm readers after reset", item.side)
    if failures:
        raise RuntimeError("复位后恢复 RealMan 后台读取器失败: " + "; ".join(failures))


def _stop_enabled_controller_drag_before_ready(robot: Any) -> None:
    """Force both controllers out of drag before exposing the READY state."""

    paused_readers = _pause_enabled_arm_sdk_readers(robot)
    try:
        current_by_side: dict[str, tuple[float, ...]] = {}
        for side, arm in _enabled_arm_items(robot):
            follower = getattr(arm, "_follower_arm", None)
            current_by_side[side] = _read_follower_joints_deg(follower, side)
            stop_drag = getattr(follower, "rm_stop_drag_teach", None)
            if not callable(stop_drag):
                raise RuntimeError(f"{side} 从臂 SDK 不支持 rm_stop_drag_teach")
            code = int(stop_drag())
            if code != 0:
                logging.warning(
                    "%s follower startup drag stop returned code=%s; "
                    "verifying non-drag control with a current-pose hold",
                    side,
                    code,
                )

        for side, arm in _enabled_arm_items(robot):
            follower = getattr(arm, "_follower_arm", None)
            hold = getattr(follower, "rm_movej_canfd", None)
            if not callable(hold):
                raise RuntimeError(f"{side} 从臂 SDK 不支持 rm_movej_canfd 当前位姿锁定")
            code = int(hold(list(current_by_side[side]), False, 0))
            if code != 0:
                raise RuntimeError(f"{side} 从臂启动时当前位姿锁定失败，SDK 错误码 {code}")
            logging.info(
                "%s follower controller drag disabled and current pose latched before READY",
                side,
            )
    finally:
        _resume_arm_sdk_readers(paused_readers)


def _stop_drag_before_reset(
    arm: Any,
    side: str,
) -> tuple[float, ...]:
    """Exit any controller-side drag state before a planned reset trajectory."""

    follower = getattr(arm, "_follower_arm", None)
    if follower is None:
        raise RuntimeError(f"{side} 从臂未连接")
    current = _read_follower_joints_deg(follower, side)
    stop_drag = getattr(follower, "rm_stop_drag_teach", None)
    if not callable(stop_drag):
        raise RuntimeError(f"{side} 从臂 SDK 不支持 rm_stop_drag_teach")
    code = int(stop_drag())
    if code != 0:
        logging.warning(
            "%s follower pre-reset drag stop returned code=%s; "
            "the following rm_movej return code and measured motion remain fail-closed",
            side,
            code,
        )
    logging.info(
        "%s follower drag disabled before reset: joints_deg=%s",
        side,
        [round(value, 3) for value in current],
    )
    return current


def _validate_follower_joint_limits(follower: Any, target_deg: tuple[float, ...], side: str) -> None:
    read_min = getattr(follower, "rm_get_joint_min_pos", None)
    read_max = getattr(follower, "rm_get_joint_max_pos", None)
    if not callable(read_min) or not callable(read_max):
        logging.warning("%s 从臂 SDK 未提供关节限位查询；仅应用 [-360, 360]° 通用保护", side)
        return
    min_result = read_min()
    max_result = read_max()
    if (
        not isinstance(min_result, tuple)
        or len(min_result) < 2
        or not isinstance(max_result, tuple)
        or len(max_result) < 2
        or int(min_result[0]) != 0
        or int(max_result[0]) != 0
    ):
        raise RuntimeError(
            f"{side} 从臂关节限位读取失败: min={min_result!r}, max={max_result!r}"
        )
    minimum = tuple(float(value) for value in min_result[1])
    maximum = tuple(float(value) for value in max_result[1])
    if len(minimum) != 7 or len(maximum) != 7:
        raise RuntimeError(f"{side} 从臂关节限位长度不是 7")
    violations = [
        f"J{index + 1}={target:.3f}° not in [{lower:.3f}, {upper:.3f}]°"
        for index, (target, lower, upper) in enumerate(zip(target_deg, minimum, maximum))
        if not (np.isfinite(lower) and np.isfinite(upper) and lower <= target <= upper)
    ]
    if violations:
        raise RuntimeError(f"{side} 从臂复位目标超出控制器关节限位: " + "; ".join(violations))


def _stop_follower_motion(follower: Any, side: str) -> None:
    slow_stop = getattr(follower, "rm_set_arm_slow_stop", None)
    hard_stop = getattr(follower, "rm_set_arm_stop", None)
    try:
        if callable(slow_stop):
            code = int(slow_stop())
            if code == 0:
                logging.warning("%s 从臂复位运动已缓停", side)
                return
            logging.error("%s 从臂复位缓停失败，错误码 %s", side, code)
        if callable(hard_stop):
            code = int(hard_stop())
            logging.error("%s 从臂复位急停返回码 %s", side, code)
    except Exception:
        logging.exception("%s 从臂复位停止命令失败", side)


def _hard_stop_enabled_robot_arms(robot: Any) -> None:
    """Issue the RealMan controller hard-stop command to every enabled arm."""

    if robot is None:
        raise RuntimeError("robot is not connected")
    stopped: list[str] = []
    failures: list[str] = []
    for side, arm in _enabled_arm_items(robot):
        follower = getattr(arm, "_follower_arm", None)
        hard_stop = getattr(follower, "rm_set_arm_stop", None)
        if not callable(hard_stop):
            failures.append(f"{side}: SDK does not provide rm_set_arm_stop")
            continue
        try:
            code = int(hard_stop())
        except Exception as exc:
            failures.append(f"{side}: {type(exc).__name__}: {exc}")
            continue
        if code != 0:
            failures.append(f"{side}: rm_set_arm_stop returned {code}")
            continue
        stopped.append(side)
    logging.critical("Controller hard stop completed for arms=%s", stopped)
    if failures:
        raise RuntimeError("; ".join(failures))


def reset_right_arm_before_episode(
    *,
    robot: Any,
    state: Any,
    commands: "queue.Queue[str]",
    app: Any,
    config: EpisodeRightArmResetConfig,
    status: str = "CALIBRATING",
    phase_label: str = "before episode",
    max_delta_env_name: str = "EPISODE_RIGHT_ARM_RESET_MAX_START_DELTA_DEG",
) -> str | None:
    """Move only the right follower while force-drag teaching is disabled."""

    if not config.enabled:
        return None
    right_arm = getattr(robot, "right_arm", None)
    follower = getattr(right_arm, "_follower_arm", None)
    if right_arm is None or follower is None:
        raise RuntimeError("右从臂未启用或未连接，无法执行回合前复位")

    target = config.joints_deg
    state.update(
        status=status,
        is_recording=False,
        message=f"Resetting right arm {phase_label}",
    )
    logging.info(
        "Right-arm reset requested %s: target_deg=%s; left arm locked/untouched",
        phase_label,
        [round(value, 3) for value in target],
    )
    paused_readers: list[_PausedArmSdkReaders] = []
    motion_started = False
    cancelled: str | None = None
    try:
        if config.pause_readers:
            # Both arms have independent SDK handles, but pausing both avoids a
            # left-arm reader calling into the vendor C library while the right
            # arm executes the reset trajectory and its verification reads.
            paused_readers = _pause_enabled_arm_sdk_readers(robot)
            current = _read_follower_joints_deg(follower, "right")
            state_source, state_age_s = "sdk-direct-readers-paused", None
        else:
            current, state_source, state_age_s = _read_arm_joints_deg(
                right_arm,
                follower,
                "right",
                max_cache_age_s=config.state_max_age_s,
            )

        if config.stop_drag_first:
            # A previous process can exit while the controller remains in
            # force-drag mode. Leave it before sending the planned trajectory.
            current = _stop_drag_before_reset(right_arm, "right")
            state_source, state_age_s = "sdk-direct-after-drag-stop", None

        if config.query_controller_limits:
            _validate_follower_joint_limits(follower, target, "right")
        errors = tuple(abs(current_value - target_value) for current_value, target_value in zip(current, target))
        max_error = max(errors)
        logging.info(
            "Right-arm reset initial state: source=%s age_ms=%s current_deg=%s max_error=%.3fdeg",
            state_source,
            "n/a" if state_age_s is None else f"{state_age_s * 1000.0:.1f}",
            [round(value, 3) for value in current],
            max_error,
        )
        if max_error > config.max_start_delta_deg:
            joint_index = errors.index(max_error) + 1
            raise RuntimeError(
                f"右臂拒绝复位：起点到目标的最大关节差为 J{joint_index}={max_error:.3f}°，"
                f"超过 {max_delta_env_name}={config.max_start_delta_deg:.3f}°"
            )
        if max_error <= config.tolerance_deg:
            logging.info(
                "Right arm already at episode reset target; max_error=%.3fdeg; drag remains disabled",
                max_error,
            )
            return None

        movej = getattr(follower, "rm_movej", None)
        if not callable(movej):
            raise RuntimeError("right 从臂 SDK 不支持 rm_movej")

        state.update(message=f"Moving right arm {phase_label}; left arm locked and untouched")
        logging.info(
            "Sending nonblocking right-arm rm_movej reset command now; force drag is disabled; "
            "RealMan background readers are paused"
        )
        motion_started = True
        code = int(movej(list(target), int(config.speed), 0, 0, 0))
        if code != 0:
            raise RuntimeError(f"右臂 rm_movej 复位指令失败，SDK 错误码 {code}")
        logging.info(
            "Right-arm episode reset started: target_deg=%s speed=%s%% initial_max_error=%.3fdeg",
            [round(value, 3) for value in target],
            config.speed,
            max_error,
        )

        deadline = time.monotonic() + config.timeout_s
        stable_count = 0
        reached = False
        while time.monotonic() < deadline:
            for command in app.drain_commands(commands):
                if command in {"stop", "finish", "discard"}:
                    cancelled = "stop"
                    break
            if cancelled is not None:
                break
            current = _read_follower_joints_deg(follower, "right")
            max_error = max(abs(a - b) for a, b in zip(current, target))
            state.update(
                status=status,
                message=f"Resetting right arm: max joint error {max_error:.2f} deg",
            )
            if max_error <= config.tolerance_deg:
                stable_count += 1
                if stable_count >= config.stable_samples:
                    reached = True
                    break
            else:
                stable_count = 0
            time.sleep(1.0 / config.poll_hz)

        if cancelled is not None:
            _stop_follower_motion(follower, "right")
            motion_started = False
            logging.warning("Right-arm episode reset cancelled by command=%s", cancelled)
            return cancelled
        if not reached:
            raise RuntimeError(
                f"右臂复位在 {config.timeout_s:.1f}s 内未到位，最后最大关节误差 {max_error:.3f}°"
            )
        motion_started = False
        if config.settle_s > 0:
            time.sleep(config.settle_s)
        final_joints = _read_follower_joints_deg(follower, "right")
        state_source, state_age_s = "sdk-direct-readers-paused", None
        final_error = max(abs(a - b) for a, b in zip(final_joints, target))
        if final_error > config.tolerance_deg:
            raise RuntimeError(
                f"右臂复位静置后超出到位容差：{final_error:.3f}° > {config.tolerance_deg:.3f}°"
            )
        logging.info(
            "Right-arm episode reset complete: max_error=%.3fdeg source=%s age_ms=%s; "
            "left arm untouched; drag may now be enabled",
            final_error,
            state_source,
            "n/a" if state_age_s is None else f"{state_age_s * 1000.0:.1f}",
        )
        return None
    except BaseException:
        if motion_started:
            _stop_follower_motion(follower, "right")
        raise
    finally:
        if paused_readers:
            _resume_arm_sdk_readers(paused_readers)


def reset_then_enable_drag_before_episode(
    *,
    robot: Any,
    state: Any,
    commands: "queue.Queue[str]",
    app: Any,
    config: EpisodeRightArmResetConfig,
    drag_teach: DragTeachSession | None,
) -> str | None:
    """Enforce reset-before-drag ordering for a drag-controlled episode."""

    if drag_teach is not None and drag_teach.is_active:
        raise RuntimeError("回合开始前检测到拖动模式仍处于启用状态，拒绝在拖动中执行复位")
    result = reset_right_arm_before_episode(
        robot=robot,
        state=state,
        commands=commands,
        app=app,
        config=config,
    )
    if result is not None:
        return result
    if drag_teach is not None:
        state.update(status="CALIBRATING", is_recording=False, message="Enabling drag after reset")
        drag_teach.start()
        logging.info("Right-arm reset verified; left/right force drag enabled for this episode")
    return None


def capture_episode_start_pose(
    *,
    robot: Any,
    state: Any,
    episode_index: int,
    task: str,
    config: EpisodeReturnToStartConfig,
) -> EpisodeStartPose | None:
    """Snapshot the live right-follower pose immediately after start is accepted."""

    if not config.enabled:
        return None
    right_arm = getattr(robot, "right_arm", None)
    follower = getattr(right_arm, "_follower_arm", None)
    if right_arm is None or follower is None:
        raise RuntimeError("右从臂未启用或未连接，无法记录本回合复位位置")

    state.update(
        status="CALIBRATING",
        is_recording=False,
        message="Capturing right-arm start pose",
    )
    paused_readers: list[_PausedArmSdkReaders] = []
    try:
        if config.pause_readers:
            paused_readers = _pause_enabled_arm_sdk_readers(robot)
        joints_deg = _read_follower_joints_deg(follower, "right")

        robot_cfg = getattr(robot, "config", None)
        arm_cfg = getattr(robot_cfg, "right_arm_config", None) or getattr(
            right_arm,
            "config",
            None,
        )
        gripper_position: int | None = None
        if bool(getattr(arm_cfg, "connect_gripper", True)):
            gripper = getattr(right_arm, "_gripper", None)
            read_position = getattr(gripper, "read_position", None)
            if gripper is None or not callable(read_position):
                raise RuntimeError("右从臂夹爪已配置但不可读，无法完整记录本回合初始位姿")
            raw_position = float(read_position())
            if not np.isfinite(raw_position) or not 0.0 <= raw_position <= 1000.0:
                raise RuntimeError(
                    f"右从臂夹爪初始位置必须在 0..1000，收到 {raw_position!r}"
                )
            gripper_position = int(round(raw_position))
    finally:
        if paused_readers:
            _resume_arm_sdk_readers(paused_readers)

    pose = EpisodeStartPose(
        episode_index=int(episode_index),
        task=str(task),
        right_joint_positions_deg=joints_deg,
        right_gripper_position=gripper_position,
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    state.update(message="Right-arm start pose captured; enabling drag")
    logging.info(
        "Captured episode=%s right-arm start pose: joints_deg=%s gripper=%s",
        pose.episode_index,
        [round(value, 3) for value in pose.right_joint_positions_deg],
        pose.right_gripper_position,
    )
    return pose


def _return_right_gripper_to_episode_start(
    *,
    robot: Any,
    state: Any,
    pose: EpisodeStartPose,
    config: EpisodeReturnToStartConfig,
) -> None:
    if not config.restore_gripper:
        return
    if pose.right_gripper_position is None:
        raise RuntimeError("本回合没有可用的右夹爪初始状态，拒绝执行夹爪复位")

    right_arm = getattr(robot, "right_arm", None)
    gripper = getattr(right_arm, "_gripper", None)
    move_to_pos = getattr(gripper, "move_to_pos", None)
    read_position = getattr(gripper, "read_position", None)
    set_speed = getattr(gripper, "set_speed", None)
    if gripper is None or not callable(move_to_pos) or not callable(read_position):
        raise RuntimeError("右从臂夹爪未连接或不支持位置复位")
    if not bool(getattr(gripper, "is_connected", False)):
        raise RuntimeError("右从臂夹爪已断开，无法恢复初始状态")

    target = int(pose.right_gripper_position)
    safety_limits = getattr(robot, "_gripper_safety_limits", None)
    if safety_limits is not None and not (
        int(safety_limits.min_position) <= target <= int(safety_limits.max_position)
    ):
        raise RuntimeError(
            "记录的右夹爪初始位置超出当前安全范围："
            f"{target} not in {safety_limits.min_position}..{safety_limits.max_position}"
        )

    normal_speed = compat.env_int("GRIPPER_SPEED", 60)
    state.update(
        status="RESETTING",
        is_recording=False,
        message=f"Restoring right gripper to {target}",
    )
    try:
        if callable(set_speed) and not bool(set_speed(config.gripper_speed)):
            raise RuntimeError(
                f"右夹爪设置复位速度 {config.gripper_speed} 失败"
            )
        if not bool(move_to_pos(target)):
            raise RuntimeError(f"右夹爪恢复初始位置 {target} 的指令失败")

        deadline = time.monotonic() + config.gripper_timeout_s
        stable_count = 0
        last_position = int(round(float(read_position())))
        while time.monotonic() < deadline:
            last_position = int(round(float(read_position())))
            error = abs(last_position - target)
            state.update(
                status="RESETTING",
                message=f"Resetting right gripper: position error {error}",
            )
            if error <= config.gripper_tolerance:
                stable_count += 1
                if stable_count >= 2:
                    break
            else:
                stable_count = 0
            time.sleep(0.05)
        else:
            raise RuntimeError(
                f"右夹爪在 {config.gripper_timeout_s:.1f}s 内未恢复初始状态："
                f"target={target}, current={last_position}"
            )
    finally:
        if callable(set_speed):
            try:
                if not bool(set_speed(normal_speed)):
                    logging.warning("Failed restoring right gripper speed=%s", normal_speed)
            except Exception:
                logging.exception("Failed restoring right gripper speed after reset")

    logging.info(
        "Right gripper returned to episode start: target=%s tolerance=%s speed=%s",
        target,
        config.gripper_tolerance,
        config.gripper_speed,
    )


def return_right_arm_to_episode_start(
    *,
    robot: Any,
    state: Any,
    commands: "queue.Queue[str]",
    app: Any,
    pose: EpisodeStartPose | None,
    config: EpisodeReturnToStartConfig,
) -> None:
    """Slowly return only the right follower after drag has been stopped."""

    if not config.enabled:
        return
    if pose is None:
        raise RuntimeError("本回合没有可用的右臂初始位姿，拒绝执行复位")
    result = reset_right_arm_before_episode(
        robot=robot,
        state=state,
        commands=commands,
        app=app,
        config=config.reset_config(pose.right_joint_positions_deg),
        status="RESETTING",
        phase_label="to the captured episode start pose",
        max_delta_env_name="EPISODE_RETURN_TO_START_MAX_DELTA_DEG",
    )
    if result is not None:
        raise RuntimeError(f"右臂回到本回合初始位姿时被命令 {result!r} 中止")
    _return_right_gripper_to_episode_start(
        robot=robot,
        state=state,
        pose=pose,
        config=config,
    )
    state.update(
        status="RESETTING",
        is_recording=False,
        message="Right arm and gripper returned to captured start state",
    )


class TrajectoryGripperControlServer:
    """Expose the capture process's initialized grippers to a trajectory process."""

    def __init__(
        self,
        robot: Any,
        *,
        host: str,
        port: int,
        safety_limits: GripperSafetyLimits,
        initial_position: int = 950,
    ) -> None:
        self.robot = robot
        self.host = host
        self.port = int(port)
        self.safety_limits = safety_limits
        self._lock = threading.RLock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        initial = self.safety_limits.clamp_position(initial_position)
        self._targets = {"left": initial, "right": initial}
        self._dispatch_active = False

    def _gripper(self, side: str) -> Any:
        if side not in {"left", "right"}:
            raise ValueError("arm 必须是 left 或 right")
        arm = getattr(self.robot, f"{side}_arm", None)
        if arm is None:
            raise RuntimeError(f"采集进程中不存在 {side} 机械臂")
        gripper = getattr(arm, "_gripper", None)
        if gripper is None:
            raise RuntimeError(f"{side} 夹爪未连接")
        return gripper

    def _status_locked(self, side: str) -> dict[str, Any]:
        gripper = self._gripper(side)
        connected = bool(getattr(gripper, "is_connected", False))
        initialized = bool(getattr(gripper, "is_initialized", False))
        return {
            "ok": True,
            "arm": side,
            "connected": connected,
            "initialized": initialized,
            "position": self._targets[side],
            "dispatch_active": self._dispatch_active,
            "command_source": "program",
            "min_position": self.safety_limits.min_position,
            "max_position": self.safety_limits.max_position,
            "torque_limit": self.safety_limits.torque_limit,
            "owner": "capture_realman_x5_force_aligned_app",
        }

    def _settings_locked(self, side: str, speed: Any, torque: Any) -> dict[str, Any]:
        gripper = self._gripper(side)
        if not bool(getattr(gripper, "is_connected", False)):
            raise RuntimeError(f"{side} 夹爪未连接")
        if not bool(getattr(gripper, "is_initialized", False)):
            raise RuntimeError(f"{side} 夹爪未初始化")
        speed_value = int(speed)
        torque_value = int(torque)
        if not 10 <= speed_value <= 100:
            raise ValueError("speed 必须在 10..100 范围内")
        if not 10 <= torque_value <= 100:
            raise ValueError("torque 必须在 10..100 范围内")
        if torque_value > self.safety_limits.torque_limit:
            raise ValueError(
                "torque 不能超过启动限制 "
                f"{self.safety_limits.torque_limit}"
            )
        if not bool(gripper.set_speed(speed_value)):
            raise RuntimeError(f"{side} 夹爪设置速度失败")
        raw_gripper = getattr(gripper, "_gripper", None)
        if raw_gripper is None or not bool(raw_gripper.set_torque_limit(torque_value)):
            raise RuntimeError(f"{side} 夹爪设置力矩失败")
        result = self._status_locked(side)
        result.update(speed=speed_value, torque=torque_value)
        return result

    def _target_locked(self, side: str, position: Any) -> dict[str, Any]:
        gripper = self._gripper(side)
        if not bool(getattr(gripper, "is_connected", False)):
            raise RuntimeError(f"{side} 夹爪未连接")
        if not bool(getattr(gripper, "is_initialized", False)):
            raise RuntimeError(f"{side} 夹爪未初始化")
        if not self._dispatch_active:
            raise RuntimeError("采集 action 循环尚未运行；请先在采集界面开始录制")
        position_value = int(position)
        if not (
            self.safety_limits.min_position
            <= position_value
            <= self.safety_limits.max_position
        ):
            raise ValueError(
                "position 必须在启动限制 "
                f"{self.safety_limits.min_position}.."
                f"{self.safety_limits.max_position} 范围内"
            )
        self._targets[side] = position_value
        result = self._status_locked(side)
        result["target"] = position_value
        return result

    def set_dispatch_active(self, active: bool) -> None:
        with self._lock:
            self._dispatch_active = bool(active)

    def apply_program_targets(self, action: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            targets = dict(self._targets)
        overridden = dict(action)
        for side, position in targets.items():
            arm = getattr(self.robot, f"{side}_arm", None)
            arm_cfg = getattr(arm, "config", None)
            if arm is None or not bool(getattr(arm_cfg, "enabled", True)):
                continue
            key = f"{side}_{arm.GRIPPER_NAME}"
            overridden[key] = self.safety_limits.position_to_leader_value(
                position,
                arm,
            )
        return overridden

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                logging.debug("trajectory-gripper " + fmt, *args)

            def _json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != "/gripper/status":
                    self.send_error(404)
                    return
                side = parse_qs(parsed.query).get("arm", [""])[0]
                try:
                    with owner._lock:
                        self._json(owner._status_locked(side))
                except Exception as exc:
                    self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)

            def do_POST(self) -> None:  # noqa: N802
                if self.path not in {"/gripper/settings", "/gripper/target"}:
                    self.send_error(404)
                    return
                try:
                    length = min(4096, max(0, int(self.headers.get("Content-Length", "0"))))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("请求 JSON 必须是对象")
                    side = str(payload.get("arm", ""))
                    with owner._lock:
                        if self.path == "/gripper/settings":
                            result = owner._settings_locked(
                                side,
                                payload.get("speed"),
                                payload.get("torque"),
                            )
                        else:
                            result = owner._target_locked(side, payload.get("position"))
                    self._json(result)
                except Exception as exc:
                    self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="trajectory-gripper-control",
            daemon=True,
        )
        self._thread.start()
        logging.info("Trajectory gripper control: http://%s:%s", self.host, self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


@dataclass(frozen=True)
class TimedSample:
    t: float
    data: Any


class TimeRingBuffer:
    def __init__(self, maxlen: int) -> None:
        self._items: deque[TimedSample] = deque(maxlen=max(1, int(maxlen)))
        self._condition = threading.Condition()

    def add(self, t: float, data: Any) -> None:
        if t <= 0:
            t = time.perf_counter()
        with self._condition:
            self._items.append(TimedSample(float(t), data))
            self._condition.notify_all()

    def clear(self) -> None:
        with self._condition:
            self._items.clear()
            self._condition.notify_all()

    def latest(self, *, max_age_s: float | None = None) -> TimedSample | None:
        now = time.perf_counter()
        with self._condition:
            if not self._items:
                return None
            sample = self._items[-1]
        if max_age_s is not None and now - sample.t > max_age_s:
            return None
        return sample

    def _nearest_locked(self, target_t: float, *, max_age_s: float) -> TimedSample | None:
        if not self._items:
            return None
        best = min(self._items, key=lambda sample: abs(sample.t - target_t))
        if abs(best.t - target_t) > max_age_s:
            return None
        return best

    def nearest(self, target_t: float, *, max_age_s: float) -> TimedSample | None:
        with self._condition:
            return self._nearest_locked(target_t, max_age_s=max_age_s)

    def wait_nearest(self, target_t: float, *, max_age_s: float, timeout_s: float) -> TimedSample | None:
        deadline = time.perf_counter() + max(0.0, timeout_s)
        with self._condition:
            while True:
                sample = self._nearest_locked(target_t, max_age_s=max_age_s)
                if sample is not None:
                    return sample
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)

    def wait_nearest_prefer(
        self,
        target_t: float,
        *,
        max_age_s: float,
        preferred_max_age_s: float,
        timeout_s: float,
    ) -> TimedSample | None:
        """Wait for a close sample instead of immediately accepting a stale-but-valid one."""
        deadline = time.perf_counter() + max(0.0, timeout_s)
        best_valid: TimedSample | None = None
        preferred_max_age_s = max(0.0, preferred_max_age_s)
        with self._condition:
            while True:
                sample = self._nearest_locked(target_t, max_age_s=max_age_s)
                if sample is not None:
                    best_valid = sample
                    if abs(sample.t - target_t) <= preferred_max_age_s:
                        return sample
                    latest = self._items[-1] if self._items else None
                    if latest is not None and latest.t >= target_t:
                        return sample
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return best_valid
                self._condition.wait(timeout=remaining)

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)

    def describe(self, target_t: float) -> str:
        now = time.perf_counter()
        with self._condition:
            if not self._items:
                return "empty"
            oldest = self._items[0]
            latest = self._items[-1]
            nearest = min(self._items, key=lambda sample: abs(sample.t - target_t))
            n = len(self._items)
        return (
            f"n={n} "
            f"oldest_age_ms={(now - oldest.t) * 1000.0:.1f} "
            f"latest_age_ms={(now - latest.t) * 1000.0:.1f} "
            f"oldest_offset_ms={(oldest.t - target_t) * 1000.0:.1f} "
            f"latest_offset_ms={(latest.t - target_t) * 1000.0:.1f} "
            f"nearest_offset_ms={(nearest.t - target_t) * 1000.0:.1f}"
        )


def _parse_env_list(value: str) -> list[str]:
    items: list[str] = []
    for chunk in value.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            items.append(chunk)
    return items


def _resolve_foot_pedal_devices() -> list[str]:
    device_text = os.environ.get("FOOT_PEDAL_DEVICE", "").strip()
    devices_text = os.environ.get("FOOT_PEDAL_DEVICES", "").strip()
    if device_text or devices_text:
        return _parse_env_list(",".join(part for part in (device_text, devices_text) if part))

    default_path = "/dev/input/by-id/usb-PCsensor_FS20Pro-event-kbd"
    if Path(default_path).exists():
        return [default_path]

    matches = sorted(Path("/dev/input/by-id").glob("*FS20Pro*event-kbd"))
    if matches:
        return [str(matches[0])]

    hint = os.environ.get("FOOT_PEDAL_NAME_HINT", "PCsensor FS20Pro").strip()
    hints = tuple(part.strip() for part in hint.split(",") if part.strip())
    return listen_foot_pedal.discover_device_paths(hints)


def _parse_foot_pedal_key(value: str, key_names: dict[int, str]) -> int:
    text = value.strip()
    if not text:
        raise ValueError("empty foot-pedal key")
    try:
        return int(text, 0)
    except ValueError:
        pass

    name_to_code = {name.upper(): code for code, name in key_names.items()}
    upper = text.upper()
    if upper.startswith("KEY_CODE_"):
        return int(upper.removeprefix("KEY_CODE_"), 0)
    if not upper.startswith(("KEY_", "BTN_")):
        upper = "KEY_" + upper
    if upper not in name_to_code:
        raise ValueError(f"unknown input key {value!r}")
    return name_to_code[upper]


class FootPedalCommandLoop:
    def __init__(
        self,
        commands: "queue.Queue[str]",
        state: Any,
        *,
        emergency_stop: Callable[[], None] | None = None,
    ) -> None:
        self.commands = commands
        self.state = state
        self.emergency_stop = emergency_stop
        self.grab = compat.env_bool("FOOT_PEDAL_GRAB", True)
        self.debounce_s = compat.env_float("FOOT_PEDAL_DEBOUNCE_S", 0.25)
        self.reconnect_s = max(
            0.1,
            compat.env_float("FOOT_PEDAL_RECONNECT_S", 0.5),
        )
        self.two_pedal_stop_enabled = compat.env_bool(
            "FOOT_PEDAL_TWO_PEDAL_STOP_ENABLED",
            False,
        )
        self.chord_window_s = compat.env_float(
            "FOOT_PEDAL_TWO_PEDAL_WINDOW_S",
            0.0,
        )
        if self.chord_window_s < 0:
            raise ValueError("FOOT_PEDAL_TWO_PEDAL_WINDOW_S must be >= 0")
        self.paths = _resolve_foot_pedal_devices()
        self.devices: list[listen_foot_pedal.EventDevice] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.key_names = listen_foot_pedal.load_key_names()
        self.primary_key_to_command = {
            _parse_foot_pedal_key(os.environ.get("FOOT_PEDAL_START_KEY", "KEY_ESC"), self.key_names): "start",
            _parse_foot_pedal_key(os.environ.get("FOOT_PEDAL_FINISH_KEY", "KEY_LEFT"), self.key_names): "finish",
            _parse_foot_pedal_key(os.environ.get("FOOT_PEDAL_DISCARD_KEY", "KEY_RIGHT"), self.key_names): "discard",
        }
        if len(self.primary_key_to_command) != 3:
            raise ValueError("foot-pedal start/finish/discard keys must be distinct")
        self.key_to_command = dict(self.primary_key_to_command)
        stop_key = os.environ.get("FOOT_PEDAL_STOP_KEY", "").strip()
        if stop_key:
            stop_code = _parse_foot_pedal_key(stop_key, self.key_names)
            if stop_code in self.primary_key_to_command:
                raise ValueError("FOOT_PEDAL_STOP_KEY must differ from start/finish/discard keys")
            self.key_to_command[stop_code] = "stop"
        self._last_press_by_code: dict[int, float] = {}
        self._pressed_primary_codes: set[int] = set()
        self._pending_primary_commands: dict[int, tuple[float, str]] = {}
        self._two_pedal_stop_latched = False

    def start(self) -> None:
        if not self.paths:
            raise RuntimeError(
                "FOOT_PEDAL_CONTROL=true but no input device was found. "
                "Set FOOT_PEDAL_DEVICE=/dev/input/by-id/usb-PCsensor_FS20Pro-event-kbd."
            )
        self.devices = listen_foot_pedal.open_devices(self.paths, grab=self.grab)
        device_text = ", ".join(f"{dev.path}({dev.name or 'unnamed'})" for dev in self.devices)
        mapping_text = ", ".join(
            f"{self.key_names.get(code, code)}->{cmd}" for code, cmd in sorted(self.key_to_command.items())
        )
        logging.info(
            "Foot pedal control enabled: devices=%s grab=%s mapping=%s "
            "two_pedal_stop=%s chord_window_ms=%.0f",
            device_text,
            self.grab,
            mapping_text,
            self.two_pedal_stop_enabled,
            self.chord_window_s * 1000.0,
        )
        self._thread = threading.Thread(target=self._run, name="foot-pedal-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.devices:
            listen_foot_pedal.close_devices(self.devices, grab=self.grab)
            self.devices = []

    def _state_snapshot(self) -> dict[str, Any]:
        snapshot = getattr(self.state, "snapshot", None)
        if callable(snapshot):
            try:
                return dict(snapshot())
            except Exception:
                logging.exception("Failed reading capture state for foot pedal")
        return {
            "status": getattr(self.state, "status", ""),
            "is_recording": bool(getattr(self.state, "is_recording", False)),
        }

    def _command_allowed(self, cmd: str) -> bool:
        snap = self._state_snapshot()
        status = str(snap.get("status", ""))
        is_recording = bool(snap.get("is_recording", False))
        if cmd == "start":
            return status == "READY" and not is_recording
        if cmd in {"finish", "discard"}:
            return is_recording
        if cmd == "stop":
            return status not in {"DONE", "STOPPING"}
        return False

    def _emit_command(self, code: int, cmd: str, *, now: float | None = None) -> None:
        if now is None:
            now = time.perf_counter()
        last_press = self._last_press_by_code.get(code, 0.0)
        if now - last_press < self.debounce_s:
            return
        self._last_press_by_code[code] = now

        key_name = self.key_names.get(code, f"KEY_CODE_{code}")
        if not self._command_allowed(cmd):
            logging.info("Foot pedal ignored key=%s cmd=%s in current state", key_name, cmd)
            snap = self._state_snapshot()
            status = str(snap.get("status", "UNKNOWN"))
            try:
                self.state.update(
                    message=f"Pedal {key_name}: {cmd} ignored while {status}",
                )
            except Exception:
                logging.exception("Failed publishing ignored foot-pedal event to UI")
            return
        self.commands.put(cmd)
        logging.info("Foot pedal command key=%s cmd=%s", key_name, cmd)
        try:
            self.state.update(message=f"Pedal {key_name}: {cmd} accepted")
        except Exception:
            logging.exception("Failed publishing foot-pedal command to UI")

    def _trigger_two_pedal_stop(self) -> None:
        if self._two_pedal_stop_latched:
            return
        self._two_pedal_stop_latched = True
        self._pending_primary_commands.clear()
        pressed_names = [
            self.key_names.get(code, f"KEY_CODE_{code}")
            for code in sorted(self._pressed_primary_codes)
        ]
        logging.critical(
            "TWO-PEDAL SOFTWARE EMERGENCY STOP: pressed=%s; "
            "requesting controller arm stop and capture shutdown",
            pressed_names,
        )
        try:
            self.state.update(
                stop_requested=True,
                message="Two-pedal emergency stop requested",
            )
        except Exception:
            logging.exception("Failed updating capture state for two-pedal stop")

        # The direct controller call is best-effort and is not a substitute
        # for the safety-rated physical emergency-stop circuit. Queue capture
        # shutdown in finally so it happens even when the SDK reports failure,
        # but only after the hard-stop attempts have completed; this prevents
        # the main thread from racing a post-drag hold command against them.
        try:
            if self.emergency_stop is not None:
                self.emergency_stop()
        except Exception as exc:
            logging.exception("Two-pedal controller arm stop failed")
            try:
                self.state.update(
                    last_error=f"Two-pedal controller arm stop failed: {exc}",
                    message="Two-pedal stop requested; controller stop reported an error",
                )
            except Exception:
                pass
        finally:
            self.commands.put("stop")

    def _handle_key_event(self, code: int, value: int, *, now: float | None = None) -> None:
        if now is None:
            now = time.perf_counter()
        is_primary = code in self.primary_key_to_command

        if value == 0:
            if is_primary:
                self._pressed_primary_codes.discard(code)
                if not self._pressed_primary_codes:
                    self._two_pedal_stop_latched = False
            return
        if value != 1:
            return

        if not is_primary:
            cmd = self.key_to_command.get(code)
            if cmd is not None:
                self._emit_command(code, cmd, now=now)
            return

        self._pressed_primary_codes.add(code)
        if self.two_pedal_stop_enabled and len(self._pressed_primary_codes) >= 2:
            self._trigger_two_pedal_stop()
            return

        cmd = self.primary_key_to_command[code]
        if self.two_pedal_stop_enabled and self.chord_window_s > 0:
            self._pending_primary_commands[code] = (now + self.chord_window_s, cmd)
        else:
            self._emit_command(code, cmd, now=now)

    def _flush_pending_primary_commands(self, *, now: float | None = None) -> None:
        if now is None:
            now = time.perf_counter()
        ready = sorted(
            (
                (deadline, code, cmd)
                for code, (deadline, cmd) in self._pending_primary_commands.items()
                if deadline <= now
            ),
        )
        for _deadline, code, cmd in ready:
            self._pending_primary_commands.pop(code, None)
            if not self._two_pedal_stop_latched:
                self._emit_command(code, cmd, now=now)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if not self.devices:
                    # Resolve the stable by-id symlink again because unplugging
                    # and reconnecting a USB pedal normally changes eventN.
                    self.paths = _resolve_foot_pedal_devices()
                    if not self.paths:
                        raise FileNotFoundError("foot pedal input device not found")
                    self.devices = listen_foot_pedal.open_devices(
                        self.paths,
                        grab=self.grab,
                    )
                    self._pressed_primary_codes.clear()
                    self._pending_primary_commands.clear()
                    self._two_pedal_stop_latched = False
                    logging.info(
                        "Foot pedal reconnected: devices=%s",
                        ", ".join(dev.path for dev in self.devices),
                    )
                    try:
                        self.state.update(
                            last_error="",
                            message="Foot pedal reconnected",
                        )
                    except Exception:
                        logging.exception("Failed publishing foot-pedal reconnect to UI")

                now = time.perf_counter()
                timeout_s = 0.2
                if self._pending_primary_commands:
                    next_deadline = min(
                        deadline
                        for deadline, _cmd in self._pending_primary_commands.values()
                    )
                    timeout_s = min(timeout_s, max(0.0, next_deadline - now))
                ready, _, _ = select.select(self.devices, [], [], timeout_s)
                for dev in ready:
                    for _sec, _usec, event_type, code, value in listen_foot_pedal.read_events(dev):
                        if event_type != listen_foot_pedal.EV_KEY:
                            continue
                        self._handle_key_event(code, value)
                self._flush_pending_primary_commands()
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logging.warning(
                    "Foot pedal disconnected; retrying in %.1fs: %s",
                    self.reconnect_s,
                    exc,
                )
                if self.devices:
                    listen_foot_pedal.close_devices(self.devices, grab=self.grab)
                    self.devices = []
                self._pressed_primary_codes.clear()
                self._pending_primary_commands.clear()
                self._two_pedal_stop_latched = False
                try:
                    self.state.update(
                        last_error=str(exc),
                        message=f"Foot pedal disconnected; reconnecting: {exc}",
                    )
                except Exception:
                    pass
                self._stop_event.wait(self.reconnect_s)


def _start_foot_pedal_control(
    commands: "queue.Queue[str]",
    state: Any,
    *,
    emergency_stop: Callable[[], None] | None = None,
) -> FootPedalCommandLoop | None:
    if not compat.env_bool("FOOT_PEDAL_CONTROL", False):
        return None
    loop = FootPedalCommandLoop(
        commands,
        state,
        emergency_stop=emergency_stop,
    )
    try:
        loop.start()
    except PermissionError as exc:
        raise RuntimeError(
            f"Cannot open foot pedal input device: {exc}. "
            "Run with input-device permission or set FOOT_PEDAL_CONTROL=false."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot open/grab foot pedal input device: {exc}") from exc
    return loop


def _is_wrist_image_key(key: str) -> bool:
    return "wrist" in key


def _is_tactile_stream_key(key: str) -> bool:
    return "tactile_" in key or key.startswith(("depth_deformation.", "deformation.", "shear.", "depth."))


def _source_key_candidates(feature_key: str) -> tuple[str, ...]:
    """Return producer keys that may carry the same physical stream."""
    keys = [feature_key]
    double_prefix = "depth_deformation.depth_deformation."
    single_prefix = "depth_deformation."
    if feature_key.startswith(double_prefix):
        keys.append(single_prefix + feature_key[len(double_prefix) :])
    elif feature_key.startswith(single_prefix + "tactile_"):
        keys.append(single_prefix + feature_key)
    return tuple(dict.fromkeys(keys))


def _tactile_frame_to_preview_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert X5 tactile arrays into an RGB heatmap for live preview only."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        hwc = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] in {1, 2, 3, 4} and arr.shape[-1] not in {1, 2, 3, 4}:
        hwc = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 3:
        hwc = arr
    else:
        raise ValueError(f"expected tactile frame ndim 2/3, got shape={arr.shape}")

    data = hwc.astype(np.float32, copy=False)
    finite = np.isfinite(data)
    if not finite.any():
        gray = np.zeros(data.shape[:2], dtype=np.uint8)
    else:
        data = np.where(finite, data, 0.0)
        channels = data.shape[-1]
        if channels >= 3 and float(np.nanmax(data)) > 1000.0:
            depth = np.abs(data[..., 0])
            dx = data[..., 1] - 30000.0
            dy = data[..., 2] - 30000.0
            intensity = depth + 0.35 * np.sqrt(dx * dx + dy * dy)
        elif channels >= 3:
            depth = np.abs(data[..., 0] - np.nanmedian(data[..., 0]))
            dx = data[..., 1] - np.nanmedian(data[..., 1])
            dy = data[..., 2] - np.nanmedian(data[..., 2])
            intensity = depth + 0.35 * np.sqrt(dx * dx + dy * dy)
        elif channels >= 2:
            dx = data[..., 0] - np.nanmedian(data[..., 0])
            dy = data[..., 1] - np.nanmedian(data[..., 1])
            intensity = np.sqrt(dx * dx + dy * dy)
        else:
            intensity = np.abs(data[..., 0] - np.nanmedian(data[..., 0]))

        intensity = np.where(np.isfinite(intensity), intensity, 0.0)
        active = intensity[intensity > 0.0]
        vmax = float(np.percentile(active, 99.0)) if active.size else 0.0
        if vmax <= 1e-6:
            gray = np.zeros(intensity.shape, dtype=np.uint8)
        else:
            gray = np.clip(intensity / vmax * 255.0, 0.0, 255.0).astype(np.uint8)

    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    bgr = cv2.applyColorMap(gray, colormap)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _tactile_sensor_name_from_key(key: str) -> str | None:
    for name in ("left_left", "left_right", "right_left", "right_right"):
        if key.endswith(name) or f"tactile_{name}" in key:
            return name
    return None


def _array_content_signature(frame: np.ndarray) -> tuple[int, float, float]:
    arr = np.asarray(frame)
    if arr.size == 0:
        return 0, 0.0, 0.0
    flat = arr.reshape(-1)
    stride = max(1, flat.size // 2048)
    sample = flat[::stride].astype(np.float64, copy=False)
    checksum = int(np.sum(sample * 1009.0) % 1_000_000_007)
    return checksum, float(np.nanmin(sample)), float(np.nanmax(sample))


class ProducerHub:
    def __init__(
        self,
        robot: Any,
        settings: Any,
        *,
        tactile_frame_publisher: TactileFramePublisher | None = None,
    ) -> None:
        self.robot = robot
        self.settings = settings
        self.tactile_frame_publisher = tactile_frame_publisher
        self.buffers: dict[str, TimeRingBuffer] = {}
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.errors: queue.Queue[BaseException] = queue.Queue()
        self.last_preview_t = 0.0
        self.previewer = compat.get_rgb_previewer()

        self.buffer_seconds = max(1.0, compat.env_float("ALIGNED_BUFFER_SECONDS", 4.0))
        self.state_hz = compat.env_float("ALIGNED_STATE_HZ", 100.0)
        self.wrist_hz = compat.env_float("ALIGNED_WRIST_POLL_HZ", 90.0)
        self.d405_hz = compat.env_float("ALIGNED_D405_POLL_HZ", 60.0)
        self.tactile_hz = compat.env_float("ALIGNED_TACTILE_POLL_HZ", 120.0)
        self.state_buffer_seconds = max(
            self.buffer_seconds, compat.env_float("ALIGNED_STATE_BUFFER_SECONDS", 8.0)
        )
        self.wrist_buffer_seconds = max(
            self.buffer_seconds, compat.env_float("ALIGNED_WRIST_BUFFER_SECONDS", 8.0)
        )
        self.d405_buffer_seconds = max(
            self.buffer_seconds, compat.env_float("ALIGNED_D405_BUFFER_SECONDS", 6.0)
        )
        self.tactile_buffer_seconds = max(
            self.buffer_seconds, compat.env_float("ALIGNED_TACTILE_BUFFER_SECONDS", 8.0)
        )
        self.preview_hz = compat.env_float("ALIGNED_PREVIEW_HZ", compat.env_float("PREVIEW_FPS", 5.0))
        previewer_has_fixed_thresholds = bool(getattr(self.previewer, "tactile_thresholds", {}) or {})
        self.tactile_preview_direct_rgb = compat.env_bool(
            "ALIGNED_TACTILE_PREVIEW_DIRECT_RGB",
            not previewer_has_fixed_thresholds,
        )
        logging.info(
            "Aligned tactile preview direct_rgb=%s fixed_thresholds=%s",
            self.tactile_preview_direct_rgb,
            previewer_has_fixed_thresholds,
        )
        self.tactile_read_direct = compat.env_bool("ALIGNED_TACTILE_READ_DIRECT", False)
        self.tactile_debug_every_s = compat.env_float("ALIGNED_TACTILE_DEBUG_EVERY_S", 1.0)
        self.tactile_monitor_during_episode = compat.env_bool("ALIGNED_TACTILE_MONITOR_DURING_EPISODE", False)
        self.tactile_reconnect_on_stall = compat.env_bool("ALIGNED_TACTILE_RECONNECT_ON_STALL", False)
        self.tactile_stall_timeout_s = compat.env_float("ALIGNED_TACTILE_STALL_TIMEOUT_S", 3.0)
        self.tactile_reconnect_during_episode = compat.env_bool(
            "ALIGNED_TACTILE_RECONNECT_DURING_EPISODE",
            False,
        )

    def _buffer_len_for_key(self, key: str) -> int:
        fps = max(1.0, float(getattr(self.settings, "fps", 30.0)))
        if key == "__state__":
            return max(60, int(self.state_hz * self.state_buffer_seconds))
        if _is_wrist_image_key(key):
            return max(60, int(fps * self.wrist_buffer_seconds))
        if _is_tactile_stream_key(key):
            return max(240, int(self.tactile_hz * self.tactile_buffer_seconds))
        if "d405" in key.lower():
            return max(120, int(self.d405_hz * self.d405_buffer_seconds))
        return max(30, int(max(fps, 30.0) * self.buffer_seconds))

    def buffer(self, key: str) -> TimeRingBuffer:
        if key not in self.buffers:
            self.buffers[key] = TimeRingBuffer(self._buffer_len_for_key(key))
        return self.buffers[key]

    def clear_buffers(self, keys: list[str]) -> None:
        for key in keys:
            for candidate in _source_key_candidates(key):
                self.buffer(candidate).clear()

    def describe_key(self, key: str, target_t: float) -> str:
        parts: list[str] = []
        for candidate in _source_key_candidates(key):
            buf = self.buffer(candidate)
            parts.append(f"{candidate}:{buf.describe(target_t)}")
        return "; ".join(parts)

    def start(self) -> None:
        self._start_thread("aligned_state_producer", self._state_loop)
        self._start_wrist_threads()
        self._start_shared_camera_threads()
        self._start_tactile_thread()
        if self.previewer is not None:
            self._start_thread("aligned_preview_loop", self._preview_loop)

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=1.0)

    def raise_if_error(self) -> None:
        try:
            err = self.errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError(f"aligned producer failed: {err}") from err

    def wait_ready(self, required_keys: list[str], *, timeout_s: float, max_age_s: float) -> None:
        deadline = time.perf_counter() + max(0.0, timeout_s)
        missing = list(required_keys)
        while time.perf_counter() < deadline:
            self.raise_if_error()
            missing = [
                key
                for key in required_keys
                if not any(
                    self.buffer(candidate).latest(max_age_s=max_age_s) is not None
                    for candidate in _source_key_candidates(key)
                )
            ]
            if not missing:
                return
            time.sleep(0.02)
        raise RuntimeError(
            "Aligned recorder sources are not ready/fresh: "
            + ", ".join(missing[:16])
            + (" ..." if len(missing) > 16 else "")
        )

    def wait_stable(
        self,
        required_keys: list[str],
        *,
        timeout_s: float,
        max_age_s: float,
        stable_s: float,
        min_updates: int,
    ) -> None:
        deadline = time.perf_counter() + max(0.0, timeout_s)
        stable_since: float | None = None
        last_t_by_key: dict[str, float] = {}
        update_count_by_key: dict[str, int] = {}
        missing = list(required_keys)
        while time.perf_counter() < deadline:
            self.raise_if_error()
            latest_by_key: dict[str, TimedSample] = {}
            missing = []
            for key in required_keys:
                sample = None
                for candidate in _source_key_candidates(key):
                    sample = self.buffer(candidate).latest(max_age_s=max_age_s)
                    if sample is not None:
                        break
                if sample is None:
                    missing.append(key)
                else:
                    latest_by_key[key] = sample

            now = time.perf_counter()
            if missing:
                stable_since = None
                update_count_by_key.clear()
                last_t_by_key.clear()
                time.sleep(0.02)
                continue

            if stable_since is None:
                stable_since = now
                update_count_by_key = {key: 0 for key in required_keys}
                last_t_by_key = {}

            for key, sample in latest_by_key.items():
                last_t = last_t_by_key.get(key)
                if last_t is None or sample.t != last_t:
                    update_count_by_key[key] = update_count_by_key.get(key, 0) + 1
                    last_t_by_key[key] = sample.t

            if now - stable_since >= max(0.0, stable_s) and all(
                update_count_by_key.get(key, 0) >= max(1, min_updates) for key in required_keys
            ):
                return
            time.sleep(0.02)

        details = [self.describe_key(key, time.perf_counter()) for key in missing[:16]] if missing else []
        raise RuntimeError(
            "Aligned recorder sources did not stay stable/fresh before episode start: "
            + ", ".join(missing[:16] or required_keys[:16])
            + (" ..." if len(missing or required_keys) > 16 else "")
            + (("; debug=" + " | ".join(details)) if details else "")
        )

    def _start_thread(self, name: str, target: Any) -> None:
        def guarded() -> None:
            try:
                target()
            except BaseException as exc:  # noqa: BLE001
                if not self.stop_event.is_set():
                    logging.exception("%s failed", name)
                    self.errors.put(exc)
                    self.stop_event.set()

        thread = threading.Thread(target=guarded, name=name, daemon=True)
        thread.start()
        self.threads.append(thread)

    def _sleep_loop(self, hz: float) -> None:
        if hz > 0:
            time.sleep(max(0.0005, 1.0 / hz))
        else:
            time.sleep(0.001)

    def _read_arm_scalar_obs(self, arm: Any) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        state_reader = getattr(arm, "_follower_state_reader", None)
        if state_reader is not None:
            joints = state_reader.get_state()
            for i, joint in enumerate(getattr(arm, "JOINT_NAMES", [])):
                obs[joint] = float(joints[i]) if i < len(joints) else 0.0
        else:
            follower = getattr(arm, "_follower_arm", None)
            if follower is not None:
                ret, joints_deg = follower.rm_get_joint_degree()
                if ret == 0:
                    use_degrees = bool(getattr(getattr(arm, "config", None), "use_degrees", False))
                    for i, joint in enumerate(getattr(arm, "JOINT_NAMES", [])):
                        value = float(joints_deg[i]) if i < len(joints_deg) else 0.0
                        obs[joint] = value if use_degrees else float(np.radians(value))
                else:
                    for joint in getattr(arm, "JOINT_NAMES", []):
                        obs[joint] = 0.0

        gripper_name = getattr(arm, "GRIPPER_NAME", "main_gripper")
        gripper = getattr(arm, "_gripper", None)
        try:
            if gripper is not None and getattr(gripper, "is_connected", False):
                obs[gripper_name] = float(gripper.read_position()) / 1000.0
            else:
                obs[gripper_name] = 0.0
        except Exception as exc:
            logging.debug("gripper state read failed: %s", exc)
            obs[gripper_name] = 0.0

        force_reader = getattr(arm, "_force_sensor_reader", None)
        if force_reader is not None:
            wrench = force_reader.get_wrench()
            for i, name in enumerate(getattr(arm, "FORCE_SENSOR_NAMES", [])):
                obs[name] = float(wrench[i]) if i < len(wrench) else 0.0
        elif bool(getattr(getattr(arm, "config", None), "connect_force_sensor", False)):
            for name in getattr(arm, "FORCE_SENSOR_NAMES", []):
                obs[name] = 0.0
        return obs

    def _state_loop(self) -> None:
        while not self.stop_event.is_set():
            t = time.perf_counter()
            obs: dict[str, Any] = {}
            if getattr(self.robot.config.left_arm_config, "enabled", False):
                left_obs = self._read_arm_scalar_obs(self.robot.left_arm)
                obs.update({f"left_{key}": value for key, value in left_obs.items()})
            if getattr(self.robot.config.right_arm_config, "enabled", False):
                right_obs = self._read_arm_scalar_obs(self.robot.right_arm)
                obs.update({f"right_{key}": value for key, value in right_obs.items()})
            self.buffer("__state__").add(t, obs)
            self._sleep_loop(self.state_hz)

    def _start_wrist_threads(self) -> None:
        for side, arm_cfg_name, arm_name in (
            ("left", "left_arm_config", "left_arm"),
            ("right", "right_arm_config", "right_arm"),
        ):
            arm_cfg = getattr(self.robot.config, arm_cfg_name)
            if not getattr(arm_cfg, "enabled", False):
                continue
            arm = getattr(self.robot, arm_name)
            receivers = getattr(arm, "_tcp_receivers", {})
            for stream_key, receiver in receivers.items():
                feature_key = f"{side}_{stream_key}"
                self._start_thread(
                    f"aligned_wrist_{feature_key}",
                    lambda r=receiver, k=feature_key: self._wrist_loop(r, k),
                )

    def _wrist_loop(self, receiver: Any, key: str) -> None:
        last_t = 0.0
        last_processed_id = 0
        timeout_s = max(0.0, compat.env_float("ALIGNED_WRIST_READ_TIMEOUT_S", 0.01))
        cache = getattr(receiver, "_kd_tacmae_processed_cache", None)
        add_processed_callback = getattr(cache, "add_processed_callback", None)
        remove_processed_callback = getattr(cache, "remove_processed_callback", None)
        if callable(add_processed_callback):
            handoff_maxsize = max(1, compat.env_int("ALIGNED_WRIST_HANDOFF_QUEUE_FRAMES", 240))
            handoff: queue.Queue[tuple[float, int, np.ndarray] | None] = queue.Queue(maxsize=handoff_maxsize)

            def on_processed(source_t: float, frame_id: int, frame_bgr: np.ndarray) -> None:
                if self.stop_event.is_set():
                    return
                try:
                    handoff.put_nowait((float(source_t), int(frame_id), frame_bgr))
                except queue.Full:
                    self.errors.put(
                        RuntimeError(
                            f"aligned wrist handoff queue overflow for {key}: "
                            f"size={handoff.qsize()} max={handoff_maxsize} frame_id={frame_id}"
                        )
                    )
                    self.stop_event.set()

            add_processed_callback(on_processed)
            logging.info("Aligned wrist producer %s using processed callback handoff queue=%s", key, handoff_maxsize)
            try:
                while not self.stop_event.is_set():
                    try:
                        item = handoff.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is None:
                        continue
                    source_t, frame_id, frame_bgr = item
                    if frame_id <= last_processed_id or not isinstance(frame_bgr, np.ndarray):
                        continue
                    last_processed_id = frame_id
                    if source_t <= 0:
                        source_t = time.perf_counter()
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    self.buffer(key).add(float(source_t), np.ascontiguousarray(frame_rgb))
            finally:
                if callable(remove_processed_callback):
                    try:
                        remove_processed_callback(on_processed)
                    except Exception:
                        pass
            return

        while not self.stop_event.is_set():
            get_frames_after = getattr(receiver, "get_processed_frames_after", None)
            if callable(get_frames_after):
                frames = get_frames_after(last_processed_id)
                if frames:
                    for source_t, frame_id, frame_bgr in frames:
                        last_processed_id = max(last_processed_id, int(frame_id))
                        if not isinstance(frame_bgr, np.ndarray):
                            continue
                        if source_t <= 0:
                            source_t = time.perf_counter()
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        self.buffer(key).add(float(source_t), np.ascontiguousarray(frame_rgb))
                self._sleep_loop(self.wrist_hz)
                continue

            frame_bgr = None
            try:
                frame_bgr = receiver.async_read(timeout_s=timeout_s, require_new=False)
            except TypeError:
                frame_bgr = receiver.async_read()
            if isinstance(frame_bgr, np.ndarray):
                source_t = 0.0
                get_t = getattr(receiver, "get_latest_frame_time_perf", None)
                if callable(get_t):
                    try:
                        source_t = float(get_t() or 0.0)
                    except Exception:
                        source_t = 0.0
                if source_t <= 0:
                    source_t = time.perf_counter()
                if source_t != last_t:
                    last_t = source_t
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    self.buffer(key).add(source_t, np.ascontiguousarray(frame_rgb))
            self._sleep_loop(self.wrist_hz)

    def _start_shared_camera_threads(self) -> None:
        for cam_name, camera in getattr(self.robot, "shared_cameras", {}).items():
            self._start_thread(
                f"aligned_shared_camera_{cam_name}",
                lambda c=camera, k=cam_name: self._shared_camera_loop(c, k),
            )

    def _shared_camera_loop(self, camera: Any, key: str) -> None:
        last_t = 0.0
        while not self.stop_event.is_set():
            image = camera.async_read()
            if isinstance(image, np.ndarray):
                source_t = getattr(camera, "latest_timestamp", None)
                if not isinstance(source_t, (int, float)) or source_t <= 0:
                    source_t = time.perf_counter()
                source_t = float(source_t)
                if source_t != last_t:
                    last_t = source_t
                    self.buffer(key).add(source_t, np.ascontiguousarray(image))
            self._sleep_loop(self.d405_hz)

    def _start_tactile_thread(self) -> None:
        reader = getattr(self.robot, "tactile_sidecar", None) or getattr(self.robot, "x5_tactile", None)
        if reader is not None:
            logging.info("Aligned tactile producer enabled: reader=%s direct_sensors=%s", type(reader).__name__, hasattr(reader, "_sensors"))
            self._start_thread("aligned_tactile_producer", lambda: self._tactile_loop(reader))

    def _read_tactile_direct(
        self,
        reader: Any,
        *,
        left_enabled: bool,
        right_enabled: bool,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        sensors = getattr(reader, "_sensors", None)
        if not isinstance(sensors, dict):
            return {}, {}

        if hasattr(reader, "_enabled_names"):
            names = reader._enabled_names(left_enabled=left_enabled, right_enabled=right_enabled)
        else:
            names = list(getattr(reader, "_active_names", []) or sensors)
        images: dict[str, np.ndarray] = {}
        times: dict[str, float] = {}
        for name in names:
            sensor = sensors.get(name)
            if sensor is None:
                continue
            sensor_images = sensor.read_images()
            images.update(sensor_images)
        return images, times

    def _tactile_loop(self, reader: Any) -> None:
        last_t_by_key: dict[str, float] = {}
        last_sig_by_key: dict[str, int] = {}
        last_changed_t_by_key: dict[str, float] = {}
        last_frame_received_t_by_key: dict[str, float] = {}
        left_enabled = bool(getattr(self.robot.config.left_arm_config, "enabled", False))
        right_enabled = bool(getattr(self.robot.config.right_arm_config, "enabled", False))
        last_empty_warn_t = 0.0
        last_debug_t = 0.0
        last_reconnect_t = 0.0
        last_seq_by_key: dict[str, int] = {}
        use_cache_history = compat.env_bool("ALIGNED_TACTILE_USE_CACHE_HISTORY", True)
        history_batch = max(1, compat.env_int("ALIGNED_TACTILE_HISTORY_BATCH", 1024))
        logged_first = False
        last_any_frame_t = time.perf_counter()

        def reconnect_tactile(reason: str, now: float) -> None:
            nonlocal last_reconnect_t, logged_first, last_any_frame_t
            if not self.tactile_reconnect_on_stall:
                return
            if now - last_reconnect_t < max(1.0, self.tactile_stall_timeout_s):
                return
            if not self.tactile_reconnect_during_episode:
                raise RuntimeError(
                    f"{reason}; aborting episode instead of reconnecting tactile streams during recording"
                )
            logging.warning("%s; reconnecting X5 tactile streams", reason)
            last_reconnect_t = now
            try:
                reconnect = getattr(compat, "reconnect_x5_tactile_for_episode", None)
                if callable(reconnect):
                    reconnect(self.robot, None)
                else:
                    reader.disconnect()
                    reader.connect(left_enabled=left_enabled, right_enabled=right_enabled)
                last_sig_by_key.clear()
                last_changed_t_by_key.clear()
                last_frame_received_t_by_key.clear()
                last_seq_by_key.clear()
                logged_first = False
                last_any_frame_t = time.perf_counter()
            except Exception as exc:
                logging.warning("Aligned tactile reconnect failed: %s", exc)

        def abort_or_reconnect_stalled_keys(now: float) -> None:
            if not self.tactile_monitor_during_episode:
                return
            if not last_frame_received_t_by_key:
                return
            stalled = [
                key
                for key, frame_t in last_frame_received_t_by_key.items()
                if now - frame_t >= self.tactile_stall_timeout_s
            ]
            if stalled:
                reconnect_tactile(
                    "Aligned tactile stream stopped producing new frames for "
                    f"{self.tactile_stall_timeout_s:.1f}s on "
                    f"{', '.join(sorted(stalled)[:8])}",
                    now,
                )

        while not self.stop_event.is_set():
            if use_cache_history:
                cache = getattr(reader, "_kd_tacmae_async_cache", None)
                get_frames_after = getattr(cache, "get_frames_after", None)
                if callable(get_frames_after):
                    frames = get_frames_after(last_seq_by_key, max_frames=history_batch)
                    now = time.perf_counter()
                    if frames:
                        last_any_frame_t = now
                        if not logged_first:
                            logging.info(
                                "Aligned tactile producer using async cache history first keys=%s",
                                sorted({key for key, _, _, _ in frames}),
                            )
                            logged_first = True
                        debug_by_key: dict[str, np.ndarray] = {}
                        for key, seq, source_t, image in frames:
                            last_seq_by_key[key] = max(int(seq), int(last_seq_by_key.get(key, 0)))
                            if not isinstance(image, np.ndarray):
                                continue
                            source_t = float(source_t) if source_t > 0 else now
                            last_t_by_key[key] = source_t
                            last_frame_received_t_by_key[key] = now
                            image = np.ascontiguousarray(image)
                            if self.tactile_frame_publisher is not None:
                                self.tactile_frame_publisher.publish(
                                    key,
                                    source_t,
                                    image,
                                )
                            for alias in _source_key_candidates(key):
                                self.buffer(alias).add(source_t, image)
                            debug_by_key[key] = image
                        if self.tactile_debug_every_s > 0 and now - last_debug_t >= self.tactile_debug_every_s:
                            debug_parts = []
                            for key, image in sorted(debug_by_key.items()):
                                sig, min_value, max_value = _array_content_signature(image)
                                changed = sig != last_sig_by_key.get(key)
                                last_sig_by_key[key] = sig
                                if changed or key not in last_changed_t_by_key:
                                    last_changed_t_by_key[key] = now
                                short = key.rsplit(".", 1)[-1]
                                debug_parts.append(
                                    f"{short}:seq={last_seq_by_key.get(key)} sig={sig} "
                                    f"range={min_value:.1f}-{max_value:.1f} changed={changed}"
                                )
                            if debug_parts:
                                logging.info("Aligned tactile history content %s", " | ".join(debug_parts))
                                last_debug_t = now
                        abort_or_reconnect_stalled_keys(now)
                        self._sleep_loop(self.tactile_hz)
                        continue
                    if self.tactile_monitor_during_episode and now - last_any_frame_t >= self.tactile_stall_timeout_s:
                        reconnect_tactile(
                            f"Aligned tactile history received no new frames for {now - last_any_frame_t:.1f}s",
                            now,
                        )
                    self._sleep_loop(self.tactile_hz)
                    continue
            if self.tactile_read_direct:
                images, times = self._read_tactile_direct(
                    reader,
                    left_enabled=left_enabled,
                    right_enabled=right_enabled,
                )
                if not images:
                    images = reader.read_images(left_enabled=left_enabled, right_enabled=right_enabled)
                    times = {}
            else:
                images = reader.read_images(left_enabled=left_enabled, right_enabled=right_enabled)
                times = {}
            now = time.perf_counter()
            if not times:
                get_times = getattr(reader, "get_last_update_times_perf", None)
                if callable(get_times):
                    try:
                        times = get_times()
                    except Exception:
                        times = {}
            if not images and now - last_empty_warn_t >= 1.0:
                logging.warning("Aligned tactile producer has no images yet")
                last_empty_warn_t = now
            elif images and not logged_first:
                logging.info("Aligned tactile producer first keys=%s", sorted(images))
                logged_first = True
            debug_parts: list[str] = []
            for key, image in images.items():
                if not isinstance(image, np.ndarray):
                    continue
                # Use PC-side producer poll time for tactile alignment. Some Flux
                # paths keep returning the first sensor timestamp even while the
                # cached image object is readable, which makes freshness checks
                # falsely fail before an episode starts.
                source_t = now
                source_t = float(source_t)
                last_t_by_key[key] = source_t
                sensor_name = _tactile_sensor_name_from_key(key)
                sensor = None
                sensors = getattr(reader, "_sensors", None)
                if isinstance(sensors, dict) and sensor_name is not None:
                    sensor = sensors.get(sensor_name)
                fid = getattr(sensor, "last_fid", None) if sensor is not None else None
                seq_key = f"__fid__.{key}"
                fid_changed = fid is None or int(fid) != int(last_seq_by_key.get(seq_key, -1))
                if fid_changed:
                    last_frame_received_t_by_key[key] = now
                    if fid is not None:
                        last_seq_by_key[seq_key] = int(fid)
                image = np.ascontiguousarray(image)
                if self.tactile_frame_publisher is not None:
                    self.tactile_frame_publisher.publish(
                        key,
                        source_t,
                        image,
                    )
                if self.tactile_debug_every_s > 0 and now - last_debug_t >= self.tactile_debug_every_s:
                    sig, min_value, max_value = _array_content_signature(image)
                    changed = sig != last_sig_by_key.get(key)
                    last_sig_by_key[key] = sig
                    if changed or key not in last_changed_t_by_key:
                        last_changed_t_by_key[key] = now
                    frame_count = getattr(sensor, "frame_count", None) if sensor is not None else None
                    short = key.rsplit(".", 1)[-1]
                    debug_parts.append(
                        f"{short}:fid={fid} n={frame_count} sig={sig} "
                        f"range={min_value:.1f}-{max_value:.1f} changed={changed}"
                    )
                for alias in _source_key_candidates(key):
                    self.buffer(alias).add(source_t, image)
            if debug_parts:
                logging.info("Aligned tactile content %s", " | ".join(debug_parts))
                last_debug_t = now
            abort_or_reconnect_stalled_keys(now)
            self._sleep_loop(self.tactile_hz)

    def _preview_loop(self) -> None:
        interval = 1.0 / max(0.1, self.preview_hz)
        while not self.stop_event.is_set():
            now = time.perf_counter()
            if now - self.last_preview_t >= interval:
                self.last_preview_t = now
                obs: dict[str, Any] = {}
                for key, buf in list(self.buffers.items()):
                    if key == "__state__":
                        continue
                    sample = buf.latest(max_age_s=2.0)
                    if sample is not None:
                        data = sample.data
                        if self.tactile_preview_direct_rgb and _is_tactile_stream_key(key):
                            try:
                                data = _tactile_frame_to_preview_rgb(sample.data)
                            except Exception as exc:
                                logging.debug("Aligned tactile preview conversion skipped %s: %s", key, exc)
                        for candidate in _source_key_candidates(key):
                            obs[candidate] = data
                if obs and self.previewer is not None:
                    self.previewer.submit(obs)
            time.sleep(0.02)


class ReadyLeaderGripperFollowSession:
    """Let the leader control follower grippers while no episode is recording."""

    def __init__(
        self,
        *,
        robot: Any,
        teleop: Any,
        settings: Any,
        app: Any,
        state: Any,
    ) -> None:
        self.robot = robot
        self.teleop = teleop
        self.settings = settings
        self.app = app
        self.state = state
        self.hz = compat.env_float("READY_GRIPPER_FOLLOW_HZ", 30.0)
        self.takeover_tolerance = compat.env_int(
            "READY_GRIPPER_TAKEOVER_TOLERANCE",
            75,
        )
        self._stop_event = threading.Event()
        self._enabled = threading.Event()
        self._paused = threading.Event()
        self._paused.set()
        self._synced = threading.Event()
        self._lock = threading.RLock()
        self._takeover_position: int | None = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._loop,
            name="ready_leader_gripper_follow",
            daemon=True,
        )

    @property
    def is_synced(self) -> bool:
        return self._synced.is_set()

    @property
    def takeover_position(self) -> int | None:
        with self._lock:
            return self._takeover_position

    def start(self) -> None:
        if bool(getattr(self.robot.config, "send_action_enabled", True)):
            raise RuntimeError(
                "READY 夹爪跟随要求 send_action_enabled=false，拒绝发送任何关节动作"
            )
        self.thread.start()
        self.resume(takeover_position=None)
        logging.info("Leader gripper follow enabled immediately after hardware connection")

    def resume(self, *, takeover_position: int | None) -> None:
        self.raise_if_error()
        with self._lock:
            self._takeover_position = takeover_position
        if takeover_position is None:
            self._synced.set()
        else:
            self._synced.clear()
        self._paused.clear()
        self._enabled.set()

    def pause(self, *, timeout_s: float = 2.0) -> None:
        self._enabled.clear()
        if not self._paused.wait(timeout=max(0.1, timeout_s)):
            raise RuntimeError("READY 夹爪跟随线程未能及时暂停")
        self.raise_if_error()
        settle_s = max(
            0.0,
            compat.env_float("READY_GRIPPER_PAUSE_SETTLE_S", 0.10),
        )
        if settle_s > 0:
            time.sleep(settle_s)

    def stop(self) -> None:
        self._enabled.clear()
        self._stop_event.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            logging.error("READY leader gripper follow thread did not stop cleanly")

    def raise_if_error(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"READY 主臂夹爪跟随失败: {self.error}") from self.error

    def _right_gripper_target(self, action: dict[str, Any]) -> int:
        right_arm = getattr(self.robot, "right_arm", None)
        key = f"right_{getattr(right_arm, 'GRIPPER_NAME', 'main_gripper')}"
        if key not in action:
            raise RuntimeError(f"主臂 action 缺少右夹爪字段 {key}")
        return int(
            round(
                GripperSafetyLimits.leader_value_to_position(
                    float(action[key]),
                    right_arm,
                )
            )
        )

    def _loop(self) -> None:
        interval = 1.0 / max(1.0, self.hz)
        last_wait_message_t = 0.0
        try:
            while not self._stop_event.is_set():
                if not self._enabled.is_set():
                    self._paused.set()
                    self._stop_event.wait(0.01)
                    continue
                self._paused.clear()
                loop_t = time.perf_counter()
                action = self.teleop.get_action()
                if self.settings.swap_teleop_actions:
                    action = self.app.swap_left_right_action(action)
                safety_limits = getattr(self.robot, "_gripper_safety_limits", None)
                if safety_limits is not None:
                    action = safety_limits.clamp_action(action, self.robot)

                with self._lock:
                    takeover_position = self._takeover_position
                if not self._synced.is_set() and takeover_position is not None:
                    leader_target = self._right_gripper_target(action)
                    error = abs(leader_target - takeover_position)
                    if error <= self.takeover_tolerance:
                        self._synced.set()
                        reset_rate_limit = getattr(
                            self.robot,
                            "reset_gripper_rate_limit",
                            None,
                        )
                        if callable(reset_rate_limit):
                            reset_rate_limit()
                        logging.info(
                            "READY leader gripper synchronized with restored follower: "
                            "leader_target=%s follower_target=%s",
                            leader_target,
                            takeover_position,
                        )
                        self.state.update(
                            status="READY",
                            message="Leader gripper synchronized. Ready.",
                        )
                    else:
                        now = time.monotonic()
                        if now - last_wait_message_t >= 1.0:
                            last_wait_message_t = now
                            self.state.update(
                                status="READY",
                                message=(
                                    "Move leader gripper to restored start state "
                                    f"(position error {error})"
                                ),
                            )
                        time.sleep(max(0.0, interval - (time.perf_counter() - loop_t)))
                        continue

                self.robot.send_action(action)
                time.sleep(max(0.0, interval - (time.perf_counter() - loop_t)))
        except BaseException as exc:
            if not self._stop_event.is_set():
                self.error = exc
                logging.exception("READY leader gripper follow failed")
                self.state.update(
                    status="ERROR",
                    last_error=str(exc),
                    message="Leader gripper follow error",
                )
        finally:
            self._paused.set()


class ActionLoop:
    def __init__(
        self,
        robot: Any,
        teleop: Any,
        dataset: Any,
        settings: Any,
        app: Any,
        *,
        state_buffer: TimeRingBuffer,
    ) -> None:
        self.robot = robot
        self.teleop = teleop
        self.dataset = dataset
        self.settings = settings
        self.app = app
        self.state_buffer = state_buffer
        self.control_mode = getattr(robot, "_capture_control_mode", "leader")
        self.buffer = TimeRingBuffer(max(60, int(settings.fps * 6)))
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.hz = compat.env_float("ALIGNED_ACTION_HZ", float(settings.fps))
        self.drag_state_max_age_s = (
            compat.env_float("DRAG_ACTION_STATE_MAX_AGE_MS", 150.0) / 1000.0
        )
        self.leader_fallback_state_max_age_s = (
            compat.env_float(
                "LEADER_FALLBACK_STATE_MAX_AGE_MS",
                compat.env_float("DRAG_ACTION_STATE_MAX_AGE_MS", 150.0),
            )
            / 1000.0
        )
        self.leader_fallback_sides = tuple(
            getattr(robot, "_leader_state_fallback_action_sides", ())
        )
        self.thread = threading.Thread(target=self._loop, name="aligned_action_loop", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def raise_if_error(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"aligned action loop failed: {self.error}") from self.error

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            self.raise_if_error()
            if self.buffer.latest(max_age_s=1.0) is not None:
                return
            time.sleep(0.01)
        raise RuntimeError("aligned action loop did not produce an action before episode start")

    def _loop(self) -> None:
        interval = 1.0 / max(1.0, self.hz)
        program_control = getattr(self.robot, "_program_gripper_control", None)
        safety_limits = getattr(self.robot, "_gripper_safety_limits", None)
        if program_control is not None:
            program_control.set_dispatch_active(True)
            logging.info(
                "Aligned action loop: gripper commands come from program targets; "
                "teleoperator joint actions are recorded only"
            )
        elif self.control_mode == "drag":
            logging.info(
                "Aligned action loop: follower joint state is copied to joint action; "
                "gripper commands continue to come from the leader"
            )
        elif self.leader_fallback_sides:
            logging.warning(
                "Aligned action loop: sides=%s use follower state because their leader USB is absent",
                list(self.leader_fallback_sides),
            )
        try:
            while not self.stop_event.is_set():
                loop_t = time.perf_counter()
                action = self.teleop.get_action()
                if self.settings.swap_teleop_actions:
                    action = self.app.swap_left_right_action(action)
                if program_control is not None:
                    action = program_control.apply_program_targets(action)
                sample_t = time.perf_counter()
                if self.control_mode == "drag":
                    state_sample = self.state_buffer.latest(
                        max_age_s=self.drag_state_max_age_s
                    )
                    if state_sample is None:
                        raise RuntimeError(
                            "drag 模式没有可用的最新从臂 state；"
                            f"最大允许延迟 {self.drag_state_max_age_s * 1000.0:.0f} ms"
                        )
                    action = replace_joint_actions_with_follower_state(
                        action,
                        state_sample.data,
                        self.robot,
                    )
                    sample_t = state_sample.t
                elif self.leader_fallback_sides:
                    state_sample = self.state_buffer.latest(
                        max_age_s=self.leader_fallback_state_max_age_s
                    )
                    if state_sample is None:
                        raise RuntimeError(
                            "leader fallback 没有新鲜从臂 state；"
                            f"最大允许延迟 {self.leader_fallback_state_max_age_s * 1000.0:.0f} ms"
                        )
                    action = replace_arm_actions_with_follower_state(
                        action,
                        state_sample.data,
                        self.robot,
                        self.leader_fallback_sides,
                    )
                if safety_limits is not None:
                    action = safety_limits.clamp_action(action, self.robot)
                self.robot.send_action(action)
                recorded_action = self.app.map_gripper_actions_for_dataset(action, self.robot, self.settings)
                self.buffer.add(sample_t, recorded_action)
                elapsed = time.perf_counter() - loop_t
                time.sleep(max(0.0, interval - elapsed))
        except BaseException as exc:  # noqa: BLE001
            if not self.stop_event.is_set():
                logging.exception("aligned action loop failed")
                self.error = exc
                self.stop_event.set()
        finally:
            if program_control is not None:
                program_control.set_dispatch_active(False)


class EpisodeWriter:
    def __init__(self, dataset: Any, *, maxsize: int) -> None:
        self.dataset = dataset
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max(1, maxsize))
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._loop, name="aligned_lerobot_writer", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def enqueue(self, frame: dict[str, Any], timeout_s: float) -> None:
        if self.error is not None:
            raise RuntimeError(f"LeRobot writer failed: {self.error}") from self.error
        self.queue.put(frame, timeout=timeout_s)

    def close(self) -> None:
        while self.thread.is_alive() and self.error is None:
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=30.0)
        if self.thread.is_alive():
            raise RuntimeError("LeRobot writer did not stop within 30s")
        if self.error is not None:
            raise RuntimeError(f"LeRobot writer failed: {self.error}") from self.error

    def _loop(self) -> None:
        try:
            while True:
                frame = self.queue.get()
                if frame is None:
                    self.queue.task_done()
                    return
                self.dataset.add_frame(frame)
                self.queue.task_done()
        except BaseException as exc:  # noqa: BLE001
            logging.exception("aligned LeRobot writer failed")
            self.error = exc


def _is_image_feature(spec: Any) -> bool:
    return isinstance(spec, tuple) and len(spec) >= 2


def _copy_frame_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value.copy())
    return value


def _zero_image(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(tuple(int(v) for v in shape), dtype=np.uint8)


def _collect_aligned_observation(
    *,
    robot: Any,
    hub: ProducerHub,
    target_t: float,
    visual_max_age_s: float,
    wrist_max_age_s: float,
    wrist_preferred_max_age_s: float,
    d405_max_age_s: float,
    tactile_max_age_s: float,
    wrist_ready_timeout_s: float,
    visual_hold_max_age_s: float,
    state_max_age_s: float,
    abort_on_missing: bool,
    hold_last_on_stale: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    obs: dict[str, Any] = {}
    offsets_ms: dict[str, float] = {}

    state_sample = hub.buffer("__state__").nearest(target_t, max_age_s=state_max_age_s)
    if state_sample is not None and isinstance(state_sample.data, dict):
        obs.update(state_sample.data)
        offsets_ms["__state__"] = (state_sample.t - target_t) * 1000.0

    missing_images: list[str] = []
    for key, spec in robot.observation_features.items():
        if not _is_image_feature(spec):
            continue
        sample = None
        is_wrist = _is_wrist_image_key(key)
        if is_wrist:
            max_age_s = wrist_max_age_s
        elif _is_tactile_stream_key(key):
            max_age_s = tactile_max_age_s
        elif "d405" in key.lower():
            max_age_s = d405_max_age_s
        else:
            max_age_s = visual_max_age_s
        for candidate in _source_key_candidates(key):
            buffer = hub.buffer(candidate)
            if is_wrist and wrist_ready_timeout_s > 0:
                sample = buffer.wait_nearest_prefer(
                    target_t,
                    max_age_s=max_age_s,
                    preferred_max_age_s=wrist_preferred_max_age_s,
                    timeout_s=wrist_ready_timeout_s,
                )
            else:
                sample = buffer.nearest(target_t, max_age_s=max_age_s)
            if sample is not None:
                break
        if sample is None and hold_last_on_stale:
            for candidate in _source_key_candidates(key):
                sample = hub.buffer(candidate).latest(max_age_s=visual_hold_max_age_s)
                if sample is not None:
                    break
        if sample is None:
            missing_images.append(key)
            if not abort_on_missing:
                obs[key] = _zero_image(spec)
            continue
        obs[key] = _copy_frame_value(sample.data)
        offsets_ms[key] = (sample.t - target_t) * 1000.0

    for key, spec in robot.observation_features.items():
        if key in obs:
            continue
        if _is_image_feature(spec):
            continue
        obs[key] = 0.0

    if missing_images and abort_on_missing:
        details = [hub.describe_key(key, target_t) for key in missing_images[:16]]
        raise RuntimeError(
            "Aligned recorder missing/freshness timeout for image streams: "
            + ", ".join(missing_images[:16])
            + (" ..." if len(missing_images) > 16 else "")
            + "; debug="
            + " | ".join(details)
        )
    return obs, offsets_ms


def _collect_aligned_action(
    action_loop: ActionLoop,
    target_t: float,
    *,
    max_age_s: float,
    abort_on_missing: bool,
) -> tuple[dict[str, Any], float]:
    sample = action_loop.buffer.nearest(target_t, max_age_s=max_age_s)
    if sample is None:
        if abort_on_missing:
            raise RuntimeError("Aligned recorder missing/freshness timeout for action")
        latest = action_loop.buffer.latest(max_age_s=None)
        if latest is None:
            return {}, 0.0
        sample = latest
    data = {key: _copy_frame_value(value) for key, value in sample.data.items()}
    return data, (sample.t - target_t) * 1000.0


def _maybe_reconnect_x5_tactile_before_episode(robot: Any, state: Any) -> None:
    mode = os.environ.get(
        "ALIGNED_TACTILE_RECONNECT_ON_EPISODE_START",
        os.environ.get("TACTILE_RECONNECT_ON_EPISODE_START", "auto"),
    ).strip().lower()
    if mode in {"0", "false", "no", "off", "never", "none"}:
        return

    should_reconnect = mode in {"1", "true", "yes", "on", "always"}
    if mode in {"", "auto"}:
        health_check = getattr(compat, "x5_tactile_streams_healthy", None)
        should_reconnect = True
        if callable(health_check):
            should_reconnect = not bool(health_check(robot))

    if should_reconnect:
        reconnect = getattr(compat, "reconnect_x5_tactile_for_episode", None)
        if callable(reconnect):
            reconnect(robot, state)


def _observation_key_from_video_key(video_key: str) -> str:
    for prefix in (
        "observation.images.",
        "observation.depth_deformation.",
        "observation.depth_shear.",
        "observation.shear.",
        "observation.depth.",
        "observation.deformation.",
        "observation.",
    ):
        if video_key.startswith(prefix):
            suffix = video_key[len(prefix) :]
            if prefix == "observation.images.":
                return suffix
            if prefix.startswith("observation.") and prefix != "observation.":
                return prefix[len("observation.") :] + suffix
            return suffix
    return video_key


def _latest_video_warm_start_samples(
    *,
    hub: ProducerHub,
    video_keys: list[str],
    max_age_s: float,
) -> dict[str, np.ndarray]:
    samples: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for video_key in video_keys:
        obs_key = _observation_key_from_video_key(video_key)
        sample = None
        for candidate in _source_key_candidates(obs_key):
            sample = hub.buffer(candidate).latest(max_age_s=max_age_s)
            if sample is not None:
                break
        if sample is None or not isinstance(sample.data, np.ndarray):
            missing.append(video_key)
            continue
        samples[video_key] = sample.data
    if missing:
        raise RuntimeError(
            "Cannot warm-start streaming encoders; missing fresh sample(s): "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )
    return samples


def _prestart_streaming_video_encoders(
    dataset: Any,
    state: Any,
    *,
    hub: ProducerHub,
    required_keys: list[str],
) -> None:
    if not compat.env_bool("ALIGNED_PRESTART_STREAMING_ENCODERS", True):
        return
    writer = getattr(dataset, "writer", None)
    encoder = getattr(writer, "_streaming_encoder", None)
    if writer is None or encoder is None:
        return
    if bool(getattr(encoder, "_episode_active", False)):
        return
    video_keys = list(getattr(getattr(writer, "_meta", None), "video_keys", []) or [])
    if not video_keys:
        return
    temp_dir = getattr(writer, "_root", None)
    if temp_dir is None:
        return
    state.update(status="CALIBRATING", is_recording=False, message="Prestarting video encoders")
    logging.info("Prestarting streaming video encoders before aligned timeline: keys=%s", video_keys)
    encoder.start_episode(video_keys=video_keys, temp_dir=temp_dir)
    warm_start = getattr(encoder, "warm_start", None)
    if compat.env_bool("ALIGNED_ENCODER_WARM_START", True) and callable(warm_start):
        sample_max_age_s = compat.env_float("ALIGNED_ENCODER_WARM_START_MAX_AGE_MS", 1000.0) / 1000.0
        samples = _latest_video_warm_start_samples(
            hub=hub,
            video_keys=video_keys,
            max_age_s=sample_max_age_s,
        )
        warm_timeout_s = compat.env_float("ALIGNED_ENCODER_WARM_START_TIMEOUT_S", 5.0)
        logging.info("Warm-starting streaming video encoders with current producer frames")
        warm_start(samples, timeout_s=warm_timeout_s)
    settle_s = max(0.0, compat.env_float("ALIGNED_ENCODER_PRESTART_SETTLE_S", 0.8))
    if settle_s > 0:
        time.sleep(settle_s)
    if compat.env_bool("ALIGNED_CLEAR_BUFFERS_AFTER_ENCODER_PRESTART", True):
        logging.info("Clearing aligned image buffers after encoder prestart")
        hub.clear_buffers(required_keys)


def _dataset_root(dataset: Any) -> Path | None:
    root = getattr(getattr(dataset, "meta", None), "root", None)
    if root is None:
        root = getattr(dataset, "root", None)
    if root is None:
        root = os.environ.get("DATASET_ROOT", "").strip() or None
    if root is None:
        return None
    return Path(str(root)).expanduser().resolve()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logging.warning("Failed removing temporary metadata file %s", temporary)


def write_episode_start_pose_metadata(
    *,
    dataset: Any,
    pose: EpisodeStartPose,
    config: EpisodeReturnToStartConfig,
    return_status: str,
    error: str | None = None,
) -> Path:
    """Persist a per-episode reset target plus a latest-pose inference shortcut."""

    if return_status not in {"pending", "completed", "failed"}:
        raise ValueError(f"invalid return status: {return_status!r}")
    dataset_root = _dataset_root(dataset)
    if dataset_root is None:
        raise RuntimeError("无法确定数据集目录，不能保存本回合初始位姿")

    relative_path = Path("meta") / "episode_start_poses" / f"episode-{pose.episode_index:06d}.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "episode_index": pose.episode_index,
        "task": pose.task,
        "arm": "right",
        "joint_names": [f"right_main_joint{index}" for index in range(1, 8)],
        "joint_positions_deg": list(pose.right_joint_positions_deg),
        "joint_positions_rad": [
            float(np.radians(value)) for value in pose.right_joint_positions_deg
        ],
        "gripper_position": pose.right_gripper_position,
        "gripper_position_range": [0, 1000],
        "captured_at_utc": pose.captured_at_utc,
        "return_to_start": {
            "enabled": config.enabled,
            "speed_percent": config.speed,
            "tolerance_deg": config.tolerance_deg,
            "max_start_delta_deg": config.max_start_delta_deg,
            "restore_gripper": config.restore_gripper,
            "gripper_speed_percent": config.gripper_speed,
            "gripper_tolerance": config.gripper_tolerance,
            "status": return_status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    if error is not None:
        payload["return_to_start"]["error"] = str(error)

    output_path = dataset_root / relative_path
    _write_json_atomic(output_path, payload)
    latest_payload = dict(payload)
    latest_payload["source"] = str(relative_path)
    _write_json_atomic(dataset_root / "meta" / "inference_start_pose.json", latest_payload)
    logging.info(
        "Saved episode=%s start-pose metadata status=%s path=%s",
        pose.episode_index,
        return_status,
        output_path,
    )
    return output_path


def _snapshot_base(dataset_root: Path) -> Path:
    return dataset_root / ".aligned_recorder_snapshots"


def _validate_parquet_file(path: Path) -> tuple[bool, str]:
    try:
        if not path.exists():
            return False, "missing"
        if path.stat().st_size < 8:
            return False, f"too small ({path.stat().st_size} bytes)"
        with path.open("rb") as f:
            f.seek(-4, os.SEEK_END)
            if f.read(4) != b"PAR1":
                return False, "missing PAR1 footer"
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]

            _ = pq.ParquetFile(path).metadata
        except ImportError:
            logging.warning("pyarrow is unavailable; dataset snapshot validation used footer check only")
        except Exception as exc:
            return False, f"pyarrow metadata error: {exc}"
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def _iter_dataset_metadata_files(dataset_root: Path) -> list[Path]:
    files: list[Path] = []
    for name in ("data", "meta"):
        base = dataset_root / name
        if not base.exists():
            continue
        files.extend(path for path in base.rglob("*") if path.is_file())
    return sorted(files)


def _validate_dataset_metadata(dataset_root: Path) -> tuple[bool, str]:
    files = _iter_dataset_metadata_files(dataset_root)
    if not files:
        return False, "no data/meta files found"
    failures: list[str] = []
    for path in files:
        if path.suffix != ".parquet":
            continue
        ok, reason = _validate_parquet_file(path)
        if not ok:
            failures.append(f"{path.relative_to(dataset_root)}: {reason}")
    if failures:
        return False, "; ".join(failures[:6]) + (" ..." if len(failures) > 6 else "")
    return True, "ok"


def _copy_dataset_metadata_tree(dataset_root: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    for path in _iter_dataset_metadata_files(dataset_root):
        rel = path.relative_to(dataset_root)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)


def _prune_dataset_snapshots(dataset_root: Path) -> None:
    keep = max(1, compat.env_int("ALIGNED_DATASET_SNAPSHOT_KEEP", 2))
    base = _snapshot_base(dataset_root)
    if not base.exists():
        return
    snapshots = sorted(path for path in base.glob("episode-*") if path.is_dir())
    for old in snapshots[:-keep]:
        try:
            shutil.rmtree(old)
        except Exception:
            logging.warning("Failed pruning old aligned dataset snapshot: %s", old, exc_info=True)


def _load_dataset_episodes(dataset_root: Path) -> Any:
    from lerobot.datasets.io_utils import load_episodes

    return load_episodes(dataset_root)


def _seal_dataset_episode_files_for_snapshot(dataset: Any) -> None:
    """Close current parquet writers so a saved episode is recoverable before final app exit."""
    if not compat.env_bool("ALIGNED_DATASET_SEAL_AFTER_SAVE", True):
        return
    dataset_root = _dataset_root(dataset)
    if dataset_root is None:
        logging.warning("Aligned dataset episode seal skipped: cannot infer dataset root")
        return

    writer = getattr(dataset, "writer", None)
    if writer is not None:
        close_writer = getattr(writer, "close_writer", None)
        if callable(close_writer):
            close_writer()
        elif getattr(writer, "_pq_writer", None) is not None:
            writer._pq_writer.close()
            writer._pq_writer = None

    meta = getattr(dataset, "meta", None)
    if meta is not None:
        close_meta_writer = getattr(meta, "_close_writer", None)
        if callable(close_meta_writer):
            close_meta_writer()
        elif getattr(meta, "_pq_writer", None) is not None:
            meta._pq_writer.close()
            meta._pq_writer = None
        try:
            meta.episodes = _load_dataset_episodes(dataset_root)
        except Exception:
            logging.warning("Aligned dataset episode seal could not reload episodes metadata", exc_info=True)
        # Force the next episode onto a new metadata parquet instead of reopening
        # and truncating the just-sealed file.
        if hasattr(meta, "latest_episode"):
            meta.latest_episode = None

    if writer is not None:
        # Force the next episode onto a new data parquet instead of reopening and
        # truncating the just-sealed file.
        if hasattr(writer, "_latest_episode"):
            writer._latest_episode = None
        if hasattr(writer, "_current_file_start_frame"):
            writer._current_file_start_frame = None


def _snapshot_dataset_after_save(dataset: Any) -> None:
    _seal_dataset_episode_files_for_snapshot(dataset)
    if not compat.env_bool("ALIGNED_DATASET_SNAPSHOT_AFTER_SAVE", True):
        return
    dataset_root = _dataset_root(dataset)
    if dataset_root is None:
        logging.warning("Aligned dataset snapshot skipped: cannot infer dataset root")
        return
    ok, reason = _validate_dataset_metadata(dataset_root)
    if not ok:
        restored = _restore_dataset_snapshot_if_needed(dataset, force=True)
        restore_text = "restored previous good snapshot" if restored else "no previous good snapshot to restore"
        raise RuntimeError(f"Saved dataset metadata is invalid ({reason}); {restore_text}")

    episode_count = int(getattr(getattr(dataset, "meta", None), "total_episodes", 0) or 0)
    if episode_count <= 0:
        return
    base = _snapshot_base(dataset_root)
    base.mkdir(parents=True, exist_ok=True)
    dst = base / f"episode-{episode_count - 1:06d}"
    tmp = base / f".episode-{episode_count - 1:06d}.tmp-{os.getpid()}"
    _copy_dataset_metadata_tree(dataset_root, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    _prune_dataset_snapshots(dataset_root)
    logging.info("Aligned dataset metadata snapshot saved: %s", dst)


def _restore_dataset_snapshot_if_needed(dataset: Any, *, force: bool = False) -> bool:
    if not compat.env_bool("ALIGNED_DATASET_RESTORE_ON_ERROR", True):
        return False
    dataset_root = _dataset_root(dataset)
    if dataset_root is None:
        return False
    if not force:
        ok, _ = _validate_dataset_metadata(dataset_root)
        if ok:
            return False
    base = _snapshot_base(dataset_root)
    snapshots = sorted(path for path in base.glob("episode-*") if path.is_dir()) if base.exists() else []
    if not snapshots:
        logging.warning("Aligned dataset restore skipped: no metadata snapshot found under %s", base)
        return False

    snapshot = snapshots[-1]
    for name in ("data", "meta"):
        target = dataset_root / name
        source = snapshot / name
        if target.exists():
            shutil.rmtree(target)
        if source.exists():
            shutil.copytree(source, target)
    ok, reason = _validate_dataset_metadata(dataset_root)
    if not ok:
        raise RuntimeError(f"Restored aligned dataset snapshot is still invalid: {reason}")
    try:
        setattr(dataset, "_aligned_metadata_restored", True)
    except Exception:
        pass
    logging.warning("Restored aligned dataset metadata from snapshot: %s", snapshot)
    return True


def _finalize_dataset_with_snapshot_restore(dataset: Any) -> None:
    try:
        dataset.finalize()
    except Exception:
        logging.exception("dataset.finalize() failed; trying aligned metadata snapshot restore")
        _restore_dataset_snapshot_if_needed(dataset, force=True)
        raise
    dataset_root = _dataset_root(dataset)
    if dataset_root is None:
        return
    ok, reason = _validate_dataset_metadata(dataset_root)
    if not ok:
        restored = _restore_dataset_snapshot_if_needed(dataset, force=True)
        if restored:
            return
        raise RuntimeError(f"Finalized dataset metadata is invalid: {reason}")


def _sleep_until_with_commands(
    deadline: float,
    app: Any,
    commands: "queue.Queue[str]",
) -> str | None:
    while True:
        for cmd in app.drain_commands(commands):
            if cmd in {"finish", "discard", "stop"}:
                return cmd
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return None
        time.sleep(min(0.01, remaining))


def capture_episode_aligned(
    *,
    app: Any,
    robot: Any,
    teleop: Any,
    dataset: Any,
    settings: Any,
    state: Any,
    commands: "queue.Queue[str]",
    hub: ProducerHub,
    right_arm_reset: EpisodeRightArmResetConfig,
    drag_teach: DragTeachSession | None,
) -> str:
    reset_result = reset_then_enable_drag_before_episode(
        robot=robot,
        state=state,
        commands=commands,
        app=app,
        config=right_arm_reset,
        drag_teach=drag_teach,
    )
    if reset_result is not None:
        return reset_result
    _maybe_reconnect_x5_tactile_before_episode(robot, state)
    if compat.env_bool("FORCE_SENSOR_CLEAR_ON_EPISODE_START", False):
        compat.clear_force_sensors_for_episode(robot, state)
    if (
        os.environ.get("GRIPPER_COMMAND_SOURCE", "leader").strip().lower() != "program"
        and compat.env_bool("GRIPPER_PREWARM_ON_EPISODE_START", True)
    ):
        compat.prewarm_gripper_for_episode(robot, teleop, state)

    fps = float(settings.fps)
    interval = 1.0 / fps
    writer_delay_s = compat.env_float("ALIGNED_WRITER_DELAY_S", 0.45)
    start_delay_s = compat.env_float("ALIGNED_START_DELAY_S", 0.10)
    visual_max_age_s = compat.env_float("ALIGNED_VISUAL_MAX_AGE_MS", 120.0) / 1000.0
    wrist_max_age_s = compat.env_float("ALIGNED_WRIST_MAX_AGE_MS", compat.env_float("ALIGNED_VISUAL_MAX_AGE_MS", 120.0)) / 1000.0
    wrist_preferred_max_age_s = compat.env_float("ALIGNED_WRIST_PREFERRED_MAX_OFFSET_MS", 40.0) / 1000.0
    d405_max_age_s = compat.env_float("ALIGNED_D405_MAX_AGE_MS", 200.0) / 1000.0
    tactile_max_age_s = compat.env_float("ALIGNED_TACTILE_MAX_AGE_MS", 250.0) / 1000.0
    wrist_ready_timeout_s = compat.env_float("ALIGNED_WRIST_READY_TIMEOUT_S", 0.2)
    visual_hold_max_age_s = compat.env_float("ALIGNED_VISUAL_HOLD_MAX_AGE_MS", 1000.0) / 1000.0
    state_max_age_s = compat.env_float("ALIGNED_STATE_MAX_AGE_MS", 150.0) / 1000.0
    action_max_age_s = compat.env_float("ALIGNED_ACTION_MAX_AGE_MS", 250.0) / 1000.0
    abort_on_missing = compat.env_bool("ALIGNED_ABORT_ON_MISSING", True)
    strict_no_hold = compat.env_bool("ALIGNED_STRICT_NO_HOLD", True)
    hold_last_on_stale = False if strict_no_hold else compat.env_bool("ALIGNED_HOLD_LAST_ON_STALE", True)
    writer_queue_size = compat.env_int("ALIGNED_WRITER_QUEUE_MAXSIZE", 240)
    writer_put_timeout_s = compat.env_float("ALIGNED_WRITER_PUT_TIMEOUT_S", 1.0)
    profile_every = compat.env_int("ALIGNED_PROFILE_EVERY", 60)

    required_keys = [key for key, spec in robot.observation_features.items() if _is_image_feature(spec)]
    state.update(status="CALIBRATING", is_recording=False, message="Waiting for aligned producer buffers")
    hub.wait_ready(
        required_keys,
        timeout_s=compat.env_float("ALIGNED_STARTUP_TIMEOUT_S", 5.0),
        max_age_s=compat.env_float("ALIGNED_STARTUP_MAX_AGE_MS", 500.0) / 1000.0,
    )
    _prestart_streaming_video_encoders(
        dataset,
        state,
        hub=hub,
        required_keys=required_keys,
    )
    hub.wait_stable(
        required_keys,
        timeout_s=compat.env_float("ALIGNED_POST_ENCODER_STARTUP_TIMEOUT_S", 5.0),
        max_age_s=compat.env_float("ALIGNED_STARTUP_MAX_AGE_MS", 500.0) / 1000.0,
        stable_s=compat.env_float("ALIGNED_POST_ENCODER_STABLE_S", 0.6),
        min_updates=compat.env_int("ALIGNED_POST_ENCODER_MIN_UPDATES", 3),
    )

    action_loop = ActionLoop(
        robot,
        teleop,
        dataset,
        settings,
        app,
        state_buffer=hub.buffer("__state__"),
    )
    writer = EpisodeWriter(dataset, maxsize=writer_queue_size)
    frame_count = 0
    result = "finish"
    action_loop.start()
    writer.start()
    try:
        action_loop.wait_ready(timeout_s=compat.env_float("ALIGNED_ACTION_STARTUP_TIMEOUT_S", 2.0))
        state.update(status="RECORDING", is_recording=True, current_frame=0, message="Recording")
        app.drain_commands(commands)
        max_frames = int(round(float(settings.episode_time_s) * fps))
        # Start the 30 Hz recording timeline only after all producers/actors are
        # ready. If we anchor it before action_loop startup, a slow RealMan/gripper
        # startup can put target_t more than a second behind live wrist frames and
        # strict timestamp matching will correctly abort.
        start_t = time.perf_counter() + max(0.0, start_delay_s)

        while frame_count < max_frames:
            hub.raise_if_error()
            action_loop.raise_if_error()
            target_t = start_t + frame_count * interval
            cmd = _sleep_until_with_commands(target_t + writer_delay_s, app, commands)
            if cmd is not None:
                result = cmd
                state.update(is_recording=False)
                break

            obs, offsets = _collect_aligned_observation(
                robot=robot,
                hub=hub,
                target_t=target_t,
                visual_max_age_s=visual_max_age_s,
                wrist_max_age_s=wrist_max_age_s,
                wrist_preferred_max_age_s=wrist_preferred_max_age_s,
                d405_max_age_s=d405_max_age_s,
                tactile_max_age_s=tactile_max_age_s,
                wrist_ready_timeout_s=wrist_ready_timeout_s,
                visual_hold_max_age_s=visual_hold_max_age_s,
                state_max_age_s=state_max_age_s,
                abort_on_missing=abort_on_missing,
                hold_last_on_stale=hold_last_on_stale,
            )
            recorded_action, action_offset_ms = _collect_aligned_action(
                action_loop,
                target_t,
                max_age_s=action_max_age_s,
                abort_on_missing=abort_on_missing,
            )
            if getattr(robot, "_capture_control_mode", "leader") == "drag":
                # Use the exact state sample selected for this dataset frame so
                # observation.state joints and action joints are identical.
                recorded_action = replace_joint_actions_with_follower_state(
                    recorded_action,
                    obs,
                    robot,
                )
            fallback_sides = tuple(
                getattr(robot, "_leader_state_fallback_action_sides", ())
            )
            if fallback_sides:
                if "__state__" not in offsets:
                    raise RuntimeError(
                        "leader fallback 的目标帧没有有效从臂 state，拒绝写入全零 action"
                    )
                # Use this frame's exact aligned state so the unavailable
                # leader side has action == observation.state in the dataset.
                recorded_action = replace_arm_actions_with_follower_state(
                    recorded_action,
                    obs,
                    robot,
                    fallback_sides,
                    dataset_gripper_coordinates=bool(
                        settings.save_gripper_action_as_target
                    ),
                )
            offsets["__action__"] = action_offset_ms
            for key, spec in robot.action_features.items():
                if key not in recorded_action and not _is_image_feature(spec):
                    recorded_action[key] = 0.0

            observation_frame = app.build_dataset_frame(dataset.features, obs, prefix=app.OBS_STR)
            action_frame = app.build_dataset_frame(dataset.features, recorded_action, prefix=app.ACTION)
            writer.enqueue(
                {**observation_frame, **action_frame, "task": settings.single_task},
                timeout_s=writer_put_timeout_s,
            )

            frame_count += 1
            if profile_every > 0 and frame_count % profile_every == 0 and offsets:
                max_abs = max(abs(v) for v in offsets.values())
                latest = ", ".join(
                    f"{key}={value:+.1f}ms"
                    for key, value in sorted(offsets.items())
                    if key != "__state__"
                )
                logging.info(
                    "aligned frame=%s writer_q=%s max_abs_offset=%.1fms offsets=%s",
                    frame_count,
                    writer.queue.qsize(),
                    max_abs,
                    latest,
                )
            state.update(
                current_frame=frame_count,
                last_loop_hz=fps,
                message=f"Aligned recording frame {frame_count}",
            )

        if frame_count >= max_frames:
            result = "finish"
        return result
    finally:
        state.update(is_recording=False)
        action_loop.stop()
        writer.close()
        logging.info("Aligned episode stopped result=%s frames=%s", result, frame_count)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s:%(message)s")
    compat.configure_opencv_runtime()
    app = compat.load_capture_app()
    compat.patch_make_robot(app)
    compat.patch_parse_args(app)
    compat.start_tactile_sidecar_if_requested()

    settings = app.parse_args()
    default_control_mode = (
        "program"
        if compat.env_bool("TRAJECTORY_GRIPPER_CONTROL_ENABLED", False)
        else "leader"
    )
    capture_control_mode = resolve_capture_control_mode(default_control_mode)
    right_arm_reset = EpisodeRightArmResetConfig.from_env()
    return_to_start = EpisodeReturnToStartConfig.from_env()
    if right_arm_reset.enabled and capture_control_mode != "drag":
        raise ValueError(
            "EPISODE_RIGHT_ARM_RESET_ENABLED=true 目前只允许与 "
            "CAPTURE_CONTROL_MODE=drag 一起使用"
        )
    if return_to_start.enabled and capture_control_mode != "drag":
        raise ValueError(
            "EPISODE_RETURN_TO_START_ENABLED=true 目前只允许与 "
            "CAPTURE_CONTROL_MODE=drag 一起使用"
        )
    if return_to_start.enabled and int(settings.task_index) != 0:
        raise ValueError(
            "EPISODE_RETURN_TO_START_ENABLED=true 只允许用于右臂单臂任务 "
            "(TASK_INDEX=right)"
        )
    if right_arm_reset.enabled and return_to_start.enabled:
        raise ValueError(
            "固定回合前复位与动态回到回合起点不能同时启用；请保持 "
            "EPISODE_RIGHT_ARM_RESET_ENABLED=false"
        )
    if right_arm_reset.enabled:
        logging.info(
            "Per-episode right-arm reset enabled: target_deg=%s speed=%s%% "
            "tolerance=%.3fdeg max_start_delta=%.3fdeg timeout=%.1fs "
            "stop_drag_first=%s pause_readers=%s; left arm untouched",
            [round(value, 3) for value in right_arm_reset.joints_deg],
            right_arm_reset.speed,
            right_arm_reset.tolerance_deg,
            right_arm_reset.max_start_delta_deg,
            right_arm_reset.timeout_s,
            right_arm_reset.stop_drag_first,
            right_arm_reset.pause_readers,
        )
    if return_to_start.enabled:
        logging.info(
            "Dynamic right-arm return enabled: capture pose on start; return after "
            "finish/discard speed=%s%% tolerance=%.3fdeg max_delta=%.3fdeg "
            "timeout=%.1fs pause_readers=%s; left arm untouched",
            return_to_start.speed,
            return_to_start.tolerance_deg,
            return_to_start.max_start_delta_deg,
            return_to_start.timeout_s,
            return_to_start.pause_readers,
        )
    os.environ["CAPTURE_CONTROL_MODE"] = capture_control_mode
    if capture_control_mode == "program":
        os.environ["GRIPPER_COMMAND_SOURCE"] = "program"
        os.environ["TRAJECTORY_GRIPPER_CONTROL_ENABLED"] = "true"
        os.environ["SEND_ACTION_ENABLED"] = "false"
        os.environ["SEND_GRIPPER_ACTION_ENABLED"] = "true"
        os.environ["GRIPPER_PREWARM_ON_EPISODE_START"] = "false"
        settings.send_action_enabled = False
        settings.send_gripper_action_enabled = True
    elif capture_control_mode == "drag":
        os.environ["GRIPPER_COMMAND_SOURCE"] = "leader"
        os.environ["TRAJECTORY_GRIPPER_CONTROL_ENABLED"] = "false"
        os.environ["SEND_ACTION_ENABLED"] = "false"
        os.environ["SEND_GRIPPER_ACTION_ENABLED"] = "true"
        settings.send_action_enabled = False
        settings.send_gripper_action_enabled = True
    else:
        os.environ["GRIPPER_COMMAND_SOURCE"] = "leader"
        os.environ["TRAJECTORY_GRIPPER_CONTROL_ENABLED"] = "false"
    app.apply_runtime_env(settings)
    state = app.CaptureState(settings)
    commands: queue.Queue[str] = queue.Queue()
    ui = app.CaptureControlServer(state, commands)
    ui.start()
    logging.info("Aligned capture UI: http://127.0.0.1:%s", settings.ui_port)
    app.open_capture_ui(settings)

    foot_pedal: FootPedalCommandLoop | None = None
    robot = None
    teleop = None
    dataset = None
    hub: ProducerHub | None = None
    gripper_control: TrajectoryGripperControlServer | None = None
    drag_teach: DragTeachSession | None = None
    ready_gripper_follow: ReadyLeaderGripperFollowSession | None = None
    tactile_bridge_server: TactileFrameBridgeServer | None = None
    interrupted = False

    def stop_robot_from_two_pedal_chord() -> None:
        try:
            _hard_stop_enabled_robot_arms(robot)
        finally:
            # A controller hard stop invalidates the normal drag-stop +
            # current-pose-hold sequence. Never send that hold afterward.
            if drag_teach is not None:
                drag_teach.mark_controller_emergency_stopped()

    try:
        gripper_safety_limits = GripperSafetyLimits(
            min_position=compat.env_int("GRIPPER_MIN_POSITION", 0),
            max_position=compat.env_int("GRIPPER_MAX_POSITION", 1000),
            torque_limit=compat.env_int("GRIPPER_TORQUE_LIMIT", 90),
        )
        foot_pedal = _start_foot_pedal_control(
            commands,
            state,
            emergency_stop=stop_robot_from_two_pedal_chord,
        )
        state.update(status="CONNECTING", message="Connecting hardware")
        robot = app.make_robot(settings)
        teleop_config = app.BiRealmanRM75bLeaderConfig()
        configure_teleop_for_capture_task(teleop_config, settings)
        teleop = app.BiRealmanRM75bLeader(teleop_config)
        robot.connect()
        gripper_safety_limits.apply_torque_limit(robot)
        setattr(robot, "_gripper_safety_limits", gripper_safety_limits)
        logging.info(
            "Gripper safety limits: position=%s..%s torque<=%s",
            gripper_safety_limits.min_position,
            gripper_safety_limits.max_position,
            gripper_safety_limits.torque_limit,
        )
        setattr(robot, "_capture_control_mode", capture_control_mode)
        if capture_control_mode == "program":
            robot.config.send_action_enabled = False
            robot.config.send_gripper_action_enabled = True
            logging.info(
                "Capture control mode=program; follower joints are record-only "
                "and initialized grippers use program target values"
            )
            gripper_control = TrajectoryGripperControlServer(
                robot,
                host=os.environ.get(
                    "TRAJECTORY_GRIPPER_CONTROL_HOST",
                    "127.0.0.1",
                ).strip()
                or "127.0.0.1",
                port=compat.env_int("TRAJECTORY_GRIPPER_CONTROL_PORT", 8767),
                safety_limits=gripper_safety_limits,
                initial_position=compat.env_int(
                    "PROGRAM_GRIPPER_INITIAL_POSITION",
                    950,
                ),
            )
            setattr(robot, "_program_gripper_control", gripper_control)
            gripper_control.start()
        elif capture_control_mode == "drag":
            robot.config.send_action_enabled = False
            robot.config.send_gripper_action_enabled = True
            drag_precise = compat.env_bool("DRAG_FORCE_PRECISE", True)
            drag_singular_wall = compat.env_bool("DRAG_SINGULAR_WALL", True)
            drag_teach = DragTeachSession(
                robot,
                precise=drag_precise,
                singular_wall=drag_singular_wall,
            )
            _stop_enabled_controller_drag_before_ready(robot)
            logging.info(
                "Capture control mode=drag; force drag remains disabled while READY. "
                "After each start command, the right arm resets first, then both follower "
                "arms enter %s 6-axis force position+orientation drag for recording; "
                "joint actions come from follower state and grippers follow the leader",
                "precise" if drag_precise else "fast",
            )
        else:
            logging.info(
                "Capture control mode=leader; teleoperator joint/gripper values use "
                "the original capture path"
            )
        if capture_control_mode == "leader":
            missing_leader_sides = connect_leaders_with_missing_port_fallback(
                teleop,
                allow_partial=compat.env_bool(
                    "LEADER_PARTIAL_FALLBACK_TO_STATE",
                    True,
                ),
            )
            fallback_action_sides = leader_fallback_action_sides(
                missing_leader_sides,
                swap_teleop_actions=bool(settings.swap_teleop_actions),
            )
            setattr(
                robot,
                "_leader_state_fallback_action_sides",
                fallback_action_sides,
            )
            if fallback_action_sides:
                skip_prefixes = _append_send_skip_sides(fallback_action_sides)
                logging.warning(
                    "主臂缺失映射 physical=%s -> follower_action=%s；"
                    "该从臂只记录 state，不发送动作，skip_prefixes=%s",
                    list(missing_leader_sides),
                    list(fallback_action_sides),
                    list(skip_prefixes),
                )
        else:
            teleop.connect()
            if (
                capture_control_mode == "drag"
                and compat.env_bool("READY_GRIPPER_FOLLOW_ENABLED", True)
            ):
                ready_gripper_follow = ReadyLeaderGripperFollowSession(
                    robot=robot,
                    teleop=teleop,
                    settings=settings,
                    app=app,
                    state=state,
                )
                ready_gripper_follow.start()
        dataset = app.build_dataset(robot, settings)

        tactile_frame_publisher: TactileFramePublisher | None = None
        if compat.env_bool("TACTILE_FRAME_BRIDGE_ENABLED", True):
            tactile_frame_publisher = TactileFramePublisher(
                max_pair_skew_ms=compat.env_float(
                    "TACTILE_FRAME_BRIDGE_MAX_PAIR_SKEW_MS",
                    50.0,
                )
            )
            tactile_bridge_server = TactileFrameBridgeServer(
                tactile_frame_publisher,
                host=os.environ.get(
                    "TACTILE_FRAME_BRIDGE_HOST",
                    "127.0.0.1",
                ).strip()
                or "127.0.0.1",
                port=compat.env_int("TACTILE_FRAME_BRIDGE_PORT", 8769),
            )
            tactile_bridge_server.start()
            logging.info(
                "Tactile frame bridge: http://%s:%s",
                tactile_bridge_server.host,
                tactile_bridge_server.port,
            )

        hub = ProducerHub(
            robot,
            settings,
            tactile_frame_publisher=tactile_frame_publisher,
        )
        hub.start()
        episode = int(getattr(dataset.meta, "total_episodes", 0) or 0)
        state.update(
            status="READY",
            current_episode=episode,
            saved_episodes=episode,
            message="Ready. Press start in UI.",
        )

        while episode < settings.num_episodes:
            state.update(status="READY", current_episode=episode, current_frame=0, message="Ready")
            cmd = app.wait_for_command(commands, state, {"start", "stop"})
            if cmd == "stop":
                break

            if ready_gripper_follow is not None:
                ready_gripper_follow.pause()
                if not ready_gripper_follow.is_synced:
                    takeover_position = ready_gripper_follow.takeover_position
                    ready_gripper_follow.resume(
                        takeover_position=takeover_position,
                    )
                    state.update(
                        status="READY",
                        is_recording=False,
                        message=(
                            "Start ignored: move leader gripper to the restored "
                            "start state, then press start again"
                        ),
                    )
                    continue

            episode_start_pose = capture_episode_start_pose(
                robot=robot,
                state=state,
                episode_index=episode,
                task=str(getattr(settings, "task_right", "")),
                config=return_to_start,
            )

            try:
                result = capture_episode_aligned(
                    app=app,
                    robot=robot,
                    teleop=teleop,
                    dataset=dataset,
                    settings=settings,
                    state=state,
                    commands=commands,
                    hub=hub,
                    right_arm_reset=right_arm_reset,
                    drag_teach=drag_teach,
                )
            finally:
                if drag_teach is not None:
                    drag_teach.stop(strict=True)
                    logging.info("Episode ended; left/right force drag disabled before READY/save")
            if result == "stop":
                dataset.clear_episode_buffer(delete_images=True)
                break
            if result == "discard":
                dataset.clear_episode_buffer(delete_images=True)
                return_right_arm_to_episode_start(
                    robot=robot,
                    state=state,
                    commands=commands,
                    app=app,
                    pose=episode_start_pose,
                    config=return_to_start,
                )
                if ready_gripper_follow is not None:
                    ready_gripper_follow.resume(
                        takeover_position=(
                            episode_start_pose.right_gripper_position
                            if (
                                episode_start_pose is not None
                                and return_to_start.restore_gripper
                            )
                            else None
                        ),
                    )
                app.drain_commands(commands)
                state.update(
                    status="READY",
                    message="Discarded and reset. Ready to rerecord.",
                )
                continue

            state.update(status="SAVING", message="Saving episode")
            dataset.save_episode()
            episode = dataset.meta.total_episodes
            if episode_start_pose is not None:
                write_episode_start_pose_metadata(
                    dataset=dataset,
                    pose=episode_start_pose,
                    config=return_to_start,
                    return_status="pending",
                )
                # Protect the completed recording before commanding real motion.
                _snapshot_dataset_after_save(dataset)
                try:
                    return_right_arm_to_episode_start(
                        robot=robot,
                        state=state,
                        commands=commands,
                        app=app,
                        pose=episode_start_pose,
                        config=return_to_start,
                    )
                except BaseException as exc:
                    write_episode_start_pose_metadata(
                        dataset=dataset,
                        pose=episode_start_pose,
                        config=return_to_start,
                        return_status="failed",
                        error=str(exc),
                    )
                    _snapshot_dataset_after_save(dataset)
                    raise
                write_episode_start_pose_metadata(
                    dataset=dataset,
                    pose=episode_start_pose,
                    config=return_to_start,
                    return_status="completed",
                )
            _snapshot_dataset_after_save(dataset)
            if ready_gripper_follow is not None:
                ready_gripper_follow.resume(
                    takeover_position=(
                        episode_start_pose.right_gripper_position
                        if (
                            episode_start_pose is not None
                            and return_to_start.restore_gripper
                        )
                        else None
                    ),
                )
            state.update(
                saved_episodes=episode,
                current_episode=episode,
                current_frame=0,
                status="READY",
                message=f"Saved episode {episode - 1}",
            )
            if settings.reset_time_s > 0:
                time.sleep(settings.reset_time_s)

        state.update(status="STOPPING", stop_requested=True, message="Finalizing dataset")
        if dataset is not None:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer(delete_images=True)
            _finalize_dataset_with_snapshot_restore(dataset)
        state.update(status="DONE", message="Stopped")
        return 0
    except KeyboardInterrupt:
        interrupted = True
        state.update(status="STOPPING", message="Keyboard interrupt")
        return 130
    except Exception as exc:
        logging.exception("Aligned capture app failed")
        state.update(status="ERROR", last_error=str(exc), message="Error")
        return 1
    finally:
        if ready_gripper_follow is not None:
            ready_gripper_follow.stop()
        if gripper_control is not None:
            gripper_control.stop()
            if robot is not None:
                setattr(robot, "_program_gripper_control", None)
        if drag_teach is not None:
            drag_teach.stop()
        if hub is not None:
            hub.stop()
        if tactile_bridge_server is not None:
            tactile_bridge_server.stop()
        compat.cleanup_wrist_processed_caches()
        compat.cleanup_tactile_read_caches()
        try:
            for device in (teleop, robot):
                if device is not None:
                    try:
                        if getattr(device, "is_connected", False):
                            device.disconnect()
                    except Exception:
                        logging.exception("Failed disconnecting %s", device)
        finally:
            compat.stop_tactile_sidecar()
            compat.close_rgb_preview()
            compat.cleanup_active_d405_cameras()
        try:
            if dataset is not None and not getattr(dataset, "_is_finalized", False):
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer(delete_images=True)
                if getattr(dataset, "_aligned_metadata_restored", False):
                    logging.info("Skipping dataset.finalize() after aligned metadata snapshot restore")
                elif interrupted:
                    logging.info("Skipping dataset.finalize() after KeyboardInterrupt")
                    _restore_dataset_snapshot_if_needed(dataset)
                else:
                    _finalize_dataset_with_snapshot_restore(dataset)
        except Exception:
            logging.exception("Failed finalizing dataset")
            try:
                if dataset is not None:
                    _restore_dataset_snapshot_if_needed(dataset, force=True)
            except Exception:
                logging.exception("Failed restoring aligned dataset metadata snapshot")
        if foot_pedal is not None:
            foot_pedal.stop()
        ui.stop()


if __name__ == "__main__":
    raise SystemExit(main())
