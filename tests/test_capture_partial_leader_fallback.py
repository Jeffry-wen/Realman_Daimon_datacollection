from __future__ import annotations

from pathlib import Path
import os
import queue
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capture_realman_x5_force_aligned_app as capture  # noqa: E402


class _Arm:
    JOINT_NAMES = tuple(f"main_joint{i}" for i in range(1, 8))
    GRIPPER_NAME = "main_gripper"
    config = SimpleNamespace(
        leader_gripper_min=0.066,
        leader_gripper_max=0.971,
        gripper_gain=1.1,
    )


class _CaptureState:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []

    def update(self, **values: object) -> None:
        self.updates.append(values)


class _Follower:
    def __init__(
        self,
        joints: list[float],
        *,
        move_return: int = 0,
        drag_stop_return: int = 0,
        drag_peers: tuple["_Follower", ...] = (),
    ) -> None:
        self.joints = list(joints)
        self.move_return = int(move_return)
        self.drag_stop_return = int(drag_stop_return)
        self.drag_peers = drag_peers
        self.move_calls: list[tuple[list[float], int, int, int, int]] = []
        self.hold_calls: list[tuple[list[float], bool, int]] = []
        self.drag_start_calls = 0
        self.drag_stop_calls = 0
        self.drag_active = False

    def rm_get_robot_info(self):
        return 0, {"force_type": "6F"}

    def rm_set_force_drag_mode(self, mode: int) -> int:
        return 0

    def rm_start_multi_drag_teach(self, mode: int, singular_wall: int) -> int:
        self.drag_start_calls += 1
        self.drag_active = True
        return 0

    def rm_stop_drag_teach(self) -> int:
        self.drag_stop_calls += 1
        if self.drag_stop_return == 0:
            self.drag_active = False
        return self.drag_stop_return

    def rm_get_joint_degree(self):
        return 0, list(self.joints)

    def rm_get_joint_min_pos(self):
        return 0, [-180.0] * 7

    def rm_get_joint_max_pos(self):
        return 0, [180.0] * 7

    def rm_movej(self, joints, speed, radius, connect, block) -> int:
        if self.drag_active or any(peer.drag_active for peer in self.drag_peers):
            raise AssertionError("rm_movej must run before either arm enters drag mode")
        self.move_calls.append((list(joints), speed, radius, connect, block))
        if self.move_return == 0:
            self.joints = list(joints)
        return self.move_return

    def rm_movej_canfd(self, joints, follow, trajectory_mode) -> int:
        if self.drag_active:
            raise AssertionError("drag must be stopped before a joint command")
        self.hold_calls.append((list(joints), bool(follow), int(trajectory_mode)))
        return 0


class _CachedStateReader:
    def __init__(self, joints, *, age_s: float = 0.01) -> None:
        self.joints = list(joints)
        self.age_s = float(age_s)

    def get_state(self):
        return list(self.joints)

    def get_state_age(self) -> float:
        return self.age_s


class _PausableReader(_CachedStateReader):
    def __init__(self, joints, *, age_s: float = 0.01) -> None:
        super().__init__(joints, age_s=age_s)
        self.running = True
        self.stop_calls = 0
        self.join_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False

    def join(self, timeout=None) -> None:
        self.join_calls += 1

    def is_alive(self) -> bool:
        return self.running


class _CommandApp:
    @staticmethod
    def drain_commands(commands):
        drained = []
        while True:
            try:
                drained.append(commands.get_nowait())
            except queue.Empty:
                return drained


class _Gripper:
    def __init__(self, position: int) -> None:
        self.position = int(position)
        self.is_connected = True
        self.is_initialized = True
        self.move_calls: list[int] = []
        self.speed_calls: list[int] = []

    def read_position(self) -> int:
        return self.position

    def move_to_pos(self, position: int) -> bool:
        self.move_calls.append(int(position))
        self.position = int(position)
        return True

    def set_speed(self, speed: int) -> bool:
        self.speed_calls.append(int(speed))
        return True


class PartialLeaderFallbackTest(unittest.TestCase):
    def test_tactile_health_ignores_disabled_arm_placeholders(self) -> None:
        class Tactile:
            @staticmethod
            def get_debug_status():
                healthy = {
                    "connected": True,
                    "has_frame": True,
                    "frame_count": 100,
                    "fps": 30.0,
                    "age_ms": 10.0,
                }
                disabled = {
                    "connected": False,
                    "has_frame": False,
                    "frame_count": 0,
                    "fps": 0.0,
                    "age_ms": None,
                }
                return {
                    "left_left": dict(disabled),
                    "left_right": dict(disabled),
                    "right_left": dict(healthy),
                    "right_right": dict(healthy),
                }

        robot = SimpleNamespace(
            config=SimpleNamespace(
                left_arm_config=SimpleNamespace(enabled=False),
                right_arm_config=SimpleNamespace(enabled=True),
            ),
            x5_tactile=Tactile(),
        )

        self.assertTrue(capture.compat.x5_tactile_streams_healthy(robot))

    def test_right_arm_wrist_patch_skips_disabled_left_placeholder(self) -> None:
        calls: list[str] = []
        left_arm = SimpleNamespace(
            config=SimpleNamespace(wrist_cam_name="cam_left_wrist")
        )
        right_arm = SimpleNamespace(
            config=SimpleNamespace(wrist_cam_name="cam_right_wrist")
        )
        robot = SimpleNamespace(
            config=SimpleNamespace(
                left_arm_config=SimpleNamespace(
                    enabled=False,
                    connect_wrist_camera=True,
                ),
                right_arm_config=SimpleNamespace(
                    enabled=True,
                    connect_wrist_camera=True,
                ),
            ),
            left_arm=left_arm,
            right_arm=right_arm,
            observation_features={
                "right_cam_right_wrist": (1080, 1920, 3),
            },
        )

        def mark_patched(arm, side):
            calls.append(side)
            arm._kd_tacmae_wrist_undistort_crop_patch = True
            return True

        with (
            patch.dict(
                os.environ,
                {"WRIST_UNDISTORT": "true", "WRIST_UNDISTORT_CROP_SIZE": "896"},
                clear=False,
            ),
            patch.object(
                capture.compat,
                "_patch_arm_wrist_undistort_crop",
                side_effect=mark_patched,
            ),
        ):
            capture.compat.patch_wrist_undistort_crop(robot)

        self.assertEqual(calls, ["right"])
        self.assertNotIn("left_cam_left_wrist", robot.observation_features)
        self.assertEqual(
            robot.observation_features["right_cam_right_wrist"],
            (896, 896, 3),
        )

    def test_gripper_safety_ignores_disabled_arm_placeholder(self) -> None:
        torque_calls: list[int] = []

        class RawGripper:
            def set_torque_limit(self, value: int) -> bool:
                torque_calls.append(value)
                return True

        left_arm = SimpleNamespace(
            GRIPPER_NAME="main_gripper",
            config=SimpleNamespace(connect_gripper=True),
            _gripper=None,
        )
        right_arm = SimpleNamespace(
            GRIPPER_NAME="main_gripper",
            config=SimpleNamespace(connect_gripper=True),
            _gripper=SimpleNamespace(_gripper=RawGripper()),
        )
        robot = SimpleNamespace(
            config=SimpleNamespace(
                left_arm_config=SimpleNamespace(enabled=False, connect_gripper=True),
                right_arm_config=SimpleNamespace(enabled=True, connect_gripper=True),
            ),
            left_arm=left_arm,
            right_arm=right_arm,
        )

        capture.GripperSafetyLimits(100, 1000, 60).apply_torque_limit(robot)

        self.assertEqual(torque_calls, [60])

    def test_right_arm_capture_connects_only_effective_gripper_leader(self) -> None:
        swapped_config = SimpleNamespace(
            left_arm_config=SimpleNamespace(enabled=True),
            right_arm_config=SimpleNamespace(enabled=True),
        )
        enabled = capture.configure_teleop_for_capture_task(
            swapped_config,
            SimpleNamespace(task_index=0, swap_teleop_actions=True),
        )
        self.assertEqual(enabled, ("left",))
        self.assertTrue(swapped_config.left_arm_config.enabled)
        self.assertFalse(swapped_config.right_arm_config.enabled)

        direct_config = SimpleNamespace(
            left_arm_config=SimpleNamespace(enabled=True),
            right_arm_config=SimpleNamespace(enabled=True),
        )
        enabled = capture.configure_teleop_for_capture_task(
            direct_config,
            SimpleNamespace(task_index=0, swap_teleop_actions=False),
        )
        self.assertEqual(enabled, ("right",))
        self.assertFalse(direct_config.left_arm_config.enabled)
        self.assertTrue(direct_config.right_arm_config.enabled)

    def test_right_arm_reset_config_parses_exact_degree_target(self) -> None:
        values = {
            "EPISODE_RIGHT_ARM_RESET_ENABLED": "true",
            "EPISODE_RIGHT_ARM_RESET_JOINTS_DEG": "0.251,-0.385,5.442,90.016,0.627,89.769,0.167",
        }
        with patch.dict(os.environ, values, clear=False):
            config = capture.EpisodeRightArmResetConfig.from_env()
        self.assertTrue(config.enabled)
        self.assertEqual(
            config.joints_deg,
            (0.251, -0.385, 5.442, 90.016, 0.627, 89.769, 0.167),
        )

    def test_dynamic_return_config_defaults_to_slow_speed(self) -> None:
        values = {
            "EPISODE_RETURN_TO_START_ENABLED": "true",
            "EPISODE_RETURN_TO_START_SPEED": "5",
            "EPISODE_RETURN_TO_START_MAX_DELTA_DEG": "60",
        }
        with patch.dict(os.environ, values, clear=False):
            config = capture.EpisodeReturnToStartConfig.from_env()
        self.assertTrue(config.enabled)
        self.assertEqual(config.speed, 5)
        self.assertEqual(config.max_start_delta_deg, 60.0)

    def test_start_pedal_pose_capture_reads_only_right_follower_and_gripper(self) -> None:
        joints = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        left_follower = _Follower([99.0] * 7)
        right_follower = _Follower(list(joints))
        robot = SimpleNamespace(
            config=SimpleNamespace(
                left_arm_config=SimpleNamespace(enabled=False),
                right_arm_config=SimpleNamespace(enabled=True, connect_gripper=True),
            ),
            left_arm=SimpleNamespace(_follower_arm=left_follower),
            right_arm=SimpleNamespace(
                _follower_arm=right_follower,
                _gripper=_Gripper(925),
            ),
        )
        pose = capture.capture_episode_start_pose(
            robot=robot,
            state=_CaptureState(),
            episode_index=3,
            task="Pick up the three pens on the table and place them into the container.",
            config=capture.EpisodeReturnToStartConfig(
                enabled=True,
                pause_readers=False,
            ),
        )
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertEqual(pose.episode_index, 3)
        self.assertEqual(pose.right_joint_positions_deg, joints)
        self.assertEqual(pose.right_gripper_position, 925)
        self.assertEqual(left_follower.move_calls, [])

    def test_finish_or_discard_returns_only_right_arm_to_captured_pose(self) -> None:
        start = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        left_follower = _Follower([10.0] * 7)
        right_follower = _Follower([8.0] * 7)
        gripper = _Gripper(400)
        robot = SimpleNamespace(
            config=SimpleNamespace(
                left_arm_config=SimpleNamespace(enabled=True),
                right_arm_config=SimpleNamespace(enabled=True),
            ),
            left_arm=SimpleNamespace(_follower_arm=left_follower),
            right_arm=SimpleNamespace(
                _follower_arm=right_follower,
                _gripper=gripper,
            ),
            _gripper_safety_limits=capture.GripperSafetyLimits(100, 1000, 60),
        )
        config = capture.EpisodeReturnToStartConfig(
            enabled=True,
            speed=5,
            max_start_delta_deg=180.0,
            stable_samples=1,
            settle_s=0.0,
            pause_readers=False,
            restore_gripper=True,
        )
        pose = capture.EpisodeStartPose(
            episode_index=0,
            task="test",
            right_joint_positions_deg=start,
            right_gripper_position=900,
            captured_at_utc="2026-09-17T00:00:00+00:00",
        )
        capture.return_right_arm_to_episode_start(
            robot=robot,
            state=_CaptureState(),
            commands=queue.Queue(),
            app=_CommandApp(),
            pose=pose,
            config=config,
        )
        self.assertEqual(left_follower.move_calls, [])
        self.assertEqual(right_follower.move_calls, [(list(start), 5, 0, 0, 0)])
        self.assertEqual(gripper.move_calls, [900])
        self.assertEqual(gripper.speed_calls, [30, 60])

    def test_ready_session_starts_leader_gripper_control_before_recording(self) -> None:
        sent: list[dict[str, float]] = []
        right_arm = _Arm()
        leader_action = {"right_main_gripper": 0.9}
        robot = SimpleNamespace(
            config=SimpleNamespace(send_action_enabled=False),
            right_arm=right_arm,
            _gripper_safety_limits=capture.GripperSafetyLimits(100, 1000, 60),
            send_action=lambda action: sent.append(dict(action)),
        )
        teleop = SimpleNamespace(
            get_action=lambda: dict(leader_action),
        )
        session = capture.ReadyLeaderGripperFollowSession(
            robot=robot,
            teleop=teleop,
            settings=SimpleNamespace(swap_teleop_actions=False),
            app=SimpleNamespace(),
            state=_CaptureState(),
        )
        session.start()
        try:
            deadline = __import__("time").monotonic() + 1.0
            while not sent and __import__("time").monotonic() < deadline:
                __import__("time").sleep(0.01)
            session.pause()
            self.assertTrue(session.is_synced)
            sent.clear()
            session.resume(takeover_position=100)
            __import__("time").sleep(0.08)
            self.assertFalse(session.is_synced)
            self.assertEqual(sent, [])
            leader_action["right_main_gripper"] = (
                capture.GripperSafetyLimits.position_to_leader_value(100, right_arm)
            )
            deadline = __import__("time").monotonic() + 1.0
            while not session.is_synced and __import__("time").monotonic() < deadline:
                __import__("time").sleep(0.01)
            session.pause()
        finally:
            session.stop()
        self.assertTrue(session.is_synced)
        self.assertTrue(sent)

    def test_start_pose_metadata_is_saved_per_episode_and_for_inference(self) -> None:
        pose = capture.EpisodeStartPose(
            episode_index=2,
            task="Pick up the three pens on the table and place them into the container.",
            right_joint_positions_deg=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
            right_gripper_position=950,
            captured_at_utc="2026-09-17T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            dataset = SimpleNamespace(meta=SimpleNamespace(root=directory))
            path = capture.write_episode_start_pose_metadata(
                dataset=dataset,
                pose=pose,
                config=capture.EpisodeReturnToStartConfig(enabled=True, speed=5),
                return_status="completed",
            )
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
            latest = __import__("json").loads(
                (Path(directory) / "meta" / "inference_start_pose.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(payload["joint_positions_deg"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(payload["gripper_position"], 950)
        self.assertEqual(payload["return_to_start"]["status"], "completed")
        self.assertEqual(latest["source"], "meta/episode_start_poses/episode-000002.json")

    def test_episode_reset_finishes_before_both_arms_enter_drag(self) -> None:
        target = (0.251, -0.385, 5.442, 90.016, 0.627, 89.769, 0.167)
        left_follower = _Follower([1.0] * 7)
        right_follower = _Follower([0.0] * 7, drag_peers=(left_follower,))
        robot = SimpleNamespace(
            config=SimpleNamespace(
                left_arm_config=SimpleNamespace(enabled=True),
                right_arm_config=SimpleNamespace(enabled=True),
            ),
            left_arm=SimpleNamespace(_follower_arm=left_follower),
            right_arm=SimpleNamespace(_follower_arm=right_follower),
        )
        drag = capture.DragTeachSession(robot)
        self.assertFalse(drag.is_active)
        result = capture.reset_then_enable_drag_before_episode(
            robot=robot,
            state=_CaptureState(),
            commands=queue.Queue(),
            app=_CommandApp(),
            config=capture.EpisodeRightArmResetConfig(
                enabled=True,
                joints_deg=target,
                speed=10,
                tolerance_deg=0.5,
                max_start_delta_deg=180.0,
                timeout_s=1.0,
                poll_hz=100.0,
                stable_samples=1,
                settle_s=0.0,
                stop_drag_first=False,
                pause_readers=False,
            ),
            drag_teach=drag,
        )
        self.assertIsNone(result)
        self.assertTrue(drag.is_active)
        self.assertEqual(left_follower.move_calls, [])
        self.assertEqual(left_follower.drag_stop_calls, 0)
        self.assertEqual(left_follower.drag_start_calls, 1)
        self.assertEqual(len(right_follower.move_calls), 1)
        self.assertEqual(right_follower.move_calls[0], (list(target), 10, 0, 0, 0))
        self.assertEqual(right_follower.drag_stop_calls, 0)
        self.assertEqual(right_follower.drag_start_calls, 1)

        drag.stop()
        self.assertFalse(left_follower.drag_active)
        self.assertFalse(right_follower.drag_active)
        self.assertEqual(left_follower.hold_calls, [([1.0] * 7, False, 0)])
        self.assertEqual(right_follower.hold_calls, [(list(target), False, 0)])
        right_follower.joints = [0.0] * 7
        result = capture.reset_then_enable_drag_before_episode(
            robot=robot,
            state=_CaptureState(),
            commands=queue.Queue(),
            app=_CommandApp(),
            config=capture.EpisodeRightArmResetConfig(
                enabled=True,
                joints_deg=target,
                speed=10,
                tolerance_deg=0.5,
                max_start_delta_deg=180.0,
                timeout_s=1.0,
                poll_hz=100.0,
                stable_samples=1,
                settle_s=0.0,
                stop_drag_first=False,
                pause_readers=False,
            ),
            drag_teach=drag,
        )
        self.assertIsNone(result)
        self.assertEqual(len(right_follower.move_calls), 2)
        self.assertEqual(left_follower.drag_start_calls, 2)
        self.assertEqual(right_follower.drag_start_calls, 2)

    def test_episode_reset_reads_radian_async_cache_as_degrees(self) -> None:
        target = (0.251, -0.385, 5.442, 90.016, 0.627, 89.769, 0.167)
        follower = _Follower(list(target))
        reader = _CachedStateReader([value * 3.141592653589793 / 180.0 for value in target])
        arm = SimpleNamespace(
            _follower_arm=follower,
            _follower_state_reader=reader,
            config=SimpleNamespace(use_degrees=False),
        )
        result = capture.reset_right_arm_before_episode(
            robot=SimpleNamespace(right_arm=arm),
            state=_CaptureState(),
            commands=queue.Queue(),
            app=_CommandApp(),
            config=capture.EpisodeRightArmResetConfig(
                enabled=True,
                joints_deg=target,
                stop_drag_first=False,
                pause_readers=False,
            ),
        )
        self.assertIsNone(result)
        self.assertEqual(follower.move_calls, [])

    def test_episode_reset_stops_drag_pauses_readers_and_restores_them(self) -> None:
        target = (0.251, -0.385, 5.442, 90.016, 0.627, 89.769, 0.167)
        left_follower = _Follower([2.0] * 7)
        right_follower = _Follower([0.0] * 7)
        left_state = _PausableReader([0.0] * 7)
        left_force = _PausableReader([0.0] * 6)
        right_state = _PausableReader([0.0] * 7)
        right_force = _PausableReader([0.0] * 6)

        def arm(follower, state_reader, force_reader):
            item = SimpleNamespace(
                _follower_arm=follower,
                _follower_state_reader=state_reader,
                _force_sensor_reader=force_reader,
                config=SimpleNamespace(use_degrees=True),
            )

            def start_state() -> None:
                item._follower_state_reader = _PausableReader(follower.joints)

            def start_force() -> None:
                item._force_sensor_reader = _PausableReader([0.0] * 6)

            item._start_async_state_reader = start_state
            item._start_async_force_reader = start_force
            return item

        left_arm = arm(left_follower, left_state, left_force)
        right_arm = arm(right_follower, right_state, right_force)
        robot = SimpleNamespace(
            config=SimpleNamespace(
                left_arm_config=SimpleNamespace(enabled=True),
                right_arm_config=SimpleNamespace(enabled=True),
            ),
            left_arm=left_arm,
            right_arm=right_arm,
        )
        result = capture.reset_right_arm_before_episode(
            robot=robot,
            state=_CaptureState(),
            commands=queue.Queue(),
            app=_CommandApp(),
            config=capture.EpisodeRightArmResetConfig(
                enabled=True,
                joints_deg=target,
                max_start_delta_deg=180.0,
                settle_s=0.0,
            ),
        )
        self.assertIsNone(result)
        self.assertEqual(left_follower.move_calls, [])
        self.assertEqual(right_follower.drag_stop_calls, 1)
        self.assertEqual(right_follower.move_calls, [(list(target), 10, 0, 0, 0)])
        for reader in (left_state, left_force, right_state, right_force):
            self.assertEqual(reader.stop_calls, 1)
            self.assertEqual(reader.join_calls, 1)
        self.assertIsNot(left_arm._follower_state_reader, left_state)
        self.assertIsNot(left_arm._force_sensor_reader, left_force)
        self.assertIsNot(right_arm._follower_state_reader, right_state)
        self.assertIsNot(right_arm._force_sensor_reader, right_force)

    def test_startup_stops_both_controller_drag_sides_before_ready(self) -> None:
        left_follower = _Follower([1.0] * 7)
        right_follower = _Follower([2.0] * 7)
        robot = SimpleNamespace(
            config=SimpleNamespace(
                left_arm_config=SimpleNamespace(enabled=True),
                right_arm_config=SimpleNamespace(enabled=True),
            ),
            left_arm=SimpleNamespace(_follower_arm=left_follower),
            right_arm=SimpleNamespace(_follower_arm=right_follower),
        )
        capture._stop_enabled_controller_drag_before_ready(robot)
        self.assertEqual(left_follower.drag_stop_calls, 1)
        self.assertEqual(right_follower.drag_stop_calls, 1)
        self.assertEqual(left_follower.move_calls, [])
        self.assertEqual(right_follower.move_calls, [])
        self.assertEqual(left_follower.hold_calls, [([1.0] * 7, False, 0)])
        self.assertEqual(right_follower.hold_calls, [([2.0] * 7, False, 0)])

    def test_episode_reset_rejects_large_start_delta_before_motion(self) -> None:
        follower = _Follower([0.0] * 7)
        robot = SimpleNamespace(
            right_arm=SimpleNamespace(_follower_arm=follower),
        )
        with self.assertRaisesRegex(RuntimeError, "最大关节差"):
            capture.reset_right_arm_before_episode(
                robot=robot,
                state=_CaptureState(),
                commands=queue.Queue(),
                app=_CommandApp(),
                config=capture.EpisodeRightArmResetConfig(
                    enabled=True,
                    joints_deg=(0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0),
                    max_start_delta_deg=45.0,
                    stop_drag_first=False,
                    pause_readers=False,
                ),
            )
        self.assertEqual(follower.move_calls, [])

    def test_failed_right_reset_never_enables_either_drag_side(self) -> None:
        left_follower = _Follower([0.0] * 7)
        right_follower = _Follower([0.0] * 7, move_return=1)
        robot = SimpleNamespace(
            config=SimpleNamespace(
                left_arm_config=SimpleNamespace(enabled=True),
                right_arm_config=SimpleNamespace(enabled=True),
            ),
            left_arm=SimpleNamespace(_follower_arm=left_follower),
            right_arm=SimpleNamespace(_follower_arm=right_follower),
        )
        drag = capture.DragTeachSession(robot)
        with self.assertRaisesRegex(RuntimeError, "rm_movej"):
            capture.reset_then_enable_drag_before_episode(
                robot=robot,
                state=_CaptureState(),
                commands=queue.Queue(),
                app=_CommandApp(),
                config=capture.EpisodeRightArmResetConfig(
                    enabled=True,
                    joints_deg=(1.0,) * 7,
                    max_start_delta_deg=45.0,
                    settle_s=0.0,
                    stop_drag_first=False,
                    pause_readers=False,
                ),
                drag_teach=drag,
            )
        self.assertEqual(left_follower.drag_stop_calls, 0)
        self.assertEqual(left_follower.drag_start_calls, 0)
        self.assertEqual(right_follower.drag_stop_calls, 0)
        self.assertEqual(right_follower.drag_start_calls, 0)

    def test_failed_drag_stop_remains_active_and_blocks_next_episode(self) -> None:
        left_follower = _Follower([0.0] * 7)
        right_follower = _Follower([0.0] * 7, drag_stop_return=1)
        robot = SimpleNamespace(
            config=SimpleNamespace(
                left_arm_config=SimpleNamespace(enabled=True),
                right_arm_config=SimpleNamespace(enabled=True),
            ),
            left_arm=SimpleNamespace(_follower_arm=left_follower),
            right_arm=SimpleNamespace(_follower_arm=right_follower),
        )
        drag = capture.DragTeachSession(robot)
        drag.start()
        with self.assertRaisesRegex(RuntimeError, "关闭拖动失败"):
            drag.stop(strict=True)
        self.assertTrue(drag.is_active)
        with self.assertRaisesRegex(RuntimeError, "拖动模式仍处于启用状态"):
            capture.reset_then_enable_drag_before_episode(
                robot=robot,
                state=_CaptureState(),
                commands=queue.Queue(),
                app=_CommandApp(),
                config=capture.EpisodeRightArmResetConfig(enabled=False),
                drag_teach=drag,
            )

    def test_missing_right_maps_to_left_when_capture_swaps(self) -> None:
        self.assertEqual(
            capture.leader_fallback_action_sides(
                ("right",),
                swap_teleop_actions=True,
            ),
            ("left",),
        )
        self.assertEqual(
            capture.leader_fallback_action_sides(
                ("right",),
                swap_teleop_actions=False,
            ),
            ("right",),
        )

    def test_fallback_replaces_only_selected_side_and_round_trips_gripper(self) -> None:
        robot = SimpleNamespace(left_arm=_Arm(), right_arm=_Arm())
        action = {
            f"{side}_main_joint{i}": 99.0
            for side in ("left", "right")
            for i in range(1, 8)
        }
        action.update(left_main_gripper=0.5, right_main_gripper=0.5)
        state = {f"left_main_joint{i}": float(i) for i in range(1, 8)}
        state["left_main_gripper"] = 0.45

        replaced = capture.replace_arm_actions_with_follower_state(
            action,
            state,
            robot,
            ("left",),
        )
        self.assertEqual(replaced["left_main_joint7"], 7.0)
        self.assertEqual(replaced["right_main_joint7"], 99.0)
        mapped = capture.GripperSafetyLimits.leader_value_to_position(
            replaced["left_main_gripper"],
            robot.left_arm,
        )
        self.assertAlmostEqual(mapped / 1000.0, 0.45, places=7)

        dataset_action = capture.replace_arm_actions_with_follower_state(
            action,
            state,
            robot,
            ("left",),
            dataset_gripper_coordinates=True,
        )
        self.assertEqual(dataset_action["left_main_gripper"], 0.45)

    def test_absent_port_is_disabled_before_atomic_connect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "leader-left"
            existing.touch()
            config = SimpleNamespace(
                left_arm_config=SimpleNamespace(enabled=True, port=str(existing)),
                right_arm_config=SimpleNamespace(enabled=True, port=str(existing) + "-missing"),
            )

            class Teleop:
                def __init__(self) -> None:
                    self.config = config
                    self.connected = False

                def connect(self) -> None:
                    self.connected = True

            teleop = Teleop()
            missing = capture.connect_leaders_with_missing_port_fallback(
                teleop,
                allow_partial=True,
            )
            self.assertTrue(teleop.connected)
            self.assertEqual(missing, ("right",))
            self.assertTrue(config.left_arm_config.enabled)
            self.assertFalse(config.right_arm_config.enabled)


if __name__ == "__main__":
    unittest.main()
