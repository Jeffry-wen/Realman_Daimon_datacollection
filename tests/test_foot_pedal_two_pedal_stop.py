from __future__ import annotations

from pathlib import Path
import os
import queue
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capture_realman_x5_force_aligned_app as capture  # noqa: E402


KEY_A = 30
KEY_B = 48
KEY_C = 46


class _State:
    def __init__(self, *, status: str = "READY", is_recording: bool = False) -> None:
        self.status = status
        self.is_recording = is_recording
        self.updates: list[dict[str, object]] = []

    def snapshot(self) -> dict[str, object]:
        return {
            "status": self.status,
            "is_recording": self.is_recording,
        }

    def update(self, **values: object) -> None:
        self.updates.append(values)


def _make_loop(
    *,
    state: _State | None = None,
    emergency_stop=None,
) -> capture.FootPedalCommandLoop:
    values = {
        "FOOT_PEDAL_DEVICE": "/dev/null",
        "FOOT_PEDAL_START_KEY": "KEY_A",
        "FOOT_PEDAL_FINISH_KEY": "KEY_B",
        "FOOT_PEDAL_DISCARD_KEY": "KEY_C",
        "FOOT_PEDAL_TWO_PEDAL_STOP_ENABLED": "true",
        "FOOT_PEDAL_TWO_PEDAL_WINDOW_S": "0.12",
        "FOOT_PEDAL_DEBOUNCE_S": "0.0",
    }
    key_names = {KEY_A: "KEY_A", KEY_B: "KEY_B", KEY_C: "KEY_C"}
    with patch.dict(os.environ, values, clear=False), patch.object(
        capture.listen_foot_pedal,
        "load_key_names",
        return_value=key_names,
    ):
        return capture.FootPedalCommandLoop(
            queue.Queue(),
            state or _State(),
            emergency_stop=emergency_stop,
        )


class TwoPedalStopTest(unittest.TestCase):
    def test_single_pedal_is_immediate_when_two_pedal_stop_is_disabled(self) -> None:
        values = {
            "FOOT_PEDAL_DEVICE": "/dev/null",
            "FOOT_PEDAL_START_KEY": "KEY_A",
            "FOOT_PEDAL_FINISH_KEY": "KEY_B",
            "FOOT_PEDAL_DISCARD_KEY": "KEY_C",
            "FOOT_PEDAL_TWO_PEDAL_STOP_ENABLED": "false",
            "FOOT_PEDAL_TWO_PEDAL_WINDOW_S": "0.12",
            "FOOT_PEDAL_DEBOUNCE_S": "0.0",
        }
        key_names = {KEY_A: "KEY_A", KEY_B: "KEY_B", KEY_C: "KEY_C"}
        with patch.dict(os.environ, values, clear=False), patch.object(
            capture.listen_foot_pedal,
            "load_key_names",
            return_value=key_names,
        ):
            loop = capture.FootPedalCommandLoop(queue.Queue(), _State())

        loop._handle_key_event(KEY_A, 1, now=1.0)

        self.assertEqual(loop.commands.get_nowait(), "start")
        self.assertTrue(
            any("start accepted" in str(update.get("message")) for update in loop.state.updates)
        )

    def test_ignored_pedal_is_published_to_ui_state(self) -> None:
        state = _State(status="READY", is_recording=False)
        loop = _make_loop(state=state)

        loop._emit_command(KEY_B, "finish", now=1.0)

        self.assertTrue(loop.commands.empty())
        self.assertTrue(
            any(
                "finish ignored while READY" in str(update.get("message"))
                for update in state.updates
            )
        )

    def test_hard_stop_is_sent_to_both_enabled_arm_controllers(self) -> None:
        class _Follower:
            def __init__(self) -> None:
                self.stop_calls = 0

            def rm_set_arm_stop(self) -> int:
                self.stop_calls += 1
                return 0

        left = _Follower()
        right = _Follower()
        robot = SimpleNamespace(
            left_arm=SimpleNamespace(_follower_arm=left),
            right_arm=SimpleNamespace(_follower_arm=right),
        )

        capture._hard_stop_enabled_robot_arms(robot)

        self.assertEqual(left.stop_calls, 1)
        self.assertEqual(right.stop_calls, 1)

    def test_each_two_pedal_pair_triggers_stop_and_suppresses_single_commands(self) -> None:
        for first, second in ((KEY_A, KEY_B), (KEY_A, KEY_C), (KEY_B, KEY_C)):
            with self.subTest(pair=(first, second)):
                stop_calls: list[bool] = []
                loop = _make_loop(emergency_stop=lambda: stop_calls.append(True))

                loop._handle_key_event(first, 1, now=10.00)
                loop._handle_key_event(second, 1, now=10.02)
                loop._flush_pending_primary_commands(now=11.00)

                self.assertEqual(stop_calls, [True])
                self.assertEqual(loop.commands.get_nowait(), "stop")
                self.assertTrue(loop.commands.empty())
                self.assertTrue(
                    any(update.get("stop_requested") is True for update in loop.state.updates)
                )

    def test_single_pedal_command_is_emitted_after_chord_window(self) -> None:
        loop = _make_loop()

        loop._handle_key_event(KEY_A, 1, now=20.00)
        loop._handle_key_event(KEY_A, 0, now=20.02)
        loop._flush_pending_primary_commands(now=20.11)
        self.assertTrue(loop.commands.empty())

        loop._flush_pending_primary_commands(now=20.12)
        self.assertEqual(loop.commands.get_nowait(), "start")

    def test_chord_is_latched_until_all_pedals_are_released(self) -> None:
        stop_calls: list[bool] = []
        loop = _make_loop(emergency_stop=lambda: stop_calls.append(True))

        loop._handle_key_event(KEY_A, 1, now=30.00)
        loop._handle_key_event(KEY_B, 1, now=30.01)
        loop._handle_key_event(KEY_C, 1, now=30.02)
        self.assertEqual(stop_calls, [True])

        loop._handle_key_event(KEY_A, 0, now=30.03)
        loop._handle_key_event(KEY_B, 0, now=30.04)
        loop._handle_key_event(KEY_C, 0, now=30.05)
        loop._handle_key_event(KEY_B, 1, now=30.06)
        loop._handle_key_event(KEY_C, 1, now=30.07)
        self.assertEqual(stop_calls, [True, True])

    def test_controller_failure_still_queues_capture_stop(self) -> None:
        state = _State()

        def fail() -> None:
            raise RuntimeError("controller unavailable")

        loop = _make_loop(state=state, emergency_stop=fail)
        loop._handle_key_event(KEY_A, 1, now=40.00)
        loop._handle_key_event(KEY_B, 1, now=40.01)

        self.assertEqual(loop.commands.get_nowait(), "stop")
        self.assertTrue(any("controller unavailable" in str(update) for update in state.updates))


if __name__ == "__main__":
    unittest.main()
