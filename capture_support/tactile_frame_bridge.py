"""Local binary bridge from the X5-owning capture process to visualizers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from http import server
import json
import re
import threading
import time
from typing import Any
from urllib import parse

import numpy as np


BRIDGE_SCHEMA = "kd_tacmae_tactile_bridge_v1"
_TACTILE_KEY = re.compile(
    r"^(?:observation\.)?depth_deformation\.tactile_"
    r"(?P<arm>left|right)_(?P<pad>left|right)$"
)


@dataclass(frozen=True)
class TactileBridgeFrame:
    arm: str
    sequence: int
    source_times: tuple[float, float]
    packed_u16: np.ndarray

    @property
    def pair_skew_ms(self) -> float:
        return abs(self.source_times[0] - self.source_times[1]) * 1000.0


@dataclass(frozen=True)
class _PadFrame:
    generation: int
    source_time: float
    packed_u16: np.ndarray


class TactileFramePublisher:
    """Pair fresh left/right pad frames into an immutable bridge snapshot."""

    def __init__(self, *, max_pair_skew_ms: float = 50.0) -> None:
        self.max_pair_skew_s = max(0.0, float(max_pair_skew_ms) / 1000.0)
        self.condition = threading.Condition()
        self._generation = 0
        self._sequence = {"left": 0, "right": 0}
        self._latest: dict[str, dict[str, _PadFrame]] = {
            "left": {},
            "right": {},
        }
        self._last_paired_generation = {
            "left": {"left": 0, "right": 0},
            "right": {"left": 0, "right": 0},
        }
        self._paired: dict[str, TactileBridgeFrame | None] = {
            "left": None,
            "right": None,
        }
        self._pair_times: dict[str, deque[float]] = {
            "left": deque(maxlen=90),
            "right": deque(maxlen=90),
        }

    def publish(self, key: str, source_time: float, image: np.ndarray) -> bool:
        match = _TACTILE_KEY.fullmatch(str(key))
        if match is None:
            return False
        packed = np.asarray(image)
        if (
            packed.ndim != 3
            or packed.shape[-1] != 3
            or packed.dtype != np.uint16
        ):
            return False
        packed = np.ascontiguousarray(packed)
        arm = match.group("arm")
        pad = match.group("pad")
        timestamp = float(source_time) if source_time > 0 else time.perf_counter()
        with self.condition:
            self._generation += 1
            self._latest[arm][pad] = _PadFrame(
                generation=self._generation,
                source_time=timestamp,
                packed_u16=packed,
            )
            latest = self._latest[arm]
            if set(latest) != {"left", "right"}:
                return True
            last = self._last_paired_generation[arm]
            if any(
                latest[name].generation <= last[name]
                for name in ("left", "right")
            ):
                return True
            left = latest["left"]
            right = latest["right"]
            if abs(left.source_time - right.source_time) > self.max_pair_skew_s:
                return True
            if left.packed_u16.shape != right.packed_u16.shape:
                return True
            self._sequence[arm] += 1
            frame = TactileBridgeFrame(
                arm=arm,
                sequence=self._sequence[arm],
                source_times=(left.source_time, right.source_time),
                packed_u16=np.stack(
                    [left.packed_u16, right.packed_u16],
                    axis=0,
                ),
            )
            self._paired[arm] = frame
            self._last_paired_generation[arm] = {
                "left": left.generation,
                "right": right.generation,
            }
            self._pair_times[arm].append(time.perf_counter())
            self.condition.notify_all()
        return True

    def wait_next(
        self,
        arm: str,
        after_sequence: int,
        *,
        timeout_s: float,
    ) -> TactileBridgeFrame | None:
        if arm not in self._paired:
            raise ValueError("arm must be left or right")
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self.condition:
            while True:
                frame = self._paired[arm]
                if frame is not None and frame.sequence > int(after_sequence):
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(timeout=remaining)

    def status(self, arm: str) -> dict[str, Any]:
        if arm not in self._paired:
            raise ValueError("arm must be left or right")
        with self.condition:
            frame = self._paired[arm]
            pair_times = self._pair_times[arm]
            fps = (
                (len(pair_times) - 1)
                / max(pair_times[-1] - pair_times[0], 1e-6)
                if len(pair_times) > 1
                else 0.0
            )
            return {
                "ok": True,
                "schema": BRIDGE_SCHEMA,
                "arm": arm,
                "ready": frame is not None,
                "sequence": 0 if frame is None else frame.sequence,
                "shape": None if frame is None else list(frame.packed_u16.shape),
                "dtype": None if frame is None else str(frame.packed_u16.dtype),
                "pair_skew_ms": (
                    None if frame is None else frame.pair_skew_ms
                ),
                "publisher_fps": fps,
                "hardware_owner": "capture_process",
            }


class _ReusableThreadingHTTPServer(server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _handler_for(publisher: TactileFramePublisher):
    class Handler(server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = parse.urlsplit(self.path)
            query = parse.parse_qs(parsed.query)
            arm = query.get("arm", ["right"])[0]
            try:
                if parsed.path == "/healthz":
                    self._json(
                        {
                            "ok": True,
                            "schema": BRIDGE_SCHEMA,
                            "hardware_owner": "capture_process",
                        }
                    )
                    return
                if parsed.path == "/tactile/status":
                    self._json(publisher.status(arm))
                    return
                if parsed.path != "/tactile/frame":
                    self.send_error(404)
                    return
                after = int(query.get("after", ["0"])[0])
                timeout_ms = min(
                    5000,
                    max(0, int(query.get("timeout_ms", ["1000"])[0])),
                )
                frame = publisher.wait_next(
                    arm,
                    after,
                    timeout_s=timeout_ms / 1000.0,
                )
                if frame is None:
                    self.send_response(204)
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return
                body = memoryview(frame.packed_u16).cast("B")
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Tactile-Schema", BRIDGE_SCHEMA)
                self.send_header("X-Tactile-Arm", frame.arm)
                self.send_header("X-Tactile-Sequence", str(frame.sequence))
                self.send_header(
                    "X-Tactile-Shape",
                    ",".join(str(value) for value in frame.packed_u16.shape),
                )
                self.send_header("X-Tactile-Dtype", str(frame.packed_u16.dtype))
                self.send_header(
                    "X-Tactile-Source-Times",
                    ",".join(f"{value:.9f}" for value in frame.source_times),
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, TypeError) as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


class TactileFrameBridgeServer:
    def __init__(
        self,
        publisher: TactileFramePublisher,
        *,
        host: str = "127.0.0.1",
        port: int = 8769,
    ) -> None:
        self.publisher = publisher
        self.host = str(host)
        self.port = int(port)
        self._server: _ReusableThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._server = _ReusableThreadingHTTPServer(
            (self.host, self.port),
            _handler_for(self.publisher),
        )
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="tactile-frame-bridge-http",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
