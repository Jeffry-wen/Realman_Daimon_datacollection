#!/usr/bin/env python3
"""Reconnect a USB foot pedal to an already-running capture web UI.

This sidecar is intentionally scoped to one capture PID.  It exits when that
process exits and reconnects the input device if USB re-enumeration invalidates
the current file descriptor.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import listen_foot_pedal


def _request_json(url: str, *, data: bytes | None = None) -> dict[str, object]:
    request = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(request, timeout=2.0) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-pid", type=int, required=True)
    parser.add_argument(
        "--device",
        default="/dev/input/by-id/usb-PCsensor_FS20Pro-event-kbd",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--grab", action="store_true")
    parser.add_argument("--debounce", type=float, default=0.25)
    args = parser.parse_args()

    key_names = listen_foot_pedal.load_key_names()
    name_to_code = {name: code for code, name in key_names.items()}
    mapping = {
        name_to_code["KEY_A"]: "start",
        name_to_code["KEY_B"]: "finish",
        name_to_code["KEY_C"]: "discard",
    }
    allowed = {
        "start": lambda status: status == "READY",
        "finish": lambda status: status == "RECORDING",
        "discard": lambda status: status == "RECORDING",
    }
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    devices: list[listen_foot_pedal.EventDevice] = []
    last_press: dict[int, float] = {}
    reconnect_after = 0.0
    print(
        f"bridge started capture_pid={args.capture_pid} device={args.device} url={args.url}",
        flush=True,
    )
    try:
        while not stopping and Path(f"/proc/{args.capture_pid}").exists():
            if not devices:
                now = time.monotonic()
                if now < reconnect_after:
                    time.sleep(min(0.2, reconnect_after - now))
                    continue
                try:
                    devices = listen_foot_pedal.open_devices([args.device], grab=args.grab)
                    print(f"pedal connected: {args.device}", flush=True)
                except OSError as exc:
                    print(f"pedal reconnect pending: {exc}", flush=True)
                    reconnect_after = now + 1.0
                    continue

            try:
                ready, _, _ = select.select(devices, [], [], 0.25)
                for device in ready:
                    for _sec, _usec, event_type, code, value in listen_foot_pedal.read_events(device):
                        if event_type != listen_foot_pedal.EV_KEY or value != 1 or code not in mapping:
                            continue
                        now = time.monotonic()
                        if now - last_press.get(code, 0.0) < args.debounce:
                            continue
                        last_press[code] = now
                        command = mapping[code]
                        status_data = _request_json(f"{args.url}/status")
                        status = str(status_data.get("status", ""))
                        key_name = key_names.get(code, str(code))
                        if not allowed[command](status):
                            print(
                                f"ignored key={key_name} command={command} status={status}",
                                flush=True,
                            )
                            continue
                        payload = urllib.parse.urlencode({"cmd": command}).encode("ascii")
                        result = _request_json(f"{args.url}/command", data=payload)
                        print(
                            f"forwarded key={key_name} command={command} status={status} result={result}",
                            flush=True,
                        )
            except (OSError, ValueError) as exc:
                print(f"pedal disconnected: {exc}", flush=True)
                listen_foot_pedal.close_devices(devices, grab=args.grab)
                devices = []
                reconnect_after = time.monotonic() + 0.5
            except Exception as exc:
                print(f"bridge request error: {exc}", flush=True)
                time.sleep(0.5)
    finally:
        listen_foot_pedal.close_devices(devices, grab=args.grab)
    print("bridge stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
