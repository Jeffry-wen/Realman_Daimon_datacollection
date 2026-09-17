#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import os
import re
import select
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_NAME_HINTS = ("SEMICO USB Keyboard",)
EV_KEY = 0x01
EV_REL = 0x02
EV_ABS = 0x03
EV_MSC = 0x04
EV_SYN = 0x00
EVIOCGRAB = 0x40044590
EVENT_STRUCT = struct.Struct("@llHHi")


@dataclass
class EventDevice:
    path: str
    name: str
    fd: int

    def fileno(self) -> int:
        return self.fd


def _event_name(path: Path) -> str:
    name_path = Path("/sys/class/input") / path.name / "device/name"
    try:
        return name_path.read_text(errors="replace").strip()
    except OSError:
        return ""


def discover_device_paths(name_hints: tuple[str, ...]) -> list[str]:
    paths: list[str] = []
    for event_path in sorted(Path("/dev/input").glob("event*")):
        name = _event_name(event_path)
        if any(hint.lower() in name.lower() for hint in name_hints):
            paths.append(str(event_path))
    return paths


def discover_all_event_paths() -> list[str]:
    return [str(path) for path in sorted(Path("/dev/input").glob("event*"))]


def open_devices(paths: list[str], *, grab: bool) -> list[EventDevice]:
    devices: list[EventDevice] = []
    for path in paths:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        if grab:
            fcntl.ioctl(fd, EVIOCGRAB, 1)
        devices.append(EventDevice(path=path, name=_event_name(Path(path)), fd=fd))
    return devices


def close_devices(devices: list[EventDevice], *, grab: bool) -> None:
    for dev in devices:
        if grab:
            try:
                fcntl.ioctl(dev.fd, EVIOCGRAB, 0)
            except OSError:
                pass
        try:
            os.close(dev.fd)
        except OSError:
            pass


def load_key_names() -> dict[int, str]:
    names: dict[int, str] = {}
    for path in ("/usr/include/linux/input-event-codes.h", "/usr/include/linux/input.h"):
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text(errors="ignore").splitlines():
            match = re.match(r"#define\s+((?:KEY|BTN)_[A-Za-z0-9_]+)\s+((?:0x)?[0-9A-Fa-f]+)\b", line)
            if not match:
                continue
            name, code_text = match.groups()
            code = int(code_text, 0)
            names.setdefault(code, name)
    return names


def event_type_name(event_type: int) -> str:
    return {
        EV_KEY: "KEY",
        EV_REL: "REL",
        EV_ABS: "ABS",
        EV_MSC: "MSC",
    }.get(event_type, f"EV_{event_type}")


def read_events(dev: EventDevice) -> list[tuple[int, int, int, int]]:
    events: list[tuple[int, int, int, int]] = []
    while True:
        try:
            data = os.read(dev.fd, EVENT_STRUCT.size * 64)
        except BlockingIOError:
            break
        if not data:
            break
        usable = len(data) - (len(data) % EVENT_STRUCT.size)
        for offset in range(0, usable, EVENT_STRUCT.size):
            sec, usec, event_type, code, value = EVENT_STRUCT.unpack_from(data, offset)
            events.append((sec, usec, event_type, code, value))
    return events


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Listen to a three-button USB foot pedal and print key events."
    )
    parser.add_argument(
        "--device",
        action="append",
        default=[],
        help="Specific /dev/input/eventX path. May be repeated. Default: auto-detect SEMICO USB Keyboard.",
    )
    parser.add_argument(
        "--name-hint",
        action="append",
        default=[],
        help="Device name substring for auto-detect. Default: SEMICO USB Keyboard.",
    )
    parser.add_argument("--presses", type=int, default=3, help="Stop after this many key down events. Use 0 forever.")
    parser.add_argument("--grab", action="store_true", help="Exclusively grab the device while listening.")
    parser.add_argument("--all", action="store_true", help="Listen to every /dev/input/event* device.")
    parser.add_argument("--raw", action="store_true", help="Print non-key events too.")
    parser.add_argument(
        "--no-mouse-motion",
        action="store_true",
        help="Suppress EV_REL mouse movement events; useful when scanning all devices.",
    )
    parser.add_argument("--list", action="store_true", help="List /dev/input/event* devices and exit.")
    args = parser.parse_args()

    if args.list:
        for path in discover_all_event_paths():
            print(f"{path}: {_event_name(Path(path))}")
        return 0

    if args.device:
        paths = args.device
    elif args.all:
        paths = discover_all_event_paths()
    else:
        paths = discover_device_paths(tuple(args.name_hint) if args.name_hint else DEFAULT_NAME_HINTS)
    if not paths:
        print("No matching input devices found.", file=sys.stderr)
        print("Try: ls -l /dev/input/by-id /dev/input/by-path", file=sys.stderr)
        return 1

    try:
        devices = open_devices(paths, grab=args.grab)
    except PermissionError as exc:
        print(f"Permission denied opening input device: {exc}", file=sys.stderr)
        print("Try: sudo -E python scripts/listen_foot_pedal.py --presses 3 --grab", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Failed opening/grabbing input device: {exc}", file=sys.stderr)
        return 2

    key_names = load_key_names()
    print("Listening devices:")
    for dev in devices:
        print(f"  {dev.path}: {dev.name}")
    print()
    print("现在从左到右踩三下。DOWN 是踩下，UP 是松开。")
    print("Press Ctrl+C to stop.")
    print()

    press_count = 0
    started = time.perf_counter()
    try:
        while True:
            ready, _, _ = select.select(devices, [], [], 0.5)
            for dev in ready:
                for _sec, _usec, event_type, code, value in read_events(dev):
                    if event_type == EV_SYN:
                        continue
                    if args.no_mouse_motion and event_type == EV_REL:
                        continue
                    if event_type != EV_KEY and not args.raw:
                        continue
                    elapsed = time.perf_counter() - started
                    if event_type == EV_KEY:
                        if value == 1:
                            state = "DOWN"
                        elif value == 0:
                            state = "UP"
                        elif value == 2:
                            state = "HOLD"
                        else:
                            state = str(value)
                        name = key_names.get(code, f"KEY_CODE_{code}")
                        print(
                            f"{elapsed:8.3f}s device={Path(dev.path).name:<7} name={dev.name!r} "
                            f"type=KEY state={state:<4} code={code:<4} key={name}",
                            flush=True,
                        )
                        if value == 1:
                            press_count += 1
                            print(f"  press #{press_count}: {name} ({code}) from {dev.path}", flush=True)
                            if args.presses > 0 and press_count >= args.presses:
                                return 0
                    else:
                        print(
                            f"{elapsed:8.3f}s device={Path(dev.path).name:<7} name={dev.name!r} "
                            f"type={event_type_name(event_type):<4} code={code:<4} value={value}",
                            flush=True,
                        )
    except KeyboardInterrupt:
        return 130
    finally:
        close_devices(devices, grab=args.grab)


if __name__ == "__main__":
    raise SystemExit(main())
