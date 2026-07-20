#!/usr/bin/env python3
"""Interactive command terminal for ardy_sonic/general.

Type env-local/global XY goals, such as:

    0 3
    1 -1

Each command is written atomically into the shared runtime directory. The
container-side LiveGoalReplanCallback picks it up and replans from the robot's
current physical pose toward that absolute target. Pass --relative to send a
current-position delta instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_runtime() -> Path:
    env = os.environ.get("ARDY_SONIC_RUNTIME")
    if env:
        return Path(env)
    # Match ardy_sonic/run_planner.sh on the host and run_ardy_sonic.sh in the
    # container: both sides normally share <workspace>/shared_io/runtime.
    if Path("/workspace/shared_io").exists():
        return Path("/workspace/shared_io/runtime")
    return _workspace_root() / "shared_io" / "runtime"


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _write_goal(path: Path, x: float, y: float, mode: str, note: str = "") -> None:
    now = time.time()
    data = {
        "seq": time.time_ns(),
        "created_at": now,
        "command_mode": mode,
        "note": note,
    }
    if mode == "absolute":
        data["goal_xy"] = [float(x), float(y)]
    elif mode == "relative":
        data["goal_delta_xy"] = [float(x), float(y)]
    else:
        raise ValueError(f"unknown command mode: {mode}")
    _atomic_write_json(path, data)
    key = "goal_xy" if mode == "absolute" else "goal_delta_xy"
    print(f"[goal_console] wrote {key}={[round(x, 4), round(y, 4)]} -> {path}", flush=True)


def _parse_xy(text: str) -> tuple[float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("expected exactly two numbers, e.g. 0 3")
    return float(parts[0]), float(parts[1])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xy", nargs="*", help="Optional one-shot command: X Y")
    ap.add_argument("--runtime", default=None, help="Shared runtime dir. Default matches run_planner.sh.")
    ap.add_argument("--file", default="general_goal_command.json", help="Command filename inside the runtime dir.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--relative",
        action="store_true",
        help="Interpret X Y as a current-position delta in env-local coordinates.",
    )
    mode.add_argument(
        "--absolute",
        action="store_true",
        help="Interpret X Y as a literal env-local/global goal coordinate (default).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    runtime = Path(args.runtime) if args.runtime else _default_runtime()
    path = runtime / args.file
    mode = "relative" if args.relative else "absolute"

    if args.xy:
        if len(args.xy) != 2:
            raise SystemExit("one-shot mode expects exactly two args: X Y")
        _write_goal(path, float(args.xy[0]), float(args.xy[1]), mode=mode, note="one-shot")
        return 0

    print("[goal_console] live ARDY goal console")
    print(f"[goal_console] runtime={runtime}")
    print(f"[goal_console] command_file={path}")
    if mode == "absolute":
        print("[goal_console] type `x y` to set an absolute env-local goal, or `q` to quit.")
    else:
        print("[goal_console] type `dx dy` to move relative to the current env-local position, or `q` to quit.")
    while True:
        try:
            line = input("goal xy> " if mode == "absolute" else "goal delta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in {"q", "quit", "exit"}:
            return 0
        try:
            x, y = _parse_xy(line)
        except ValueError as exc:
            print(f"[goal_console] {exc}", file=sys.stderr)
            continue
        _write_goal(path, x, y, mode=mode, note="interactive")


if __name__ == "__main__":
    raise SystemExit(main())
