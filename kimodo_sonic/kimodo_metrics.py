#!/usr/bin/env python3
"""Isaac Sim/WebRTC viewer for Kimodo-generated G1 target trajectories.

Run from the GR00T-WholeBodyControl container:

    LIVESTREAM=2 /workspace/isaaclab/isaaclab.sh -p kimodo_sonic/kimodo_metrics.py --livestream 2
"""

from __future__ import annotations

from pathlib import Path

from motion_sonic import motionbricks_metrics as viewer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "kimodo_sonic" / "motion"
DEFAULT_MOTION_NAME = "kimodo_to_target_forward_5m_hand_raise"
DEFAULT_TARGET_NAME = "forward_5m_hand_raise_target"


def build_parser():
    parser = viewer.build_parser()
    parser.description = "Isaac Sim WebRTC viewer for a Kimodo 5 m zero-DOF target trajectory."
    parser.set_defaults(
        qpos=str(DEFAULT_RUN_DIR / "qpos" / f"{DEFAULT_MOTION_NAME}.npy"),
        target=str(DEFAULT_RUN_DIR / "target_reference" / f"{DEFAULT_TARGET_NAME}.npz"),
        marker_json=str(DEFAULT_RUN_DIR / "visualization" / f"{DEFAULT_MOTION_NAME}_trajectory_markers.json"),
        loop=0,
        strip_target_hold=1,
        show_dof_panel=1,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    viewer.run_viewer(args)


if __name__ == "__main__":
    main()
