import math

import numpy as np

from ardy_sonic.base_state_estimator import FootOdometryBaseEstimator


def _quat_from_yaw(yaw: float):
    return [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]


def _rot_inv_xy(yaw: float, xy):
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = float(xy[0]), float(xy[1])
    return np.asarray([c * x + s * y, -s * x + c * y], dtype=np.float32)


def test_foot_odometry_recovers_xy_from_single_stance_anchor():
    est = FootOdometryBaseEstimator(default_height=0.72)
    yaw = 0.25
    left_anchor = np.asarray([0.0, 0.10], dtype=np.float32)

    for i in range(80):
        true_xy = np.asarray([0.01 * i, 0.02 * math.sin(0.05 * i)], dtype=np.float32)
        foot_left_base = _rot_inv_xy(yaw, left_anchor - true_xy)
        out = est.update(
            root_quat_wxyz=_quat_from_yaw(yaw),
            foot_pos_base={"left": [foot_left_base[0], foot_left_base[1], -0.72]},
            contacts={"left": True},
            base_z=0.72,
            initial_xy=true_xy if i == 0 else None,
        )

    np.testing.assert_allclose(out.root_pos[:2], true_xy, atol=1e-5)
    assert out.num_contact_feet == 1


def test_foot_odometry_handles_stance_switch():
    est = FootOdometryBaseEstimator(default_height=0.72)
    yaw = -0.15
    left_anchor = np.asarray([0.0, 0.10], dtype=np.float32)
    right_anchor = None

    for i in range(120):
        true_xy = np.asarray([0.008 * i, 0.05], dtype=np.float32)
        if i < 60:
            contacts = {"left": True, "right": False}
            right_foot_map = true_xy + np.asarray([0.0, -0.10], dtype=np.float32)
        elif i == 60:
            contacts = {"left": True, "right": True}
            right_anchor = true_xy + np.asarray([0.0, -0.10], dtype=np.float32)
            right_foot_map = right_anchor
        else:
            contacts = {"left": False, "right": True}
            right_foot_map = right_anchor
        left_base = _rot_inv_xy(yaw, left_anchor - true_xy)
        right_base = _rot_inv_xy(yaw, right_foot_map - true_xy)
        out = est.update(
            root_quat_wxyz=_quat_from_yaw(yaw),
            foot_pos_base={
                "left": [left_base[0], left_base[1], -0.72],
                "right": [right_base[0], right_base[1], -0.72],
            },
            contacts=contacts,
            base_z=0.72,
            initial_xy=true_xy if i == 0 else None,
        )

    np.testing.assert_allclose(out.root_pos[:2], true_xy, atol=1e-5)
    assert out.contacts == {"left": False, "right": True}
