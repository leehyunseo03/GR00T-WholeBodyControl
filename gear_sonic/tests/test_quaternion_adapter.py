import numpy as np
import pytest
import torch

from gear_sonic.isaac_utils.quaternion_adapter import wxyz_to_xyzw, xyzw_to_wxyz


def test_xyzw_to_wxyz_identity_numpy():
    q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    np.testing.assert_array_equal(xyzw_to_wxyz(q), np.array([1.0, 0.0, 0.0, 0.0]))


def test_wxyz_to_xyzw_identity_tensor_preserves_device_dtype():
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    out = wxyz_to_xyzw(q)

    assert out.dtype == q.dtype
    assert out.device == q.device
    torch.testing.assert_close(out, torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=q.dtype))


def test_round_trip_batched_tensor():
    q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.70710678, 0.70710678, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    assert wxyz_to_xyzw(q).shape == (2, 4)
    torch.testing.assert_close(xyzw_to_wxyz(wxyz_to_xyzw(q)), q)


def test_rejects_non_quaternion_last_dim():
    with pytest.raises(ValueError, match="last dimension must be 4"):
        xyzw_to_wxyz(torch.zeros(2, 3))
