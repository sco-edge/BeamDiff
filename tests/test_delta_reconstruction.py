import numpy as np
import pytest


def test_current_modular_delta_behavior_is_explicit(delta_functions):
    x = np.zeros((1, 10, 234, 4), dtype=np.uint16)
    x[:, 0, :, :] = [510, 510, 126, 126]
    x[:, 1:, :, :] = [2, 2, 2, 2]
    delta = delta_functions['frame_delta'](x)
    np.testing.assert_array_equal(delta[0, 0, 0], [4, 4, 4, 4])
    linear = delta_functions['frame_delta'](x, 'linear')
    np.testing.assert_array_equal(linear[0, 0, 0], [-508, -508, -124, -124])


def test_reconstruction_preserves_anchor_wraps_phi_and_clips_psi(delta_functions):
    anchor = np.broadcast_to([510, 2, 126, 1], (1, 234, 4)).astype(np.uint16).copy()
    delta = np.zeros((1, 9, 234, 4), dtype=np.float32)
    delta[:, 0] = [4, -4, 4, -4]
    result = delta_functions['reconstruct'](anchor, delta)
    np.testing.assert_array_equal(result[:, 0], anchor)
    np.testing.assert_array_equal(result[0, 1, 0], [2, 510, 127, 0])
    assert result.shape == (1, 10, 234, 4)
    assert result.dtype == np.uint16


def test_anchor_features_are_finite_and_six_channel(delta_functions):
    anchor = np.zeros((2, 1, 234, 4), dtype=np.uint16)
    features = delta_functions['anchor_features'](anchor)
    assert features.shape == (2, 1, 234, 6)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features[..., :2] ** 2 + features[..., 2:4] ** 2, 1, atol=1e-6)


def test_invalid_delta_mode(delta_functions):
    with pytest.raises(ValueError, match='unknown delta mode'):
        delta_functions['frame_delta'](np.zeros((1, 10, 234, 4)), 'unsupported')
