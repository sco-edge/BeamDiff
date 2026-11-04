import numpy as np
import pytest

from scripts.audit_bfa_generation_fidelity import js_divergence, signed_delta


def test_identical_histograms_have_zero_js_divergence():
    counts = np.asarray([0, 1, 4, 0])
    assert js_divergence(counts, counts) == pytest.approx(0)


def test_js_is_symmetric_bounded_and_handles_empty_bins():
    left, right = np.asarray([1, 0]), np.asarray([0, 1])
    value = js_divergence(left, right)
    assert value == pytest.approx(js_divergence(right, left))
    assert 0 <= value <= 1
    assert value == pytest.approx(1, abs=1e-8)


def test_circular_phi_boundary_crossing():
    values = np.asarray([[510, 2, 510]], dtype=np.uint16)
    np.testing.assert_array_equal(signed_delta(values, 512), [[4, -4]])
