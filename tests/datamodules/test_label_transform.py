"""Tests for LabelTransform."""

import numpy as np
import pytest

from matcha.datamodules.classic.label_transform import (
    LabelTransform,
    ForwardTransformRegistry,
    BackwardTransformRegistry,
    norm_log10,
    inv_log10,
    norm_log2,
    inv_log2,
    norm_log1p,
    inv_log1p,
    norm_ln,
    inv_ln,
    norm_logk100,
    inv_logk100,
    log10p,
    inv_log10p,
    no_scale,
)


# ===================================================================
# Individual transform functions
# ===================================================================


class TestNoScale:
    def test_returns_copy(self):
        x = np.array([1.0, 2.0, 3.0])
        out = no_scale(x)
        np.testing.assert_array_equal(x, out)

    def test_does_not_modify_input(self):
        x = np.array([1.0, 2.0, 3.0])
        original = x.copy()
        no_scale(x)
        np.testing.assert_array_equal(x, original)


class TestLog10Transform:
    def test_forward(self):
        x = np.array([10.0, 100.0, 1000.0])
        out = norm_log10(x)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])

    def test_inverse(self):
        x = np.array([1.0, 2.0, 3.0])
        out = inv_log10(x)
        np.testing.assert_allclose(out, [10.0, 100.0, 1000.0])

    def test_roundtrip(self):
        x = np.array([5.0, 50.0, 500.0])
        out = inv_log10(norm_log10(x))
        np.testing.assert_allclose(out, x, rtol=1e-6)


class TestLog2Transform:
    def test_forward(self):
        x = np.array([2.0, 4.0, 8.0])
        out = norm_log2(x)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])

    def test_inverse(self):
        x = np.array([1.0, 2.0, 3.0])
        out = inv_log2(x)
        np.testing.assert_allclose(out, [2.0, 4.0, 8.0])

    def test_roundtrip(self):
        x = np.array([3.0, 7.0, 15.0])
        out = inv_log2(norm_log2(x))
        np.testing.assert_allclose(out, x, rtol=1e-6)


class TestLnTransform:
    def test_forward(self):
        x = np.array([np.e, np.e**2])
        out = norm_ln(x)
        np.testing.assert_allclose(out, [1.0, 2.0], rtol=1e-6)

    def test_inverse(self):
        x = np.array([1.0, 2.0])
        out = inv_ln(x)
        np.testing.assert_allclose(out, [np.e, np.e**2], rtol=1e-6)

    def test_roundtrip(self):
        x = np.array([5.0, 50.0])
        out = inv_ln(norm_ln(x))
        np.testing.assert_allclose(out, x, rtol=1e-6)


class TestLog1pTransform:
    def test_forward(self):
        x = np.array([np.e - 1, np.e**2 - 1])
        out = norm_log1p(x)
        np.testing.assert_allclose(out, [1.0, 2.0], rtol=1e-6)

    def test_roundtrip(self):
        x = np.array([5.0, 50.0])
        out = inv_log1p(norm_log1p(x))
        np.testing.assert_allclose(out, x, rtol=1e-6)


class TestLogk100Transform:
    def test_forward_at_50(self):
        """logk100(50) = log10((100-50)/50) = log10(1) = 0"""
        x = np.array([50.0])
        out = norm_logk100(x)
        np.testing.assert_allclose(out, [0.0], atol=1e-6)

    def test_roundtrip(self):
        x = np.array([20.0, 50.0, 80.0])
        out = inv_logk100(norm_logk100(x))
        np.testing.assert_allclose(out, x, rtol=1e-4)


class TestLog10pTransform:
    def test_forward(self):
        x = np.array([9.0, 99.0])
        out = log10p(x)
        np.testing.assert_allclose(out, [1.0, 2.0], rtol=1e-6)

    def test_roundtrip(self):
        x = np.array([9.0, 99.0, 999.0])
        out = inv_log10p(log10p(x))
        np.testing.assert_allclose(out, x, rtol=1e-6)


# ===================================================================
# Registry classes
# ===================================================================


class TestForwardTransformRegistry:
    @pytest.mark.parametrize(
        "method",
        ["log10", "log2", "log1p", "ln", "logk100", "log10p", "none"],
    )
    def test_all_methods_registered(self, method):
        assert method in ForwardTransformRegistry.mapping

    def test_scale_dispatches_correctly(self):
        x = np.array([10.0, 100.0])
        out = ForwardTransformRegistry.scale(x, "log10")
        np.testing.assert_allclose(out, [1.0, 2.0])


class TestBackwardTransformRegistry:
    @pytest.mark.parametrize(
        "method",
        ["log10", "log2", "log1p", "ln", "logk100", "log10p", "none"],
    )
    def test_all_methods_registered(self, method):
        assert method in BackwardTransformRegistry.mapping

    def test_scale_dispatches_correctly(self):
        x = np.array([1.0, 2.0])
        out = BackwardTransformRegistry.scale(x, "log10")
        np.testing.assert_allclose(out, [10.0, 100.0])


# ===================================================================
# LabelTransform class
# ===================================================================


class TestLabelTransformInit:
    def test_init_none(self):
        t = LabelTransform()
        assert t.params.transform_map is None

    def test_init_string(self):
        t = LabelTransform(transform_map="log10")
        assert t.params.transform_map == "log10"

    def test_init_list(self):
        t = LabelTransform(transform_map=["log10", "log2"])
        assert t.params.transform_map == {0: "log10", 1: "log2"}

    def test_init_dict_int_keys(self):
        t = LabelTransform(transform_map={0: "log10", 1: "log2"})
        assert t.params.transform_map == {0: "log10", 1: "log2"}

    def test_init_dict_string_keys(self):
        t = LabelTransform(transform_map={"log10": [0, 1], "log2": [2]})
        assert t.params.transform_map[0] == "log10"
        assert t.params.transform_map[1] == "log10"
        assert t.params.transform_map[2] == "log2"


class TestLabelTransformProcess:
    def test_none_map_is_identity(self):
        t = LabelTransform()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = t.process(x, forward=True)
        np.testing.assert_array_equal(out, x)

    def test_single_string_forward_and_backward(self):
        t = LabelTransform(transform_map="log10")
        x = np.array([[10.0], [100.0]])
        fwd = t.process(x, forward=True)
        np.testing.assert_allclose(fwd, [[1.0], [2.0]], rtol=1e-6)
        bwd = t.process(fwd, forward=False)
        np.testing.assert_allclose(bwd, x, rtol=1e-6)

    def test_per_task_forward(self):
        t = LabelTransform(transform_map=["none", "log10"])
        x = np.array([[5.0, 10.0], [7.0, 100.0]])
        fwd = t.process(x, forward=True)
        np.testing.assert_allclose(fwd[:, 0], [5.0, 7.0])
        np.testing.assert_allclose(fwd[:, 1], [1.0, 2.0], rtol=1e-6)

    def test_does_not_modify_input(self):
        t = LabelTransform(transform_map="log10")
        x = np.array([[10.0], [100.0]])
        original = x.copy()
        t.process(x, forward=True)
        np.testing.assert_array_equal(x, original)


class TestLabelTransformClipping:
    def test_set_clipping_bounds(self):
        t = LabelTransform()
        t.set_clipping_bounds({"Min": -5.0, "Max": 5.0})
        assert t.params.y_clip == {"Min": -5.0, "Max": 5.0}

    def test_clipping_applied_on_backward(self):
        t = LabelTransform()
        t.set_clipping_bounds({"Min": 0.0, "Max": 10.0})
        x = np.array([[-5.0], [5.0], [15.0]])
        out = t.process(x, forward=False)
        np.testing.assert_array_equal(out, [[0.0], [5.0], [10.0]])

    def test_clipping_not_applied_on_forward(self):
        t = LabelTransform()
        t.set_clipping_bounds({"Min": 0.0, "Max": 10.0})
        x = np.array([[-5.0], [5.0], [15.0]])
        out = t.process(x, forward=True)
        np.testing.assert_array_equal(out, x)
