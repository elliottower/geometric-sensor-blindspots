"""Tests for the validation hierarchy, especially Level 4 equivariance.

The critical test: equivariance_accuracy must distinguish blind metrics
(geodesic, which is translation-invariant) from sensitive metrics
(T_loc, which breaks translation invariance). If it can't tell them
apart, Level 4 is meaningless and shouldn't be in the confirmatory claim.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nv_diamond import DecoherenceParams, generate_dataset
from designed_metrics import designed_l1_loc, designed_l1_scale
from methods import method_08_grassmannian_geodesic, method_21_vn_entropy
from validation_hierarchy import equivariance_accuracy, equivariance_accuracy_gain


@pytest.fixture
def moderate_offset_data():
    params = DecoherenceParams(sigma_device=0.03)
    return generate_dataset(n_per_class=200, params=params, seed=77, n_devices=5)


@pytest.fixture
def moderate_gain_data():
    params = DecoherenceParams(sigma_gain=0.08)
    return generate_dataset(n_per_class=200, params=params, seed=77, n_devices=5)


def test_equivariance_geodesic_is_blind_to_offset(moderate_offset_data):
    X, y, dev = moderate_offset_data
    acc = equivariance_accuracy(
        method_08_grassmannian_geodesic, X, y, dev,
        expect_sensitive=False, n_offsets=50,
    )
    assert acc > 0.7, (
        f"Geodesic should be blind to additive offset (equivariance acc > 0.7): "
        f"got {acc:.2f}"
    )


def test_equivariance_t_loc_is_sensitive_to_offset(moderate_offset_data):
    X, y, dev = moderate_offset_data
    acc = equivariance_accuracy(
        designed_l1_loc, X, y, dev,
        expect_sensitive=True, n_offsets=50,
    )
    assert acc > 0.7, (
        f"T_loc should be sensitive to additive offset (equivariance acc > 0.7): "
        f"got {acc:.2f}"
    )


def test_equivariance_distinguishes_blind_from_sensitive(moderate_offset_data):
    X, y, dev = moderate_offset_data
    acc_blind = equivariance_accuracy(
        method_08_grassmannian_geodesic, X, y, dev,
        expect_sensitive=False, n_offsets=50,
    )
    acc_sensitive = equivariance_accuracy(
        designed_l1_loc, X, y, dev,
        expect_sensitive=True, n_offsets=50,
    )
    assert acc_blind > 0.7 and acc_sensitive > 0.7, (
        f"Level 4 must distinguish blind from sensitive: "
        f"geodesic(blind)={acc_blind:.2f}, T_loc(sensitive)={acc_sensitive:.2f}"
    )


def test_equivariance_gain_geodesic_blind(moderate_gain_data):
    X, y, dev = moderate_gain_data
    acc = equivariance_accuracy_gain(
        method_08_grassmannian_geodesic, X, y, dev,
        expect_sensitive=False, n_trials=50,
    )
    assert acc > 0.7, (
        f"Geodesic should be blind to gain (equivariance acc > 0.7): got {acc:.2f}"
    )


def test_equivariance_gain_t_scale_sensitive(moderate_gain_data):
    X, y, dev = moderate_gain_data
    acc = equivariance_accuracy_gain(
        designed_l1_scale, X, y, dev,
        expect_sensitive=True, n_trials=50,
    )
    assert acc > 0.7, (
        f"T_scale should be sensitive to gain (equivariance acc > 0.7): got {acc:.2f}"
    )
