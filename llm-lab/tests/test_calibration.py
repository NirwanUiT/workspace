import numpy as np
import pytest

from llmlab import calibration as cal


def two_class(conf_correct):
    """Build 2-way probs where the top label has confidence c and is right iff ok."""
    probs, labels = [], []
    for c, ok in conf_correct:
        probs.append([c, 1 - c])
        labels.append(0 if ok else 1)
    return np.array(probs), np.array(labels)


def test_hand_computed_ece():
    # 5 equal-width bins of 0.2. Bin (0.6, 0.8]: conf 0.7, 0.7 -> acc 0.5, gap 0.2.
    # Bin (0.8, 1.0]: conf 0.9, 1.0 -> mean 0.95, acc 1.0, gap 0.05.
    probs, labels = two_class([(0.7, True), (0.7, False), (0.9, True), (1.0, True)])
    assert cal.ece(probs, labels, n_bins=5) == pytest.approx(0.5 * 0.2 + 0.5 * 0.05)


def test_bin_edges_are_right_closed():
    probs, labels = two_class([(0.6, True), (1.0, True)])
    b = cal.reliability_bins(probs, labels, n_bins=5)
    assert b["count"].tolist() == [0, 0, 1, 0, 1]  # 0.6 -> (0.4, 0.6], 1.0 -> last bin


def test_brier_bounds():
    onehot = np.eye(4)
    labels = np.arange(4)
    assert cal.brier(onehot, labels) == 0
    assert cal.brier(np.full((4, 4), 0.25), labels) == pytest.approx(0.75)
    assert cal.brier(np.roll(onehot, 1, axis=1), labels) == pytest.approx(2.0)


def test_calibrated_sampler_has_small_ece():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2, size=(50_000, 4))
    probs = cal.softmax_t(logits, 1.0)
    labels = np.array([rng.choice(4, p=p) for p in probs])
    assert cal.ece(probs, labels, scheme="width") < 0.01
    assert cal.ece(probs, labels, scheme="mass") < 0.01


def test_overconfidence_is_detected_and_fixed_by_temperature():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2, size=(20_000, 4))
    labels = np.array([rng.choice(4, p=p) for p in cal.softmax_t(logits, 2.0)])
    # the model reports logits/1 but the truth is logits/2: overconfident
    assert cal.ece(cal.softmax_t(logits, 1.0), labels) > 0.05
    t = cal.fit_temperature(logits, labels)
    assert t == pytest.approx(2.0, rel=0.08)
    assert cal.ece(cal.softmax_t(logits, t), labels) < 0.015


def test_mass_bins_are_balanced():
    rng = np.random.default_rng(2)
    probs = cal.softmax_t(rng.normal(0, 1.5, size=(3000, 4)), 1.0)
    labels = rng.integers(0, 4, 3000)
    counts = cal.reliability_bins(probs, labels, n_bins=10, scheme="mass")["count"]
    assert counts.min() >= 290 and counts.max() <= 310


def test_temperature_does_not_change_accuracy():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(500, 4))
    labels = rng.integers(0, 4, 500)
    assert cal.accuracy(cal.softmax_t(logits, 0.3), labels) == cal.accuracy(cal.softmax_t(logits, 3.0), labels)
