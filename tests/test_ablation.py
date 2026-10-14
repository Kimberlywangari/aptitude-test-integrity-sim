import pytest

from eval.common import load_reference_and_test
from eval.run_ablation import VARIANTS, combine, component_flags
from src.detectors import baseline_flags, fit_baseline, fit_improved, improved_flags


@pytest.fixture(scope="module")
def setup():
    reference, test = load_reference_and_test()
    baseline_thresholds = fit_baseline(reference)
    improved_thresholds = fit_improved(reference)
    components = component_flags(test, baseline_thresholds, improved_thresholds)
    return test, baseline_thresholds, improved_thresholds, components


def test_avg_time_plus_focus_loss_variant_is_exactly_the_baseline(setup):
    test, baseline_thresholds, _, components = setup
    flags = combine(components, avg_time=True, z_score=False, focus_loss=True)
    assert flags.equals(baseline_flags(test, baseline_thresholds))


def test_z_score_plus_focus_loss_variant_is_exactly_the_improved_detector(setup):
    test, _, improved_thresholds, components = setup
    flags = combine(components, avg_time=False, z_score=True, focus_loss=True)
    assert flags.equals(improved_flags(test, improved_thresholds))


def test_single_rule_variants_equal_that_rule_alone(setup):
    *_, components = setup
    assert combine(components, True, False, False).equals(components["avg_time"])
    assert combine(components, False, True, False).equals(components["z_score"])
    assert combine(components, False, False, True).equals(components["focus_loss"])


def test_switching_on_another_rule_never_removes_a_flag(setup):
    *_, components = setup
    smaller = combine(components, False, True, False)
    larger = combine(components, False, True, True)
    assert (smaller & ~larger).sum() == 0


def test_every_variant_has_at_least_one_rule_switched_on():
    assert len(VARIANTS) == 7
    assert all(avg or z or focus for _, avg, z, focus in VARIANTS)
    assert len({(avg, z, focus) for _, avg, z, focus in VARIANTS}) == 7
