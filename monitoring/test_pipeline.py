from .pipeline import DriftMonitor, Example, evaluate, psi, train_threshold


def test_training_and_evaluation_are_deterministic():
    rows = [Example(v, int(v >= 0.5)) for v in (0.1, 0.2, 0.7, 0.9)]
    model = train_threshold(rows)
    assert model.threshold == 0.7
    assert evaluate(model, rows).f1 == 1.0


def test_psi_is_zero_for_identical_windows_and_positive_for_shift():
    reference = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
    assert psi(reference, reference) == 0.0
    assert psi(reference, [0.8, 0.9, 1.0, 1.0, 1.0, 1.0]) > 0.2


def test_two_bad_windows_trigger_retraining_and_reset_reference():
    baseline = [Example(v, int(v >= 0.5)) for v in (0.1, 0.2, 0.7, 0.9)]
    model = train_threshold(baseline)
    monitor = DriftMonitor([e.signal_density for e in baseline], consecutive_windows=2)
    shifted = [0.1, 0.2, 0.3, 0.4]
    labels = [1, 1, 1, 1]
    assert not monitor.observe(shifted, labels, model)["retraining_recommended"]
    assert monitor.observe(shifted, labels, model)["retraining_recommended"]
    retrained = monitor.retrain(shifted, labels, version=2)
    assert retrained.version == 2
    assert monitor.bad_windows == 0
