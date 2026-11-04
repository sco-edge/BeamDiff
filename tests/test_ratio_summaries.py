import json

import pytest

from scripts.summarize_beamdiff_ratio_sweep import sample_sd, summarize


def write_run(root, ratio, seed, accuracy):
    path = root / f'ratio{ratio:g}_seed{seed}' / 'random_window_metrics.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'seed': seed, 'accuracy': accuracy,
                               'macro_f1': accuracy, 'macro_recall': accuracy,
                               'synthetic_train_samples': 10 if ratio else 0}))


def test_sample_sd_uses_sample_not_population_definition():
    assert sample_sd([1, 3]) == pytest.approx(2 ** 0.5)
    assert sample_sd([1]) == 0


def test_summary_uses_paired_percentage_point_gains(tmp_path):
    for seed, base, augmented in [(42, .8, .9), (111, .9, .95)]:
        write_run(tmp_path, 0, seed, base)
        write_run(tmp_path, 1, seed, augmented)
    rows, _ = summarize(tmp_path, [0, 1], [42, 111])
    assert rows[1]['accuracy_mean_percent'] == pytest.approx(92.5)
    assert rows[1]['accuracy_gain_mean_pp'] == pytest.approx(7.5)
    assert rows[1]['accuracy_seeds_improved'] == 2
    assert rows[0]['accuracy_gain_mean_pp'] == 0


def test_summarizing_without_baseline_is_rejected(tmp_path):
    write_run(tmp_path, 1, 42, .9)
    with pytest.raises(ValueError, match='include 0'):
        summarize(tmp_path, [1], [42])
