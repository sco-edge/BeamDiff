import csv

import pytest

from scripts.plot_augmentation_ratio_comparison import load


def test_plot_input_is_sorted_by_ratio(tmp_path):
    path = tmp_path / 'summary.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'ratio_percent', 'accuracy_mean_percent', 'accuracy_sd_percent'])
        writer.writeheader()
        writer.writerows([
            {'ratio_percent': 100, 'accuracy_mean_percent': 96, 'accuracy_sd_percent': 1},
            {'ratio_percent': 0, 'accuracy_mean_percent': 92, 'accuracy_sd_percent': 2},
        ])
    assert load(path, 'accuracy') == ([0., 100.], [92., 96.], [2., 1.])


def test_empty_plot_input_is_rejected(tmp_path):
    path = tmp_path / 'empty.csv'
    path.write_text('ratio_percent,accuracy_mean_percent,accuracy_sd_percent\n')
    with pytest.raises(ValueError, match='Empty summary'):
        load(path, 'accuracy')
