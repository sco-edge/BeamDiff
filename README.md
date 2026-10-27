# BeamDiff

**BeamDiff: Sensing-Aware Diffusion for Wi-Fi Beamforming Feedback Augmentation**

BeamDiff generates synthetic BFA windows for Wi-Fi human activity recognition. It preserves a measured first-frame anchor and generates activity-conditioned transitions, with supervision from a frozen BeamSense classifier.

## Setup

```bash
pip install -r requirements.txt
```

## Code

- `scripts/`: preprocessing, training, generation, evaluation, and visualization
- `tfdiff/`: BFI parsing and BFA window construction
- `tests/`: basic validation tests
- `results/`: recorded BeamDiff results

Run scripts with `--help` for available arguments. See [Usage and implementation notes](docs/USAGE.md) for detailed instructions and code–manuscript differences.

## Dataset

[CSI-BFI-HAR](https://doi.org/10.21227/nbp8-7r47) · [Dataset files](https://huggingface.co/datasets/foysalhaque/CSI-BFI-HAR-Dataset)

Datasets and pretrained checkpoints are not included. See [Attribution](ATTRIBUTION.md) for prior-work references.
