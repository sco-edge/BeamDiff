# Environment and reproducibility

## Capture the actual experimental environment

Activate the environment used for the experiment, then run:

```bash
python scripts/report_environment.py --gpu
python -m pip freeze
```

The first command reports Python, OS family, architecture, installed dependency
versions, and optional NVIDIA GPU name, VRAM capacity (MiB), and driver version.
It does not initialize PyTorch or TensorFlow. Missing packages are reported as
null; unavailable NVIDIA tools are reported without failing the command.

These commands describe the active environment, not necessarily the one used
for previous runs. Review output before sharing it: full `pip freeze` output
may contain local paths or private package URLs. No environment report is
automatically uploaded.

## Version specification status

`requirements.txt` lists direct dependencies but is not a verified lockfile for
the original experiments. Do not substitute locally installed or latest package
versions for missing experimental records. Exact Python, framework, CUDA runtime,
and GPU details must be recovered from the actual training environment and run
records before claiming exact reproducibility.

The CUDA version displayed in the standard `nvidia-smi` header indicates driver
compatibility, not necessarily the runtime used by the installed framework.

## CPU checks versus training

The unit suite needs only NumPy and pytest:

```bash
python -m pip install numpy pytest
python -m pytest -q
```

These tests cover preprocessing, split integrity, augmentation sampling,
reconstruction, and statistics. They do not validate a full GPU training run,
teacher weight conversion, or reproduction of the published accuracy values.

## Timing measurements

Measure and report generator training, synthetic generation, downstream
classifier training, and test inference separately. Include dataset/sample count,
batch size, epochs, diffusion steps, device, and synchronization/warm-up policy.
Do not infer historical runtime from file timestamps or from a different machine.
