# Environment and reproducibility

## Capture the actual experimental environment

Activate the environment used for the experiment, then run:

```bash
python scripts/report_environment.py --gpu > environment.json
python -m pip freeze > requirements-freeze.txt
```

`environment.json` contains these fields:

| Field | Description |
| --- | --- |
| `python` | Active Python interpreter version. |
| `os` | Operating-system family reported by Python. |
| `machine` | Machine architecture reported by Python. |
| `packages` | Installed versions of the direct scientific, training, plotting, and test packages checked by the script. A missing package is `null`. |
| `note` | Reminder that the report describes only the current environment. |
| `gpu` | Present only with `--gpu`. Reports NVIDIA device name, total VRAM in MiB, and driver version, or an `available: false` reason if `nvidia-smi` cannot be queried. |

The reporting script does not initialize PyTorch or TensorFlow. It exits
successfully when an optional package is missing or NVIDIA tooling is
unavailable, so inspect the JSON values rather than treating command success as
proof that the environment is training-ready. Run without `--gpu` when NVIDIA
metadata is not needed.

These commands describe the active environment, not necessarily the one used
for previous runs. Review both files before sharing them: full `pip freeze`
output may contain local paths or private package URLs. No environment report
is automatically uploaded.

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
