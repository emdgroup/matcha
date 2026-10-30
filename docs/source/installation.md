# Installation

MATCHA is published to PyPI as `emd-matcha` and requires **Python 3.12** or
newer (`>=3.12,<3.14`).

## Base install

The installation is available either for GPUs or CPUs, depending on
what you have available on your machine. Keep in mind, the default
GPU installation requires CUDA 12.9.

```bash
# CPU-only (portable, works on any machine)
pip install "emd-matcha[cpu]"

# CUDA 12.9 GPU installation
pip install "emd-matcha[gpu]"
```

You can also install via [`uv`](https://docs.astral.sh/uv/):

```bash
uv add "emd-matcha[cpu]"
# or
uv add "emd-matcha[gpu]"
```

## Optional extras

- `cli` — full MLflow client plus fastparquet, required for the `matcha` CLI
  and foundation-model training workflows.

Extras can be combined with the CPU or GPU extras, for example:

```bash
pip install "emd-matcha[cpu,cli]"
```

## Custom GPU install

The install without either CPU or GPU is useful if you need to install
the GPU dependencies with different CUDA versions. In that case, the best
way is to install `emd-matcha` without any extra, then add to the environment
the other packages with the desired CUDA version.

```bash
pip install emd-matcha

# add pytorch with a different CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu132

# install the other GPU dependencies, for example
pip install torch-geometric
pip install torch-geometric
pip install lightning
pip install optuna
pip install optuna-integration
pip install einops
pip install transformers
pip install chemprop
```
