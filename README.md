# BOA carbon benchmarks

Measures the energy use and carbon footprint of compressing CMS detector data with
[BOA Constrictor](https://github.com/boa-collaboration/boa-constrictor), a neural compressor,
and compares it with classical codecs (ZSTD, LZMA). Energy and emissions are measured with
[CodeCarbon](https://codecarbon.io).

## Contents

| Path | What it is |
|---|---|
| `boa_energy_sweep.py` | BOA sweep: trains the model and compresses/decompresses at each input size (GPU) |
| `bench_classical_sweep.py` | Classical sweep: ZSTD and LZMA at the same input sizes (CPU) |
| `results/` | CSVs from the runs: `boa_energy_sweep_<gpu>.csv`, `classical_sweep_<machine>.csv` |
| `Plots.ipynb` | Figures built from the CSVs in `results/` (saved to `figures/`) |
| `CodeCarbon_Learning_Guide.ipynb` | Introduction to measuring energy with CodeCarbon |

Both scripts use the same input: `CMS_DATA_float32.bin` (~50 MB), downloaded automatically.
The sweep sizes are 10, 25, 50, 100, 250, 500 and 1000 MB. Inputs larger than 50 MB repeat
the file, which makes compression ratios look better than they are; every row records
`source_repeats` so you can tell which ones.

## Setup

```bash
pip install -r requirements.txt
```

For the BOA sweep on a GPU, install the PyTorch build that matches your CUDA version
(see [pytorch.org](https://pytorch.org/get-started/locally/)). The default `mambav1` backbone
also needs `mambapy` or `mamba_ssm`.

## Running the classical sweep

Runs on CPU. It needs no GPU and no BOA repo.

```bash
python bench_classical_sweep.py
```

This downloads the CMS file to `data/` on its first run and writes `results/classical_sweep.csv`.
The full sweep takes about 45 minutes on a laptop (LZMA at 1 GB alone is ~20 minutes).

Useful options:

```bash
python bench_classical_sweep.py --sizes 10 50 --codecs zstd            # quick test
python bench_classical_sweep.py --label macbook-i5-8257u \
    --out results/classical_sweep_macbook-i5-8257u.csv                  # name the run
```

| Option | Default | Meaning |
|---|---|---|
| `--sizes` | `10 25 50 100 250 500 1000` | input sizes in MB |
| `--codecs` | `zstd lzma` | codecs to run |
| `--zstd-level` / `--lzma-preset` | `3` / `6` | compression levels |
| `--country` | `GBR` | ISO3 code for the grid carbon intensity |
| `--data`, `--work`, `--out` | `data/…`, `data/`, `results/classical_sweep.csv` | file locations |

On macOS, CodeCarbon cannot read CPU power, so energy is estimated from the CPU's TDP.

## Running the BOA sweep

Needs an NVIDIA GPU; it falls back to CPU, but that is very slow.

```bash
python boa_energy_sweep.py
```

On its first run it:
1. looks for the BOA repo next to this folder, and clones it into `boa-constrictor/` if it is not there;
2. compiles the GPU range coder (if that fails, it uses the CPU coder and prints a warning);
3. records a 30-second idle baseline, then runs the sweep.

The output is `results/boa_energy_sweep_<gpu-name>.csv`. Each size gets one row per training
epoch plus a compression and a decompression row. On an RTX 5090 the full sweep takes about
35 minutes; on a Tesla T4, several hours.

Useful options:

```bash
python boa_energy_sweep.py --sizes 10 25 --epochs 1 --idle 0             # quick test
python boa_energy_sweep.py --epochs 15 --codec-at 3 5 10                  # compress at several epochs
python boa_energy_sweep.py --repo /path/to/boa-constrictor                # use an existing clone
```

| Option | Default | Meaning |
|---|---|---|
| `--sizes` | `10 25 50 100 250 500 1000` | input sizes in MB |
| `--epochs` | `3` | training epochs per size |
| `--codec-at` | none | extra epochs at which to compress/decompress (the last epoch always is) |
| `--gpu-streams` | `5000` | chunks coded in parallel; lower this if you run out of GPU memory |
| `--precision` | `fp32` | `fp32` or `bf16` |
| `--idle` | `30` | seconds of idle baseline (`0` skips it) |
| `--label`, `--out` | GPU name | run name and output CSV |
| `--country` | `GBR` | ISO3 code for the grid carbon intensity |

If the output CSV already exists with different columns, the old file is renamed to
`…_old_<timestamp>.csv` rather than mixed with the new rows.

## Plotting

```bash
jupyter notebook Plots.ipynb
```

The notebook reads the CSVs in `results/`. Note that `boa_energy_sweep_v100x4.csv` came from
an earlier multi-GPU version of the BOA script and has two extra columns (`codec_gpus`,
`worker_max_s`).


## Generative AI Disclosure

See [AI_DISCLOSURE.md](AI_DISCLOSURE.md) for the authors' declaration regarding the use of generative AI.
