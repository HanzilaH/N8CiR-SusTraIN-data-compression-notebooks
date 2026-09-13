#!/usr/bin/env python3
"""BOA input-size sweep (10 MB -> 1 GB) with per-epoch energy tracking. Output: one CSV.

    python boa_energy_sweep.py                                    # default sweep, 3 epochs
    python boa_energy_sweep.py --sizes 10 50 200 --epochs 15 --codec-at 3 5 10 15

Rows: one `training_epoch` per epoch per size, a `compression`/`decompression` pair per
size (or per --codec-at epoch), plus one `idle` baseline. Energy/CO2e come from CodeCarbon;
every row carries cumulative training energy and its own hardware fingerprint, so a
carbon-vs-ratio (Pareto) plot can be built from the CSV alone.

Needs: torch, codecarbon>=3.2, pandas, numpy (mamba backbones also need mambapy / mamba_ssm).
The BOA repo is used from next to this script, from --repo, or cloned if neither exists; the
CMS test data comes from the repo, or is downloaded when the repo was supplied without it.
"""
import argparse, gc, hashlib, os, subprocess, sys, time, types, urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_URL = "https://github.com/AkkiG2401/boa-constrictor.git"
CMS_IN_REPO = "experiments/cms_experiment/CMS_DATA_float32.bin"
DATA_URL = f"https://raw.githubusercontent.com/AkkiG2401/boa-constrictor/main/{CMS_IN_REPO}"

p = argparse.ArgumentParser()
p.add_argument("--sizes", type=float, nargs="+", default=[10, 25, 50, 100, 250, 500, 1000],
               help="input sizes in MB")
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--codec-at", type=int, nargs="+", default=[],
               help="also compress/decompress after these epoch counts (the final epoch is always done)")
p.add_argument("--repeats", type=int, default=1, help="compress/decompress repeats per codec point")
p.add_argument("--repo", type=Path, default=None,
               help="path to the BOA repo (default: found next to the script, else cloned)")
p.add_argument("--repo-url", default=REPO_URL, help="clone from here when no repo is found")
p.add_argument("--out", type=Path, default=None, help="output CSV (default: results/boa_energy_sweep_<gpu>.csv)")
p.add_argument("--data", type=Path, default=None, help="source .bin (downloaded if missing)")
p.add_argument("--label", default=None, help="run label (default: GPU name)")
p.add_argument("--country", default="GBR", help="ISO3 code for the grid carbon intensity")
p.add_argument("--backbone", default="mambav1")
p.add_argument("--d-model", type=int, default=64)
p.add_argument("--num-layers", type=int, default=2)
p.add_argument("--seq-len", type=int, default=10000)
p.add_argument("--batch-size", type=int, default=5)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
p.add_argument("--chunk-bytes", type=int, default=2048)
p.add_argument("--gpu-streams", type=int, default=5000, help="chunks coded in parallel; lower if OOM")
p.add_argument("--val-batches", type=int, default=20, help="batches used for the per-epoch val bpp")
p.add_argument("--idle", type=int, default=30, help="seconds of idle baseline (0 = skip)")
p.add_argument("--keep-inputs", action="store_true", help="keep the generated input files")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

VOCAB, SPLIT = 256, 0.8
os.environ["BOA_GPU_STREAMS"] = str(args.gpu_streams)

# --- repo + paths -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR, RESULTS_DIR, ART_DIR = ROOT / "data", ROOT / "results", ROOT / "artifacts"
for d in (DATA_DIR, RESULTS_DIR, ART_DIR):
    d.mkdir(parents=True, exist_ok=True)


def get_repo():
    """--repo, else a copy sitting next to this script, else clone it (~170 MB, incl. the CMS data)."""
    if args.repo:
        return args.repo
    here = Path(__file__).resolve().parent
    for folder in [here, *here.parents]:
        for c in (folder / "boa-constrictor-main", folder / "boa-constrictor", folder):
            if (c / "boa.py").exists() and (c / "model.py").exists():
                return c
    target = ROOT / "boa-constrictor"
    print(f"BOA repo not found; cloning {args.repo_url} -> {target}", flush=True)
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", args.repo_url, str(target)])
    except (OSError, subprocess.CalledProcessError) as err:
        sys.exit(f"Clone failed ({err}). Install git, or pass --repo /path/to/boa-constrictor")
    return target


REPO = get_repo()
sys.path.insert(0, str(REPO))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
LABEL = args.label or "".join(c if c.isalnum() else "-" for c in GPU_NAME.lower()).strip("-")
CSV = args.out or RESULTS_DIR / f"boa_energy_sweep_{LABEL}.csv"
SOURCE = args.data or (REPO / CMS_IN_REPO if (REPO / CMS_IN_REPO).exists()
                       else DATA_DIR / "CMS_DATA_float32.bin")

from codecarbon import OfflineEmissionsTracker
from model import BoaConstrictor, ByteDataloader
from boa import BOA

# --- codec backend ----------------------------------------------------------------------------
CODEC_DEVICE = "cpu"
if DEVICE == "cuda":
    try:
        import gpu_range_coder            # compiles a CUDA extension on first import
        CODEC_DEVICE = "cuda"
    except Exception as err:
        print(f"[codec] GPU range coder unavailable ({err}); using the CPU coder")
        sys.modules["gpu_range_coder"] = types.ModuleType("gpu_range_coder")

# mamba_ssm's fused kernels cannot run under the CPU coder, so the two choices go together.
FACTORY_DEVICE = DEVICE if CODEC_DEVICE == "cuda" else "cpu"
AMP = args.precision == "bf16" and DEVICE == "cuda"

# --- measurement ------------------------------------------------------------------------------
HW_KEYS = ["cpu_model", "cpu_count", "gpu_model", "gpu_count", "ram_total_size", "os",
           "python_version", "codecarbon_version", "tracking_mode", "country_name",
           "country_iso_code", "region", "cloud_provider", "cloud_region", "on_cloud"]
HARDWARE = dict.fromkeys(HW_KEYS + ["torch_version", "cuda_version", "device", "gpu_name"])
CUM = {"energy": 0.0, "emissions": 0.0}   # cumulative training cost for the current size


@contextmanager
def track():
    """Measure the block with CodeCarbon; the yielded dict is filled in on exit."""
    m = {}
    tracker = OfflineEmissionsTracker(country_iso_code=args.country, measure_power_secs=1,
                                      tracking_mode="machine", log_level="error",
                                      save_to_file=False, save_to_api=False)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    tracker.start()
    t0 = time.perf_counter()
    try:
        yield m
    finally:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        runtime = time.perf_counter() - t0
        kg = tracker.stop()
        d = tracker.final_emissions_data
        m.update(runtime_s=runtime, energy_kWh=d.energy_consumed, cpu_kWh=d.cpu_energy,
                 gpu_kWh=d.gpu_energy, ram_kWh=d.ram_energy, cpu_power_W=d.cpu_power,
                 gpu_power_W=d.gpu_power, ram_power_W=d.ram_power,
                 emissions_kgCO2eq=float(kg) if kg else d.emissions)
        HARDWARE.update({k: getattr(d, k, None) for k in HW_KEYS})
        HARDWARE.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                        device=DEVICE, gpu_name=GPU_NAME)


def row(stage, m=None, size_MB=0.0, input_bytes=0, **kw):
    """One CSV row. The schema is fixed here so rows can be appended one at a time."""
    m = m or {}
    gb = input_bytes / 1e9
    r = dict(
        run_label=LABEL, timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        stage=stage, status=kw.pop("status", "ok"), size_MB=size_MB, input_bytes=input_bytes,
        source_repeats=kw.pop("source_repeats", None), epoch=kw.pop("epoch", None),
        repeat=kw.pop("repeat", None),
        runtime_s=m.get("runtime_s"), energy_kWh=m.get("energy_kWh"), cpu_kWh=m.get("cpu_kWh"),
        gpu_kWh=m.get("gpu_kWh"), ram_kWh=m.get("ram_kWh"), cpu_power_W=m.get("cpu_power_W"),
        gpu_power_W=m.get("gpu_power_W"), ram_power_W=m.get("ram_power_W"),
        emissions_kgCO2eq=m.get("emissions_kgCO2eq"),
        energy_kWh_per_GB=m["energy_kWh"] / gb if m and gb else None,
        emissions_kgCO2eq_per_GB=m["emissions_kgCO2eq"] / gb if m and gb else None,
        MB_per_s=input_bytes / 1e6 / m["runtime_s"] if m and input_bytes else None,
        train_bpp=kw.pop("train_bpp", None), val_bpp=kw.pop("val_bpp", None),
        val_ratio_est=kw.pop("val_ratio_est", None),
        cum_train_energy_kWh=CUM["energy"], cum_train_emissions_kgCO2eq=CUM["emissions"],
        compressed_bytes=kw.pop("compressed_bytes", None),
        compression_ratio=kw.pop("compression_ratio", None),
        bits_per_byte=kw.pop("bits_per_byte", None), roundtrip_ok=kw.pop("roundtrip_ok", None),
        total_epochs=args.epochs, seq_len=args.seq_len, batch_size=args.batch_size, lr=args.lr,
        chunk_bytes=args.chunk_bytes, gpu_streams=args.gpu_streams, backbone=args.backbone,
        d_model=args.d_model, num_layers=args.num_layers, precision=args.precision,
        codec_device=CODEC_DEVICE, **HARDWARE)
    assert not kw, f"unused: {list(kw)}"
    return r


def append(rows):
    df = pd.DataFrame(rows)
    if CSV.exists() and list(pd.read_csv(CSV, nrows=0).columns) != list(df.columns):
        CSV.rename(CSV.with_name(f"{CSV.stem}_old_{int(CSV.stat().st_mtime)}.csv"))
    df.to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
    for r in rows:
        print(f"    {r['stage']:14s} ep{r['epoch'] or 0:<3} {r['runtime_s'] or 0:7.1f} s  "
              f"{r['energy_kWh'] or 0:.3e} kWh  {r['emissions_kgCO2eq'] or 0:.3e} kgCO2e"
              + (f"  ratio {r['compression_ratio']:.3f}x" if r["compression_ratio"] else ""),
              flush=True)


# --- data -------------------------------------------------------------------------------------
if not SOURCE.exists():
    print(f"Downloading {DATA_URL}\n  -> {SOURCE}")
    urllib.request.urlretrieve(DATA_URL, SOURCE)
FULL = SOURCE.stat().st_size
BLOCK = args.seq_len * args.batch_size


def build_input(size_MB):
    """Input file of exactly size_MB, by repeating the ~50 MB source when it is too small.

    Repetition leaves runtime and energy unchanged (chunks are coded independently) but makes
    the compression ratio optimistic, so every row records source_repeats.
    """
    target = int(size_MB * 1e6) // BLOCK * BLOCK
    if target < 2 * BLOCK:
        raise ValueError(f"{size_MB:g} MB is below the {2 * BLOCK / 1e6:.2f} MB minimum "
                         f"(one train + one val batch at this seq_len/batch_size)")
    path = DATA_DIR / f"sweep_input_{size_MB:g}MB.bin"
    if not (path.exists() and path.stat().st_size == target):
        with path.open("wb") as out:
            left = target
            while left > 0:
                with SOURCE.open("rb") as src:
                    while left > 0:
                        blk = src.read(min(8 << 20, left))
                        if not blk:
                            break
                        out.write(blk)
                        left -= len(blk)
    return path, target, target / FULL


def make_splits(path):
    """Train/val byte buffers cut to whole batches; memmapped so 1 GB stays cheap."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    n = len(mm) // BLOCK * BLOCK
    n_train = min(max(BLOCK, int(n * SPLIT) // BLOCK * BLOCK), n - BLOCK)
    n_val = max(BLOCK, (n - n_train) // 2 // BLOCK * BLOCK)
    return mm[:n_train].tobytes(), mm[n_train:n_train + n_val].tobytes()


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.digest()


# --- train / codec ----------------------------------------------------------------------------
def train_epoch(model, loader, opt, crit):
    model.train()
    total = toks = 0
    for batch in loader:
        x, y = batch[:, :-1], batch[:, 1:]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=AMP):
            loss = crit(model(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        opt.step()
        total += loss.item() * y.numel()
        toks += y.numel()
    return total / max(toks, 1) / np.log(2)


@torch.inference_mode()
def val_bpp(model, loader, crit):
    model.eval()
    total = toks = 0
    for i, batch in enumerate(loader):
        if i >= args.val_batches:
            break
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=AMP):
            loss = crit(model(x).reshape(-1, VOCAB), y.reshape(-1))
        total += loss.item() * y.numel()
        toks += y.numel()
    loader.pos = 0
    return total / max(toks, 1) / np.log(2)


def codec(model, path, n_bytes, src_hash, epoch, repeat, source_repeats):
    boa_path = ART_DIR / f"sweep_{n_bytes}.boa"
    boa_path.unlink(missing_ok=True)
    model.eval()
    handle = BOA(CODEC_DEVICE, str(boa_path), model)

    with track() as m_c:
        handle.compress(data_path=str(path), seq_size=args.chunk_bytes, progress=False)
    comp_bytes = boa_path.stat().st_size
    with track() as m_d:
        restored = handle.decompress(progress=False)

    ok = hashlib.sha256(restored).digest() == src_hash
    del restored, handle
    boa_path.unlink(missing_ok=True)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return [row(stage, m, size_MB=n_bytes / 1e6, input_bytes=n_bytes, epoch=epoch, repeat=repeat,
                source_repeats=source_repeats, compressed_bytes=comp_bytes,
                compression_ratio=n_bytes / comp_bytes, bits_per_byte=8 * comp_bytes / n_bytes,
                roundtrip_ok=ok, status="ok" if ok else "roundtrip_failed")
            for stage, m in (("compression", m_c), ("decompression", m_d))]


# --- warm-up, idle baseline, plan -------------------------------------------------------------
warm = BoaConstrictor(d_model=args.d_model, num_layers=args.num_layers, vocab_size=VOCAB,
                      device=FACTORY_DEVICE, backbone=args.backbone).to(DEVICE)
wx = torch.randint(0, 256, (2, 1024), device=DEVICE)
for _ in range(3):
    torch.nn.functional.cross_entropy(warm(wx[:, :-1]).reshape(-1, VOCAB),
                                      wx[:, 1:].reshape(-1)).backward()
del warm, wx
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

CODEC_EPOCHS = sorted({e for e in args.codec_at if 1 <= e <= args.epochs} | {args.epochs})
T4 = {"train": 0.98, "compress": 1.63, "decompress": 1.53}   # MB/s, for the estimate only
est = sum(s * args.epochs / T4["train"] +
          args.repeats * len(CODEC_EPOCHS) * (s / T4["compress"] + s / T4["decompress"])
          for s in args.sizes)
print(f"torch {torch.__version__} on {DEVICE} ({GPU_NAME}), codec={CODEC_DEVICE}, "
      f"precision={args.precision}")
print(f"sizes {[f'{s:g}' for s in args.sizes]} MB x {args.epochs} epochs, "
      f"codec at epochs {CODEC_EPOCHS} x{args.repeats}")
print(f"peak disk for inputs ~{max(args.sizes)/1000:.2f} GB | estimate ~{est/3600:.1f} h on a T4")
print(f"CSV: {CSV}\n")

if args.idle:
    with track() as m_idle:
        time.sleep(args.idle)
    append([row("idle", m_idle)])

# --- sweep ------------------------------------------------------------------------------------
t_start = time.perf_counter()
for i, size_MB in enumerate(args.sizes, 1):
    print(f"=== [{i}/{len(args.sizes)}] {size_MB:g} MB "
          f"({(time.perf_counter() - t_start) / 60:.1f} min elapsed) ===", flush=True)
    CUM["energy"] = CUM["emissions"] = 0.0
    model = opt = None
    try:
        path, n_bytes, src_repeats = build_input(size_MB)
        src_hash = file_sha256(path)
        train_bytes, val_bytes = make_splits(path)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = BoaConstrictor(d_model=args.d_model, num_layers=args.num_layers, vocab_size=VOCAB,
                               device=FACTORY_DEVICE, backbone=args.backbone).to(DEVICE)
        train_loader = ByteDataloader(train_bytes, seq_len=args.seq_len,
                                      batch_size=args.batch_size, device=DEVICE)
        val_loader = ByteDataloader(val_bytes, seq_len=args.seq_len,
                                    batch_size=args.batch_size, device=DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        crit = torch.nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            with track() as m:
                bpp = train_epoch(model, train_loader, opt, crit)
            CUM["energy"] += m["energy_kWh"]
            CUM["emissions"] += m["emissions_kgCO2eq"]
            vbpp = val_bpp(model, val_loader, crit)
            append([row("training_epoch", m, size_MB=n_bytes / 1e6, input_bytes=n_bytes,
                        epoch=epoch, source_repeats=src_repeats, train_bpp=bpp, val_bpp=vbpp,
                        val_ratio_est=8 / vbpp if vbpp else None)])

            if epoch in CODEC_EPOCHS:
                for r in range(1, args.repeats + 1):
                    append(codec(model, path, n_bytes, src_hash, epoch, r, src_repeats))

    except Exception as err:
        print(f"    FAILED: {type(err).__name__}: {err}", flush=True)
        append([row("failed", size_MB=size_MB, status=f"{type(err).__name__}: {err}")])
    finally:
        model = opt = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if not args.keep_inputs:
            (DATA_DIR / f"sweep_input_{size_MB:g}MB.bin").unlink(missing_ok=True)

print(f"\nDone in {(time.perf_counter() - t_start) / 60:.1f} min -> {CSV}")
