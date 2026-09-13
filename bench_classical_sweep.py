#!/usr/bin/env python3
"""Input-size sweep for the classical codecs (LZMA, ZSTD) — CPU + RAM only. Output: one CSV.

Mirrors boa_energy_sweep.py: same input sizes, same repeat-the-CMS-file inputs, same
CodeCarbon setup, comparable columns — so the two CSVs can be plotted together.

    python bench_classical_sweep.py                  # downloads the CMS file on first run
    python bench_classical_sweep.py --sizes 10 50 --codecs zstd

Everything is streamed in 8 MB blocks, so a 1 GB input never needs 1 GB of RAM.
Fast measurements are repeated inside one tracker until the window is long enough for
CodeCarbon to sample it, then divided back down (`repeats_in_window`).
"""
import argparse, csv, hashlib, lzma, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

import zstandard as zstd
from codecarbon import OfflineEmissionsTracker

p = argparse.ArgumentParser()
p.add_argument("--sizes", type=float, nargs="+", default=[10, 25, 50, 100, 250, 500, 1000])
p.add_argument("--codecs", nargs="+", default=["zstd", "lzma"], choices=["zstd", "lzma"])
HERE = Path(__file__).resolve().parent
DATA_URL = ("https://raw.githubusercontent.com/boa-collaboration/boa-constrictor/main/"
            "experiments/cms_experiment/CMS_DATA_float32.bin")
p.add_argument("--data", type=Path, default=HERE / "data" / "CMS_DATA_float32.bin",
               help="source .bin (the ~50 MB CMS file; downloaded if missing)")
p.add_argument("--work", type=Path, default=HERE / "data", help="scratch dir for built inputs")
p.add_argument("--out", type=Path, default=HERE / "results" / "classical_sweep.csv", help="output CSV")
p.add_argument("--lzma-preset", type=int, default=6)
p.add_argument("--zstd-level", type=int, default=3)
p.add_argument("--country", default="GBR")
p.add_argument("--label", default=None)
p.add_argument("--min-window-s", type=float, default=3.0)
args = p.parse_args()

BLOCK = 8 << 20
LABEL = args.label or "cpu"
args.work.mkdir(parents=True, exist_ok=True)
args.out.parent.mkdir(parents=True, exist_ok=True)
if not args.data.exists():
    print(f"Downloading {DATA_URL}\n  -> {args.data}", flush=True)
    args.data.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, args.data)
FULL = args.data.stat().st_size

COLUMNS = ["run_label", "timestamp_utc", "codec", "level", "stage", "status", "size_MB",
           "input_bytes", "source_repeats", "repeats_in_window", "runtime_s", "energy_kWh",
           "cpu_kWh", "gpu_kWh", "ram_kWh", "cpu_power_W", "ram_power_W", "emissions_kgCO2eq",
           "energy_kWh_per_GB", "emissions_kgCO2eq_per_GB", "MB_per_s", "compressed_bytes",
           "compression_ratio", "bits_per_byte", "roundtrip_ok", "cpu_model", "cpu_count",
           "ram_total_size", "os", "python_version", "codecarbon_version", "country_iso_code",
           "energy_source"]


def build_input(size_MB):
    """Input file of exactly size_MB, repeating the source file — same as the BOA sweep."""
    target = int(size_MB * 1e6) // 50_000 * 50_000        # same byte counts as the BOA sweep
    path = args.work / f"sweep_input_{size_MB:g}MB.bin"
    if not (path.exists() and path.stat().st_size == target):
        with path.open("wb") as out:
            left = target
            while left > 0:
                with args.data.open("rb") as src:
                    while left > 0:
                        blk = src.read(min(BLOCK, left))
                        if not blk:
                            break
                        out.write(blk)
                        left -= len(blk)
    return path, target, target / FULL


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(BLOCK), b""):
            h.update(blk)
    return h.digest()


def compress(codec, src, dst):
    with src.open("rb") as fi, dst.open("wb") as fo:
        if codec == "lzma":
            c = lzma.LZMACompressor(preset=args.lzma_preset)
            for blk in iter(lambda: fi.read(BLOCK), b""):
                fo.write(c.compress(blk))
            fo.write(c.flush())
        else:
            zstd.ZstdCompressor(level=args.zstd_level).copy_stream(fi, fo, read_size=BLOCK)


def decompress(codec, src, dst):
    with src.open("rb") as fi, dst.open("wb") as fo:
        if codec == "lzma":
            d = lzma.LZMADecompressor()
            for blk in iter(lambda: fi.read(BLOCK), b""):
                fo.write(d.decompress(blk))
        else:
            zstd.ZstdDecompressor().copy_stream(fi, fo, read_size=BLOCK)


def measure(fn, reps):
    """Run fn() `reps` times inside one CodeCarbon window; return per-call numbers."""
    tracker = OfflineEmissionsTracker(country_iso_code=args.country, measure_power_secs=1,
                                      tracking_mode="machine", log_level="error",
                                      save_to_file=False, save_to_api=False)
    tracker.start()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    elapsed = time.perf_counter() - t0
    kg = tracker.stop()
    d = tracker.final_emissions_data
    return {"runtime_s": elapsed / reps, "energy_kWh": d.energy_consumed / reps,
            "cpu_kWh": d.cpu_energy / reps, "gpu_kWh": d.gpu_energy / reps,
            "ram_kWh": d.ram_energy / reps, "cpu_power_W": d.cpu_power,
            "ram_power_W": d.ram_power,
            "emissions_kgCO2eq": (float(kg) if kg else d.emissions) / reps,
            "repeats_in_window": reps, "_hw": d}


def row(codec, stage, m, n_bytes, src_repeats, comp_bytes, ok):
    gb, d = n_bytes / 1e9, m["_hw"]
    return {"run_label": LABEL, "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "codec": codec, "level": args.lzma_preset if codec == "lzma" else args.zstd_level,
            "stage": stage, "status": "ok" if ok else "roundtrip_failed",
            "size_MB": n_bytes / 1e6, "input_bytes": n_bytes, "source_repeats": src_repeats,
            "repeats_in_window": m["repeats_in_window"], "runtime_s": m["runtime_s"],
            "energy_kWh": m["energy_kWh"], "cpu_kWh": m["cpu_kWh"], "gpu_kWh": m["gpu_kWh"],
            "ram_kWh": m["ram_kWh"], "cpu_power_W": m["cpu_power_W"], "ram_power_W": m["ram_power_W"],
            "emissions_kgCO2eq": m["emissions_kgCO2eq"],
            "energy_kWh_per_GB": m["energy_kWh"] / gb,
            "emissions_kgCO2eq_per_GB": m["emissions_kgCO2eq"] / gb,
            "MB_per_s": n_bytes / 1e6 / m["runtime_s"], "compressed_bytes": comp_bytes,
            "compression_ratio": n_bytes / comp_bytes, "bits_per_byte": 8 * comp_bytes / n_bytes,
            "roundtrip_ok": ok, "cpu_model": d.cpu_model, "cpu_count": d.cpu_count,
            "ram_total_size": d.ram_total_size, "os": d.os, "python_version": d.python_version,
            "codecarbon_version": d.codecarbon_version, "country_iso_code": d.country_iso_code,
            "energy_source": "codecarbon (CPU power = TDP estimate on macOS, no RAPL)"}


def append(rows):
    new = not args.out.exists()
    with args.out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
            print(f"  {r['codec']:5s} {r['stage']:13s} {r['size_MB']:>7.0f} MB  "
                  f"{r['runtime_s']:8.2f} s  {r['MB_per_s']:6.1f} MB/s  "
                  f"{r['energy_kWh']:.3e} kWh  x{r['repeats_in_window']}"
                  + (f"  ratio {r['compression_ratio']:.3f}" if r["stage"] == "compression" else ""),
                  flush=True)


rate = {}                                   # codec+stage -> MB/s seen so far
t_start = time.perf_counter()
print(f"source {FULL/1e6:.1f} MB | sizes {[f'{s:g}' for s in args.sizes]} MB | "
      f"lzma preset {args.lzma_preset}, zstd level {args.zstd_level}\nCSV: {args.out}\n", flush=True)

for size_MB in args.sizes:
    path, n_bytes, src_repeats = build_input(size_MB)
    src_hash = sha256(path)
    print(f"=== {size_MB:g} MB ({(time.perf_counter()-t_start)/60:.1f} min elapsed) ===", flush=True)

    for codec in args.codecs:
        comp_path = args.work / f"{path.stem}.{codec}"
        back_path = args.work / f"{path.stem}.restored"

        def plan(stage):                     # enough repeats to fill a measurable window
            mb_s = rate.get((codec, stage))
            if not mb_s:
                return 1
            return max(1, min(20, int(args.min_window_s / max(n_bytes / 1e6 / mb_s, 1e-9)) + 1))

        m_c = measure(lambda: compress(codec, path, comp_path), plan("compression"))
        rate[(codec, "compression")] = n_bytes / 1e6 / m_c["runtime_s"]
        if m_c["runtime_s"] * m_c["repeats_in_window"] < args.min_window_s:   # first, tiny run
            m_c = measure(lambda: compress(codec, path, comp_path), plan("compression"))
        comp_bytes = comp_path.stat().st_size

        m_d = measure(lambda: decompress(codec, comp_path, back_path), plan("decompression"))
        rate[(codec, "decompression")] = n_bytes / 1e6 / m_d["runtime_s"]
        if m_d["runtime_s"] * m_d["repeats_in_window"] < args.min_window_s:
            m_d = measure(lambda: decompress(codec, comp_path, back_path), plan("decompression"))

        ok = sha256(back_path) == src_hash
        append([row(codec, "compression", m_c, n_bytes, src_repeats, comp_bytes, ok),
                row(codec, "decompression", m_d, n_bytes, src_repeats, comp_bytes, ok)])
        comp_path.unlink(missing_ok=True)
        back_path.unlink(missing_ok=True)

    path.unlink(missing_ok=True)

print(f"\nDone in {(time.perf_counter()-t_start)/60:.1f} min -> {args.out}")
