"""Benchmark efisiensi model final V4 (di luar hash metode; dijalankan pada GPU RTX 4050 dan CPU lokal).

Ruang lingkup latensi: inferensi model (citra mentah global + crop lokal + fitur tabular ->
probabilitas) untuk satu window, batch 1, setelah warm-up. Tidak mencakup I/O,
pra-pemrosesan sinyal, rendering citra, maupun ekstraksi fitur tabular.
"""
from __future__ import annotations

import io
import json
import platform
import statistics
import time

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.flop_counter import FlopCounterMode

import v4_pipeline as v4
import v4_runner as R
import v4_xai as X
from ecgtl.hierarchical.data import read_manifest
from ecgtl.hierarchical.protocol import atomic_json
from ecgtl.hierarchical.training import to_device


def reference_window(exp) -> str:
    """Window holdout tetap dengan jumlah crop = median holdout (sama untuk semua model)."""
    manifest = read_manifest(R.protocol().prepared / "global_manifest.csv")
    holdout = v4.label_valid(manifest[manifest.split == "holdout"].reset_index(drop=True))
    local = read_manifest(R.protocol().prepared / "local_manifest.csv")
    counts = local[local.parent_window_id.isin(holdout.window_id)].groupby("parent_window_id").size()
    median = int(counts.median())
    return str(sorted(counts[counts == median].index)[0]), median


def _inputs(batch):
    return (batch["global"], batch["local"], batch["mask"], *[batch[n] for n in X.FEATURES])


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def latency(model, batch, device, repeats=300, warmup=30) -> dict:
    model = model.to(device).eval(); inputs = [x.to(device) for x in _inputs(batch)]
    for _ in range(warmup):
        model(*inputs)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = psutil.Process().memory_info().rss; times = []
    for _ in range(repeats):
        start = time.perf_counter(); model(*inputs); _sync(device); times.append(1000 * (time.perf_counter() - start))
    result = {"median_ms": statistics.median(times), "p95_ms": float(np.percentile(times, 95)),
              "mean_ms": statistics.fmean(times), "throughput_window_per_s": 1000 / statistics.median(times),
              "repeats": repeats, "warmup": warmup}
    if device.type == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
    else:
        result["rss_increase_mb"] = (psutil.Process().memory_info().rss - rss_before) / 2 ** 20
        result["threads"] = torch.get_num_threads()
    return result


def _size_mb(model) -> float:
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 2 ** 20


@torch.inference_mode()
def flops(model, batch, device) -> float:
    model = model.to(device).eval()
    with FlopCounterMode(display=False) as counter:
        model(*[x.to(device) for x in _inputs(batch)])
    return counter.get_total_flops() / 1e9


def training_cost(arch, scenario) -> dict:
    out = R.protocol().out; rows = []
    for fold in range(R.FOLDS):
        result = json.loads((out / "cv" / arch / scenario / f"fold_{fold}" / "result.json").read_text())
        rows.append({"training_s": result["training_s"], "epochs": result["epochs_run"]})
    final = json.loads((out / "final_models" / arch / scenario / "final" / "result.json").read_text())
    frame = pd.DataFrame(rows)
    return {"cv_total_jam": float(frame.training_s.sum() / 3600), "detik_per_epoch_cv": float((frame.training_s / frame.epochs).mean()),
            "final_total_menit": final["training_s"] / 60, "final_epoch": final["epochs_run"],
            "parameter_dilatih": final["trainable_parameters"], "perangkat_training": final.get("device")}


def training_peak_memory(arch, scenario, steps=3) -> float:
    """VRAM puncak beberapa langkah training dengan konfigurasi asli skenario (MB)."""
    exp = R.experiment(arch, scenario); device = exp.device
    train = exp.dev.sample(n=exp.cfg["batch_size"] * steps, random_state=0).reset_index(drop=True)
    model, scaler, cached = exp._model_scaler(train, 0, False, True) if scenario != "S3" else (None, None, False)
    if scenario == "S3":
        import v4_s3
        model = v4_s3.FullFineTuneNet(arch, exp.p.cfg, exp._head()).to(device)
        scaler = v4.V4Scaler(exp.store, train.window_id, exp.record_template, exp.beat_record_template)
    weights, _ = exp.base.weights(train, exp.tau); optimizer = exp._optimizer(model); model.train()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    for batch in exp.loader(train, scaler, cached, True, 0):
        batch = to_device(batch, device)
        with torch.autocast(device.type, enabled=exp.cfg["amp"] and device.type == "cuda"):
            loss, _ = exp.loss(exp.forward(model, batch), batch, weights)
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
    del model, optimizer; torch.cuda.empty_cache()
    return peak


def benchmark(arch: str, scenario: str, repeats=300, warmup=30) -> dict:
    exp, model, scaler, _ = X.load_final(arch, scenario)
    window_id, crops = reference_window(exp)
    batch, _ = X.batch_for(exp, scaler, [window_id]); batch = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
    total = sum(p.numel() for p in model.parameters())
    row = {"arch": arch, "skenario": scenario, "window_referensi": window_id, "crop_per_window": crops,
           "parameter_total": total, "ukuran_model_mb": _size_mb(model), "gflops_per_window": flops(model, batch, torch.device("cpu")),
           **training_cost(arch, scenario)}
    if torch.cuda.is_available():
        row.update({f"gpu_{k}": v for k, v in latency(model, batch, torch.device("cuda"), repeats, warmup).items()})
        row["gpu_nama"] = torch.cuda.get_device_name(0)
        row["training_vram_puncak_mb"] = training_peak_memory(arch, scenario)
    row.update({f"cpu_{k}": v for k, v in latency(model, batch, torch.device("cpu"), max(50, repeats // 3), 10).items()})
    row["cpu_nama"] = platform.processor()
    model.to(exp.device)
    folder = R.protocol().out / "benchmark"; folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / f"{arch}_{scenario}.json", row)
    return row


def table() -> pd.DataFrame:
    folder = R.protocol().out / "benchmark"
    return pd.DataFrame([json.loads(p.read_text()) for p in sorted(folder.glob("*_S*.json"))])
