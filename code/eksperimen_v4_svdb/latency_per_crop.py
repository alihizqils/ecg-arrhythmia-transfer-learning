"""Uji B1b: latensi CPU CNN path terhadap jumlah crop lokal per window (pelengkap B1).

Benchmark resmi (v4_benchmark.py) mengukur satu window acuan hold-out dengan jumlah crop = median hold-out (15).
Jumlah crop per window bergantung pada jumlah beat, sehingga pada rekaman uji B1 rata-ratanya 10-18,7. Di sini
latensi CPU (fungsi latency yang sama: batch 1, warm-up 10, 50 ulangan, thread default torch) diukur pada window
hold-out dengan beberapa jumlah crop, lalu latensi CNN per rekaman B1 dihitung dari regresi linear latensi terhadap
jumlah crop pada rerata crop rekaman (latensi linear dalam jumlah crop sehingga rerata dapat dipakai).
Keluaran (baru, tidak menimpa): additional_tests/b1_latensi/cnn_vs_crop/
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("V4_OUTPUT_ROOT", str(Path(__file__).resolve().parent / "outputs_colab_r15"))

import numpy as np
import pandas as pd
import torch

import v4_benchmark as BM
import v4_pipeline as v4
import v4_runner as R
import v4_xai as X
from ecgtl.hierarchical.data import read_manifest

OUT = R.protocol().out
DEST = OUT / "additional_tests" / "b1_latensi" / "cnn_vs_crop"
ARCHS = ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"]
TARGET = [6, 10, 15, 19, 24]


def windows_by_crop():
    m = read_manifest(R.protocol().prepared / "global_manifest.csv")
    hold = v4.label_valid(m[m.split == "holdout"].reset_index(drop=True))
    local = read_manifest(R.protocol().prepared / "local_manifest.csv")
    n = local[local.parent_window_id.isin(hold.window_id)].groupby("parent_window_id").size()
    out = {}
    for t in TARGET:
        c = int(n.iloc[(n - t).abs().argsort()].iloc[0])
        out[c] = str(sorted(n[n == c].index)[0])
    return out


def main():
    if (DEST / "complete.json").exists():
        print((DEST / "complete.json").read_text()); return
    DEST.mkdir(parents=True, exist_ok=True)
    wins = windows_by_crop(); print("window per jumlah crop:", wins, "| thread:", torch.get_num_threads(), flush=True)
    rows = []
    for arch in ARCHS:
        exp, model, scaler, _ = X.load_final(arch, "S2")
        ref = json.loads((OUT / "benchmark" / f"{arch}_S2.json").read_text())
        for c, wid in sorted(wins.items()) + [(ref["crop_per_window"], ref["window_referensi"])]:
            batch, _ = X.batch_for(exp, scaler, [wid])
            batch = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
            assert int(batch["local"].shape[1]) == c, (batch["local"].shape, c)
            r = BM.latency(model, batch, torch.device("cpu"), 50, 10)
            rows.append({"arch": arch, "window_id": wid, "crop": c, "acuan_resmi": wid == ref["window_referensi"],
                         "cpu_median_ms": r["median_ms"], "cpu_p95_ms": r["p95_ms"], "threads": r["threads"]})
            print(rows[-1], flush=True)
        del model, exp; torch.cuda.empty_cache()
    T = pd.DataFrame(rows); T.to_csv(DEST / "latensi_cpu_per_crop.csv", index=False)
    fit = {}
    for arch, d in T[~T.acuan_resmi].groupby("arch"):
        b, a = np.polyfit(d.crop, d.cpu_median_ms, 1)
        pred = a + b * d.crop; r2 = 1 - ((d.cpu_median_ms - pred) ** 2).sum() / ((d.cpu_median_ms - d.cpu_median_ms.mean()) ** 2).sum()
        fit[arch] = {"intersep_ms": float(a), "ms_per_crop": float(b), "r2": float(r2)}
    rec = pd.read_csv(OUT / "additional_tests" / "b1_latensi" / "latensi_per_rekaman.csv", dtype={"record": str})
    pw = pd.read_csv(OUT / "additional_tests" / "b1_latensi" / "latensi_per_window_tahap.csv")
    summ = []
    for arch in ARCHS:
        f = fit[arch]
        for (_, r), (_, w) in zip(rec.iterrows(), pw.iterrows()):
            cnn = f["intersep_ms"] + f["ms_per_crop"] * r.crop_per_window_rerata; tot = w.total_non_cnn + cnn
            summ.append({"arch": arch, "rekaman": f"{r.database}/{r.record}", "crop_rerata": r.crop_per_window_rerata,
                         "ms_non_cnn": w.total_non_cnn, "ms_cnn_path_sesuai_crop": cnn, "ms_total": tot,
                         "porsi_cnn_path": cnn / tot, "faktor_realtime": tot * r.n_window / 1000 / r.durasi_s})
    S = pd.DataFrame(summ); S.to_csv(DEST / "latensi_end_to_end_sesuai_crop.csv", index=False)
    res = {"fit_linear": fit, "thread": int(T.threads.iloc[0]), "catatan": "CPU, batch 1, 50 ulangan, warm-up 10"}
    (DEST / "complete.json").write_text(json.dumps(res, indent=2))
    pd.set_option("display.width", 220); print(T.round(1).to_string(index=False)); print(json.dumps(fit, indent=2))
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
