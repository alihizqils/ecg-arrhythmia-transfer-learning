"""Inferensi ulang hold-out dengan template rekaman bebas label.

Untuk setiap model final (9 konfigurasi, bobot tidak diubah):
  1. reproduksi: inferensi hold-out dengan fitur lama harus sama dengan probabilities.npy resmi;
  2. inferensi hold-out dengan blok template rekaman (lead 1, RR dominan tingkat rekaman, lead 2) yang dihitung
     dari seluruh beat rekaman lengkap (full_record_template.py). Fitur lain tidak berubah. Scaler tetap
     scaler final (fit pada data development), sehingga tidak ada statistik hold-out yang dipakai.
Penghapusan modalitas (citra / tabular / keduanya) dihitung pada pass yang sama.
Keluaran: outputs_colab_r15/main_results/holdout_labelfree/<arch>/<sc>/ (berkas baru).
Jalankan dengan V4_OUTPUT_ROOT=outputs_colab_r15.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
os.environ.setdefault("V4_OUTPUT_ROOT", str(HERE / "outputs_colab_r15"))
sys.path.insert(0, str(HERE))
import v4_pipeline as v4  # noqa: E402
import v4_xai as X  # noqa: E402
from tools.a5_physiology import LOCAL_PHYSIOLOGY_NAMES  # noqa: E402
from v4_record_template import (BEAT_RECORD_TEMPLATE_NAMES, DOMINANT_RR_NAMES, RECORD_TEMPLATE_NAMES,  # noqa: E402
                                dominant_rr_features)

OUT = HERE / "outputs_colab_r15"
DEST = OUT / "main_results" / "holdout_labelfree"
NRT, NDOM, NL2, NBEAT = len(RECORD_TEMPLATE_NAMES), len(DOMINANT_RR_NAMES), 9, len(BEAT_RECORD_TEMPLATE_NAMES)


def load_new():
    z = np.load(OUT / "main_results" / "template_rekaman_penuh.npz")
    ids = z["window_ids"].astype(str); off = z["beat_offsets"]
    rt, l2, beat = z["rt"], z["lead2"], z["beat"]
    return {w: (rt[i], l2[i], beat[off[i]:off[i + 1]]) for i, w in enumerate(ids)}


def patch(exp, ids, new):
    corr = LOCAL_PHYSIOLOGY_NAMES.index("qrs_template_correlation")
    rec, beat = dict(exp.record_template), dict(exp.beat_record_template)
    for w in ids:
        rt, l2, b = new[w]; old_r, old_b = rec[w], beat[w]
        assert len(b) == len(old_b), w
        _, phys = exp.store.physiology_by_window[w]; _, lrr = exp.store.by_window[w]
        dom = dominant_rr_features(phys[:, corr].astype(np.float64), b[:, 0].astype(np.float64), lrr[:, 0].astype(np.float64))
        rec[w] = np.concatenate([rt, dom, l2, old_r[NRT + NDOM + NL2:]]).astype(old_r.dtype)
        beat[w] = np.concatenate([b, old_b[:, NBEAT:]], axis=1).astype(old_b.dtype)
    return rec, beat


def main(configs):
    new = load_new()
    for arch, sc in configs:
        dest = DEST / arch / sc
        if (dest / "complete.json").exists():
            print(arch, sc, "sudah ada"); continue
        exp, model, scaler, _ = X.load_final(arch, sc)
        ref_m = pd.read_csv(OUT / "holdout" / arch / sc / "manifest.csv", dtype={"window_id": str})
        rows = exp.all_rows.set_index(exp.all_rows.window_id.astype(str)).loc[ref_m.window_id].reset_index(drop=True)
        ref_p = np.load(OUT / "holdout" / arch / sc / "probabilities.npy")
        old = X.modality_knockout(exp, model, scaler, rows)
        p_old = old["skor"]["gabungan"].pop("_prob")
        rep = {"selisih_maks_prob": float(np.abs(p_old - ref_p).max()),
               "argmax_sama": float((p_old.argmax(1) == ref_p.argmax(1)).mean())}
        print(arch, sc, "reproduksi", rep, flush=True)
        exp.record_template, exp.beat_record_template = patch(exp, list(ref_m.window_id), new)
        scaler.record_template, scaler.beat_record_template = exp.record_template, exp.beat_record_template
        res = X.modality_knockout(exp, model, scaler, rows)
        p_new = res["skor"]["gabungan"].pop("_prob")
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "probabilities_labelfree.npy", p_new.astype(np.float32))
        np.save(dest / "labels.npy", res["y"])
        json.dump({"reproduksi_fitur_lama": rep, "knockout_fitur_lama": old["skor"], "knockout_labelfree": res["skor"],
                   "shapley_labelfree": res["shapley"],
                   "argmax_berubah_vs_resmi": float((p_new.argmax(1) != ref_p.argmax(1)).mean())},
                  open(dest / "complete.json", "w"), indent=2, default=float)
        print(arch, sc, "macro lama", round(old["skor"]["gabungan"]["macro"], 4), "baru",
              round(res["skor"]["gabungan"]["macro"], 4), flush=True)
        del exp, model
        import torch; torch.cuda.empty_cache()


if __name__ == "__main__":
    cfg = [(a, s) for a in ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"] for s in ["S1", "S2", "S3"]]
    if len(sys.argv) > 1:
        cfg = [c for c in cfg if f"{c[0]}:{c[1]}" in sys.argv[1:]]
    main(cfg)
