"""Sensitivitas template untuk S1-S3 (salinan cv_sensitivitas_template dengan probabilitas tersimpan).

Tujuan: pengaruh template rekaman bebas label pada fold validasi.

Bobot fold pada E* tidak tersimpan (hanya probabilitas per epoch), sehingga CV tidak dapat dihitung ulang persis.
Pendekatan: checkpoint fold (best.pt, epoch terbaik fold itu) dievaluasi pada fold validasinya dua kali,
dengan template lama dan template bebas label (scaler fold tetap). Pass lama harus mereproduksi probabilitas
epoch tersimpan pada epoch checkpoint. Selisih Macro-F1 antara kedua pass mengukur pengaruh perubahan fitur.
Keluaran: main_results/cv_sensitivitas_template.csv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
os.environ.setdefault("V4_OUTPUT_ROOT", str(HERE / "outputs_colab_r15"))
sys.path.insert(0, str(HERE))
import v4_holdout  # noqa: E402
import v4_pipeline as v4  # noqa: E402
import v4_runner as R  # noqa: E402
import v4_xai as X  # noqa: E402
from ecgtl.hierarchical.training import load_checkpoint  # noqa: E402
from holdout_labelfree import load_new, patch  # noqa: E402

OUT = HERE / "outputs_colab_r15"
DEST = OUT / "main_results" / "cv_sensitivitas_semua.csv"
PROB = OUT / "main_results" / "cv_sensitivitas_prob"


def main(configs):
    new = load_new(); rows = pd.read_csv(DEST).to_dict("records") if DEST.exists() else []
    done = {(r["arch"], r["sc"], r["fold"]) for r in rows}
    for arch, sc in configs:
        R.protocol(); exp = R.experiment(arch, sc)
        rec0, beat0 = exp.record_template, exp.beat_record_template
        for k in range(5):
            if (arch, sc, k) in done:
                continue
            d = OUT / "cv" / arch / sc / f"fold_{k}"; ck = load_checkpoint(d / "best.pt")
            exp.record_template, exp.beat_record_template = rec0, beat0
            scaler = v4.V4Scaler(exp.store, pd.Series(ck["train_ids"]).astype(str), rec0, beat0)
            if scaler.state_dict() != ck["rr_scaler"]:
                raise ValueError("Scaler fold berbeda.")
            model = v4_holdout.build_final_model(exp, sc, ck)
            m = pd.read_csv(d / "manifest.csv", dtype={"window_id": str})
            val = exp.all_rows.set_index(exp.all_rows.window_id.astype(str)).loc[m.window_id].reset_index(drop=True)
            a = X.modality_knockout(exp, model, scaler, val)["skor"]["gabungan"]["_prob"]
            ref = np.load(d / "epoch_probabilities" / f"epoch_{int(ck['epoch']):03d}.npy")
            rec1, beat1 = patch(exp, list(m.window_id), new)
            exp.record_template, exp.beat_record_template = rec1, beat1
            scaler.record_template, scaler.beat_record_template = rec1, beat1
            b = X.modality_knockout(exp, model, scaler, val)["skor"]["gabungan"]["_prob"]
            y = np.load(d / "labels.npy")
            PROB.mkdir(exist_ok=True)
            np.savez(PROB / f"{arch}_{sc}_fold{k}.npz", lama=a, labelfree=b, y=y, window_id=m.window_id.to_numpy().astype(str))
            rs = lambda p: float(((p.argmax(1) == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
            f = lambda p: f1_score(y, p.argmax(1), labels=range(5), average="macro", zero_division=0)
            r = {"arch": arch, "sc": sc, "fold": k, "epoch_checkpoint": int(ck["epoch"]),
                 "reproduksi_argmax": float((a.argmax(1) == ref.argmax(1)).mean()),
                 "reproduksi_selisih_maks": float(np.abs(a - ref).max()),
                 "macro_lama": f(a), "macro_labelfree": f(b), "selisih": f(b) - f(a),
                 "argmax_berubah": float((a.argmax(1) != b.argmax(1)).mean()),
                 "recallS_lama": rs(a), "recallS_labelfree": rs(b)}
            rows.append(r); pd.DataFrame(rows).to_csv(DEST, index=False); print(r, flush=True)
            del model; torch.cuda.empty_cache()
        exp.record_template, exp.beat_record_template = rec0, beat0


if __name__ == "__main__":
    cfg = [(a, s) for s in ["S2", "S1", "S3"] for a in ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"]]
    if len(sys.argv) > 1:
        cfg = [c for c in cfg if f"{c[0]}:{c[1]}" in sys.argv[1:]]
    main(cfg)
