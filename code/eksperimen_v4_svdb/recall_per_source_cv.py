"""Uji tambahan U1: recall per kelas dan database sumber pada CV development (tanpa training).

Hold-out hanya memuat AF dari AFDB dan hampir semua VT dari VFDB, tetapi development juga memuat AF dari MITDB
(7 pasien) dan VFDB (3 pasien) serta VT dari MITDB (3 pasien). Prediksi out-of-fold dibangun ulang persis dengan
prosedur yang sudah dipakai paper (cv_isolated.py):
  - CNN path terisolasi: fold k pada E*_k nested (dipilih dari empat fold lain);
  - model lengkap terisolasi: CNN path di atas + specialist tanpa fold k + fusi leave-one-patient-out di dalam fold k.
Pemeriksaan: macro-F1 per fold kedua prosedur harus sama dengan main_results/cv_terisolasi_per_fold.csv.
Batas: pasien lain dari database yang sama ikut di data latih fold lain (uji lintas pasien per sumber, bukan lintas
database; uji lintas sumber adalah U2). Hold-out tidak dibaca.
Keluaran (baru): outputs_additional_tests/u1_recall_per_sumber/
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import v4_model_lengkap as M
from cv_isolated import ARCHS, DEST as R3, OUT, curves, pick

DEST = M.HERE / "outputs_additional_tests" / "u1_recall_per_sumber"
C = ["N", "S", "V", "AF", "VT"]


def main():
    if (DEST / "complete.json").exists():
        print((DEST / "complete.json").read_text()); return
    DEST.mkdir(parents=True, exist_ok=True)
    g = M.windows(); Z, _ = M.paket()
    assert np.array_equal(Z["window_id"], g.window_id.to_numpy().astype(str))
    y = g.label_idx.to_numpy(); fold = g.fold.to_numpy(); dev = (g.split == "development").to_numpy()
    pid = (g.database + "/" + g.record).to_numpy(); db = g.database.to_numpy(); pat = g.patient_id.to_numpy()
    pos = pd.Series(np.arange(len(g)), index=g.window_id)
    ref = pd.read_csv(R3 / "cv_terisolasi_per_fold.csv")
    rows, preds, maxdiff = [], {}, 0.0
    for arch in ARCHS:
        for sc in ["S1", "S2", "S3"]:
            estar = M.estar_resmi(OUT, arch, sc); cs = curves(arch, sc); assert pick(cs, range(5)) == estar
            w = M.BOBOT[sc]; v = "W" if w else "U"; L = np.zeros((len(g), 5))
            for k in range(5):
                d = OUT / "cv" / arch / sc / f"fold_{k}"
                m = pd.read_csv(d / "manifest.csv", dtype={"window_id": str}); ix = pos.loc[m.window_id].to_numpy()
                ek = pick(cs, [j for j in range(5) if j != k])
                L[ix] = np.log(np.clip(np.load(d / "epoch_probabilities" / f"epoch_{ek:03d}.npy"), 1e-6, 1))
            cnn = np.full(len(g), -1); full = np.full(len(g), -1)
            for k in range(5):
                val = dev & (fold == k); iv = np.flatnonzero(val)
                F = np.hstack([L, Z[f"cv_{v}"][k]]); pk = pid[iv]
                for p in np.unique(pk):
                    te = pk == p; tr = iv[~te]
                    full[iv[te]] = M.decide(M.fuser(F[tr], y[tr], w), F[iv[te]], L[iv[te]])
                cnn[iv] = L[iv].argmax(1)
                for name, pr, proc in [("cnn", cnn, "CNN path terisolasi (E* nested)"),
                                       ("lengkap", full, "terisolasi (E* nested, fusi LOPO dalam fold)")]:
                    got = M.ringkas(y[val], pr[val])["macro_F1"]
                    exp_ = ref[(ref.arch == arch) & (ref.skenario == sc) & (ref.fold == k) & (ref.prosedur == proc)].macro_F1
                    assert len(exp_) == 1; maxdiff = max(maxdiff, abs(got - float(exp_.iloc[0])))
            assert (cnn[dev] >= 0).all() and (full[dev] >= 0).all()
            preds[f"{arch}_{sc}_cnn"] = cnn; preds[f"{arch}_{sc}_lengkap"] = full
            for name, pr in [("CNN path", cnn), ("model lengkap", full)]:
                for c in range(5):
                    for d_ in np.unique(db[dev & (y == c)]):
                        sel = dev & (y == c) & (db == d_)
                        hit = pr[sel] == c
                        per_p = pd.Series(hit).groupby(pat[sel]).mean()
                        rows.append({"arch": arch, "skenario": sc, "prediksi": name, "kelas": C[c], "database": d_,
                                     "n_window": int(sel.sum()), "n_pasien": int(len(per_p)), "recall": float(hit.mean()),
                                     "recall_pasien_median": float(per_p.median()), "recall_pasien_min": float(per_p.min())})
                    # alarm palsu per sumber: window bukan kelas c yang diprediksi c
                    for d_ in np.unique(db[dev]):
                        sel = dev & (y != c) & (db == d_)
                        rows.append({"arch": arch, "skenario": sc, "prediksi": name, "kelas": f"prediksi {C[c]} dari non-{C[c]}",
                                     "database": d_, "n_window": int(sel.sum()), "n_pasien": int(len(np.unique(pat[sel]))),
                                     "recall": float((pr[sel] == c).mean()), "recall_pasien_median": np.nan,
                                     "recall_pasien_min": np.nan})
            print(arch, sc, "selesai", flush=True)
    assert maxdiff < 1e-9, f"tidak mereproduksi cv_terisolasi_per_fold ({maxdiff})"
    T = pd.DataFrame(rows); T.to_csv(DEST / "recall_per_sumber_cv.csv", index=False)
    np.savez_compressed(DEST / "prediksi_oof.npz", window_id=g.window_id.to_numpy().astype(str), y=y, database=db,
                        pasien=pat, dev=dev, **preds)
    key = T[T.kelas.isin(["AF", "VT"])]
    S = key.groupby(["prediksi", "kelas", "database"]).agg(n_window=("n_window", "first"), n_pasien=("n_pasien", "first"),
                                                           recall_min=("recall", "min"), recall_max=("recall", "max"),
                                                           pasien_min_terendah=("recall_pasien_min", "min")).reset_index()
    S.to_csv(DEST / "ringkas_af_vt.csv", index=False)
    res = {"reproduksi_cv_terisolasi_selisih_maks": maxdiff, "holdout_dibaca": False}
    (DEST / "complete.json").write_text(json.dumps(res, indent=2))
    pd.set_option("display.width", 220); print(S.round(3).to_string(index=False)); print(res)


if __name__ == "__main__":
    main()
