"""Kontrol: memisahkan efek kebocoran dari efek ukuran data latih fusi.

Fusi terisolasi (LOPO di dalam fold k) dilatih pada satu fold, sedangkan fusi lama pada empat fold. Untuk
membandingkan pada ukuran yang sama, fusi lama dilatih pada SATU fold lain j saja (masukan CNN fold j berasal
dari CNN yang melihat fold k = bocor), dirata-rata atas j != k. Selisih "bocor-satu-fold" dengan
"terisolasi-LOPO" pada E* yang sama mengukur kebocoran pada ukuran data yang setara.

Uji analog pada specialist: masukan specialist untuk fold latih j diganti dari GBM (k, j) (bersih, tanpa fold k)
ke GBM (j, j) (bocor, melihat fold k), CNN dan semua lainnya tetap; empat fold latih (resep lama).
Holdout tidak dibaca.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import v4_model_lengkap as M
from cv_isolated import ARCHS, DEST, OUT, curves, pick


def main():
    g = M.windows(); Z, _ = M.paket()
    y = g.label_idx.to_numpy(); fold = g.fold.to_numpy(); dev = (g.split == "development").to_numpy()
    pid = (g.database + "/" + g.record).to_numpy(); pos = pd.Series(np.arange(len(g)), index=g.window_id)
    rows = []
    for arch in ARCHS:
        for sc in ["S1", "S2", "S3"]:
            estar = M.estar_resmi(OUT, arch, sc); w = M.BOBOT[sc]; v = "W" if w else "U"
            L = np.zeros((len(g), 5))
            for k in range(5):
                d = OUT / "cv" / arch / sc / f"fold_{k}"
                m = pd.read_csv(d / "manifest.csv", dtype={"window_id": str})
                L[pos.loc[m.window_id].to_numpy()] = np.log(np.clip(
                    np.load(d / "epoch_probabilities" / f"epoch_{estar:03d}.npy"), 1e-6, 1))
            # specialist bocor: baris fold j memakai GBM (j, j) = cv[j] pada fold j
            Zleak = np.zeros_like(Z[f"cv_{v}"][0])
            for j in range(5):
                Zleak[fold == j] = Z[f"cv_{v}"][j][fold == j]
            for k in range(5):
                val = dev & (fold == k); yv = y[val]; iv = np.flatnonzero(val)
                F = np.hstack([L, Z[f"cv_{v}"][k]])
                # (1) bocor, satu fold latih
                one = []
                for j in range(5):
                    if j == k:
                        continue
                    tr = dev & (fold == j)
                    one.append(M.ringkas(yv, M.decide(M.fuser(F[tr], y[tr], w), F[val], L[val])))
                rows.append({"arch": arch, "skenario": sc, "fold": k, "prosedur": "bocor_satu_fold",
                             **pd.DataFrame(one).mean().to_dict()})
                # (2) terisolasi LOPO dalam fold, E* resmi (ukuran setara)
                p_new = np.full(len(iv), -1); pk = pid[iv]
                for p in np.unique(pk):
                    te = pk == p; tr = iv[~te]
                    p_new[te] = M.decide(M.fuser(F[tr], y[tr], w), F[iv[te]], L[iv[te]])
                rows.append({"arch": arch, "skenario": sc, "fold": k, "prosedur": "terisolasi_LOPO_Eresmi",
                             **M.ringkas(yv, p_new)})
                # (3) analog specialist: empat fold latih, specialist bersih vs bocor
                trn = dev & (fold != k)
                Fl = F.copy(); Fl[trn, 5:] = Zleak[trn]
                rows.append({"arch": arch, "skenario": sc, "fold": k, "prosedur": "specialist_bersih_4fold",
                             **M.ringkas(yv, M.decide(M.fuser(F[trn], y[trn], w), F[val], L[val]))})
                rows.append({"arch": arch, "skenario": sc, "fold": k, "prosedur": "specialist_bocor_4fold",
                             **M.ringkas(yv, M.decide(M.fuser(Fl[trn], y[trn], w), F[val], L[val]))})
            print(arch, sc, flush=True)
    T = pd.DataFrame(rows); T.to_csv(DEST / "kontrol_kebocoran_per_fold.csv", index=False)
    S = T.pivot_table(index=["arch", "skenario"], columns="prosedur", values="macro_F1", aggfunc="mean")
    S["bocor-terisolasi (ukuran setara)"] = S.bocor_satu_fold - S.terisolasi_LOPO_Eresmi
    S["specialist bocor-bersih"] = S.specialist_bocor_4fold - S.specialist_bersih_4fold
    S.to_csv(DEST / "kontrol_kebocoran_ringkas.csv")
    pd.set_option("display.width", 250)
    print(S.round(4).to_string())


if __name__ == "__main__":
    main()
