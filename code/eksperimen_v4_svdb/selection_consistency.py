"""Apakah pemilihan strategi S2 tetap sama bila fitur dibentuk dengan template bebas label?

Tanpa melatih atau menginferensi ulang. Memakai tiga hasil yang sudah ada:
  1. cv_sensitivitas_template.csv: checkpoint fold S2 dievaluasi dengan template lama dan bebas label.
  2. Prediksi hold-out CNN path setiap konfigurasi dengan template lama (holdout/<arch>/<sc>/probabilities.npy)
     dan bebas label (main_results/holdout_labelfree/<arch>/<sc>/probabilities_labelfree.npy), model sama.
     Dipakai sebagai uji ketahanan arah perbandingan, bukan untuk memilih.
  3. CV CNN path terisolasi (cv_terisolasi_ringkas.csv): margin S2 terhadap toleransi praktis
     (SD fold strategi terbaik dikurangi selisih S2 dari strategi terbaik).
Keluaran: main_results/konsistensi_seleksi_template.csv dan konsistensi_seleksi_ringkas.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from cv_isolated import ARCHS, DEST, OUT

SC = ["S1", "S2", "S3"]


def holdout_rows():
    rows = []
    for a in ARCHS:
        for s in SC:
            old = np.load(OUT / "holdout" / a / s / "probabilities.npy")
            new = np.load(DEST / "holdout_labelfree" / a / s / "probabilities_labelfree.npy")
            y = np.load(DEST / "holdout_labelfree" / a / s / "labels.npy")
            assert len(old) == len(new) == len(y)
            for name, p in [("lama", old), ("bebas_label", new)]:
                pred = p.argmax(1)
                rows.append({"arch": a, "sc": s, "template": name,
                             "macro_F1": f1_score(y, pred, labels=range(5), average="macro", zero_division=0),
                             "recall_S": recall_score(y, pred, labels=[1], average="macro", zero_division=0)})
    return pd.DataFrame(rows)


def main():
    h = holdout_rows()
    h.to_csv(DEST / "konsistensi_seleksi_template.csv", index=False)
    w = h.pivot_table(index=["arch", "sc"], columns="template", values=["macro_F1", "recall_S"])
    cv = pd.read_csv(DEST / "cv_terisolasi_ringkas.csv", header=[0, 1], index_col=[0, 1, 2])
    cv = cv[cv.index.get_level_values(2).str.startswith("CNN path terisolasi")].droplevel(2)
    sens = pd.read_csv(DEST / "cv_sensitivitas_template.csv")
    out = []
    for a in ARCHS:
        mean = cv.loc[a][("macro_F1", "mean")]; sd = cv.loc[a][("macro_F1", "std")]
        best = mean.idxmax(); tol = float(sd[best])
        within = {s: bool(mean[best] - mean[s] < tol) for s in SC}
        sa = sens[(sens.arch == a) & (sens.sc == "S2")]
        r = {"arch": a, "CV_terbaik": best, "toleransi_SD": tol,
             "S1_dalam_toleransi": within["S1"], "S2_dalam_toleransi": within["S2"], "S3_dalam_toleransi": within["S3"],
             "margin_S2": tol - float(mean[best] - mean["S2"]),
             "sens_S2_selisih_min": float(sa.selisih.min()), "sens_S2_selisih_maks": float(sa.selisih.max()),
             "sens_S2_selisih_rerata": float(sa.selisih.mean())}
        for t in ["lama", "bebas_label"]:
            r[f"holdout_recallS_S2-S1_{t}"] = float(w.loc[(a, "S2"), ("recall_S", t)] - w.loc[(a, "S1"), ("recall_S", t)])
            r[f"holdout_macro_S3-S2_{t}"] = float(w.loc[(a, "S3"), ("macro_F1", t)] - w.loc[(a, "S2"), ("macro_F1", t)])
        r["holdout_macro_perubahan_maks"] = float(max(abs(w.loc[(a, s), ("macro_F1", "bebas_label")]
                                                         - w.loc[(a, s), ("macro_F1", "lama")]) for s in SC))
        out.append(r)
    o = pd.DataFrame(out)
    o.to_csv(DEST / "konsistensi_seleksi_ringkas.csv", index=False)
    pd.set_option("display.width", 250)
    print(o.T.to_string())


if __name__ == "__main__":
    main()
