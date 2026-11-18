"""Uji B5: aturan seleksi strategi diterapkan dengan template window terpilih dan bebas label.

Sumber: main_results/cv_sensitivitas_semua.csv (checkpoint fold epoch terbaik S1-S3, 45 baris, tiap checkpoint
dievaluasi pada fold validasinya dengan template lama dan bebas label; scaler fold tetap).
Bobot fold pada E*_k hanya tersimpan untuk 7 dari 45 kombinasi, sehingga CV pada E* dengan template bebas label tidak
dapat dihitung persis. Dua pendekatan dilaporkan:
  (a) checkpoint: aturan diterapkan pada Macro-F1 checkpoint epoch terbaik, sekali dengan template lama, sekali
      dengan template bebas label (kedua pass memakai bobot yang sama, sehingga perbandingan setara);
  (b) pergeseran: CV CNN path resmi pada E* (template lama, cv_terisolasi_ringkas) ditambah pergeseran per fold
      (bebas label - lama) dari checkpoint yang sama. Ini perkiraan, bukan evaluasi pada E*.
Aturan (sama dengan paper): toleransi = SD fold strategi terbaik; strategi dalam toleransi bila selisih rerata dari
terbaik < toleransi; di antara strategi dalam toleransi dipilih biaya per epoch terendah (S1 = S2 < S3); S1 vs S2
diputus oleh recall S CV yang lebih tinggi. Hold-out tidak dipakai.
Keluaran: additional_tests/b5_seleksi_template.csv
"""
from __future__ import annotations

import pandas as pd

from cv_isolated import ARCHS, DEST, OUT

SC = ["S1", "S2", "S3"]


def rule(mean, sd, recS):
    best = mean.idxmax(); tol = float(sd[best])
    within = [s for s in SC if mean[best] - mean[s] < tol]
    cheap = [s for s in within if s in ("S1", "S2")]
    pick = max(cheap, key=lambda s: recS[s]) if cheap else "S3"
    return {"terbaik": best, "toleransi": tol, "dalam_toleransi": "+".join(within), "terpilih": pick,
            "margin_S2": tol - float(mean[best] - mean["S2"]) if "S2" in mean else float("nan"),
            **{f"macro_{s}": float(mean[s]) for s in SC}, **{f"recallS_{s}": float(recS[s]) for s in SC}}


def main():
    d = pd.read_csv(DEST / "cv_sensitivitas_semua.csv")
    n = d.groupby(["arch", "sc"]).size()
    assert len(d) == 45 and (n == 5).all(), n
    assert (d.reproduksi_argmax >= 0.99).all(), "reproduksi checkpoint gagal"
    cv = pd.read_csv(DEST / "cv_terisolasi_ringkas.csv", header=[0, 1], index_col=[0, 1, 2])
    cv = cv[cv.index.get_level_values(2).str.startswith("CNN path terisolasi")].droplevel(2)
    per = pd.read_csv(DEST / "cv_terisolasi_per_fold.csv")
    rows = []
    for a in ARCHS:
        x = d[d.arch == a]
        for t, col, rc in [("lama", "macro_lama", "recallS_lama"), ("bebas_label", "macro_labelfree", "recallS_labelfree")]:
            g = x.groupby("sc")
            rows.append({"arch": a, "pendekatan": "a_checkpoint", "template": t,
                         **rule(g[col].mean(), g[col].std(), g[rc].mean())})
        # (b) CV resmi pada E* + pergeseran per fold dari checkpoint
        mean = cv.loc[a][("macro_F1", "mean")]; sd = cv.loc[a][("macro_F1", "std")]
        recS = cv.loc[a][("recall_S", "mean")]
        rows.append({"arch": a, "pendekatan": "b_pergeseran", "template": "lama", **rule(mean, sd, recS)})
        shift = x.assign(dm=x.macro_labelfree - x.macro_lama, dr=x.recallS_labelfree - x.recallS_lama)
        pf = per[(per.arch == a) & per.prosedur.str.startswith("CNN path terisolasi")].rename(columns={"skenario": "sc"})
        g0 = pf.groupby("sc")   # pemeriksaan: rerata per fold harus sama dengan ringkasan resmi
        assert (g0.macro_F1.mean() - mean).abs().max() < 1e-9 and (g0.macro_F1.std() - sd).abs().max() < 1e-9
        pf = pf.merge(shift[["sc", "fold", "dm", "dr"]], on=["sc", "fold"], how="inner"); assert len(pf) == 15
        pf = pf.assign(m=pf.macro_F1 + pf.dm, r=pf.recall_S + pf.dr); g = pf.groupby("sc")
        rows.append({"arch": a, "pendekatan": "b_pergeseran", "template": "bebas_label",
                     **rule(g.m.mean(), g.m.std(), g.r.mean())})
    T = pd.DataFrame(rows)
    dest = OUT / "additional_tests" / "b5_seleksi_template.csv"
    T.to_csv(dest, index=False)
    pd.set_option("display.width", 250); print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
