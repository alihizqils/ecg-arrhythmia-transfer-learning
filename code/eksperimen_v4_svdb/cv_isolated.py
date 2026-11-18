"""CV model lengkap dengan isolasi fold penuh.

Masalah prosedur lama (v4_model_lengkap.jalankan): fusi untuk fold luar k dilatih pada fold j != k memakai
log-prob CNN out-of-fold fold j, yang berasal dari CNN fold j (dilatih termasuk fold k); E* juga dipilih dari
rerata kelima fold, termasuk fold k.

Prosedur baru (tanpa melatih ulang CNN, memakai probabilitas per epoch yang tersimpan):
  1. E*_k nested: epoch bersama terkecil yang rerata Macro-F1 validasinya pada empat fold lain (tanpa k)
     >= maksimum - 1e-4 (aturan resmi, tie_tolerance sama). Fold k tidak ikut memilih epoch-nya sendiri.
  2. CNN fold k pada E*_k: model fold k tidak pernah melihat fold k.
  3. Specialist: cv_<v>[k] pada window fold k = GBM yang dilatih tanpa fold k (paket terkunci).
  4. Fusi: leave-one-patient-out DI DALAM fold k. Prediksi pasien p dari fusi yang dilatih pada pasien lain
     fold k, dengan masukan (1)-(3). Label pasien p tidak dipakai oleh model mana pun yang menilai p, dan label
     fold k tidak dipakai oleh CNN maupun specialist.
Resep fusi, bobot, dan keputusan VT lock sama persis dengan v4_model_lengkap. Holdout tidak dibaca.
Keluaran ditulis ke outputs_colab_r15/main_results/ (berkas baru; tidak menimpa apa pun).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import v4_model_lengkap as M

OUT = M.HERE / "outputs_colab_r15"
DEST = OUT / "main_results"
ARCHS = ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"]
TOL = 1e-4


def curves(arch, sc):
    res = []
    for k in range(5):
        d = OUT / "cv" / arch / sc / f"fold_{k}"
        y = np.load(d / "labels.npy")
        res.append({int(e.stem.split("_")[1]): f1_score(y, np.load(e).argmax(1), labels=range(5), average="macro",
                                                         zero_division=0)
                    for e in sorted((d / "epoch_probabilities").glob("epoch_*.npy"))})
    return res


def pick(cs, folds):
    common = sorted(set.intersection(*[set(cs[j]) for j in folds]))
    means = {e: float(np.mean([cs[j][e] for j in folds])) for e in common}
    best = max(means.values())
    return min(e for e in common if means[e] >= best - TOL - 1e-12)


def main():
    DEST.mkdir(exist_ok=True)
    g = M.windows(); Z, sha = M.paket()
    assert np.array_equal(Z["window_id"], g.window_id.to_numpy().astype(str))
    y = g.label_idx.to_numpy(); fold = g.fold.to_numpy(); dev = (g.split == "development").to_numpy()
    pid = (g.database + "/" + g.record).to_numpy()
    pos = pd.Series(np.arange(len(g)), index=g.window_id)
    rows, epochs = [], []
    for arch in ARCHS:
        for sc in ["S1", "S2", "S3"]:
            estar = M.estar_resmi(OUT, arch, sc); cs = curves(arch, sc)
            assert pick(cs, range(5)) == estar
            w = M.BOBOT[sc]; v = "W" if w else "U"
            L_old = np.zeros((len(g), 5))
            L_new = np.zeros((len(g), 5))
            for k in range(5):
                d = OUT / "cv" / arch / sc / f"fold_{k}"
                m = pd.read_csv(d / "manifest.csv", dtype={"window_id": str}); ix = pos.loc[m.window_id].to_numpy()
                ek = pick(cs, [j for j in range(5) if j != k])
                epochs.append({"arch": arch, "skenario": sc, "fold": k, "E*_resmi": estar, "E*_nested": ek})
                L_old[ix] = np.log(np.clip(np.load(d / "epoch_probabilities" / f"epoch_{estar:03d}.npy"), 1e-6, 1))
                L_new[ix] = np.log(np.clip(np.load(d / "epoch_probabilities" / f"epoch_{ek:03d}.npy"), 1e-6, 1))
            for k in range(5):
                val = dev & (fold == k); trn = dev & (fold != k)
                # (a) prosedur lama, direproduksi persis
                F = np.hstack([L_old, Z[f"cv_{v}"][k]])
                p_old = M.decide(M.fuser(F[trn], y[trn], w), F[val], L_old[val])
                # (b) prosedur terisolasi: E*_k nested + fusi leave-one-patient-out di dalam fold k
                F = np.hstack([L_new, Z[f"cv_{v}"][k]]); iv = np.flatnonzero(val); p_new = np.full(len(iv), -1)
                pk = pid[iv]
                for p in np.unique(pk):
                    te = pk == p; tr = iv[~te]
                    p_new[te] = M.decide(M.fuser(F[tr], y[tr], w), F[iv[te]], L_new[iv[te]])
                yv = y[val]
                rows += [{"arch": arch, "skenario": sc, "fold": k, "prosedur": "lama (fusi lintas fold, E* semua fold)",
                          **M.ringkas(yv, p_old)},
                         {"arch": arch, "skenario": sc, "fold": k, "prosedur": "terisolasi (E* nested, fusi LOPO dalam fold)",
                          **M.ringkas(yv, p_new)},
                         {"arch": arch, "skenario": sc, "fold": k, "prosedur": "CNN path lama (E* semua fold)",
                          **M.ringkas(yv, L_old[val].argmax(1))},
                         {"arch": arch, "skenario": sc, "fold": k, "prosedur": "CNN path terisolasi (E* nested)",
                          **M.ringkas(yv, L_new[val].argmax(1))}]
            print(arch, sc, "selesai", flush=True)
    T = pd.DataFrame(rows); T.to_csv(DEST / "cv_terisolasi_per_fold.csv", index=False)
    E = pd.DataFrame(epochs); E.to_csv(DEST / "cv_terisolasi_epoch.csv", index=False)
    met = [c for c in T.columns if c not in ("arch", "skenario", "fold", "prosedur")]
    S = T.groupby(["arch", "skenario", "prosedur"])[met].agg(["mean", "std"])
    S.to_csv(DEST / "cv_terisolasi_ringkas.csv")
    # cek reproduksi prosedur lama terhadap hasil resmi
    rep = []
    for arch in ARCHS:
        for sc in ["S1", "S2", "S3"]:
            off = pd.read_csv(OUT / "model_lengkap" / arch / sc / "cv_per_fold.csv")
            off = off[off.model == "lengkap"].set_index("fold").macro_F1
            mine = T[(T.arch == arch) & (T.skenario == sc) & T.prosedur.str.startswith("lama")].set_index("fold").macro_F1
            rep.append(float((off - mine).abs().max()))
    (DEST / "README.json").write_text(json.dumps({
        "dibuat_oleh": "cv_isolated.py", "paket_specialist_sha256": sha,
        "reproduksi_cv_lama_selisih_maks_macroF1": max(rep), "holdout_dibaca": False}, indent=2))
    print("reproduksi lama, selisih maks:", max(rep))
    pd.set_option("display.width", 250)
    print(S.xs("mean", axis=1, level=1)[["macro_F1", "recall_S", "F1_S"]].round(4).to_string())
    print(S.xs("std", axis=1, level=1)[["macro_F1"]].round(4).to_string())
    print(E.pivot_table(index=["arch", "skenario"], columns="fold", values="E*_nested").join(
        E.groupby(["arch", "skenario"])["E*_resmi"].first()).to_string())


if __name__ == "__main__":
    main()
