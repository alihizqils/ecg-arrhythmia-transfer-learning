"""Class weighting ujung ke ujung untuk model terbaik (CNN EfficientNet-B0 R15 E1-E4 + specialist klaster pasien).

    python eksperimen_v4_svdb/fase_e_ns/e6_bobot_ujung_ke_ujung.py --cnn S2          # tahap 1 (CNN S2 saja)
    python eksperimen_v4_svdb/fase_e_ns/e6_bobot_ujung_ke_ujung.py --cnn S1 S2       # tahap 2 (setelah S1 5 fold)

Keputusan desain 10 Nov 2026: perbandingan S1 vs S2 dibuat bersih, yaitu class weighting berlaku di SEMUA
tahap: loss CNN, specialist beat GBM, dan fusi regresi logistik.
  S1 murni = CNN S1 (tau 1, tanpa bobot) + specialist tanpa bobot + fusi tanpa bobot
  S2 murni = CNN S2 (tau 2)              + specialist berbobot     + fusi berbobot   (= model terbaik E4 saat ini)
Bobot specialist dan fusi pada E4: (1 / frekuensi kelas)^0,5. "Tanpa bobot" = sample_weight tidak dipakai.
Kombinasi silang (CNN x specialist x fusi) dilaporkan untuk memisahkan kontribusi tiap tahap.
Protokol sama dengan E4: data development, fold pasien, nested leave-one-fold-out (specialist pasangan (k, j) dilatih
tanpa fold k dan j; fusi fold luar k dilatih pada fold development lain). Holdout tidak dibaca.
Epoch CNN: E* dipilih dengan aturan resmi V4 (argmax rerata Macro-F1 validasi pada epoch yang tersedia di semua fold,
tidak kurang dari minimal epoch); untuk S2 aturan ini harus menghasilkan E* = 20 (diperiksa).
Pemeriksaan: specialist berbobot harus mereproduksi peluang specialist E4 tersimpan, dan S2 murni = F1 S 0,7235.
Keluaran: fase_e_ns/hasil/e6/bobot_ujung_ke_ujung/<cnn>/ (tidak menimpa).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from e5_common import (BEAT_HP, OUTROOT as E5ROOT, PAIRS, RUN, agg13, beat_features, decide, kl_probs, load_base,
                       log, metrics_full, nested_view)

OUTB = E5ROOT.parent / "e6" / "bobot_ujung_ke_ujung"
CACHE = E5ROOT.parent / "e6" / "_cache_bobot"
MIN_EPOCH = 20


def sqrt_w(y):
    return pd.Series(y).map(1 / pd.Series(y).value_counts(normalize=True)).to_numpy() ** 0.5


def gbm(X, y, weighted):
    return HistGradientBoostingClassifier(**{**BEAT_HP, "random_state": 0}).fit(
        X, y, sample_weight=sqrt_w(y) if weighted else None)


def fuser(X, y, weighted):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    m.fit(X, y, **({"logisticregression__sample_weight": sqrt_w(y)} if weighted else {}))
    return m


def official_estar(sc):
    """E* dari receipt resmi V4 (selection/<arch>/<sc>_final_epoch.json): argmax rerata Macro validasi pada SEMUA
    epoch bersama, tanpa batas bawah. Koreksi 10 Nov 2026: versi awal skrip ini memakai batas bawah 20 (MIN_EPOCH)
    yang bukan bagian aturan resmi; untuk S2 hasilnya sama (E* = 20), untuk S1 berbeda (resmi 7)."""
    return int(json.loads((RUN / "selection" / "efficientnet_b0" / f"{sc}_final_epoch.json").read_text())["selected_epoch"])


def load_cnn(B, sc, resmi=False):
    """Log-probabilitas CNN out-of-fold pada E*, plus E* dan kurva Macro validasi per epoch.
    resmi=False: perilaku awal (batas bawah MIN_EPOCH, keliru untuk S1); resmi=True: E* dari receipt resmi."""
    g = B["g"]; pos = pd.Series(np.arange(len(g)), index=g.window_id); curves = {}; files = {}
    for k in range(5):
        d = RUN / "cv" / "efficientnet_b0" / sc / f"fold_{k}"
        if not (d / "result.json").exists():
            raise FileNotFoundError(f"{sc} fold {k} belum selesai")
        y = np.load(d / "labels.npy"); eps = sorted((d / "epoch_probabilities").glob("epoch_*.npy"))
        curves[k] = {int(e.stem.split("_")[1]): f1_score(y, np.load(e).argmax(1), labels=range(5), average="macro",
                                                          zero_division=0) for e in eps}
        files[k] = d
    common = sorted(set.intersection(*[set(c) for c in curves.values()]))
    common = [e for e in common if e >= MIN_EPOCH] or common
    estar = max(common, key=lambda e: np.mean([curves[k][e] for k in range(5)]))
    if resmi:
        estar = official_estar(sc)
        allc = sorted(set.intersection(*[set(c) for c in curves.values()]))
        assert estar == max(allc, key=lambda e: np.mean([curves[k][e] for k in range(5)])), "receipt E* tidak cocok"
    L = np.full((len(g), 5), np.nan)
    for k, d in files.items():
        m = pd.read_csv(d / "manifest.csv", dtype={"window_id": str})
        L[pos.loc[m.window_id].to_numpy()] = np.load(d / "epoch_probabilities" / f"epoch_{estar:03d}.npy")
    return np.log(np.clip(np.nan_to_num(L, nan=1.0), 1e-6, 1)), estar, common


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cnn", nargs="+", default=["S2"])
    ap.add_argument("--estar-resmi", action="store_true"); a = ap.parse_args()
    out = OUTB / ("_".join(a.cnn) + ("_estar_resmi" if a.estar_resmi else ""))
    if (out / "ringkasan.csv").exists():
        sys.exit(f"{out} sudah ada; tidak ditimpa.")
    out.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    B = load_base(); y, dev, fold = B["y"], B["dev"], B["fold"]
    XA, KL, _ = beat_features(B); bi, y4 = B["bi"], B["y4"]; X = np.hstack([XA[bi], KL[bi]]); del XA, KL
    pdev, pfold = B["pdev"], B["pfold"]

    def spec(weighted):
        f = CACHE / f"spec_{'W' if weighted else 'U'}.npy"
        if f.exists():
            return np.load(f)
        Q = np.full((5, 5, len(bi), 2), np.nan, np.float32)
        for k, j in PAIRS:
            tr = pdev & ~np.isin(pfold, [k, j]) & (y4 >= 0); tg = np.flatnonzero(pdev & np.isin(pfold, [k, j]))
            mdl = gbm(X[tr], y4[tr], weighted); pr = mdl.predict_proba(X[tg]); cl = list(mdl.classes_)
            Q[k, j, tg, 0] = pr[:, cl.index(1)]; Q[k, j, tg, 1] = pr[:, cl.index(3)]; Q[j, k] = Q[k, j]
        np.save(f, Q); log(f"specialist {'berbobot' if weighted else 'tanpa bobot'} selesai"); return Q
    SP = {"W": spec(True), "U": spec(False)}
    PKL = kl_probs(B); m = pdev
    repro = float(np.nanmax(np.abs(SP["W"][:, :, m] - PKL[:, :, m])))
    log(f"reproduksi specialist E4: selisih maks peluang = {repro:.2e}")

    CNN = {}; info = {"reproduksi_specialist_selisih_maks": repro}
    for sc in a.cnn:
        L, estar, common = load_cnn(B, sc, a.estar_resmi); CNN[sc] = L
        info[f"E*_{sc}"] = int(estar); info[f"epoch_umum_{sc}"] = [int(common[0]), int(common[-1])]
        log(f"CNN {sc}: E* = {estar}")
    if "S2" in CNN:
        assert info["E*_S2"] == 20, f"aturan E* tidak mereproduksi S2 E* = 20 ({info['E*_S2']})"
        assert np.allclose(CNN["S2"][dev], B["L"][dev]), "log-prob CNN S2 berbeda dari baseline"

    preds = {}
    for sc, L in CNN.items():
        preds[f"{sc} | CNN saja"] = L.argmax(1)
        for sw in ["W", "U"]:
            feats = {k: np.hstack([L, agg13(B, nested_view(SP[sw], B, k))]) for k in range(5)}
            for fw in ["W", "U"]:
                p = np.full(len(y), -1)
                for k in range(5):
                    trn, val = dev & (fold != k), dev & (fold == k)
                    mdl = fuser(feats[k][trn], y[trn], fw == "W"); p[val] = decide(mdl, feats[k][val], L[val], "FUSI_KUNCI_VT")
                preds[f"{sc} | specialist {sw} | fusi {fw}"] = p
    rows = []
    for name, p in preds.items():
        for k in range(5):
            v = dev & (fold == k); rows.append({"varian": name, "fold": k, **metrics_full(y[v], p[v])})
    T = pd.DataFrame(rows); T.to_csv(out / "cv_per_fold.csv", index=False)
    cols = ["macro", "F1_N", "F1_S", "rec_S", "prec_S", "F1_V", "F1_AF", "F1_VT", "rec_VT"]
    S = T.groupby("varian", sort=False)[cols].mean().join(
        T.groupby("varian", sort=False)[["S_to_V", "S_to_AF", "V_to_S", "AF_to_S"]].sum())
    S.to_csv(out / "ringkasan.csv"); np.savez(out / "prediksi_dev.npz", **{k.replace(" ", ""): v for k, v in preds.items()})
    PF = T.pivot_table(index="varian", columns="fold", values="F1_S", sort=False); PF.to_csv(out / "f1s_per_fold.csv")
    if "S2" in CNN:
        info["F1_S_S2_murni"] = float(S.loc["S2 | specialist W | fusi W", "F1_S"])
    (out / "info.json").write_text(json.dumps(info, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    log("== ringkasan\n" + S.round(4).to_string()); log("== F1 S per fold\n" + PF.round(4).to_string())


if __name__ == "__main__":
    main()
