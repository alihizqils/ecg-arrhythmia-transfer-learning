"""Versi 2 (8 Nov 2026). Versi 1 (spesialis_s.py) tetap disimpan.

    python eksperimen_v4_svdb/fase_e_ns/spesialis_s_v2.py

Perbedaan dari versi 1:
  1. Kalibrasi S: hanya satu batasan, recall VT pada data fitting tidak lebih rendah dari tanpa bias
     (versi 1 menambah syarat Macro-F1 tidak turun, yang tidak diminta).
  2. Specialist N/S/V hanya memakai fitur per beat: RR (e0_ RR dan e1c_ RR lokal), morfologi QRS, dan
     gelombang P (e0_ dan e2_). Tanpa fitur ringkasan blok 10 detik (e0w_) dan tanpa konteks multi-beat (e3).
  3. Yang digabung hanya peluang S per beat (agregat window), bersama skor lima kelas EfficientNet.
  4. Dua tafsiran "model utama dipertahankan untuk AF dan VT":
       FUSI_KUNCI_AF_VT  keputusan AF dan VT dari CNN dikunci; penggabung memilih di antara N/S/V
       FUSI_KUNCI_VT     hanya VT dikunci (recall VT terlindungi); penggabung boleh mengoreksi AF -> S
     ditambah FUSI5 (tanpa kunci) sebagai pembanding.

Validasi sama dengan versi 1: fold pasien V4, leave-one-fold-out, semua komponen baru di-fit hanya dari fold lain;
skor CNN = prediksi out-of-fold epoch E* = 20. Aturan pemilihan (ditulis sebelum dijalankan; holdout tidak dipakai):
varian dengan Macro-F1 CV (rerata fold) tertinggi di antara yang F1 S CV-nya naik dan recall VT CV-nya tidak turun
dibanding baseline. Semua varian dibekukan (fit ulang pada seluruh development) dan dilaporkan pada holdout;
keputusan tetap mengikuti aturan CV.
Keluaran: fase_e_ns/hasil/spesialis_s_v2/ (tidak menimpa).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proksi_window_ns as PW  # noqa: E402
from ablasi_ns import HP as BEAT_HP  # noqa: E402
from spesialis_s import CODE, E_STAR, FIT, GRID, RUN, fuser, metrics  # noqa: E402

OUT = HERE / "hasil" / "spesialis_s_v2"
BEAT_PREFIX = ("e0_", "e1c_", "e2_")
ARMS = ["BASELINE", "KALIBRASI_S", "FUSI_KUNCI_AF_VT", "FUSI_KUNCI_VT", "FUSI5"]


def fit_bias_vt(L, y):
    m0 = metrics(y, L.argmax(1)); best = (m0["S"], 0.0)
    for b in GRID:
        m = metrics(y, (L + np.eye(5)[1] * b).argmax(1))
        if m["recall_VT"] >= m0["recall_VT"] and m["S"] > best[0] + 1e-9:
            best = (m["S"], float(b))
    return best[1]


def s_aggregate(members, ps):
    out = np.zeros((len(members), 6))
    for i, m in enumerate(members):
        if len(m) == 0:
            continue
        s = ps[m]; ss = np.sort(s)[::-1]
        out[i] = [ss[0], ss[:2].mean(), s.mean(), (s >= 0.5).sum(), (s >= 0.5).mean(), len(m)]
    return out


def decide(model, X, L, arm):
    pr = model.predict_proba(X); full = np.full((len(X), 5), -np.inf)
    full[:, model.classes_] = np.log(np.clip(pr, 1e-9, 1)); cnn = L.argmax(1)
    if arm == "FUSI5":
        return full.argmax(1)
    if arm == "FUSI_KUNCI_AF_VT":
        return np.where(np.isin(cnn, [3, 4]), cnn, full[:, :3].argmax(1))
    return np.where(cnn == 4, 4, full[:, :4].argmax(1))   # FUSI_KUNCI_VT


def main():
    if (OUT / "ringkasan.csv").exists():
        sys.exit(f"{OUT} sudah ada; tidak ditimpa.")
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    g = pd.read_csv(CODE / "data/processed/v4_svdb/prepared/global_manifest.csv", dtype={"window_id": str, "record": str})
    g = g[~((g.label == "N") & g.database.isin(["afdb", "vfdb"]))].reset_index(drop=True)
    L = np.full((len(g), 5), np.nan); pos = pd.Series(np.arange(len(g)), index=g.window_id)
    for k in range(5):
        d = RUN / "cv/efficientnet_b0/S2" / f"fold_{k}"
        m = pd.read_csv(d / "manifest.csv", dtype={"window_id": str})
        L[pos.loc[m.window_id].to_numpy()] = np.load(d / "epoch_probabilities" / f"epoch_{E_STAR:03d}.npy")
    hd = RUN / "holdout/efficientnet_b0/S2"
    mh = pd.read_csv(hd / "manifest.csv", dtype={"window_id": str})
    L[pos.loc[mh.window_id].to_numpy()] = np.load(hd / "probabilities.npy")
    assert np.isfinite(L).all()
    L = np.log(np.clip(L, 1e-6, 1))
    y = g.label_idx.to_numpy(); fold = g.fold.to_numpy()
    dev = (g.split == "development").to_numpy(); hold = (g.split == "holdout").to_numpy()

    meta = pd.read_csv(FIT / "meta.csv", dtype={"record": str})
    z = np.load(FIT / "fitur.npz", allow_pickle=True); names = [str(n) for n in z["names"]]
    cols = [i for i, n in enumerate(names) if n.startswith(BEAT_PREFIX)]
    Xb = z["X"][:, cols]; print("fitur specialist:", len(cols), flush=True)
    sym = meta.symbol.astype(str)
    yb = np.select([sym == "N", sym.isin(["A", "a", "J", "S"]), sym.isin(["V", "E"])], [0, 1, 2], -1)
    bfold = meta.fold.to_numpy(); bdev = (meta.split == "development").to_numpy()

    def train_beat(mask):
        tr = mask & (yb >= 0)
        w = pd.Series(yb[tr]).map(1 / pd.Series(yb[tr]).value_counts(normalize=True)).to_numpy() ** 0.5
        return HistGradientBoostingClassifier(**BEAT_HP).fit(Xb[tr], yb[tr], sample_weight=w)

    PB = np.full((5, 5, len(meta), 3), np.nan, np.float32)
    for k in range(5):
        for j in range(k, 5):
            mdl = train_beat(bdev & (bfold != k) & (bfold != j))
            tgt = np.flatnonzero(bdev & np.isin(bfold, [k, j]))
            PB[k, j, tgt] = mdl.predict_proba(Xb[tgt]); PB[j, k] = PB[k, j]
            print(f"specialist k={k} j={j} ({time.time() - t0:.0f} dtk)", flush=True)
    PH = np.full((len(meta), 3), np.nan, np.float32); th = np.flatnonzero(~bdev)
    PH[th] = train_beat(bdev).predict_proba(Xb[th])
    np.savez_compressed(OUT / "peluang_beat.npz", PB=PB, PH=PH)
    members = PW.window_members(g, meta)

    def feats(k):
        P = np.zeros(len(meta))
        for j in range(5):
            sel = bfold == j
            P[sel] = np.nan_to_num(PB[j, j, sel, 1] if k is None else PB[k, j, sel, 1])
        if k is None:
            P[~bdev] = np.nan_to_num(PH[~bdev, 1])
        return np.hstack([L, s_aggregate(members, P)])

    bo = np.full(len(meta), -1)
    for k in range(5):
        sel = bdev & (bfold == k) & (yb >= 0); bo[sel] = PB[k, k, sel].argmax(1)
    selb = bdev & (yb >= 0)
    beat_f1 = dict(zip("NSV", f1_score(yb[selb], bo[selb], average=None, labels=[0, 1, 2]).round(4)))
    print("F1 specialist tingkat beat (CV):", beat_f1, flush=True)

    rows, preds, biases = [], {a: np.full(len(g), -1) for a in ARMS}, {}
    for k in range(5):
        trn, val = dev & (fold != k), dev & (fold == k)
        preds["BASELINE"][val] = L[val].argmax(1)
        b = fit_bias_vt(L[trn], y[trn]); biases[k] = b
        preds["KALIBRASI_S"][val] = (L[val] + np.eye(5)[1] * b).argmax(1)
        X = feats(k); mdl = fuser(X[trn], y[trn])
        for a in ARMS[2:]:
            preds[a][val] = decide(mdl, X[val], L[val], a)
        for a in ARMS:
            rows.append({"fold": k, "arm": a, **metrics(y[val], preds[a][val])})
        print(k, "bias S", b, {a: round(r["S"], 3) for a, r in zip(ARMS, rows[-len(ARMS):])}, flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT / "cv_per_fold.csv", index=False)
    meanf = d.groupby("arm")[["macro", "N", "S", "V", "AF", "VT", "recall_S", "recall_VT"]].mean().loc[ARMS]
    ref = d[d.arm == "BASELINE"].set_index("fold")
    meanf["fold_S_naik"] = [int((d[d.arm == a].set_index("fold").S > ref.S).sum()) for a in ARMS]
    b0 = meanf.loc["BASELINE"]
    cand = [a for a in ARMS[1:] if meanf.loc[a, "S"] > b0.S and meanf.loc[a, "recall_VT"] >= b0.recall_VT]
    chosen = max(cand, key=lambda a: meanf.loc[a, "macro"]) if cand else None
    print("== CV rerata fold\n", meanf.round(4).to_string(), "\nkandidat:", cand, "-> dipilih:", chosen, flush=True)

    hp = {"BASELINE": L[hold].argmax(1)}
    bH = fit_bias_vt(L[dev], y[dev]); hp["KALIBRASI_S"] = (L[hold] + np.eye(5)[1] * bH).argmax(1)
    XH = feats(None); mH = fuser(XH[dev], y[dev])
    for a in ARMS[2:]:
        hp[a] = decide(mH, XH[hold], L[hold], a)
    H = pd.DataFrame({a: metrics(y[hold], p) for a, p in hp.items()}).T.loc[ARMS]
    print("== HOLDOUT (konfirmatori)\n", H.round(4).to_string(), flush=True)
    meanf.round(4).join(H.add_prefix("holdout_").round(4)).to_csv(OUT / "ringkasan.csv")
    np.savez(OUT / "confusion_holdout.npz", **{a: confusion_matrix(y[hold], p, labels=range(5)) for a, p in hp.items()})
    (OUT / "info.json").write_text(json.dumps({"bias_S_per_fold": biases, "bias_S_final": bH, "kandidat": cand,
                                               "chosen": chosen, "beat_f1_cv": beat_f1, "beat_prefix": BEAT_PREFIX,
                                               "n_fitur_beat": len(cols), "detik": round(time.time() - t0)}, indent=2))
    print(f"selesai {time.time() - t0:.0f} dtk")


if __name__ == "__main__":
    main()
