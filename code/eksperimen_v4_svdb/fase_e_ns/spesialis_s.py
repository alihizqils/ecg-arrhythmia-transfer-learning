"""Dua eksperimen terarah untuk kelas S di atas model EfficientNet-B0 S2 + E1-E4 yang sudah dilatih
(tanpa retraining CNN, tanpa backbone baru).

    python eksperimen_v4_svdb/fase_e_ns/spesialis_s.py

Eksp. 1  S-specific calibration
    skor = log p_CNN + b * [kelas == S]. b dipilih dari grid [-3, 3] untuk memaksimalkan F1 S pada data
    fitting, dengan batasan recall VT dan Macro-F1 tidak lebih rendah dari b = 0 pada data fitting.
Eksp. 2  Beat-level S specialist + penggabungan skor
    Specialist: GBM tiga kelas N/S/V per beat (R-peak GQRS, bukan anotasi) dengan fitur beat set E3
    studi beat (RR, RR lokal kausal, morfologi QRS, gelombang P terhadap template kausal yang dipilih
    dari ritme/bentuk saja). Anotasi beat asli MITDB/SVDB hanya dipakai sebagai label latih:
    N = "N", S = A/a/J/S, V = V/E. Saat inferensi tidak ada anotasi atau template ground truth.
    Penggabungan: regresi logistik multinomial pada log p_CNN (5) + agregat peluang S/V beat dalam window,
    dilatih dengan label window V4 (label_valid). Dua varian:
      FUSI5      penggabung lima kelas
      FUSI_NSV   model utama dipertahankan untuk AF/VT: bila CNN memilih AF atau VT keputusan tetap,
                 selain itu penggabung memilih di antara N/S/V.

Validasi (tanpa holdout): fold pasien V4, leave-one-fold-out. Untuk fold evaluasi k, semua komponen baru
(bias, specialist, penggabung) hanya di-fit dari fold lain:
    - skor CNN: prediksi out-of-fold model CV (epoch E* = 20, sama dengan model final);
    - peluang beat untuk window fold j != k dari specialist yang dilatih tanpa fold k dan j;
      untuk window fold k dari specialist yang dilatih tanpa fold k.
Aturan pemilihan (ditetapkan sebelum melihat holdout): varian dengan Macro-F1 CV tertinggi di antara yang
F1 S-nya naik dan recall VT-nya tidak turun dibanding baseline. Mekanisme dan parameter lalu dibekukan
(di-fit ulang pada seluruh development) dan dievaluasi SEKALI pada holdout (model final CNN).
Keluaran: fase_e_ns/hasil/spesialis_s/ (tidak menimpa).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
CODE = HERE.parents[1]
sys.path.insert(0, str(HERE))
import proksi_window_ns as PW  # noqa: E402
from ablasi_ns import HP as BEAT_HP  # noqa: E402

RUN = V4 / "outputs_pilot_r15"
OUT = HERE / "hasil" / "spesialis_s"
FIT = HERE / "fitur" / "250hz_semua"
C = ["N", "S", "V", "AF", "VT"]
E_STAR = 20
BEAT_PREFIX = ("e0_", "e0w_", "e1c_", "e2_", "e3c7_")
GRID = np.round(np.arange(-3, 3.0001, 0.05), 2)


def metrics(y, p):
    f = f1_score(y, p, average=None, labels=range(5))
    return {"macro": f.mean(), **dict(zip(C, f)), "recall_S": recall_score(y, p, labels=[1], average=None)[0],
            "recall_VT": recall_score(y, p, labels=[4], average=None)[0]}


def fit_bias(L, y):
    base = L.argmax(1); m0 = metrics(y, base); best = (m0["S"], 0.0)
    for b in GRID:
        p = (L + np.eye(5)[1] * b).argmax(1); m = metrics(y, p)
        if m["recall_VT"] >= m0["recall_VT"] and m["macro"] >= m0["macro"] and m["S"] > best[0] + 1e-9:
            best = (m["S"], float(b))
    return best[1]


def beat_aggregate(members, P):
    out = np.full((len(members), 10), np.nan)
    for i, m in enumerate(members):
        if len(m) == 0:
            continue
        s, v = P[m, 1], P[m, 2]; ss = np.sort(s)[::-1]
        out[i] = [ss[0], ss[:2].mean(), s.mean(), (s >= 0.5).sum(), (s >= 0.5).mean(),
                  v.max(), v.mean(), (v >= 0.5).mean(), len(m), (P[m, 0] >= 0.5).mean()]
    return np.nan_to_num(out, nan=0.0)


def fuser(X, y):
    w = pd.Series(y).map(1 / pd.Series(y).value_counts(normalize=True)).to_numpy() ** 0.5
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    m.fit(X, y, logisticregression__sample_weight=w)
    return m


def predict_variant(model, X, L, variant):
    pr = model.predict_proba(X); cls = model.classes_
    full = np.full((len(X), 5), -np.inf); full[:, cls] = np.log(np.clip(pr, 1e-9, 1))
    if variant == "FUSI5":
        return full.argmax(1)
    cnn = L.argmax(1); nsv = full[:, :3].argmax(1)
    return np.where(np.isin(cnn, [3, 4]), cnn, nsv)


def main():
    if (OUT / "ringkasan.csv").exists():
        sys.exit(f"{OUT} sudah ada; tidak ditimpa.")
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    g = pd.read_csv(CODE / "data/processed/v4_svdb/prepared/global_manifest.csv", dtype={"window_id": str, "record": str})
    valid = ~((g.label == "N") & g.database.isin(["afdb", "vfdb"]))
    g = g[valid].reset_index(drop=True)
    # skor CNN: OOF (epoch E*) untuk development, model final untuk holdout
    L = np.full((len(g), 5), np.nan); pos = pd.Series(np.arange(len(g)), index=g.window_id)
    for k in range(5):
        d = RUN / "cv/efficientnet_b0/S2" / f"fold_{k}"
        m = pd.read_csv(d / "manifest.csv", dtype={"window_id": str})
        L[pos.loc[m.window_id].to_numpy()] = np.load(d / "epoch_probabilities" / f"epoch_{E_STAR:03d}.npy")
    hd = RUN / "holdout/efficientnet_b0/S2"
    mh = pd.read_csv(hd / "manifest.csv", dtype={"window_id": str})
    L[pos.loc[mh.window_id].to_numpy()] = np.load(hd / "probabilities.npy")
    assert np.isfinite(L).all(), "ada window tanpa skor CNN"
    L = np.log(np.clip(L, 1e-6, 1))
    y = g.label_idx.to_numpy(); fold = g.fold.to_numpy()
    dev = (g.split == "development").to_numpy(); hold = (g.split == "holdout").to_numpy()
    print("window dev/holdout:", dev.sum(), hold.sum(), flush=True)

    # specialist beat N/S/V
    meta = pd.read_csv(FIT / "meta.csv", dtype={"record": str})
    z = np.load(FIT / "fitur.npz", allow_pickle=True); names = [str(n) for n in z["names"]]
    Xb = z["X"][:, [i for i, n in enumerate(names) if n.startswith(BEAT_PREFIX)]]
    sym = meta.symbol.astype(str)
    yb = np.select([sym == "N", sym.isin(["A", "a", "J", "S"]), sym.isin(["V", "E"])], [0, 1, 2], -1)
    bfold = meta.fold.to_numpy(); bdev = (meta.split == "development").to_numpy()
    print("beat latih (dev):", {c: int(((yb == i) & bdev).sum()) for i, c in enumerate("NSV")}, flush=True)

    def train_beat(mask):
        tr = mask & (yb >= 0)
        w = pd.Series(yb[tr]).map(1 / pd.Series(yb[tr]).value_counts(normalize=True)).to_numpy() ** 0.5
        return HistGradientBoostingClassifier(**BEAT_HP).fit(Xb[tr], yb[tr], sample_weight=w)

    cache = OUT / "peluang_beat.npz"
    PB = np.full((5, 5, len(meta), 3), np.nan, np.float32)
    for k in range(5):
        for j in range(5):
            if j < k:
                PB[k, j] = PB[j, k]; continue
            mdl = train_beat(bdev & (bfold != k) & (bfold != j))
            tgt = np.flatnonzero(bdev & np.isin(bfold, [k, j]))
            PB[k, j, tgt] = mdl.predict_proba(Xb[tgt])
            print(f"specialist k={k} j={j} ({time.time() - t0:.0f} dtk)", flush=True)
    final_beat = train_beat(bdev)
    PH = np.full((len(meta), 3), np.nan, np.float32); th = np.flatnonzero(~bdev)
    PH[th] = final_beat.predict_proba(Xb[th])
    np.savez_compressed(cache, PB=PB, PH=PH)

    members = PW.window_members(g, meta)

    def beat_feats(k):           # k = fold evaluasi; None = holdout/final
        P = np.full((len(meta), 3), np.nan)
        for j in range(5):
            sel = bfold == j
            P[sel] = PB[j, j, sel] if k is None else PB[k, j, sel]
        if k is None:
            P[~bdev] = PH[~bdev]
        return beat_aggregate(members, np.nan_to_num(P, nan=0.0))

    # beat-level kualitas specialist (OOF fold k model tanpa k)
    bo = np.full(len(meta), -1)
    for k in range(5):
        sel = bdev & (bfold == k) & (yb >= 0); bo[sel] = PB[k, k, sel].argmax(1)
    selb = bdev & (yb >= 0)
    beat_f1 = dict(zip("NSV", f1_score(yb[selb], bo[selb], average=None, labels=[0, 1, 2]).round(4)))
    print("F1 specialist tingkat beat (CV):", beat_f1, flush=True)

    # validasi LOFO
    rows, preds = [], {a: np.full(len(g), -1) for a in ["BASELINE", "KALIBRASI_S", "FUSI5", "FUSI_NSV", "FUSI5+KAL"]}
    biases = {}
    for k in range(5):
        trn, val = dev & (fold != k), dev & (fold == k)
        preds["BASELINE"][val] = L[val].argmax(1)
        b = fit_bias(L[trn], y[trn]); biases[k] = b
        preds["KALIBRASI_S"][val] = (L[val] + np.eye(5)[1] * b).argmax(1)
        B = beat_feats(k); X = np.hstack([L, B])
        mdl = fuser(X[trn], y[trn])
        for v in ["FUSI5", "FUSI_NSV"]:
            preds[v][val] = predict_variant(mdl, X[val], L[val], v)
        # FUSI5 + bias S yang di-fit pada skor penggabung data latih (in-sample penggabung; hanya pembanding)
        lp = np.log(np.clip(mdl.predict_proba(X[trn]), 1e-9, 1)); bf = fit_bias(lp, y[trn])
        preds["FUSI5+KAL"][val] = (np.log(np.clip(mdl.predict_proba(X[val]), 1e-9, 1)) + np.eye(5)[1] * bf).argmax(1)
        for a in preds:
            rows.append({"fold": k, "arm": a, **metrics(y[val], preds[a][val])})
        print(k, "bias S", b, {a: round(rows[-len(preds) + i]["S"], 3) for i, a in enumerate(preds)}, flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT / "cv_per_fold.csv", index=False)
    pooled = pd.DataFrame({a: metrics(y[dev], p[dev]) for a, p in preds.items()}).T
    meanf = d.groupby("arm")[["macro"] + C + ["recall_S", "recall_VT"]].mean()
    ref = d[d.arm == "BASELINE"].set_index("fold")
    meanf["fold_S_naik"] = [int((d[d.arm == a].set_index("fold").S > ref.S).sum()) for a in meanf.index]
    print("== CV rerata fold\n", meanf.round(4).to_string(), "\n== CV pooled\n", pooled.round(4).to_string(), flush=True)

    # pemilihan (tanpa holdout)
    b0 = meanf.loc["BASELINE"]
    cand = [a for a in ["KALIBRASI_S", "FUSI5", "FUSI_NSV"] if meanf.loc[a, "S"] > b0.S and meanf.loc[a, "recall_VT"] >= b0.recall_VT]
    chosen = max(cand, key=lambda a: meanf.loc[a, "macro"]) if cand else None
    print("kandidat lolos:", cand, "-> dipilih:", chosen, flush=True)

    # konfirmasi holdout: semua lengan dibekukan (fit pada seluruh development), dilaporkan; keputusan = 'chosen'
    hold_rows = {"BASELINE": metrics(y[hold], L[hold].argmax(1))}
    bH = fit_bias(L[dev], y[dev])
    hold_rows["KALIBRASI_S"] = metrics(y[hold], (L[hold] + np.eye(5)[1] * bH).argmax(1))
    BH = beat_feats(None); XH = np.hstack([L, BH]); mH = fuser(XH[dev], y[dev])
    hp = {}
    for v in ["FUSI5", "FUSI_NSV"]:
        hp[v] = predict_variant(mH, XH[hold], L[hold], v); hold_rows[v] = metrics(y[hold], hp[v])
    H = pd.DataFrame(hold_rows).T
    print("== HOLDOUT (konfirmatori)\n", H.round(4).to_string(), flush=True)
    summary = meanf.round(4).join(H.add_prefix("holdout_").round(4), how="left")
    summary.to_csv(OUT / "ringkasan.csv")
    cms = {"BASELINE": confusion_matrix(y[hold], L[hold].argmax(1), labels=range(5)),
           "KALIBRASI_S": confusion_matrix(y[hold], (L[hold] + np.eye(5)[1] * bH).argmax(1), labels=range(5)),
           **{v: confusion_matrix(y[hold], hp[v], labels=range(5)) for v in hp}}
    np.savez(OUT / "confusion_holdout.npz", **cms)
    (OUT / "info.json").write_text(json.dumps({
        "e_star": E_STAR, "bias_S_per_fold": biases, "bias_S_final": bH, "chosen": chosen, "kandidat": cand,
        "beat_f1_cv": beat_f1, "beat_prefix": BEAT_PREFIX, "beat_hp": BEAT_HP, "grid": [float(GRID[0]), float(GRID[-1]), 0.05],
        "detik": round(time.time() - t0)}, indent=2, default=str))
    print(f"selesai {time.time() - t0:.0f} dtk")


if __name__ == "__main__":
    main()
