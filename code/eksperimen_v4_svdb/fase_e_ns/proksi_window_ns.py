"""Proksi tingkat window V4 (5 kelas): apakah fitur E1/E3 dan bukti S tingkat beat menaikkan F1 S?

    python eksperimen_v4_svdb/fase_e_ns/proksi_window_ns.py

Sama dengan proksi versi V4 sebelumnya: GBM lima kelas pada fitur tabular window,
fold pasien development V4, label_valid V4 (window N dari afdb/vfdb dibuang), bobot kelas ~ (1/frekuensi)^0,5,
3 seed. Hanya development; holdout tidak dipakai.

Set fitur:
  B      fitur tabular yang dipakai V4 R12: masukan expert window (RR global, fisiologi, gelombang,
         template rekaman, RR dominan, lead kedua) + agregat min/max/mean/beat-paling-prematur dari fitur beat V4
  B+E13  B + agregat window fitur E1 (RR lokal kausal, baseline dari riwayat rekaman) dan E3 (konteks 7 beat)
  B+PS   B + agregat peluang S tingkat beat (model beat N/S E3; nested: model beat untuk window fold j
         dilatih tanpa pasien fold j dan tanpa fold validasi)
  B+E13+PS
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
FS_DIR = HERE.parent / "arsip_run_laptop_rtx4050" / "outputs" / "feature_store"
OUT = HERE / "hasil" / "proksi_window"
C = ["N", "S", "V", "AF", "VT"]
sys.path.insert(0, str(HERE))
from ablasi_ns import HP as BEAT_HP  # noqa: E402

BEAT_PREFIX = ("e0_", "e0w_", "e1c_", "e2_", "e3c7_")  # set fitur E3 studi beat (dipilih dari CV beat)
NEW_PREFIX = ("e1c_", "e3c7_")


def load_v4():
    F = np.load(FS_DIR / "5ef71c976478fd22" / "features.npz")
    R = np.load(FS_DIR / "5ef71c976478fd22" / "rr_sequence.npz")
    ids, off = F["window_ids"].astype(str), F["local_offsets"]
    temps = sorted((p for p in glob.glob(str(FS_DIR / "record_template_*" / "record_template.npz"))
                    if "beat_offsets" in np.load(p).files), key=lambda p: Path(p).stat().st_mtime)
    T = np.load(temps[-1])
    assert (T["window_ids"].astype(str) == ids).all() and (T["beat_offsets"] == off).all()
    L2 = np.load(glob.glob(str(FS_DIR / "second_lead_*" / "second_lead.npz"))[0])
    assert (L2["window_ids"].astype(str) == ids).all()
    PL, LR, BR = F["physiology_local"], R["local_rr"], T["beat_features"]
    local = np.hstack([PL, F["wave_local"], LR, R["sequence"], BR]).astype(np.float64)
    col = PL.shape[1] + F["wave_local"].shape[1] + 2   # rr_previous_over_median
    agg, dom = [], []
    for i in range(len(ids)):
        a, b = off[i], off[i + 1]; x = local[a:b]
        if b == a:
            agg.append(np.full(local.shape[1] * 4, np.nan)); dom.append(np.full(8, np.nan)); continue
        with np.errstate(all="ignore"):
            k = np.nanargmin(np.where(np.isfinite(x[:, col]), x[:, col], np.inf))
            agg.append(np.concatenate([np.nanmin(x, 0), np.nanmax(x, 0), np.nanmean(x, 0), x[k]]))
            rr = LR[a:b, 0]; f = np.full(8, np.nan)
            for j, dm in enumerate([PL[a:b, 5] >= 0.9, BR[a:b, 0] >= 0.9]):
                pair = dm & np.r_[False, dm[:-1]] & np.isfinite(rr); d = rr[pair]; f[4 * j] = dm.mean()
                if len(d) >= 3:
                    f[4*j+1] = d.std() / d.mean(); f[4*j+2] = np.mean(np.abs(np.diff(d))) / np.median(d)
                    f[4*j+3] = np.mean(np.abs(np.diff(d)) > 0.05)
            dom.append(f)
    base = np.hstack([F["physiology_global"], F["wave_global"], R["global_rr"], T["features"], np.array(dom),
                      L2["features"], np.array(agg)])
    g = pd.read_csv(CODE / "data/processed/v4_svdb/prepared/global_manifest.csv", dtype={"window_id": str, "record": str})
    g = g.set_index("window_id").loc[ids].reset_index()
    return g, base


def window_members(g, meta):
    key = (meta.database + "/" + meta.record).to_numpy()
    peaks = meta.rpeak_sample.to_numpy()
    order = {}
    for k in np.unique(key):
        order[k] = np.flatnonzero(key == k)
    members = []
    for row in g.itertuples():
        sel = order.get(f"{row.database}/{row.record}", np.empty(0, int))
        p = peaks[sel]
        members.append(sel[(p >= row.start_target_fs) & (p < row.start_target_fs + 2500)])
    return members


def aggregate(members, B, rel_col, qrs_col):
    """min/max/mean + nilai pada beat paling prematur (relatif baseline lokal) + hitungan beat 'mirip S'."""
    out = []
    for m in members:
        if len(m) == 0:
            out.append(np.full(B.shape[1] * 4 + 6, np.nan)); continue
        x = B[m].astype(np.float64)
        with np.errstate(all="ignore"):
            rel = x[:, rel_col]; q = x[:, qrs_col]
            k = np.nanargmin(np.where(np.isfinite(rel), rel, np.inf))
            prem = rel < 0.85
            out.append(np.concatenate([np.nanmin(x, 0), np.nanmax(x, 0), np.nanmean(x, 0), x[k], [
                prem.sum(), prem.mean(), (prem & (q >= 0.9)).sum(), (prem & (q < 0.8)).sum(),
                np.nanmin(rel) if np.isfinite(rel).any() else np.nan,
                np.nanmedian(rel) if np.isfinite(rel).any() else np.nan]]))
    return np.array(out)


def ps_aggregate(members, prob):
    out = []
    for m in members:
        p = prob[m]; p = p[np.isfinite(p)]
        if len(p) == 0:
            out.append(np.full(6, np.nan)); continue
        s = np.sort(p)[::-1]
        out.append([s[0], s[:2].mean(), p.mean(), (p >= 0.5).sum(), (p >= 0.3).sum(), (p >= 0.5).mean()])
    return np.array(out)


def main():
    global OUT
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    g, base = load_v4()
    fit = HERE / "fitur" / "250hz_semua"
    meta = pd.read_csv(fit / "meta.csv", dtype={"record": str})
    z = np.load(fit / "fitur.npz", allow_pickle=True)
    names = [str(n) for n in z["names"]]; X = z["X"]
    new_cols = [i for i, n in enumerate(names) if n.startswith(NEW_PREFIX)]
    beat_cols = [i for i, n in enumerate(names) if n.startswith(BEAT_PREFIX)]
    members = window_members(g, meta)
    nb = np.array([len(m) for m in members])
    print("window tanpa beat:", int((nb == 0).sum()), "beat/window median:", np.median(nb), flush=True)
    Bn = X[:, new_cols]
    sub = [names[i] for i in new_cols]
    e13 = aggregate(members, Bn, sub.index("e1c_rr_relative_b10"), sub.index("e3c7_qrs_sim_+0"))

    y = g.label_idx.to_numpy(); fold = g.fold.to_numpy()
    dev = ((g.split == "development") & ~((g.label == "N") & g.database.isin(["afdb", "vfdb"]))).to_numpy()
    print("window development (label_valid):", dev.sum(), flush=True)

    # Peluang S tingkat beat, nested per (fold validasi k, fold window j)
    yb = np.where(meta.label == "S", 1, np.where(meta.label == "N", 0, -1))
    beat_fold = meta.fold.to_numpy(); beat_dev = (meta.split == "development").to_numpy() & (yb >= 0)
    ps_cache = OUT / "peluang_s_beat.npz"
    if "--e6" in sys.argv:
        PS = None
    elif ps_cache.exists():
        PS = np.load(ps_cache)["PS"]
    else:
        PS = np.full((5, 5, len(meta)), np.nan, dtype=np.float32)  # [k, j]: model tanpa fold k dan j
        for k in range(5):
            for j in range(5):
                if j < k and np.isfinite(PS[j, k]).any():
                    PS[k, j] = PS[j, k]; continue
                tr = beat_dev & (beat_fold != k) & (beat_fold != j)
                m = HistGradientBoostingClassifier(**BEAT_HP).fit(X[tr][:, beat_cols], yb[tr])
                target = np.flatnonzero(np.isin(beat_fold, [k, j]))
                PS[k, j, target] = m.predict_proba(X[target][:, beat_cols])[:, 1]
                print(f"model beat k={k} j={j} ({time.time() - t0:.0f} dtk)", flush=True)
        np.savez_compressed(ps_cache, PS=PS)

    def ps_features(k):
        prob = np.full(len(meta), np.nan)
        for j in range(5):
            sel = beat_fold == j
            prob[sel] = PS[k, j, sel]   # j == k: model tanpa fold k (untuk window validasi)
        return ps_aggregate(members, prob)

    if "--e6" in sys.argv:  # uji lanjutan E6 (acuan ritme jangka panjang)
        OUT = HERE / "hasil" / "proksi_window_e6"; OUT.mkdir(parents=True, exist_ok=True)
        e6 = np.load(fit / "e6.npz", allow_pickle=True)
        e6names = [str(n) for n in e6["names"]]
        Bx = np.hstack([Bn, e6["X"]]); subx = sub + e6names
        e136 = aggregate(members, Bx, subx.index("e1c_rr_relative_b10"), subx.index("e3c7_qrs_sim_+0"))
        sets = {"B": lambda k: base, "B+E13": lambda k: np.hstack([base, e13]), "B+E13+E6": lambda k: np.hstack([base, e136])}
    else:
      sets = {"B": lambda k: base, "B+E13": lambda k: np.hstack([base, e13]),
            "B+PS": lambda k: np.hstack([base, ps_features(k)]),
            "B+E13+PS": lambda k: np.hstack([base, e13, ps_features(k)])}
    rows, cms = [], {}
    for k in range(5):
        trn, val = dev & (fold != k), dev & (fold == k)
        w = pd.Series(y[trn]).map(1 / pd.Series(y[trn]).value_counts(normalize=True)).to_numpy() ** 0.5
        for name, make in sets.items():
            Xw = make(k)
            fs, preds = [], []
            for seed in range(3):
                c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                   l2_regularization=1.0, random_state=seed)
                c.fit(Xw[trn], y[trn], sample_weight=w)
                pred = c.predict(Xw[val]); preds.append(pred)
                fs.append(f1_score(y[val], pred, average=None, labels=range(5)))
            f = np.mean(fs, 0)
            cm = sum(confusion_matrix(y[val], p, labels=range(5)) for p in preds) / 3
            cms[name] = cms.get(name, 0) + cm
            rows.append({"fold": k, "set": name, "n_fitur": Xw.shape[1], "macro": f.mean(), **dict(zip(C, f))})
            print(k, name, {c_: round(v, 3) for c_, v in zip(["macro"] + C, [f.mean(), *f])}, flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT / "proksi_per_fold.csv", index=False)
    summary = d.groupby("set")[["n_fitur", "macro"] + C].mean().round(4)
    summary["fold_S_naik_vs_B"] = [int((d[d.set == s].S.to_numpy() > d[d.set == "B"].S.to_numpy()).sum()) for s in summary.index]
    if "B+E13+E6" in summary.index:
        summary.loc["B+E13+E6", "fold_S_naik_vs_B+E13"] = int((d[d.set == "B+E13+E6"].S.to_numpy() > d[d.set == "B+E13"].S.to_numpy()).sum())
    summary.to_csv(OUT / "proksi_ringkasan.csv")
    err = pd.DataFrame({s: {f"S->{C[j]}": cm[1, j] for j in range(5)} | {f"{C[j]}->S": cm[j, 1] for j in range(5) if j != 1}
                        for s, cm in cms.items()}).T.round(1)
    err.to_csv(OUT / "proksi_kesalahan_S.csv")
    print(summary.to_string()); print(err.to_string())
    (OUT / "info.json").write_text(json.dumps({"detik": round(time.time() - t0), "seed": [0, 1, 2],
                                               "window_dev": int(dev.sum()), "beat_hp": BEAT_HP}, indent=2, default=str))


if __name__ == "__main__":
    main()
