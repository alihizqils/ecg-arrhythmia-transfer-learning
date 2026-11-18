"""Audit data latih Beat-Level Specialist N/S/V/AF (di atas AF_AGG, terpilih sebelumnya untuk F1 S).

    python eksperimen_v4_svdb/fase_e_ns/spesialis_s_audit.py

Temuan audit:
  1. Specialist hanya dilatih pada beat yang kebetulan berada di window terpilih (label_valid): 4.417 dari 11.826
     beat S beranotasi di development (37 %). Sisanya data asli yang belum terpakai.
  2. Label N specialist hanya simbol "N"; L, R, e, j (AAMI N) tidak ikut dilatih, padahal muncul saat inferensi.
Varian (inferensi, agregasi, fusi dan kunci VT sama dengan AF_AGG; yang berubah hanya data latih specialist):
  AF_AGG       pembanding (pasangan window-beat, N = "N")
  AUD_AAMI     pasangan window-beat, N = N, L, R, e, j (AAMI)
  AUD_SEMUA    semua beat beranotasi development (mitdb + svdb, unik), N = AAMI; label AF per beat dari anotasi
               ritme rekaman ((AFIB / (AFL; beat N saja); beat di episode VT/VFL dibuang; ditambah beat afdb/vfdb di
               episode AF, dibatasi 1.500 beat per rekaman (pengambilan acak tetap, bukan duplikasi)
  AUD_SEMUA_E3 AUD_SEMUA + fitur konteks urutan beat E3 (jendela 7 beat: RR relatif, deviasi P, kemiripan QRS,
               pola pendek-panjang) yang sudah diekstrak tetapi belum dipakai specialist
Tidak ada augmentasi, data sintetis, atau oversampling. Fold mengikuti pasien (fold rekaman = fold window).
Nested leave-one-fold-out: specialist untuk fold k, j dilatih hanya dari rekaman development di luar fold k, j.
Aturan pemilihan (ditulis sebelum dijalankan; holdout tidak dipakai): Macro-F1 CV tertinggi di antara varian baru
yang F1 S CV > AF_AGG dan recall VT CV tidak turun; bila tidak ada, AF_AGG tetap.
Keluaran: fase_e_ns/hasil/spesialis_s_audit/ (tidak menimpa).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proksi_window_ns as PW  # noqa: E402
from ablasi_ns import HP as BEAT_HP  # noqa: E402
from spesialis_s import CODE, E_STAR, FIT, RUN, fuser, metrics  # noqa: E402
from spesialis_s_v2 import BEAT_PREFIX, decide  # noqa: E402

OUT = HERE / "hasil" / "spesialis_s_audit"
RAW = CODE / "data" / "raw"
SPECS = ["AF_AGG", "AUD_AAMI", "AUD_SEMUA", "AUD_SEMUA_E3"]
ARMS = ["BASELINE"] + SPECS
REF = "AF_AGG"
S_SYM, V_SYM, N_AAMI = ["A", "a", "J", "S"], ["V", "E"], ["N", "L", "R", "e", "j"]
AF_CAP = 1500


def gbm(X, y):
    w = pd.Series(y).map(1 / pd.Series(y).value_counts(normalize=True)).to_numpy() ** 0.5
    return HistGradientBoostingClassifier(**BEAT_HP).fit(X, y, sample_weight=w)


def longest_run(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def beat_rhythm(meta):
    """Kelas ritme per beat GQRS dari anotasi ritme rekaman: 'AF', 'VT', atau ''."""
    out = np.full(len(meta), "", dtype=object)
    for (db, rec), ix in meta.groupby(["database", "record"]).indices.items():
        an = wfdb.rdann(str(RAW / db / rec), "atr"); fs = wfdb.rdheader(str(RAW / db / rec)).fs
        notes = [str(a).replace("\x00", "").strip() for a in an.aux_note]
        ch = [(int(round(s * 250 / fs)), n[1:]) for s, n in zip(an.sample, notes) if n.startswith("(") and n[1:] != "NOISE"]
        if not ch:
            continue
        starts = np.array([c[0] for c in ch]); labs = np.array([c[1] for c in ch])
        p = meta.rpeak_sample.to_numpy()[ix]; k = np.searchsorted(starts, p, side="right") - 1
        r = np.where(k >= 0, labs[np.maximum(k, 0)], "")
        out[ix] = np.select([np.isin(r, ["AFIB", "AFL"]), np.isin(r, ["VT", "VFL", "VF", "VFIB"])], ["AF", "VT"], "")
    return out


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
    L = np.log(np.clip(L, 1e-6, 1))
    y = g.label_idx.to_numpy(); fold = g.fold.to_numpy()
    dev = (g.split == "development").to_numpy(); hold = (g.split == "holdout").to_numpy()

    meta = pd.read_csv(FIT / "meta.csv", dtype={"record": str})
    z = np.load(FIT / "fitur.npz", allow_pickle=True); names = [str(n) for n in z["names"]]
    sel = [i for i, n in enumerate(names) if n.startswith(BEAT_PREFIX)]
    sel3 = [i for i, n in enumerate(names) if n.startswith("e3c7_")]
    XA = z["X"][:, sel].astype(np.float32); X3 = z["X"][:, sel3].astype(np.float32); del z
    sym_all = meta.symbol.astype(str).to_numpy(); bfold = meta.fold.to_numpy(); bdev = (meta.split == "development").to_numpy()
    rhy = beat_rhythm(meta)
    print("ritme beat:", pd.Series(rhy[bdev]).value_counts().to_dict(), f"({time.time() - t0:.0f} dtk)", flush=True)

    members = PW.window_members(g, meta)
    wi = np.concatenate([np.full(len(m), i) for i, m in enumerate(members)]).astype(int)
    bi = np.concatenate(members).astype(int)
    pdev = dev[wi]; pfold = fold[wi]
    sym = sym_all[bi]; wl = g.label.to_numpy()[wi]

    def pair_labels(nset):
        y3 = np.select([np.isin(sym, nset), np.isin(sym, S_SYM), np.isin(sym, V_SYM)], [0, 1, 2], -1)
        y4 = y3.copy()
        y4[(wl == "AF") & ((sym == "N") | (y3 < 0)) & ~np.isin(sym, V_SYM + S_SYM)] = 3
        y4[wl == "VT"] = -1
        return y4

    # himpunan latih beat-unik untuk AUD_SEMUA
    ann_db = meta.database.isin(["mitdb", "svdb"]).to_numpy()
    yb = np.select([np.isin(sym_all, N_AAMI), np.isin(sym_all, S_SYM), np.isin(sym_all, V_SYM)], [0, 1, 2], -1)
    yb = np.where(ann_db, yb, -1)
    yb[ann_db & (rhy == "AF") & np.isin(sym_all, N_AAMI)] = 3
    yb[rhy == "VT"] = -1
    rng = np.random.default_rng(0)
    for (db, rec), ix in meta[~ann_db & bdev].groupby(["database", "record"]).indices.items():
        gi = meta[~ann_db & bdev].index.to_numpy()[ix]; af = gi[rhy[gi] == "AF"]
        if len(af):
            yb[rng.choice(af, min(AF_CAP, len(af)), replace=False)] = 3
    yb[~bdev] = -1
    print("latih beat-unik dev N/S/V/AF:", np.bincount(yb[yb >= 0], minlength=4),
          "| pasangan AF_AGG:", np.bincount(pair_labels(["N"])[pdev & (pair_labels(["N"]) >= 0)], minlength=4),
          "| pasangan AAMI:", np.bincount(pair_labels(N_AAMI)[pdev & (pair_labels(N_AAMI) >= 0)], minlength=4), flush=True)

    def fit_predict(spec, excl, tgt_pairs):
        """excl: fold yang dikeluarkan dari pelatihan; kembalikan (pS, pAF) pada pasangan tgt."""
        if spec in ("AF_AGG", "AUD_AAMI"):
            y4 = pair_labels(["N"] if spec == "AF_AGG" else N_AAMI)
            tr = pdev & ~np.isin(pfold, excl) & (y4 >= 0)
            m = gbm(XA[bi[tr]], y4[tr]); pr = m.predict_proba(XA[bi[tgt_pairs]])
        else:
            tr = np.flatnonzero((yb >= 0) & ~np.isin(bfold, excl))
            F = (lambda ix: np.hstack([XA[ix], X3[ix]])) if spec == "AUD_SEMUA_E3" else (lambda ix: XA[ix])
            m = gbm(F(tr), yb[tr]); pr = m.predict_proba(F(bi[tgt_pairs]))
        return pr[:, 1], pr[:, 3]

    P = {s: np.full((5, 5, len(bi), 2), np.nan, np.float32) for s in SPECS}
    PH = {s: np.full((len(bi), 2), np.nan, np.float32) for s in SPECS}
    for s in SPECS:
        for k in range(5):
            for j in range(k, 5):
                tgt = np.flatnonzero(pdev & np.isin(pfold, [k, j])); ps, pa = fit_predict(s, [k, j], tgt)
                P[s][k, j, tgt, 0] = ps; P[s][k, j, tgt, 1] = pa; P[s][j, k] = P[s][k, j]
            print(f"{s} k={k} ({time.time() - t0:.0f} dtk)", flush=True)
        th = np.flatnonzero(~pdev); ps, pa = fit_predict(s, [-1], th); PH[s][th, 0] = ps; PH[s][th, 1] = pa
    np.savez(OUT / "peluang_beat.npz", wi=wi, bi=bi, **{f"P_{s}": P[s] for s in SPECS}, **{f"PH_{s}": PH[s] for s in SPECS})

    order = np.argsort(wi, kind="stable"); bounds = np.searchsorted(wi[order], np.arange(len(g) + 1))

    def feats(s, k):
        p = np.zeros((len(bi), 2))
        for j in range(5):
            m = pfold == j; p[m] = np.nan_to_num(P[s][j, j, m] if k is None else P[s][k, j, m])
        if k is None:
            p[~pdev] = np.nan_to_num(PH[s][~pdev])
        po = p[order]; A = np.zeros((len(g), 13))
        for i in range(len(g)):
            v = po[bounds[i]:bounds[i + 1], 0]; a = po[bounds[i]:bounds[i + 1], 1]
            if len(v):
                vs = np.sort(v)[::-1]
                A[i] = [vs[0], vs[:2].mean(), v.mean(), (v >= 0.5).sum(), (v >= 0.5).mean(), len(v), a.mean(), (a >= 0.5).mean(),
                        v.std(), vs[:3].mean(), longest_run(v >= 0.5), a.max(), np.median(a)]
        return np.hstack([L, A])

    rows, preds = [], {a: np.full(len(g), -1) for a in ARMS}
    for k in range(5):
        trn, val = dev & (fold != k), dev & (fold == k)
        preds["BASELINE"][val] = L[val].argmax(1)
        for s in SPECS:
            X = feats(s, k); preds[s][val] = decide(fuser(X[trn], y[trn]), X[val], L[val], "FUSI_KUNCI_VT")
        for a in ARMS:
            rows.append({"fold": k, "arm": a, **metrics(y[val], preds[a][val])})
        print(k, {a: round(r["S"], 3) for a, r in zip(ARMS, rows[-len(ARMS):])}, flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT / "cv_per_fold.csv", index=False)
    meanf = d.groupby("arm")[["macro", "N", "S", "V", "AF", "VT", "recall_S", "recall_VT"]].mean().loc[ARMS]
    ref = d[d.arm == REF].set_index("fold")
    meanf["fold_S_naik_vs_AF_AGG"] = [int((d[d.arm == a].set_index("fold").S > ref.S).sum()) for a in ARMS]
    r0 = meanf.loc[REF]
    cand = [a for a in SPECS[1:] if meanf.loc[a, "S"] > r0.S and meanf.loc[a, "recall_VT"] >= r0.recall_VT]
    chosen = max(cand, key=lambda a: meanf.loc[a, "macro"]) if cand else REF
    print("== CV rerata fold\n", meanf.round(4).to_string(), "\nkandidat:", cand, "-> dipilih:", chosen, flush=True)
    hp = {"BASELINE": L[hold].argmax(1)}
    for s in SPECS:
        XH = feats(s, None); hp[s] = decide(fuser(XH[dev], y[dev]), XH[hold], L[hold], "FUSI_KUNCI_VT")
    H = pd.DataFrame({a: metrics(y[hold], p) for a, p in hp.items()}).T.loc[ARMS]
    print("== HOLDOUT (konfirmatori)\n", H.round(4).to_string(), flush=True)
    meanf.round(4).join(H.add_prefix("holdout_").round(4)).to_csv(OUT / "ringkasan.csv")
    np.savez(OUT / "confusion_holdout.npz", **{a: confusion_matrix(y[hold], p, labels=range(5)) for a, p in hp.items()})
    np.savez(OUT / "prediksi_holdout.npz", window_id=g.window_id.to_numpy()[hold], y=y[hold], **hp)
    (OUT / "info.json").write_text(json.dumps({"kandidat": cand, "chosen": chosen, "detik": round(time.time() - t0)}, indent=2))
    print(f"selesai {time.time() - t0:.0f} dtk")


if __name__ == "__main__":
    main()
