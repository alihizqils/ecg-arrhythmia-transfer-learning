"""Modul bersama E5 (pengembangan setelah E1-E4 + Beat-Level Specialist klaster pasien).

Baseline E4 = CNN EfficientNet-B0 S2 R15 (E1-E4, E* = 20) + specialist GBM N/S/V/AF + fitur klaster pasien (KL)
+ 13 agregat window + fusi regresi logistik + kunci VT dari CNN (fase_e_ns/spesialis_s_pasien.py).
Seluruh E5 hanya memakai data development dengan nested leave-one-fold-out pasien yang sama dengan baseline:
model tingkat beat untuk pasangan fold (k, j) dilatih tanpa fold k dan j; fusi fold luar k dilatih pada fold dev != k.
Holdout TIDAK dibaca untuk pemilihan apa pun (tidak ada evaluasi holdout pada pilot E5).
Tidak ada augmentasi, data sintetis, atau oversampling; tidak ada pelatihan ulang CNN.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proksi_window_ns as PW  # noqa: E402
from ablasi_ns import HP as BEAT_HP  # noqa: E402,F401
from spesialis_s import CODE, E_STAR, FIT, RUN, fuser  # noqa: E402,F401
from spesialis_s_v2 import BEAT_PREFIX, decide  # noqa: E402,F401

OUTROOT = HERE / "hasil" / "e5"
CACHE = OUTROOT / "_cache"
C = ["N", "S", "V", "AF", "VT"]
SEEDS = [0, 1, 2]
PAIRS = [(k, j) for k in range(5) for j in range(k, 5)]


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.0f} dtk] {msg}", flush=True)


def metrics_full(y, p):
    f = f1_score(y, p, average=None, labels=range(5), zero_division=0)
    r = recall_score(y, p, average=None, labels=range(5), zero_division=0)
    pr = precision_score(y, p, average=None, labels=range(5), zero_division=0)
    out = {"macro": f.mean()}
    for i, c in enumerate(C):
        out[f"F1_{c}"] = f[i]; out[f"rec_{c}"] = r[i]; out[f"prec_{c}"] = pr[i]
    out["S_to_V"] = int(((y == 1) & (p == 2)).sum()); out["S_to_AF"] = int(((y == 1) & (p == 3)).sum())
    out["V_to_S"] = int(((y == 2) & (p == 1)).sum()); out["AF_to_S"] = int(((y == 3) & (p == 1)).sum())
    return out


def longest_run(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def load_base():
    """Muat data window, skor CNN, pasangan window-beat, label beat/ritme. Holdout ikut dimuat hanya agar indeks
    konsisten; tidak ada fungsi E5 yang memakainya."""
    g = pd.read_csv(CODE / "data/processed/v4_svdb/prepared/global_manifest.csv", dtype={"window_id": str, "record": str})
    g = g[~((g.label == "N") & g.database.isin(["afdb", "vfdb"]))].reset_index(drop=True)
    L = np.full((len(g), 5), np.nan); pos = pd.Series(np.arange(len(g)), index=g.window_id)
    for k in range(5):
        d = RUN / "cv/efficientnet_b0/S2" / f"fold_{k}"
        m = pd.read_csv(d / "manifest.csv", dtype={"window_id": str})
        L[pos.loc[m.window_id].to_numpy()] = np.load(d / "epoch_probabilities" / f"epoch_{E_STAR:03d}.npy")
    L = np.log(np.clip(np.nan_to_num(L, nan=1.0), 1e-6, 1))
    y = g.label_idx.to_numpy(); fold = g.fold.to_numpy(); dev = (g.split == "development").to_numpy()
    meta = pd.read_csv(FIT / "meta.csv", dtype={"record": str})
    members = PW.window_members(g, meta)
    wi = np.concatenate([np.full(len(m), i) for i, m in enumerate(members)]).astype(int)
    bi = np.concatenate(members).astype(int)
    sym_all = meta.symbol.astype(str).to_numpy()
    # label beat lama (baseline): AF diturunkan dari label ritme WINDOW (dipakai specialist E4)
    sym = sym_all[bi]; wl = g.label.to_numpy()[wi]
    y3 = np.select([sym == "N", np.isin(sym, ["A", "a", "J", "S"]), np.isin(sym, ["V", "E"])], [0, 1, 2], -1)
    y4 = y3.copy()
    y4[(wl == "AF") & ((sym == "N") | (y3 < 0)) & ~np.isin(sym, ["V", "E", "A", "a", "J", "S"])] = 3
    y4[wl == "VT"] = -1
    # label E5: tingkat beat dari anotasi beat (mitdb/svdb), tingkat ritme dari anotasi ritme rekaman
    rhy_path = CACHE / "ritme_beat.npy"
    if not rhy_path.exists():
        from spesialis_s_audit import beat_rhythm
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(rhy_path, beat_rhythm(meta).astype(str))
    rhy_all = np.load(rhy_path)
    beat_lab_all = np.select([sym_all == "N", np.isin(sym_all, ["A", "a", "J", "S"]), np.isin(sym_all, ["V", "E"])], [0, 1, 2], -1)
    beat_lab_all[~meta.database.isin(["mitdb", "svdb"]).to_numpy()] = -1     # AFDB/VFDB: beat tidak dianotasi
    rhy_lab_all = np.where(rhy_all == "AF", 1, np.where(rhy_all == "VT", -1, 0))  # 1 AF, 0 bukan AF, -1 VT (dikeluarkan)
    B = dict(g=g, L=L, y=y, fold=fold, dev=dev, meta=meta, wi=wi, bi=bi, y4=y4, pfold=fold[wi], pdev=dev[wi],
             beat_lab_all=beat_lab_all, rhy_lab_all=rhy_lab_all, rec_all=(meta.database + "/" + meta.record).to_numpy(),
             bfold_all=meta.fold.to_numpy(), bdev_all=(meta.split == "development").to_numpy())
    order = np.argsort(wi, kind="stable")
    B["order"] = order; B["bounds"] = np.searchsorted(wi[order], np.arange(len(g) + 1))
    return B


def beat_features(B):
    z = np.load(FIT / "fitur.npz", allow_pickle=True); names = [str(n) for n in z["names"]]
    sel = [i for i, n in enumerate(names) if n.startswith(BEAT_PREFIX)]
    XA = z["X"][:, sel].astype(np.float32); sub = [names[i] for i in sel]; del z
    KL = np.load(HERE / "hasil" / "spesialis_s_pasien" / "fitur_pasien.npz")["B"].astype(np.float32)
    return XA, KL, sub


def embeddings(B):
    E = np.load(HERE / "fitur" / "crop_adaptif" / "embedding_beat.npz")
    assert (E["wi"] == B["wi"]).all() and (E["bi"] == B["bi"]).all()
    return E["tetap"]   # float16, per pasangan window-beat; potongan beat berpusat di R (tidak bergantung window)


def kl_probs(B):
    Z = np.load(HERE / "hasil" / "spesialis_s_pasien" / "peluang_beat.npz")
    assert (Z["wi"] == B["wi"]).all()
    return Z["P_KL"]    # [k, j, pasangan, (pS, pAF)]


def nested_view(Q, B, k):
    """Peluang pasangan untuk fold luar k: tiap fold dev j memakai model (k, j) (dilatih tanpa k dan j)."""
    out = np.zeros(Q.shape[2:], dtype=np.float64)
    for j in range(5):
        m = B["pdev"] & (B["pfold"] == j); out[m] = np.nan_to_num(Q[k, j, m])
    return out


def inner_view(Q, B, k, m):
    """Untuk OOF dalam (fold luar k, fold dalam m): fold j memakai model (k, j); identik dengan nested_view."""
    return nested_view(Q, B, k)


def agg13(B, p):
    """13 agregat baseline dari (pS, pAF) per pasangan."""
    po = p[B["order"]]; bd = B["bounds"]; A = np.zeros((len(bd) - 1, 13))
    for i in range(len(bd) - 1):
        v = po[bd[i]:bd[i + 1], 0]; a = po[bd[i]:bd[i + 1], 1]
        if len(v):
            vs = np.sort(v)[::-1]
            A[i] = [vs[0], vs[:2].mean(), v.mean(), (v >= 0.5).sum(), (v >= 0.5).mean(), len(v), a.mean(), (a >= 0.5).mean(),
                    v.std(), vs[:3].mean(), longest_run(v >= 0.5), a.max(), np.median(a)]
    return A


def fusion_cv(B, feats_by_k, arm="FUSI_KUNCI_VT"):
    """Prediksi OOF fold luar + probabilitas fusi. feats_by_k[k] = matriks fitur window (sudah memuat L)."""
    y, dev, fold, L = B["y"], B["dev"], B["fold"], B["L"]
    pred = np.full(len(y), -1); prob = np.full((len(y), 5), np.nan)
    for k in range(5):
        trn, val = dev & (fold != k), dev & (fold == k); X = feats_by_k[k]
        mdl = fuser(X[trn], y[trn]); pred[val] = decide(mdl, X[val], L[val], arm)
        pr = np.zeros((val.sum(), 5)); pr[:, mdl.classes_] = mdl.predict_proba(X[val]); prob[val] = pr
    return pred, prob


def fusion_inner(B, feats_by_k, k):
    """Prediksi OOF dalam untuk fold dev != k (fusi dalam dilatih tanpa fold k dan m)."""
    y, dev, fold, L = B["y"], B["dev"], B["fold"], B["L"]
    pred = np.full(len(y), -1); prob = np.full((len(y), 5), np.nan); X = feats_by_k[k]
    for m in range(5):
        if m == k:
            continue
        trn, val = dev & ~np.isin(fold, [k, m]), dev & (fold == m)
        mdl = fuser(X[trn], y[trn]); pred[val] = decide(mdl, X[val], L[val], "FUSI_KUNCI_VT")
        pr = np.zeros((val.sum(), 5)); pr[:, mdl.classes_] = mdl.predict_proba(X[val]); prob[val] = pr
    return pred, prob


def per_fold_table(B, preds: dict):
    y, dev, fold = B["y"], B["dev"], B["fold"]; rows = []
    for name, p in preds.items():
        for k in range(5):
            v = dev & (fold == k); rows.append({"metode": name, "fold": k, **metrics_full(y[v], p[v])})
    return pd.DataFrame(rows)


def pooled_confusion(B, p):
    v = B["dev"]
    return confusion_matrix(B["y"][v], p[v], labels=range(5))
