"""Model normal per pasien dan pengelompokan beat per pasien, tanpa label (di atas AF_AGG).

    python eksperimen_v4_svdb/fase_e_ns/spesialis_s_pasien.py

Dasar literatur:
  A  Zhai, Zhou & Tin, Expert Syst. Appl. 2020: beat normal pasien uji dipilih otomatis tanpa label dan dipakai
     sebagai acuan khusus pasien.
  B  Teijeiro dkk., IEEE JBHI 22(2):409-420, 2018: beat dalam satu rekaman dikelompokkan menurut kemiripan, lalu
     keputusan memakai ciri kelompok plus konteks ritme.
Diagnosis sebelumnya: beat S di pasien sulit (mis. svdb/881) hanya sedikit prematur sehingga tampak normal menurut
acuan populasi, padahal bentuk P-nya jelas berbeda dari beat normal pasien itu sendiri.
Fitur baru per beat (seluruh beat GQRS rekaman, TANPA label; tersedia saat inferensi Holter):
  A  PN : beat pseudo-normal rekaman dipilih dari aturan ritme/bentuk saja (RR lokal 0,90-1,10, rasio RR
          berikut/sebelum 0,85-1,15, korelasi template QRS >= 0,90; minimal 30 beat, bila kurang memakai semua beat).
          Tiap fitur beat dinyatakan sebagai z robust terhadap median/MAD beat pseudo-normal itu (59 kolom, dipotong
          +-20) ditambah jarak RMS per kelompok fitur P, QRS, RR (3 kolom).
  B  KL : MiniBatchKMeans (k = 8, seed 0) per rekaman pada fitur bentuk P + QRS yang distandardisasi per rekaman.
          Per beat: fraksi ukuran klasternya, apakah klaster terbesar, jarak ke pusat klaster, dan median klaster
          untuk prematuritas (RR/median lokal, RR lokal relatif 10 beat), fraksi beat prematur (< 0,90), kehadiran
          P, korelasi P rekaman, korelasi template QRS (9 kolom).
Varian (label latih N/S/V/AF, agregasi 13 fitur, fusi regresi logistik, kunci VT sama dengan AF_AGG):
  AF_AGG  pembanding (peluang beat diambil dari run spesialis_s_adaptif, model identik)
  PN      fitur dasar + A
  KL      fitur dasar + B
  PN_KL   fitur dasar + A + B
Nested leave-one-fold-out pasien. Aturan pemilihan (ditulis sebelum dijalankan; holdout tidak dipakai): Macro-F1 CV
tertinggi di antara varian baru yang F1 S CV > AF_AGG dan recall VT CV tidak turun; bila tidak ada, AF_AGG tetap.
Keluaran: fase_e_ns/hasil/spesialis_s_pasien/ (tidak menimpa).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proksi_window_ns as PW  # noqa: E402
from ablasi_ns import HP as BEAT_HP  # noqa: E402
from spesialis_s import CODE, E_STAR, FIT, RUN, fuser, metrics  # noqa: E402
from spesialis_s_v2 import BEAT_PREFIX, decide  # noqa: E402

OUT = HERE / "hasil" / "spesialis_s_pasien"
SPECS = ["PN", "KL", "PN_KL"]
ARMS = ["BASELINE", "AF_AGG", *SPECS]
REF = "AF_AGG"


def gbm(X, y):
    w = pd.Series(y).map(1 / pd.Series(y).value_counts(normalize=True)).to_numpy() ** 0.5
    return HistGradientBoostingClassifier(**BEAT_HP).fit(X, y, sample_weight=w)


def longest_run(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def group_of(n):
    if any(t in n for t in ["p_", "pre_qrs", "atrial", "pr_interval", "pr_relative"]):
        return "P"
    if "qrs" in n:
        return "QRS"
    return "RR"


def patient_features(XA, sub, rec_ix):
    """Kembalikan (A: n x 62, B: n x 9) untuk seluruh beat."""
    c = {n: i for i, n in enumerate(sub)}
    grp = np.array([group_of(n) for n in sub])
    morph = np.flatnonzero(np.isin(grp, ["P", "QRS"]))
    A = np.full((len(XA), len(sub) + 3), np.nan, np.float32); B = np.full((len(XA), 9), np.nan, np.float32)
    rng = np.random.default_rng(0)
    for r, ix in rec_ix.items():
        X = XA[ix]
        rrl = X[:, c["e1c_rr_relative_b10"]]; nop = X[:, c["e0_rr_next_over_previous"]]; qc = X[:, c["e0_qrs_template_correlation"]]
        pn = (rrl > 0.90) & (rrl < 1.10) & (nop > 0.85) & (nop < 1.15) & (qc >= 0.90)
        if pn.sum() < 30:
            pn = np.ones(len(ix), bool)
        with np.errstate(all="ignore"):
            med = np.nanmedian(X[pn], 0); mad = 1.4826 * np.nanmedian(np.abs(X[pn] - med), 0)
            scale = np.where(np.isfinite(mad) & (mad > 1e-6), mad, np.nanstd(X, 0) + 1e-6)
            Z = np.clip((X - med) / scale, -20, 20)
        A[ix, :len(sub)] = Z
        for gi, gname in enumerate(["P", "QRS", "RR"]):
            with np.errstate(all="ignore"):
                A[ix, len(sub) + gi] = np.sqrt(np.nanmean(Z[:, grp == gname] ** 2, 1))
        # klaster bentuk per rekaman
        M = X[:, morph]
        with np.errstate(all="ignore"):
            mm = np.nanmedian(M, 0); M = np.where(np.isfinite(M), M, np.where(np.isfinite(mm), mm, 0))
            M = (M - M.mean(0)) / (M.std(0) + 1e-6)
        k = int(min(8, max(2, len(ix) // 50)))
        fit_ix = rng.choice(len(ix), min(len(ix), 20000), replace=False)
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=2048).fit(M[fit_ix])
        lab = km.predict(M); dist = np.linalg.norm(M - km.cluster_centers_[lab], axis=1)
        size = np.bincount(lab, minlength=k) / len(lab); big = int(np.argmax(size))
        prem = X[:, c["e0_rr_previous_over_median"]]
        stats = np.full((k, 6), np.nan)
        for q in range(k):
            s = lab == q
            if s.any():
                with np.errstate(all="ignore"):
                    stats[q] = [np.nanmedian(prem[s]), np.nanmedian(rrl[s]), np.nanmean(prem[s] < 0.90),
                                np.nanmean(X[s, c["e0_p_present"]]), np.nanmedian(X[s, c["e0_beat_rec_p_correlation"]]),
                                np.nanmedian(qc[s])]
        B[ix] = np.column_stack([size[lab], (lab == big).astype(float), dist, stats[lab]])
    return A, B


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
    sel = [i for i, n in enumerate(names) if n.startswith(BEAT_PREFIX)]; sub = [names[i] for i in sel]
    XA = z["X"][:, sel].astype(np.float32); del z
    key = (meta.database + "/" + meta.record).to_numpy()
    rec_ix = {r: np.flatnonzero(key == r) for r in np.unique(key)}
    FA, FB = patient_features(XA, sub, rec_ix)
    print("fitur pasien", FA.shape, FB.shape, f"({time.time() - t0:.0f} dtk)", flush=True)
    np.savez_compressed(OUT / "fitur_pasien.npz", A=FA, B=FB)

    members = PW.window_members(g, meta)
    wi = np.concatenate([np.full(len(m), i) for i, m in enumerate(members)]).astype(int)
    bi = np.concatenate(members).astype(int)
    pdev = dev[wi]; pfold = fold[wi]
    sym = meta.symbol.astype(str).to_numpy()[bi]; wl = g.label.to_numpy()[wi]
    y3 = np.select([sym == "N", np.isin(sym, ["A", "a", "J", "S"]), np.isin(sym, ["V", "E"])], [0, 1, 2], -1)
    y4 = y3.copy()
    y4[(wl == "AF") & ((sym == "N") | (y3 < 0)) & ~np.isin(sym, ["V", "E", "A", "a", "J", "S"])] = 3
    y4[wl == "VT"] = -1
    FX = {"PN": np.hstack([XA[bi], FA[bi]]), "KL": np.hstack([XA[bi], FB[bi]]), "PN_KL": np.hstack([XA[bi], FA[bi], FB[bi]])}
    del XA, FA, FB

    P = {s: np.full((5, 5, len(bi), 2), np.nan, np.float32) for s in SPECS}
    PH = {s: np.full((len(bi), 2), np.nan, np.float32) for s in SPECS}
    for s in SPECS:
        X = FX[s]
        for k in range(5):
            for j in range(k, 5):
                tr = pdev & ~np.isin(pfold, [k, j]) & (y4 >= 0); tgt = np.flatnonzero(pdev & np.isin(pfold, [k, j]))
                pr = gbm(X[tr], y4[tr]).predict_proba(X[tgt])
                P[s][k, j, tgt, 0] = pr[:, 1]; P[s][k, j, tgt, 1] = pr[:, 3]; P[s][j, k] = P[s][k, j]
            print(f"{s} k={k} ({time.time() - t0:.0f} dtk)", flush=True)
        tr = pdev & (y4 >= 0); th = np.flatnonzero(~pdev); pr = gbm(X[tr], y4[tr]).predict_proba(X[th])
        PH[s][th, 0] = pr[:, 1]; PH[s][th, 1] = pr[:, 3]
    G = np.load(HERE / "hasil" / "spesialis_s_adaptif" / "peluang_beat_semua.npz")
    P["AF_AGG"] = np.full((5, 5, len(bi), 2), np.nan, np.float32); PH["AF_AGG"] = np.full((len(bi), 2), np.nan, np.float32)
    for k in range(5):
        for j in range(5):
            m = pfold == j; P["AF_AGG"][k, j, m, 0] = G["PS"][k, j][bi[m]]; P["AF_AGG"][k, j, m, 1] = G["PA"][k, j][bi[m]]
    m = ~pdev; PH["AF_AGG"][m, 0] = G["PSH"][bi[m]]; PH["AF_AGG"][m, 1] = G["PAH"][bi[m]]
    np.savez(OUT / "peluang_beat.npz", wi=wi, bi=bi, **{f"P_{s}": P[s] for s in SPECS}, **{f"PH_{s}": PH[s] for s in SPECS})

    # kualitas deteksi beat S (diagnostik, beat unik)
    u, first = np.unique(bi, return_index=True); isS = np.isin(sym, ["A", "a", "J", "S"])[first]
    ann = meta.database.isin(["mitdb", "svdb"]).to_numpy()[u]; dv = pdev[first]; diag = []
    for a in ["AF_AGG", *SPECS]:
        ps = np.full(len(bi), np.nan)
        for j in range(5):
            mm = pfold == j; ps[mm] = P[a][j, j, mm, 0]
        ps[~pdev] = PH[a][~pdev, 0]; pu = ps[first]
        for lab, mm in [("dev_oof", dv & ann), ("holdout", ~dv & ann)]:
            diag.append({"spesialis": a, "data": lab, "auc": roc_auc_score(isS[mm], pu[mm]), "ap": average_precision_score(isS[mm], pu[mm])})
    diag = pd.DataFrame(diag); diag.to_csv(OUT / "kualitas_beat_s.csv", index=False)
    print("== kualitas beat S\n", diag.round(3).to_string(), flush=True)

    order = np.argsort(wi, kind="stable"); bounds = np.searchsorted(wi[order], np.arange(len(g) + 1))

    def feats(s, k):
        p = np.zeros((len(bi), 2))
        for j in range(5):
            mm = pfold == j; p[mm] = np.nan_to_num(P[s][j, j, mm] if k is None else P[s][k, j, mm])
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
        for a in ARMS[1:]:
            X = feats(a, k); preds[a][val] = decide(fuser(X[trn], y[trn]), X[val], L[val], "FUSI_KUNCI_VT")
        for a in ARMS:
            rows.append({"fold": k, "arm": a, **metrics(y[val], preds[a][val])})
        print(k, {a: round(r["S"], 3) for a, r in zip(ARMS, rows[-len(ARMS):])}, flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT / "cv_per_fold.csv", index=False)
    meanf = d.groupby("arm")[["macro", "N", "S", "V", "AF", "VT", "recall_S", "recall_VT"]].mean().loc[ARMS]
    ref = d[d.arm == REF].set_index("fold")
    meanf["fold_S_naik_vs_AF_AGG"] = [int((d[d.arm == a].set_index("fold").S > ref.S).sum()) for a in ARMS]
    r0 = meanf.loc[REF]
    cand = [a for a in SPECS if meanf.loc[a, "S"] > r0.S and meanf.loc[a, "recall_VT"] >= r0.recall_VT]
    chosen = max(cand, key=lambda a: meanf.loc[a, "macro"]) if cand else REF
    print("== CV rerata fold\n", meanf.round(4).to_string(), "\nkandidat:", cand, "-> dipilih:", chosen, flush=True)
    hp = {"BASELINE": L[hold].argmax(1)}
    for a in ARMS[1:]:
        X = feats(a, None); hp[a] = decide(fuser(X[dev], y[dev]), X[hold], L[hold], "FUSI_KUNCI_VT")
    H = pd.DataFrame({a: metrics(y[hold], p) for a, p in hp.items()}).T.loc[ARMS]
    print("== HOLDOUT (konfirmatori)\n", H.round(4).to_string(), flush=True)
    meanf.round(4).join(H.add_prefix("holdout_").round(4)).to_csv(OUT / "ringkasan.csv")
    np.savez(OUT / "confusion_holdout.npz", **{a: confusion_matrix(y[hold], p, labels=range(5)) for a, p in hp.items()})
    np.savez(OUT / "prediksi_holdout.npz", window_id=g.window_id.to_numpy()[hold], y=y[hold], **hp)
    (OUT / "info.json").write_text(json.dumps({"kandidat": cand, "chosen": chosen, "detik": round(time.time() - t0)}, indent=2))
    print(f"selesai {time.time() - t0:.0f} dtk")


if __name__ == "__main__":
    main()
