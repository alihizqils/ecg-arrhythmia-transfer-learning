"""Ablasi E0-E5 tingkat beat N vs S (7 Nov 2026).

    python eksperimen_v4_svdb/fase_e_ns/ablasi_ns.py [--fitur fitur/250hz] [--hasil hasil/250hz] [--hanya E0]

Model tetap untuk semua eksperimen (HistGradientBoosting, hyperparameter dikunci di HP, seed 42) agar
hanya kombinasi fitur yang berubah. Validasi = 5 fold pasien development V4 (sama untuk semua
eksperimen); pemilihan L konteks E3 dan kriteria lolos hanya dari CV. Uji = holdout V4 (model dilatih
pada seluruh development), dijalankan setelah CV untuk eksperimen kumulatif.
Normalisasi: model pohon invarian terhadap skala monoton, sehingga tidak ada scaler; tidak ada
parameter yang dipelajari dari data validasi/uji.
Class weighting dijalankan terpisah (EXP *_bobot); tidak ada focal loss/oversampling/threshold tuning
(di luar batasan penelitian), ambang keputusan tetap 0,5.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

HERE = Path(__file__).resolve().parent
SEED = 42
HP = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
          early_stopping=False, random_state=SEED)
CONTEXTS = (5, 7, 11)
GROUPS = {"E0": ("e0_", "e0w_"), "E1": ("e1c_",), "E1o": ("e1o_",), "E2": ("e2_",), "E4": ("e4_",), "E5": ("e5_",),
          **{f"E3c{L}": (f"e3c{L}_",) for L in CONTEXTS}, **{f"E3o{L}": (f"e3o{L}_",) for L in CONTEXTS}}


def columns(names, groups):
    prefixes = tuple(p for g in groups for p in GROUPS[g])
    return [i for i, n in enumerate(names) if n.startswith(prefixes)]


def metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"N_precision": p[0], "N_recall": r[0], "N_f1": f[0], "N_support": int(s[0]),
            "S_precision": p[1], "S_recall": r[1], "S_f1": f[1], "S_support": int(s[1]),
            "accuracy": float((pred == y).mean()), "macro_f1": float(f.mean()),
            "S_auprc": float(average_precision_score(y, prob)) if y.any() else np.nan,
            "S_auroc": float(roc_auc_score(y, prob)) if 0 < y.sum() < len(y) else np.nan,
            "TN_N_to_N": int(((y == 0) & (pred == 0)).sum()), "FP_N_to_S": int(((y == 0) & (pred == 1)).sum()),
            "FN_S_to_N": int(((y == 1) & (pred == 0)).sum()), "TP_S_to_S": int(((y == 1) & (pred == 1)).sum())}


def fit_predict(Xtr, ytr, Xte, weighted):
    model = HistGradientBoostingClassifier(**HP)
    weight = None
    if weighted:  # sama dengan aturan class weighting penelitian (bobot ~ 1/frekuensi)
        freq = np.bincount(ytr) / len(ytr)
        weight = (1 / freq)[ytr]; weight = weight / weight.mean()
    model.fit(Xtr, ytr, sample_weight=weight)
    return model.predict_proba(Xte)[:, 1]


class Runner:
    def __init__(self, fitur, hasil):
        self.meta = pd.read_csv(fitur / "meta.csv", dtype={"record": str})
        z = np.load(fitur / "fitur.npz", allow_pickle=True)
        self.names = [str(n) for n in z["names"]]
        keep = self.meta.label.isin(["N", "S"]).to_numpy()
        self.meta = self.meta[keep].reset_index(drop=True)
        self.X = z["X"][keep]
        self.y = (self.meta.label == "S").to_numpy().astype(int)
        self.dev = (self.meta.split == "development").to_numpy()
        self.hold = (self.meta.split == "holdout").to_numpy()
        self.fold = self.meta.fold.to_numpy()
        self.out = hasil; self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "prob").mkdir(exist_ok=True)
        self.rows, self.fold_rows, self.times = [], [], {}

    def cv(self, name, groups, weighted=False, database=None):
        path = self.out / "prob" / f"{name}_cv.npz"
        cols = columns(self.names, groups)
        base = self.dev & (self.meta.database == database).to_numpy() if database else self.dev
        t = time.time()
        if path.exists():
            prob = np.load(path)["prob"]
        else:
            prob = np.full(len(self.y), np.nan)
            for k in range(5):
                tr, te = base & (self.fold != k), base & (self.fold == k)
                prob[te] = fit_predict(self.X[tr][:, cols], self.y[tr], self.X[te][:, cols], weighted)
            np.savez_compressed(path, prob=prob, idx=np.flatnonzero(base))
        self.times[f"{name}_cv"] = round(time.time() - t, 1)
        for k in range(5):
            te = base & (self.fold == k)
            self.fold_rows.append({"eksperimen": name, "fold": k, **metrics(self.y[te], prob[te])})
        self.record(name, "cv_development", base, prob, groups, cols, weighted, database)
        for db in ["mitdb", "svdb"]:
            part = base & (self.meta.database == db).to_numpy()
            if part.any() and not database:
                self.record(name, f"cv_development_{db}", part, prob, groups, cols, weighted, database)
        return prob

    def test(self, name, groups, weighted=False):
        path = self.out / "prob" / f"{name}_holdout.npz"
        cols = columns(self.names, groups)
        t = time.time()
        if path.exists():
            prob = np.load(path)["prob"]
        else:
            prob = np.full(len(self.y), np.nan)
            prob[self.hold] = fit_predict(self.X[self.dev][:, cols], self.y[self.dev], self.X[self.hold][:, cols], weighted)
            np.savez_compressed(path, prob=prob, idx=np.flatnonzero(self.hold))
        self.times[f"{name}_holdout"] = round(time.time() - t, 1)
        self.record(name, "holdout", self.hold, prob, groups, cols, weighted, None)
        for db in ["mitdb", "svdb"]:
            self.record(name, f"holdout_{db}", self.hold & (self.meta.database == db).to_numpy(), prob, groups, cols,
                        weighted, None)

    def record(self, name, scope, mask, prob, groups, cols, weighted, database):
        self.rows.append({"eksperimen": name, "evaluasi": scope, "kelompok": "+".join(groups), "n_fitur": len(cols),
                          "class_weighting": weighted, "data_latih": database or "mitdb+svdb",
                          **metrics(self.y[mask], prob[mask])})

    def fold_f1(self, name):
        f = pd.DataFrame(self.fold_rows)
        return f[f.eksperimen == name].sort_values("fold")[["S_f1", "N_f1"]].to_numpy()

    def compare(self, name, reference):
        a, b = self.fold_f1(name), self.fold_f1(reference)
        return {"eksperimen": name, "pembanding": reference,
                "delta_S_f1_rerata_fold": float((a[:, 0] - b[:, 0]).mean()),
                "fold_S_f1_naik": int((a[:, 0] > b[:, 0]).sum()),
                "delta_N_f1_rerata_fold": float((a[:, 1] - b[:, 1]).mean()),
                "lolos": bool((a[:, 0] > b[:, 0]).sum() >= 4 and (a[:, 1] - b[:, 1]).mean() >= -0.01)}

    def bootstrap(self, name, reference, n=1000):
        """CI 95% beda F1 S (CV gabungan) dengan resampling pasien."""
        pa = np.load(self.out / "prob" / f"{name}_cv.npz")["prob"]
        pb = np.load(self.out / "prob" / f"{reference}_cv.npz")["prob"]
        m = self.dev & np.isfinite(pa) & np.isfinite(pb)
        pat = self.meta.patient_id[m].to_numpy(); y = self.y[m]; a = pa[m] >= 0.5; b = pb[m] >= 0.5
        codes, inv = np.unique(pat, return_inverse=True)
        stats = np.zeros((len(codes), 6))
        for arr, off in [(a, 0), (b, 3)]:
            np.add.at(stats[:, off], inv, (arr & (y == 1)))       # TP
            np.add.at(stats[:, off + 1], inv, (arr & (y == 0)))   # FP
            np.add.at(stats[:, off + 2], inv, (~arr & (y == 1)))  # FN
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(n):
            s = stats[rng.integers(0, len(codes), len(codes))].sum(0)
            f = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)
            diffs.append(f(*s[:3]) - f(*s[3:]))
        return np.percentile(diffs, [2.5, 97.5]).tolist()

    def save(self, extra):
        pd.DataFrame(self.rows).to_csv(self.out / "ringkasan_eksperimen.csv", index=False)
        pd.DataFrame(self.fold_rows).to_csv(self.out / "metrik_per_fold.csv", index=False)
        (self.out / "konfigurasi.json").write_text(json.dumps({
            "seed": SEED, "model": "sklearn.HistGradientBoostingClassifier", "hyperparameter": HP,
            "ambang": 0.5, "sklearn": sklearn.__version__, "python": platform.python_version(),
            "cpu": platform.processor(), "waktu_detik": self.times, **extra}, indent=2, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitur", type=Path, default=HERE / "fitur" / "250hz")
    ap.add_argument("--hasil", type=Path, default=HERE / "hasil" / "250hz")
    ap.add_argument("--hanya", choices=["E0", "semua"], default="semua")
    args = ap.parse_args(argv)
    R = Runner(args.fitur, args.hasil)
    print("beat N/S:", np.bincount(R.y), "dev:", R.dev.sum(), "holdout:", R.hold.sum(), flush=True)
    R.cv("E0", ["E0"])
    if args.hanya == "E0":
        R.test("E0", ["E0"]); R.save({}); return
    chain = {"E0": ["E0"], "E1": ["E0", "E1"]}
    R.cv("E1", chain["E1"])
    R.cv("E1_offline", chain["E1"] + ["E1o"])
    chain["E2"] = chain["E1"] + ["E2"]
    R.cv("E2", chain["E2"])
    scores = {}
    for L in CONTEXTS:
        R.cv(f"E3_L{L}", chain["E2"] + [f"E3c{L}"])
        scores[L] = R.fold_f1(f"E3_L{L}")[:, 0].mean()
    best = max(scores, key=scores.get)   # dipilih dari CV development saja
    chain["E3"] = chain["E2"] + [f"E3c{best}"]
    R.cv("E3", chain["E3"])
    for L in CONTEXTS:
        R.cv(f"E3_offline_L{L}", chain["E2"] + ["E1o", f"E3o{L}"])
    chain["E4"] = chain["E3"] + ["E4"]
    R.cv("E4", chain["E4"])
    chain["E5"] = chain["E4"] + ["E5"]
    R.cv("E5", chain["E5"])
    # Individual terhadap E0
    for g in ["E2", f"E3c{best}", "E4", "E5"]:
        R.cv(f"E0+{g}", ["E0", g])
    # Class weighting terpisah
    R.cv("E0_bobot", ["E0"], weighted=True)
    R.cv("E5_bobot", chain["E5"], weighted=True)
    # SVDB-only
    for name in ["E0", "E5"]:
        R.cv(f"{name}_svdb_only", chain[name], database="svdb")
    comparisons = []
    for prev, cur in [("E0", "E1"), ("E1", "E2"), ("E2", "E3"), ("E3", "E4"), ("E4", "E5"), ("E1", "E1_offline"),
                      ("E3", f"E3_offline_L{best}"), ("E0", "E5"), ("E0", "E0+E2"), ("E0", f"E0+E3c{best}"),
                      ("E0", "E0+E4"), ("E0", "E0+E5"), ("E0", "E0_bobot"), ("E5", "E5_bobot")]:
        c = R.compare(cur, prev); c["ci95_delta_S_f1_bootstrap_pasien"] = R.bootstrap(cur, prev); comparisons.append(c)
    pd.DataFrame(comparisons).to_csv(R.out / "perbandingan_bertahap.csv", index=False)
    for name in ["E0", "E1", "E2", "E3", "E4", "E5"]:
        R.test(name, chain[name])
    R.test("E5_bobot", chain["E5"], weighted=True)
    R.save({"L_konteks_terpilih": best, "S_f1_cv_per_L": scores, "rantai": chain})
    print(pd.DataFrame(comparisons).to_string())


if __name__ == "__main__":
    main()
