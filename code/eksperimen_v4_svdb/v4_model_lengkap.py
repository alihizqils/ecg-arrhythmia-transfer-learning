"""Model lengkap MA1 per skenario: CNN + Beat-Level Specialist + fusi + kunci VT (di luar hash metode CNN).

Dipanggil notebook setelah CNN satu skenario selesai (CV, E*, model final, holdout CNN):

    import v4_model_lengkap as M
    M.jalankan("efficientnet_b0", "S2")      # dev (CV nested) + holdout model lengkap
    M.bandingkan("efficientnet_b0", "S2", "S1")

Resep = MA1 (fase_e_ns/spesialis_s_pasien.py, e6_bobot_ujung_ke_ujung.py, e6_holdout_s1_s2.py), disalin persis:
  - specialist: peluang beat dari paket terkunci model_lengkap_data/spesialis_r15.npz (13 agregat per window);
  - CNN: log-probabilitas out-of-fold pada E* resmi (receipt selection/<arch>/<sc>_final_epoch.json), untuk holdout
    probabilitas model final (jumlah epoch wajib = E*);
  - fusi: StandardScaler + LogisticRegression(C=1, max_iter=2000); dev = fold luar k dilatih pada fold dev lain;
    holdout = dilatih pada seluruh window development;
  - keputusan FUSI_KUNCI_VT: bila CNN memprediksi VT, keluaran VT; selain itu argmax fusi atas N/S/V/AF.
Class weighting per skenario (konsisten di semua tahap): S1 tanpa bobot; S2 dan S3 berbobot (1/frekuensi)^0,5
pada specialist dan fusi.
Lanjut setelah terputus: hasil ditulis atomik ke <out>/model_lengkap/<arch>/<sc>/ dengan complete.json berisi hash
semua masukan. Bila sudah ada dan hash cocok, hasil hanya dibaca; bila tidak cocok, berhenti (tidak menimpa).
Holdout tidak dipakai untuk memilih apa pun.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
MANIFEST = CODE / "data" / "processed" / "v4_svdb" / "prepared" / "global_manifest.csv"
PAKET = HERE / "model_lengkap_data" / "spesialis_r15.npz"
PAKET_INFO = HERE / "model_lengkap_data" / "spesialis_r15.json"
C = ["N", "S", "V", "AF", "VT"]
BOBOT = {"S1": False, "S2": True, "S3": True}
VERSI = "model_lengkap_r15_v1"


# ---- resep MA1 (disalin persis dari fase_e_ns) ----
def sqrt_w(y):
    return pd.Series(y).map(1 / pd.Series(y).value_counts(normalize=True)).to_numpy() ** 0.5


def fuser(X, y, weighted):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    m.fit(X, y, **({"logisticregression__sample_weight": sqrt_w(y)} if weighted else {}))
    return m


def decide(model, X, L):
    """FUSI_KUNCI_VT (spesialis_s_v2.decide, arm FUSI_KUNCI_VT)."""
    pr = model.predict_proba(X); full = np.full((len(X), 5), -np.inf)
    full[:, model.classes_] = np.log(np.clip(pr, 1e-9, 1)); cnn = L.argmax(1)
    return np.where(cnn == 4, 4, full[:, :4].argmax(1))


# ---- metrik ----
def per_kelas(y, p):
    cm = confusion_matrix(y, p, labels=range(5)); tp = np.diag(cm); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp
    with np.errstate(all="ignore"):
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0); pre = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return cm, pd.DataFrame({"presisi": pre, "recall": rec, "F1": f1, "n": cm.sum(1)}, index=C)


def ringkas(y, p):
    _, t = per_kelas(y, p)
    return {"macro_F1": t.F1.mean(), "F1_N": t.F1["N"], "F1_S": t.F1["S"], "recall_S": t.recall["S"],
            "presisi_S": t.presisi["S"], "F1_V": t.F1["V"], "F1_AF": t.F1["AF"], "F1_VT": t.F1["VT"],
            "recall_VT": t.recall["VT"]}


# ---- utilitas ----
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _out_default():
    import v4_runner as R
    return R.protocol().out


def windows():
    g = pd.read_csv(MANIFEST, dtype={"window_id": str, "record": str})
    return g[~((g.label == "N") & g.database.isin(["afdb", "vfdb"]))].reset_index(drop=True)


def paket():
    info = json.loads(PAKET_INFO.read_text())
    if _sha(PAKET) != info["sha256"]:
        raise ValueError("Paket specialist (spesialis_r15.npz) tidak cocok dengan sha256 tercatat.")
    return np.load(PAKET), info["sha256"]


def estar_resmi(out, arch, sc):
    """E* dari receipt resmi, diperiksa ulang dengan aturan yang sama seperti v4_pipeline.select_final_epoch:
    epoch terkecil yang rerata Macro-F1 validasinya (epoch bersama kelima fold) >= maksimum - tie_tolerance.

    Revisi 13 Nov 2026: versi sebelumnya memakai argmax murni tanpa tie_tolerance (0,0001) sehingga menolak
    receipt sah EfficientNet-B0 S3 (receipt 4, argmax murni 6). S1/S2 MA1 (7 dan 20) tidak berubah."""
    from sklearn.metrics import f1_score
    receipt = json.loads((out / "selection" / arch / f"{sc}_final_epoch.json").read_text())
    estar = int(receipt["selected_epoch"]); tol = float(receipt["tie_tolerance"]); curves = []
    if tol != 0.0001:
        raise ValueError(f"tie_tolerance receipt {tol} berbeda dari metode terkunci (0,0001).")
    for k in range(5):
        d = out / "cv" / arch / sc / f"fold_{k}"
        y = np.load(d / "labels.npy")
        curves.append({int(e.stem.split("_")[1]): f1_score(y, np.load(e).argmax(1), labels=range(5), average="macro",
                                                            zero_division=0)
                       for e in sorted((d / "epoch_probabilities").glob("epoch_*.npy"))})
    common = sorted(set.intersection(*[set(c) for c in curves]))
    means = {e: float(np.mean([c[e] for c in curves])) for e in common}
    best = max(means.values())
    rule = min(e for e in common if means[e] >= best - tol - 1e-12)
    if rule != estar:
        raise ValueError(f"Receipt E* {sc} = {estar} tidak sama dengan aturan resmi ({rule}).")
    return estar


def _inputs(out, arch, sc, estar):
    files = [out / "cv" / arch / sc / f"fold_{k}" / "epoch_probabilities" / f"epoch_{estar:03d}.npy" for k in range(5)]
    files += [out / "cv" / arch / sc / f"fold_{k}" / "manifest.csv" for k in range(5)]
    files += [out / "holdout" / arch / sc / "probabilities.npy", out / "holdout" / arch / sc / "manifest.csv",
              out / "selection" / arch / f"{sc}_final_epoch.json"]
    return {str(f.relative_to(out)).replace("\\", "/"): _sha(f) for f in files}


def cnn_logprob(out, arch, sc, g, estar):
    """Dev: log-prob out-of-fold pada E* (window tanpa prediksi -> 0, sama dengan MA1); holdout: model final."""
    pos = pd.Series(np.arange(len(g)), index=g.window_id); L = np.full((len(g), 5), np.nan)
    for k in range(5):
        d = out / "cv" / arch / sc / f"fold_{k}"
        m = pd.read_csv(d / "manifest.csv", dtype={"window_id": str})
        L[pos.loc[m.window_id].to_numpy()] = np.load(d / "epoch_probabilities" / f"epoch_{estar:03d}.npy")
    L = np.log(np.clip(np.nan_to_num(L, nan=1.0), 1e-6, 1))
    hd = out / "holdout" / arch / sc
    if not (hd / "complete.json").exists():
        raise FileNotFoundError(f"Holdout CNN {arch} {sc} belum selesai ({hd}).")
    m = pd.read_csv(hd / "manifest.csv", dtype={"window_id": str}); Ph = np.full((len(g), 5), np.nan)
    Ph[pos.loc[m.window_id].to_numpy()] = np.load(hd / "probabilities.npy")
    hold = (g.split == "holdout").to_numpy()
    if np.isnan(Ph[hold]).any():
        raise ValueError("Sebagian window holdout tidak punya probabilitas CNN.")
    L[hold] = np.log(np.clip(Ph[hold], 1e-6, 1))
    return L


def jalankan(arch, sc, out=None, hasil=None):
    """Model lengkap satu skenario. Idempoten: dibaca ulang bila sudah selesai dengan masukan yang sama."""
    out = Path(out) if out else _out_default()
    folder = Path(hasil) if hasil else out / "model_lengkap" / arch / sc
    estar = estar_resmi(out, arch, sc)
    final = json.loads((out / "final_models" / arch / sc / "final" / "result.json").read_text())
    if int(final["epochs_run"]) != estar:
        raise ValueError(f"Model final {arch} {sc}: {final['epochs_run']} epoch, padahal E* = {estar}.")
    Z, paket_sha = paket()
    meta = {"versi": VERSI, "arch": arch, "skenario": sc, "E*": estar, "berbobot": BOBOT[sc],
            "paket_specialist_sha256": paket_sha, "checkpoint_final_sha256": final.get("checkpoint_sha256"),
            "masukan": _inputs(out, arch, sc, estar)}
    done = folder / "complete.json"
    if done.exists():
        old = json.loads(done.read_text())
        if {k: old.get(k) for k in meta} != meta:
            raise ValueError(f"{folder} sudah ada tetapi masukannya berbeda; tidak ditimpa. Periksa perubahan.")
        return old

    g = windows()
    if not np.array_equal(Z["window_id"], g.window_id.to_numpy().astype(str)):
        raise ValueError("Urutan window paket specialist berbeda dari manifest.")
    y = g.label_idx.to_numpy(); fold = g.fold.to_numpy()
    dev = (g.split == "development").to_numpy(); hold = (g.split == "holdout").to_numpy()
    L = cnn_logprob(out, arch, sc, g, estar); w = BOBOT[sc]; v = "W" if w else "U"

    pred_dev = np.full(len(g), -1); rows = []
    for k in range(5):
        F = np.hstack([L, Z[f"cv_{v}"][k]]); trn, val = dev & (fold != k), dev & (fold == k)
        pred_dev[val] = decide(fuser(F[trn], y[trn], w), F[val], L[val])
        rows.append({"fold": k, "model": "lengkap", **ringkas(y[val], pred_dev[val])})
        rows.append({"fold": k, "model": "CNN saja", **ringkas(y[val], L[val].argmax(1))})
    F = np.hstack([L, Z[f"fin_{v}"]])
    pred_hold = decide(fuser(F[dev], y[dev], w), F[hold], L[hold])
    yh = y[hold]; cnn_hold = L[hold].argmax(1)

    tmp = folder.with_name(folder.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)      # sisa tulisan yang terputus (belum pernah lengkap)
    tmp.mkdir(parents=True)
    T = pd.DataFrame(rows); T.to_csv(tmp / "cv_per_fold.csv", index=False)
    dev_mean = T.groupby("model")[list(ringkas(yh, pred_hold))].mean()
    dev_mean.to_csv(tmp / "cv_rerata.csv")
    _, ph = per_kelas(yh, pred_hold); ph.to_csv(tmp / "holdout_per_kelas.csv")
    np.save(tmp / "confusion_holdout.npy", confusion_matrix(yh, pred_hold, labels=range(5)))
    np.savez(tmp / "prediksi.npz", window_id_dev=g.window_id.to_numpy()[dev].astype(str), y_dev=y[dev], lengkap_dev=pred_dev[dev],
             cnn_dev=L[dev].argmax(1), window_id_holdout=g.window_id.to_numpy()[hold].astype(str), y_holdout=yh,
             lengkap_holdout=pred_hold, cnn_holdout=cnn_hold,
             pasien_holdout=(g.database + "/" + g.record).to_numpy()[hold].astype(str))
    result = {**meta, "dev_lengkap": dev_mean.loc["lengkap"].to_dict(), "dev_cnn_saja": dev_mean.loc["CNN saja"].to_dict(),
              "holdout_lengkap": ringkas(yh, pred_hold), "holdout_cnn_saja": ringkas(yh, cnn_hold),
              "n_window_holdout": int(hold.sum()), "holdout_dipakai_untuk_seleksi": False}
    result = json.loads(json.dumps(result, default=float))
    (tmp / "complete.json").write_text(json.dumps(result, indent=2))
    if folder.exists():
        raise ValueError(f"{folder} ada tanpa complete.json (bukan dari modul ini?); tidak ditimpa.")
    os.replace(tmp, folder)
    return result


# ---- perbandingan antar skenario (holdout, bootstrap per pasien) ----
UJI = ["macro_F1", "F1_S", "recall_S", "presisi_S", "F1_V", "F1_AF", "recall_VT", "F1_N"]  # keluarga Holm (= audit_s1_s2)


def _boot(y, a, b, pid, n=2000, seed=0):
    rng = np.random.default_rng(seed); ids = np.unique(pid); idx = {i: np.flatnonzero(pid == i) for i in ids}
    ra, rb = ringkas(y, a), ringkas(y, b); base = {k: ra[k] - rb[k] for k in UJI}; D = []
    for _ in range(n):
        s = np.concatenate([idx[i] for i in rng.choice(ids, len(ids))]); xa, xb = ringkas(y[s], a[s]), ringkas(y[s], b[s])
        D.append({k: xa[k] - xb[k] for k in UJI})
    D = pd.DataFrame(D); rows = []
    for k in base:
        p2 = 2 * min((D[k] <= 0).mean(), (D[k] >= 0).mean())
        rows.append({"metrik": k, "selisih": base[k], "ci95_bawah": D[k].quantile(.025), "ci95_atas": D[k].quantile(.975),
                     "p_dua_sisi": min(1.0, p2)})
    T = pd.DataFrame(rows); p = T.p_dua_sisi.to_numpy(); order = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    T["p_holm"] = adj
    return T


def bandingkan(arch, a, b, out=None, hasil=None, model="lengkap"):
    """Selisih a - b di holdout (bootstrap per pasien 2.000 ulangan, seed 0, koreksi Holm). Ditulis sekali."""
    out = Path(out) if out else _out_default(); root = Path(hasil) if hasil else out / "model_lengkap" / arch
    target = root / f"banding_{a}_minus_{b}_{model}.csv"
    if target.exists():
        return pd.read_csv(target)
    za, zb = (np.load(root / s / "prediksi.npz") for s in (a, b))
    if not np.array_equal(za["window_id_holdout"], zb["window_id_holdout"]):
        raise ValueError("Window holdout kedua skenario berbeda.")
    T = _boot(za["y_holdout"], za[f"{model}_holdout"], zb[f"{model}_holdout"], za["pasien_holdout"])
    tmp = target.with_name(target.name + ".tmp"); T.to_csv(tmp, index=False); os.replace(tmp, target)
    return T
