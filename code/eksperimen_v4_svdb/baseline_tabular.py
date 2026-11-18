"""Uji B4: baseline tanpa citra yang dilatih terpisah.

Penghapusan modalitas setelah training tidak sama dengan model yang dilatih tanpa modalitas itu.
Tidak ada training CNN. Dua baseline, keduanya tidak bergantung backbone:

  T  = expert tabular saja: arsitektur MLP yang sama dengan expert tabular di CNN path (64-64-5, ReLU, dropout 0,2),
       masukan fitur tabular window yang sama (RR, fisiologi, gelombang, blok template rekaman), dilatih sendiri.
  TF = T + Beat-Level Specialist + fusi regresi logistik (resep model lengkap, tanpa kunci VT karena kunci VT
       membutuhkan prediksi VT jalur citra).

Protokol mengikuti S2 (backbone beku, tau = 2): bobot kelas inverse_weights(tau=2), sampler porsi kelas asli dengan
keseimbangan pasien di dalam kelas, AdamW lr 1e-3 wd 1e-4, cosine T_max 60, EMA half-life 1,5 epoch, batch 32,
clip gradien 5, early stopping patience 10 dengan minimum 20 epoch, seed 42 + fold (final 142). E* = epoch bersama
terkecil yang rerata Macro-F1 validasi lima fold >= maksimum - 1e-4. Model final dilatih pada seluruh development
selama E* epoch. Scaler dihitung dari data latih dengan template window terpilih (sama dengan CNN path);
hold-out memakai template bebas label (sama dengan semua hasil hold-out paper). Hold-out tidak dipakai untuk memilih.

Keluaran (baru, tidak menimpa): outputs_colab_r15/additional_tests/b4_tabular_only/
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn

HERE = Path(__file__).resolve().parent
os.environ.setdefault("V4_OUTPUT_ROOT", str(HERE / "outputs_colab_r15"))
sys.path.insert(0, str(HERE))
import v4_model_lengkap as M  # noqa: E402
import v4_pipeline as v4  # noqa: E402
import v4_runner as R  # noqa: E402
from ecgtl.hierarchical.model import inverse_weights  # noqa: E402
from holdout_labelfree import load_new, patch  # noqa: E402

OUT = HERE / "outputs_colab_r15"
DEST = OUT / "additional_tests" / "b4_tabular_only"
MAX_EPOCHS, MIN_EPOCHS, PATIENCE, BATCH, LR, WD, HALF_LIFE, TOL = 60, 20, 10, 32, 1e-3, 1e-4, 1.5, 1e-4
TAU = 2.0


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def features(exp, scaler, ids):
    rows = []
    for w in ids:
        g_rr, _ = exp.store.by_window[w]
        pg, _ = exp.store.physiology_by_window[w]
        wg, _ = exp.store.wave_by_window[w]
        rows.append(np.concatenate([scaler.transform_global(g_rr),
                                    scaler.transform_extra("physiology", "global", pg),
                                    scaler.transform_extra("wave", "global", wg),
                                    scaler.transform_record(w)]))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def mlp(n_in):
    return nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 5))


def ema_update(ema, model, decay):
    with torch.no_grad():
        for e, m in zip(ema.parameters(), model.parameters()):
            e.mul_(decay).add_(m, alpha=1 - decay)


def train(exp, rows, X, epochs, seed, Xv=None, yv=None):
    """Mengembalikan model EMA akhir dan probabilitas validasi per epoch (bila ada data validasi)."""
    seed_all(seed)
    y = torch.tensor(rows.label_idx.to_numpy())
    w = inverse_weights(np.bincount(rows.label_idx.to_numpy(), minlength=5), TAU)
    p = torch.tensor(v4.V4Experiment.sampling_weights(rows.reset_index(drop=True)), dtype=torch.double)
    model = mlp(X.shape[1]); ema = copy.deepcopy(model); ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    steps = -(-len(rows) // BATCH); decay = 0.5 ** (1.0 / (HALF_LIFE * steps)); updates = 0
    probs, best, best_epoch = [], -1.0, 0
    for epoch in range(epochs):
        if Xv is not None and epoch >= MIN_EPOCHS and epoch - best_epoch >= PATIENCE:
            break
        g = torch.Generator().manual_seed(seed + epoch)
        idx = torch.multinomial(p, len(rows), replacement=True, generator=g)
        model.train()
        for b in range(0, len(idx), BATCH):
            i = idx[b:b + BATCH]
            loss = F.cross_entropy(model(X[i]), y[i], weight=w)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            updates += 1; ema_update(ema, model, min(decay, (1.0 + updates) / (10.0 + updates)))
        sch.step(); ema.eval()
        if Xv is not None:
            with torch.no_grad():
                pr = ema(Xv).softmax(-1).numpy()
            probs.append(pr)
            score = f1_score(yv, pr.argmax(1), labels=range(5), average="macro", zero_division=0)
            if score > best:
                best, best_epoch = score, epoch + 1
    return ema, probs


def main():
    if (DEST / "complete.json").exists():
        print(json.loads((DEST / "complete.json").read_text())); return
    DEST.mkdir(parents=True, exist_ok=True)
    R.protocol(); exp = R.experiment("efficientnet_b0", "S2")
    rec0, beat0 = exp.record_template, exp.beat_record_template
    g = M.windows(); y = g.label_idx.to_numpy(); fold = g.fold.to_numpy()
    dev = (g.split == "development").to_numpy(); hold = (g.split == "holdout").to_numpy()
    ids = g.window_id.astype(str).to_numpy()
    devrows = exp.dev.set_index(exp.dev.window_id.astype(str))

    # CV: probabilitas validasi per epoch
    curves, oof = [], {}
    for k in range(5):
        tr = g[dev & (fold != k)]; va = g[dev & (fold == k)]
        trr = devrows.loc[tr.window_id.astype(str)].reset_index(drop=True)
        assert not set(trr.patient_id) & set(devrows.loc[va.window_id.astype(str)].patient_id)
        sc = v4.V4Scaler(exp.store, trr.window_id.astype(str), rec0, beat0)
        _, probs = train(exp, trr, features(exp, sc, trr.window_id.astype(str)), MAX_EPOCHS, 42 + k,
                         features(exp, sc, va.window_id.astype(str)), va.label_idx.to_numpy())
        oof[k] = probs
        curves.append({e + 1: f1_score(va.label_idx, pr.argmax(1), labels=range(5), average="macro", zero_division=0)
                       for e, pr in enumerate(probs)})
        print(f"fold {k}: {len(probs)} epoch, terbaik {max(curves[-1].values()):.4f}", flush=True)
    common = sorted(set.intersection(*[set(c) for c in curves]))
    means = {e: float(np.mean([c[e] for c in curves])) for e in common}
    estar = min(e for e in common if means[e] >= max(means.values()) - TOL - 1e-12)
    pd.DataFrame([{"fold": k, "epoch": e, "macro_F1": v} for k, c in enumerate(curves) for e, v in c.items()]) \
        .to_csv(DEST / "cv_kurva.csv", index=False)

    # log-prob out-of-fold pada E* untuk fusi (sama dengan resep model lengkap)
    pos = pd.Series(np.arange(len(g)), index=ids); L = np.full((len(g), 5), np.nan)
    cv_rows = []
    for k in range(5):
        va = g[dev & (fold == k)]; pr = oof[k][estar - 1]
        L[pos.loc[va.window_id.astype(str)].to_numpy()] = pr
        cv_rows.append({"fold": k, **M.ringkas(va.label_idx.to_numpy(), pr.argmax(1))})
    pd.DataFrame(cv_rows).to_csv(DEST / "cv_tabular_per_fold.csv", index=False)

    # model final pada seluruh development selama E* epoch; hold-out dengan template bebas label
    trr = devrows.loc[g[dev].window_id.astype(str)].reset_index(drop=True)
    sc = v4.V4Scaler(exp.store, trr.window_id.astype(str), rec0, beat0)
    model, _ = train(exp, trr, features(exp, sc, trr.window_id.astype(str)), estar, 142)
    hid = list(ids[hold]); rec1, beat1 = patch(exp, hid, load_new())
    sc.record_template, sc.beat_record_template = rec1, beat1
    with torch.no_grad():
        Ph = model(features(exp, sc, hid)).softmax(-1).numpy()
    L[hold] = Ph
    L = np.log(np.clip(np.nan_to_num(L, nan=1.0), 1e-6, 1))

    Z, _ = M.paket(); assert np.array_equal(Z["window_id"], ids)
    Fh = np.hstack([L, Z["fin_W"]]); fus = M.fuser(Fh[dev], y[dev], True)
    pr = fus.predict_proba(Fh[hold]); full = np.zeros((hold.sum(), 5)); full[:, fus.classes_] = pr
    tf_hold = full.argmax(1); t_hold = L[hold].argmax(1); yh = y[hold]
    pid = (g.database + "/" + g.record).to_numpy()[hold].astype(str)
    np.savez(DEST / "prediksi_holdout.npz", window_id=np.array(hid), y=yh, tabular=t_hold, tabular_fusi=tf_hold,
             pasien=pid, prob_tabular=Ph)
    for name, p_ in [("tabular", t_hold), ("tabular_fusi", tf_hold)]:
        M.per_kelas(yh, p_)[1].to_csv(DEST / f"holdout_per_kelas_{name}.csv")

    # perbandingan berpasangan dengan model ber-citra (S2, hold-out bebas label), bootstrap per pasien + Holm
    comp = []
    for arch in ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"]:
        z = np.load(OUT / "main_results" / "model_lengkap_labelfree" / arch / "S2" / "prediksi.npz")
        assert np.array_equal(z["window_id_holdout"], np.array(hid))
        for a_name, a, b_name, b in [("CNN path", z["cnn_holdout"], "tabular", t_hold),
                                     ("model lengkap", z["lengkap_holdout"], "tabular_fusi", tf_hold)]:
            T = M._boot(yh, a, b, pid); T.insert(0, "banding", f"{a_name} {arch} - {b_name}")
            comp.append(T)
    pd.concat(comp).to_csv(DEST / "banding_holdout.csv", index=False)
    res = {"E*": estar, "cv_tabular_rerata": pd.DataFrame(cv_rows).drop(columns="fold").mean().to_dict(),
           "holdout_tabular": M.ringkas(yh, t_hold), "holdout_tabular_fusi": M.ringkas(yh, tf_hold),
           "holdout_dipakai_untuk_seleksi": False}
    res = json.loads(json.dumps(res, default=float))
    (DEST / "complete.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
