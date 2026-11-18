"""Uji B4b: pelengkap B4, baseline tabular + specialist + fusi DENGAN kunci VT.

B4 (baseline_tabular.py) membandingkan model lengkap (berkunci VT dari CNN path) dengan TF tanpa kunci VT.
Agar setara, di sini kunci VT diambil dari prediksi VT expert tabular itu sendiri, dengan aturan yang sama dengan
model lengkap (v4_model_lengkap.decide): VT bila expert memprediksi VT, selain itu argmax fusi pada N/S/V/AF.

Langkah B4 diulang persis (seed, protokol, E* yang sama). Prediksi tabular dan TF hold-out harus identik dengan
b4_tabular_only/prediksi_holdout.npz; bila tidak, skrip berhenti. Keluaran baru: b4_tabular_only/kunci_vt/
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import baseline_tabular as B4
from baseline_tabular import M, R, load_new, patch, v4

DEST = B4.DEST / "kunci_vt"


def main():
    if (DEST / "complete.json").exists():
        print(json.loads((DEST / "complete.json").read_text())); return
    DEST.mkdir(parents=True, exist_ok=True)
    ref = np.load(B4.DEST / "prediksi_holdout.npz"); estar = json.loads((B4.DEST / "complete.json").read_text())["E*"]
    R.protocol(); exp = R.experiment("efficientnet_b0", "S2")
    rec0, beat0 = exp.record_template, exp.beat_record_template
    g = M.windows(); y = g.label_idx.to_numpy(); fold = g.fold.to_numpy()
    dev = (g.split == "development").to_numpy(); hold = (g.split == "holdout").to_numpy()
    ids = g.window_id.astype(str).to_numpy(); devrows = exp.dev.set_index(exp.dev.window_id.astype(str))
    pos = pd.Series(np.arange(len(g)), index=ids); L = np.full((len(g), 5), np.nan)
    for k in range(5):
        tr = g[dev & (fold != k)]; va = g[dev & (fold == k)]
        trr = devrows.loc[tr.window_id.astype(str)].reset_index(drop=True)
        sc = v4.V4Scaler(exp.store, trr.window_id.astype(str), rec0, beat0)
        _, probs = B4.train(exp, trr, B4.features(exp, sc, trr.window_id.astype(str)), B4.MAX_EPOCHS, 42 + k,
                            B4.features(exp, sc, va.window_id.astype(str)), va.label_idx.to_numpy())
        L[pos.loc[va.window_id.astype(str)].to_numpy()] = probs[estar - 1]
        print("fold", k, "selesai", flush=True)
    trr = devrows.loc[g[dev].window_id.astype(str)].reset_index(drop=True)
    sc = v4.V4Scaler(exp.store, trr.window_id.astype(str), rec0, beat0)
    model, _ = B4.train(exp, trr, B4.features(exp, sc, trr.window_id.astype(str)), estar, 142)
    hid = list(ids[hold]); rec1, beat1 = patch(exp, hid, load_new())
    sc.record_template, sc.beat_record_template = rec1, beat1
    import torch
    with torch.no_grad():
        Ph = model(B4.features(exp, sc, hid)).softmax(-1).numpy()
    L[hold] = Ph; L = np.log(np.clip(np.nan_to_num(L, nan=1.0), 1e-6, 1))
    Z, _ = M.paket(); assert np.array_equal(Z["window_id"], ids)
    Fh = np.hstack([L, Z["fin_W"]]); fus = M.fuser(Fh[dev], y[dev], True)
    pr = fus.predict_proba(Fh[hold]); full = np.zeros((hold.sum(), 5)); full[:, fus.classes_] = pr
    t_hold, tf_hold = L[hold].argmax(1), full.argmax(1)
    rep = {"tabular_sama": float((t_hold == ref["tabular"]).mean()), "tf_sama": float((tf_hold == ref["tabular_fusi"]).mean()),
           "prob_selisih_maks": float(np.abs(Ph - ref["prob_tabular"]).max())}
    print("reproduksi B4", rep, flush=True)
    if rep["tabular_sama"] < 1 or rep["tf_sama"] < 1:
        raise ValueError("B4 tidak tereproduksi; berhenti.")
    tfk = M.decide(fus, Fh[hold], L[hold]); yh = y[hold]; pid = ref["pasien"]
    np.savez(DEST / "prediksi_holdout.npz", window_id=np.array(hid), y=yh, tabular_fusi_kunci_vt=tfk)
    M.per_kelas(yh, tfk)[1].to_csv(DEST / "holdout_per_kelas_tabular_fusi_kunci_vt.csv")
    comp = []
    for arch in B4.ARCHS if hasattr(B4, "ARCHS") else ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"]:
        z = np.load(B4.OUT / "main_results" / "model_lengkap_labelfree" / arch / "S2" / "prediksi.npz")
        assert np.array_equal(z["window_id_holdout"], np.array(hid))
        T = M._boot(yh, z["lengkap_holdout"], tfk, pid); T.insert(0, "banding", f"model lengkap {arch} - tabular_fusi_kunci_vt")
        comp.append(T)
    pd.concat(comp).to_csv(DEST / "banding_holdout.csv", index=False)
    res = {"reproduksi_B4": rep, "holdout_tabular_fusi_kunci_vt": M.ringkas(yh, tfk), "holdout_dipakai_untuk_seleksi": False}
    res = json.loads(json.dumps(res, default=float))
    (DEST / "complete.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
