"""Uji B4c: variasi seed model final baseline tanpa citra (pasangan U4 untuk model ber-citra).

Hanya seed training model final yang diganti (1042 dan 2042, sama dengan U4). CV, E* (= 8), scaler, log-prob
out-of-fold untuk fusi, dan fuser tetap resmi B4, seperti U4 yang hanya melatih ulang model final dengan E* resmi.
Seed resmi 142 ikut dilatih ulang sebagai kontrol: prediksi hold-out tabular, TF, dan TF berkunci VT harus identik
dengan B4/B4b. Bila tidak, skrip berhenti sebelum menulis hasil seed lain.

Per seed dihitung: tabular-only (T), TF tanpa kunci VT, TF dengan kunci VT (aturan v4_model_lengkap.decide), lalu
selisih model lengkap resmi (S2, hold-out bebas label) - TF berkunci VT, bootstrap per pasien + Holm (sama dengan B4b).
Hold-out tidak dipakai untuk memilih apa pun.

Keluaran baru (tidak menimpa B4/B4b): outputs_colab_r15/additional_tests/b4_tabular_only/seed_final/
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import baseline_tabular as B4
from baseline_tabular import M, R, load_new, patch, v4

DEST = B4.DEST / "seed_final"
SEED_RESMI, SEEDS = 142, (1042, 2042)
ARCHS = ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"]


def main():
    if (DEST / "complete.json").exists():
        print((DEST / "complete.json").read_text()); return
    DEST.mkdir(parents=True, exist_ok=True)
    ref = np.load(B4.DEST / "prediksi_holdout.npz")
    ref_k = np.load(B4.DEST / "kunci_vt" / "prediksi_holdout.npz")
    estar = json.loads((B4.DEST / "complete.json").read_text())["E*"]
    R.protocol(); exp = R.experiment("efficientnet_b0", "S2")
    rec0, beat0 = exp.record_template, exp.beat_record_template
    g = M.windows(); y = g.label_idx.to_numpy(); fold = g.fold.to_numpy()
    dev = (g.split == "development").to_numpy(); hold = (g.split == "holdout").to_numpy()
    ids = g.window_id.astype(str).to_numpy(); devrows = exp.dev.set_index(exp.dev.window_id.astype(str))

    # CV resmi B4 (seed 42 + fold), hanya untuk log-prob out-of-fold pada E* resmi
    pos = pd.Series(np.arange(len(g)), index=ids); Ldev = np.full((len(g), 5), np.nan)
    for k in range(5):
        tr = g[dev & (fold != k)]; va = g[dev & (fold == k)]
        trr = devrows.loc[tr.window_id.astype(str)].reset_index(drop=True)
        sc = v4.V4Scaler(exp.store, trr.window_id.astype(str), rec0, beat0)
        _, probs = B4.train(exp, trr, B4.features(exp, sc, trr.window_id.astype(str)), B4.MAX_EPOCHS, 42 + k,
                            B4.features(exp, sc, va.window_id.astype(str)), va.label_idx.to_numpy())
        Ldev[pos.loc[va.window_id.astype(str)].to_numpy()] = probs[estar - 1]
        print("fold", k, "selesai", flush=True)

    trr = devrows.loc[g[dev].window_id.astype(str)].reset_index(drop=True)
    sc = v4.V4Scaler(exp.store, trr.window_id.astype(str), rec0, beat0)
    Xdev = B4.features(exp, sc, trr.window_id.astype(str))
    hid = list(ids[hold]); rec1, beat1 = patch(exp, hid, load_new())
    sc.record_template, sc.beat_record_template = rec1, beat1
    Xh = B4.features(exp, sc, hid)
    Z, _ = M.paket(); assert np.array_equal(Z["window_id"], ids)
    assert np.array_equal(ref["window_id"], np.array(hid))
    yh = y[hold]; pid = ref["pasien"]
    full_models = {}
    for arch in ARCHS:
        z = np.load(B4.OUT / "main_results" / "model_lengkap_labelfree" / arch / "S2" / "prediksi.npz")
        assert np.array_equal(z["window_id_holdout"], np.array(hid)); full_models[arch] = z["lengkap_holdout"]

    fus = None; ringkas_rows, comp, kontrol = [], [], None
    for seed in (SEED_RESMI,) + SEEDS:
        model, _ = B4.train(exp, trr, Xdev, estar, seed)
        with torch.no_grad():
            Ph = model(Xh).softmax(-1).numpy()
        L = Ldev.copy(); L[hold] = Ph; L = np.log(np.clip(np.nan_to_num(L, nan=1.0), 1e-6, 1))
        Fh = np.hstack([L, Z["fin_W"]])
        f_seed = M.fuser(Fh[dev], y[dev], True)      # bergantung hanya pada data development (out-of-fold), sama tiap seed
        if fus is None:
            fus = f_seed
        coef_sama = bool(np.array_equal(fus[-1].coef_, f_seed[-1].coef_))
        assert coef_sama, "fuser berubah antar seed (seharusnya hanya bergantung pada log-prob out-of-fold)"
        pr = fus.predict_proba(Fh[hold]); fullp = np.zeros((hold.sum(), 5)); fullp[:, fus.classes_] = pr
        t, tf, tfk = L[hold].argmax(1), fullp.argmax(1), M.decide(fus, Fh[hold], L[hold])
        if seed == SEED_RESMI:
            kontrol = {"tabular_sama": float((t == ref["tabular"]).mean()),
                       "tf_sama": float((tf == ref["tabular_fusi"]).mean()),
                       "tf_kunci_vt_sama": float((tfk == ref_k["tabular_fusi_kunci_vt"]).mean()),
                       "prob_selisih_maks": float(np.abs(Ph - ref["prob_tabular"]).max())}
            print("kontrol seed 142 vs B4/B4b", kontrol, flush=True)
            if min(kontrol["tabular_sama"], kontrol["tf_sama"], kontrol["tf_kunci_vt_sama"]) < 1:
                raise ValueError("B4/B4b tidak tereproduksi pada seed 142; berhenti.")
        d = DEST / f"seed{seed}"; d.mkdir(exist_ok=True)
        np.savez(d / "prediksi_holdout.npz", window_id=np.array(hid), y=yh, tabular=t, tabular_fusi=tf,
                 tabular_fusi_kunci_vt=tfk, prob_tabular=Ph, pasien=pid)
        for name, p_ in [("tabular", t), ("tabular_fusi", tf), ("tabular_fusi_kunci_vt", tfk)]:
            ringkas_rows.append({"seed_final": seed, "baseline": name, **M.ringkas(yh, p_)})
        for arch in ARCHS:
            T = M._boot(yh, full_models[arch], tfk, pid)
            T.insert(0, "banding", f"model lengkap {arch} - tabular_fusi_kunci_vt"); T.insert(0, "seed_final", seed)
            comp.append(T)
        print(f"seed {seed}: tabular {M.ringkas(yh, t)['macro_F1']:.4f}, TF {M.ringkas(yh, tf)['macro_F1']:.4f}, "
              f"TF+kunci VT {M.ringkas(yh, tfk)['macro_F1']:.4f}", flush=True)

    R_ = pd.DataFrame(ringkas_rows); R_.to_csv(DEST / "ringkas_per_seed.csv", index=False)
    B = pd.concat(comp); B.to_csv(DEST / "banding_holdout_per_seed.csv", index=False)
    rentang = R_.groupby("baseline")["macro_F1"].agg(["min", "max", "mean", "std"]).to_dict(orient="index")
    mac = B[B.metrik == "macro_F1"]
    res = {"E*": estar, "seed_final": [SEED_RESMI, *SEEDS], "kontrol_seed_142": kontrol,
           "macro_F1_holdout_per_baseline": rentang,
           "model_lengkap_minus_tf_kunci_vt_macro": mac[["seed_final", "banding", "selisih", "ci95_bawah", "ci95_atas",
                                                        "p_holm"]].to_dict(orient="records"),
           "semua_p_holm_macro_lt_0.05": bool((mac.p_holm < 0.05).all()),
           "kode_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "holdout_dipakai_untuk_seleksi": False}
    res = json.loads(json.dumps(res, default=float))
    (DEST / "complete.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
