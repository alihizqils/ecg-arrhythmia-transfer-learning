"""Uji B2: uji lintas database pada INCART dengan model final V4 (inferensi saja).

Menguji apakah keputusan model bergantung pada database sumber (confounding kelas-database). INCART (St. Petersburg
INCART 12-lead) tidak dipakai untuk melatih, memilih, atau menyetel model final. Window INCART dilabeli dengan aturan
V4 yang sama (prepare_incart_data.py): N 1.203, S 300, V 1.183, AF 40 (2 pasien), tanpa VT.

Langkah:
  1. Template rekaman bebas label untuk window INCART dari seluruh beat GQRS rekaman lengkap (definisi sama dengan
     full_record_template.py; lead utama II, lead kedua V1).
  2. Untuk setiap model final (9 konfigurasi): scaler final dihitung ulang dari window development V4 dan harus sama
     dengan scaler di checkpoint; reproduksi probabilitas hold-out V4 resmi pada 300 window; lalu inferensi CNN path
     pada seluruh window INCART dengan template bebas label.
Pakai:  python incart_external_test.py --template   lalu   python incart_external_test.py   lalu   --ringkas
Keluaran: eksperimen_v4_svdb/outputs_colab_r15/additional_tests/b2_incart/
Model lengkap (specialist + fusi) tidak dievaluasi di INCART karena paket specialist hanya berisi window V4.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V4DIR = HERE.parent / "eksperimen_v4_svdb"
for p in [str(V4DIR / "fase_e_ns"), str(V4DIR), str(HERE.parent / "src"), str(HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("V4_OUTPUT_ROOT", str(V4DIR / "outputs_colab_r15"))

import incart_features  # noqa: E402
import v4_record_template as RT  # noqa: E402

OUTV4 = V4DIR / "outputs_colab_r15"
DEST = OUTV4 / "additional_tests" / "b2_incart"
PREP = HERE / "data_v4_incart" / "prepared"
FS, WIN = 250.0, 2500
QL, QR = int(round(RT._QRS[0] * FS)), int(round(RT._QRS[1] * FS))
PL, PR = int(round(RT._PRE[0] * FS)), int(round(RT._PRE[1] * FS))
ARCHS = ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"]
C = ["N", "S", "V", "AF", "VT"]


def _templates(signal, peaks):
    q = [RT._znorm(signal[p - QL:p + QR + 1]) for p in peaks if p - QL >= 0 and p + QR < len(signal)]
    r = [RT._znorm(signal[p - PL:p - PR + 1]) for p in peaks if p - PL >= 0 and p - PR < len(signal)]
    q = [x for x in q if x is not None]; r = [x for x in r if x is not None]
    return (RT._znorm(np.median(np.stack(q), 0)) if q else None, RT._znorm(np.median(np.stack(r), 0)) if r else None)


def _window_beats(segment, peaks, qt, pt, rr_median):
    out = []
    for order, peak in enumerate(peaks):
        qrs = segment[peak - QL:peak + QR + 1] if peak - QL >= 0 and peak + QR < len(segment) else None
        pre = segment[peak - PL:peak - PR + 1] if peak - PL >= 0 and peak - PR < len(segment) else None
        rr = (peak - peaks[order - 1]) / FS if order > 0 else np.nan
        out.append((rr / rr_median if np.isfinite(rr_median) and rr_median > 0 else np.nan,
                    RT._correlation(None if qrs is None else RT._znorm(qrs), qt),
                    RT._correlation(None if pre is None else RT._znorm(pre), pt)))
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def per_record(args):
    record, channel, rows = args
    rec = incart_features.load_record_incart("incartdb", record, channel, FS, annotations=False)
    signal, peaks = rec["signal"], rec["peaks"]
    second = np.nan_to_num(np.asarray(rec["second"], np.float64))
    rr_median = float(np.median(np.diff(peaks) / FS))
    qt, pt = _templates(signal, peaks); qt2, pt2 = _templates(second, peaks)
    res = {}
    for wid, start, wpeaks in rows:
        seg = signal[start:start + WIN]
        b1 = _window_beats(seg, wpeaks - start, qt, pt, rr_median)
        b2 = _window_beats(second[start:start + WIN], wpeaks - start, qt2, pt2, rr_median)
        res[wid] = (RT._window_features(b1), RT._beat_features(b1), RT._window_features(b2), seg)
    return record, res


def templates():
    path = DEST / "template_incart_labelfree.npz"
    if path.exists():
        return path
    g = pd.read_csv(PREP / "global_manifest.csv", dtype={"window_id": str, "record": str})
    g = g[g.database == "incartdb"]
    local = pd.read_csv(PREP / "local_manifest.csv", dtype={"parent_window_id": str})
    local = local[local.parent_window_id.isin(set(g.window_id))].sort_values(["parent_window_id", "rpeak_sample"])
    pk = {k: f.rpeak_sample.to_numpy(np.int64) for k, f in local.groupby("parent_window_id")}
    jobs = [(rec, ch, [(r.window_id, int(r.start_target_fs), pk.get(r.window_id, np.empty(0, np.int64)))
                       for r in rows.itertuples(index=False)])
            for (rec, ch), rows in g.groupby(["record", "channel"], sort=False)]
    out, seg_err = {}, 0.0
    with ProcessPoolExecutor(max_workers=8) as pool:
        for rec, res in pool.map(per_record, jobs):
            wf = np.load(PREP / "records" / "incartdb" / rec / "waveforms.npy", mmap_mode="r")
            pos = {int(v): i for i, v in enumerate(np.load(PREP / "records" / "incartdb" / rec / "window_indices.npy"))}
            idx = g.set_index("window_id").loc[list(res), "index"]
            for wid, (a, b, c, seg) in res.items():
                seg_err = max(seg_err, float(np.abs(np.asarray(wf[pos[int(idx[wid])]], np.float64) - seg).max()))
                out[wid] = (a, b, c)
            print("incartdb", rec, len(res), flush=True)
    ids = g.window_id.tolist()
    DEST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, window_ids=np.asarray(ids),
                        rt=np.stack([out[i][0] for i in ids]).astype(np.float32),
                        lead2=np.stack([out[i][2] for i in ids]).astype(np.float32),
                        beat=np.concatenate([out[i][1] for i in ids]).astype(np.float32),
                        beat_offsets=np.cumsum([0] + [len(out[i][1]) for i in ids]).astype(np.int64))
    json.dump({"selisih_maks_segmen_vs_waveforms": seg_err, "n_window": len(ids)},
              open(DEST / "template_incart_info.json", "w"), indent=2)
    print("selisih maks segmen:", seg_err)
    return path


def load_new(path):
    z = np.load(path); ids = z["window_ids"].astype(str); off = z["beat_offsets"]
    return {w: (z["rt"][i], z["lead2"][i], z["beat"][off[i]:off[i + 1]]) for i, w in enumerate(ids)}


def scaler_diff(a, b):
    """Selisih absolut maksimum antar state scaler (struktur harus identik)."""
    if isinstance(a, dict):
        assert set(a) == set(b), set(a) ^ set(b)
        return max([scaler_diff(a[k], b[k]) for k in a] + [0.0])
    if isinstance(a, (list, tuple)) and a and isinstance(a[0], (dict, list, tuple, str)):
        assert len(a) == len(b)
        return max(scaler_diff(x, y) for x, y in zip(a, b))
    if isinstance(a, str) or a is None:
        assert a == b, (a, b); return 0.0
    A, B = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert A.shape == B.shape and (np.isnan(A) == np.isnan(B)).all()
    return float(np.nanmax(np.abs(A - B))) if A.size else 0.0


def infer(configs):
    import torch
    import incart_v4_patch
    import prepare_incart_data as S
    import v4_holdout
    import v4_pipeline as v4
    import v4_runner as R
    import v4_s3
    import v4_xai as X
    from ecgtl.hierarchical.training import load_checkpoint
    from holdout_labelfree import patch

    v4_holdout.install(); incart_v4_patch.install()
    p = S.protocol(); v4.initialize(p); v4_s3.initialize_s3(p); R._PROTOCOL = p
    new = load_new(DEST / "template_incart_labelfree.npz")
    for arch, sc in configs:
        dest = DEST / arch / sc
        if (dest / "complete.json").exists():
            print(arch, sc, "sudah ada"); continue
        exp = R.experiment(arch, sc)
        rec0, beat0 = exp.record_template, exp.beat_record_template
        ck = load_checkpoint(OUTV4 / "final_models" / arch / sc / "final" / "best.pt")
        v4dev = exp.dev[exp.dev.database != "incartdb"].window_id.astype(str)
        scaler = v4.V4Scaler(exp.store, v4dev, rec0, beat0)
        dif = scaler_diff(scaler.state_dict(), ck["rr_scaler"])
        if dif > 1e-6:   # urutan penjumlahan float32 dapat memberi selisih ~1e-8
            raise ValueError(f"{arch} {sc}: scaler V4 dari protokol INCART berbeda dengan checkpoint final ({dif}).")
        model = v4_holdout.build_final_model(exp, sc, ck)
        for prm in model.parameters():
            prm.requires_grad_(False)
        rows_all = exp.all_rows.set_index(exp.all_rows.window_id.astype(str))
        # reproduksi hold-out V4 resmi (template lama) pada 300 window
        ref_m = pd.read_csv(OUTV4 / "holdout" / arch / sc / "manifest.csv", dtype={"window_id": str})
        ref_p = np.load(OUTV4 / "holdout" / arch / sc / "probabilities.npy")
        sel = np.random.default_rng(0).choice(len(ref_m), 300, replace=False)
        rows = rows_all.loc[ref_m.window_id.iloc[sel]].reset_index(drop=True)
        pr = X.modality_knockout(exp, model, scaler, rows)["skor"]["gabungan"]["_prob"]
        rep = {"selisih_maks_prob": float(np.abs(pr - ref_p[sel]).max()),
               "argmax_sama": float((pr.argmax(1) == ref_p[sel].argmax(1)).mean())}
        print(arch, sc, "reproduksi hold-out", rep, flush=True)
        if rep["argmax_sama"] < 0.99:
            raise ValueError("Reproduksi hold-out V4 gagal; inferensi INCART dihentikan.")
        inc = rows_all[rows_all.database == "incartdb"]
        ids = list(inc.index)
        exp.record_template, exp.beat_record_template = patch(exp, ids, new)
        scaler.record_template, scaler.beat_record_template = exp.record_template, exp.beat_record_template
        res = X.modality_knockout(exp, model, scaler, inc.reset_index(drop=True))
        prob = res["skor"]["gabungan"].pop("_prob")
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "probabilities.npy", prob.astype(np.float32))
        pd.DataFrame({"window_id": ids, "record": inc.record.astype(str).to_numpy(),
                      "label": inc.label.to_numpy()}).to_csv(dest / "manifest.csv", index=False)
        json.dump({"reproduksi_holdout_v4": rep, "knockout": res["skor"], "shapley": res["shapley"]},
                  open(dest / "complete.json", "w"), indent=2, default=float)
        print(arch, sc, "INCART CNN path macro (5 kelas):", round(res["skor"]["gabungan"]["macro"], 4), flush=True)
        exp.record_template, exp.beat_record_template = rec0, beat0
        del model; torch.cuda.empty_cache()


def summarize():
    from sklearn.metrics import confusion_matrix
    rows, cms = [], {}
    for arch in ARCHS:
        for sc in ["S1", "S2", "S3"]:
            d = DEST / arch / sc
            if not (d / "complete.json").exists():
                continue
            m = pd.read_csv(d / "manifest.csv", dtype={"record": str}); y = m.label.map(C.index).to_numpy()
            pred = np.load(d / "probabilities.npy").argmax(1)
            cm = confusion_matrix(y, pred, labels=range(5)); cms[f"{arch}_{sc}"] = cm
            tp = np.diag(cm); rec = tp / np.maximum(cm.sum(1), 1); pre = tp / np.maximum(cm.sum(0), 1)
            f1 = np.where(tp > 0, 2 * tp / np.maximum(cm.sum(1) + cm.sum(0), 1), 0.0)
            r = {"arch": arch, "sc": sc, "macro_F1_NSV": float(f1[:3].mean())}
            for i, c in enumerate(C[:4]):
                r[f"recall_{c}"], r[f"F1_{c}"], r[f"presisi_{c}"] = rec[i], f1[i], pre[i]
            r["pred_VT"] = int(cm[:, 4].sum()); r["pred_AF_dari_nonAF"] = int(cm[:3, 3].sum())
            rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(DEST / "ringkas_incart.csv", index=False)
    np.savez(DEST / "confusion_incart.npz", **cms)
    pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    if "--template" in sys.argv:
        templates()
    elif "--ringkas" in sys.argv:
        summarize()
    else:
        cfg = [(a, s) for a in ARCHS for s in ["S1", "S2", "S3"]]
        sel = [x for x in sys.argv[1:] if ":" in x]
        if sel:
            cfg = [c for c in cfg if f"{c[0]}:{c[1]}" in sel]
        infer(cfg)
