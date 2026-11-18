"""Uji B1: latensi dan memori seluruh pipeline model lengkap di CPU lokal.

Tabel efisiensi paper hanya mengukur CNN path (citra + fitur tabular -> probabilitas). Skrip ini mengukur tahap
lain dengan fungsi asli pipeline, pada rekaman hold-out lengkap (setting offline, satu proses, tanpa paralelisasi):

  1 baca + filter band-pass + resampling + deteksi beat GQRS            (per rekaman)
  2 rendering citra global dan crop lokal (window_views)                (per window)
  3 fitur tabular window: RR, fisiologi, delineasi gelombang            (per window)
  4 template rekaman bebas label + lead kedua (full_record_template.per_record)  (per rekaman)
  5 ritme lokal (v4_local_rhythm.record_features + window_features)     (per rekaman)
  6 fitur beat specialist (59 fitur e0/e1c/e2) + fitur klaster per rekaman (KL, MiniBatchKMeans)  (per rekaman)
  7 prediksi GBM specialist per beat + 13 agregat window                (per rekaman)
  8 fusi regresi logistik + kunci VT                                     (per window)

CNN path (tahap 9) diambil dari benchmark resmi (outputs_colab_r15/benchmark/<arch>_S2.json, CPU 8 thread, batch 1)
agar konsisten dengan Tabel efisiensi paper. Window = 10 s, tumpang tindih 50 % (sama dengan protokol), mencakup
seluruh rekaman. Waktu per window = waktu tahap / jumlah window rekaman (biaya per rekaman diamortisasi).

GBM specialist dilatih ulang sekali (--latih-gbm) dengan resep paket_spesialis_r15 (seluruh development berbobot) dan
harus mereproduksi PH_KL tersimpan; hanya dipakai untuk waktu prediksi. Model fusi dibuat dari data acak berdimensi
sama (18 fitur, 5 kelas): waktu prediksi regresi logistik tidak bergantung pada nilai bobot.
Keluaran (baru, tidak menimpa): outputs_colab_r15/additional_tests/b1_latensi/
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
os.environ.setdefault("V4_OUTPUT_ROOT", str(HERE / "outputs_colab_r15"))
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "fase_e_ns")); sys.path.insert(0, str(HERE.parent / "tools"))
OUT = HERE / "outputs_colab_r15"
DEST = OUT / "additional_tests" / "b1_latensi"
GBM_PATH = DEST / "gbm_specialist.pkl"
RECORDS = [("mitdb", "101"), ("svdb", "808"), ("vfdb", "424"), ("afdb", "05261")]
ARCHS = ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"]


def train_gbm():
    if GBM_PATH.exists():
        print("GBM sudah ada"); return
    from e5_common import beat_features, load_base
    from e6_bobot_ujung_ke_ujung import gbm
    B = load_base(); bi, y4, pdev = B["bi"], B["y4"], B["pdev"]
    XA, KL, sub = beat_features(B); X = np.hstack([XA[bi], KL[bi]]); del XA, KL
    tr = pdev & (y4 >= 0); th = np.flatnonzero(~pdev)
    t0 = time.time(); m = gbm(X[tr], y4[tr], True); dt = time.time() - t0
    pr = m.predict_proba(X[th]); cl = list(m.classes_)
    Z = np.load(HERE / "fase_e_ns" / "hasil" / "spesialis_s_pasien" / "peluang_beat.npz")
    repro = float(max(np.nanmax(np.abs(pr[:, cl.index(1)] - Z["PH_KL"][th, 0])),
                      np.nanmax(np.abs(pr[:, cl.index(3)] - Z["PH_KL"][th, 1]))))
    assert repro < 1e-6, f"GBM tidak mereproduksi PH_KL ({repro})"
    DEST.mkdir(parents=True, exist_ok=True)
    with open(GBM_PATH, "wb") as f:
        pickle.dump({"model": m, "beat_names": sub, "reproduksi_PH_KL": repro, "detik_latih": dt}, f)
    print("GBM tersimpan; reproduksi PH_KL", repro, "detik latih", round(dt, 1))


def run_record(db, rec, ch, cfg, overlap, spec, fusion):
    import fitur_ns as NS
    import full_record_template as RTP
    import v4_local_rhythm as LR
    from a5_physiology import _append_global_morphology, _beat_features, _global_rr_features
    from a6_wave_delineation import extract_wave_features
    from e5_common import longest_run
    from ecgtl.data.build_dataset import load_channel_signal
    from ecgtl.data.segmentation import preprocess_record
    from ecgtl.hierarchical.data import detected_peaks, window_views
    from spesialis_s_pasien import patient_features
    from spesialis_s_v2 import BEAT_PREFIX

    T, clock = {}, time.perf_counter
    fs = float(cfg["fs"]); W = int(cfg["window_samples"]); step = int(round(W * (1 - overlap)))

    s = clock()
    raw, fs_raw, used = load_channel_signal(NS.RAW / db, rec, [ch]); assert used == ch
    signal = preprocess_record(raw, fs_raw, cfg["fs"], cfg["bandpass"]).signal
    peaks = detected_peaks(signal, cfg)
    T["1_baca_filter_deteksi_beat"] = clock() - s
    starts = list(range(0, len(signal) - W + 1, step))
    wpk = [peaks[(peaks >= st) & (peaks < st + W)] for st in starts]

    s = clock(); n_crop = 0
    for st, pk in zip(starts, wpk):
        _, crops, _, _ = window_views(signal[st:st + W], pk - st, cfg); n_crop += len(crops)
    T["2_render_citra_global_lokal"] = clock() - s

    s = clock()
    for st, pk in zip(starts, wpk):
        seg = np.asarray(signal[st:st + W], np.float64); p = (pk - st).astype(np.int64)
        beat = _beat_features(seg, p, fs); _append_global_morphology(_global_rr_features(p, fs), beat)
        extract_wave_features(seg, p, fs)
    T["3_fitur_tabular_window"] = clock() - s

    s = clock()
    RTP.per_record((db, rec, ch, [(f"w{i}", st, pk) for i, (st, pk) in enumerate(zip(starts, wpk))]))
    T["4_template_rekaman_lead_kedua"] = clock() - s

    s = clock()
    _, _, lpk, table = LR.record_features((db, rec, ch, fs))
    for st in starts:
        LR.window_features(table[(lpk >= st) & (lpk < st + W)])
    T["5_ritme_lokal"] = clock() - s

    s = clock()
    r = NS.load_record(db, rec, ch, fs, annotations=False)
    e0, aux0 = NS.fitur_e0_baseline(r); e1, aux1 = NS.fitur_e1_rr_lokal(r)
    ql = NS._qrs_local(r["signal"], r["peaks"], fs)
    stable = NS.beat_stabil(ql, aux1["rr_rel10"], aux1["rr_next"] / aux1["base10"])
    e2 = NS.fitur_e2_deviasi_p(r, aux0, aux1, stable)
    F = pd.concat([e0, e1, e2], axis=1)
    XA = F[[n for n in F.columns if n.startswith(BEAT_PREFIX)]][spec["beat_names"]].to_numpy(np.float32)
    _, KL = patient_features(XA, spec["beat_names"], {rec: np.arange(len(XA))})
    T["6_fitur_beat_dan_klaster_specialist"] = clock() - s

    s = clock()
    m = spec["model"]; pr = m.predict_proba(np.hstack([XA, KL])); cl = list(m.classes_)
    pS, pAF = pr[:, cl.index(1)], pr[:, cl.index(3)]; bp = r["peaks"]; A = np.zeros((len(starts), 13))
    for i, st in enumerate(starts):
        sel = (bp >= st) & (bp < st + W); v, a = pS[sel], pAF[sel]
        if len(v):
            vs = np.sort(v)[::-1]
            A[i] = [vs[0], vs[:2].mean(), v.mean(), (v >= 0.5).sum(), (v >= 0.5).mean(), len(v), a.mean(),
                    (a >= 0.5).mean(), v.std(), vs[:3].mean(), longest_run(v >= 0.5), a.max(), np.median(a)]
    T["7_prediksi_gbm_dan_agregat"] = clock() - s

    s = clock()
    L = np.log(np.full((len(starts), 5), 0.2))
    out = fusion.predict_proba(np.hstack([L, A])); dec = np.where(L.argmax(1) == 4, 4, out[:, :4].argmax(1))
    T["8_fusi_dan_kunci_vt"] = clock() - s
    assert len(dec) == len(starts)

    info = {"database": db, "record": rec, "durasi_s": len(signal) / fs, "n_window": len(starts),
            "n_beat_gqrs": int(len(peaks)), "crop_per_window_rerata": n_crop / len(starts)}
    return info, T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--latih-gbm", action="store_true")
    ap.add_argument("--rekaman", nargs="*", default=None, help="db/rec, mis. mitdb/101")
    a = ap.parse_args()
    if a.latih_gbm:
        train_gbm(); return
    import v4_runner as R
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    target = DEST / "latensi_per_rekaman.csv"
    if target.exists() and not a.rekaman:
        sys.exit(f"{target} sudah ada; tidak ditimpa.")
    p = R.protocol(); cfg = p.cfg["data"]; overlap = 0.5   # configs/base.yaml windowing.overlap; jarak start manifest = 1250 sampel
    g = pd.read_csv(p.prepared / "global_manifest.csv", dtype={"record": str})
    with open(GBM_PATH, "rb") as f:
        spec = pickle.load(f)
    rng = np.random.default_rng(0)
    fusion = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(
        rng.normal(size=(5000, 18)), rng.integers(0, 4, 5000))
    recs = [tuple(x.split("/")) for x in a.rekaman] if a.rekaman else RECORDS
    proc = psutil.Process(); rows = []
    for db, rec in recs:
        sub = g[(g.database == db) & (g.record == rec)]
        assert (sub.split == "holdout").all() and len(sub), (db, rec)
        info, T = run_record(db, rec, sub.channel.iloc[0], cfg, overlap, spec, fusion)
        mem = proc.memory_info()
        row = {**info, **{f"detik_{k}": v for k, v in T.items()}, "detik_total_non_cnn": sum(T.values()),
               "rss_mb_setelah": mem.rss / 2 ** 20, "puncak_wset_mb_proses": getattr(mem, "peak_wset", np.nan) / 2 ** 20}
        rows.append(row); print(json.dumps(row, default=float), flush=True)
    if a.rekaman:
        return
    D = pd.DataFrame(rows); DEST.mkdir(parents=True, exist_ok=True); D.to_csv(target, index=False)
    stages = [c for c in D.columns if c.startswith("detik_") and c != "detik_total_non_cnn"]
    per_win = pd.DataFrame({c.replace("detik_", ""): 1000 * D[c] / D.n_window for c in stages + ["detik_total_non_cnn"]})
    per_win.insert(0, "rekaman", D.database + "/" + D.record)
    summ = []
    for arch in ARCHS:
        b = json.loads((OUT / "benchmark" / f"{arch}_S2.json").read_text())
        for _, rr in per_win.iterrows():
            tot = rr["total_non_cnn"] + b["cpu_median_ms"]
            dur = float(D.loc[D.database + "/" + D.record == rr.rekaman, "durasi_s"].iloc[0])
            nwin = int(D.loc[D.database + "/" + D.record == rr.rekaman, "n_window"].iloc[0])
            summ.append({"arch": arch, "rekaman": rr.rekaman, "ms_per_window_non_cnn": rr["total_non_cnn"],
                         "ms_per_window_cnn_path_cpu": b["cpu_median_ms"], "ms_per_window_total": tot,
                         "porsi_cnn_path": b["cpu_median_ms"] / tot,
                         "faktor_realtime_rekaman": tot * nwin / 1000 / dur})
    per_win.to_csv(DEST / "latensi_per_window_tahap.csv", index=False)
    pd.DataFrame(summ).to_csv(DEST / "latensi_end_to_end.csv", index=False)
    meta = {"cpu": psutil.cpu_count(logical=True), "proses_tunggal": True, "window_s": 10, "overlap": overlap,
            "gbm_reproduksi_PH_KL": spec["reproduksi_PH_KL"], "cnn_path": "benchmark resmi S2, CPU 8 thread, batch 1",
            "puncak_memori_proses_mb": float(D.puncak_wset_mb_proses.max())}
    (DEST / "complete.json").write_text(json.dumps(meta, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(per_win.round(2).to_string(index=False)); print(pd.DataFrame(summ).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
