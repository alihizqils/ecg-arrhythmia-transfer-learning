"""Revisi 14 (7 Nov 2026): fitur ritme lokal (E1) dan konteks 7 beat (E3) untuk kelas S.

Asal: eksperimen tambahan fitur ritme untuk kelas S (fase_e_ns/). Pada ablasi
tingkat beat N/S, hanya E1 (Local RR Dynamics; naik di 5/5 fold) dan E3 (multi-beat context L=7, naik
di 4/5 fold) yang lolos kriteria. Pada proksi window lima kelas (fase_e_ns/hasil/proksi_window/),
B+E13 menaikkan F1 S 0,638 -> 0,691 (naik di 5/5 fold) tanpa kelas lain turun. E2, E4, E5, E6
tidak lolos dan tidak dipakai.

Fitur dihitung per rekaman dari sinyal kanal V4 (preprocessing dan detektor GQRS identik dengan V4)
dengan fungsi yang sama persis dengan studi beat (`fase_e_ns/fitur_ns.py`). Semua fitur kausal
(hanya beat sebelumnya); label tidak dibaca (`annotations=False`).

- Per beat (BEAT_NAMES, 39): masuk proyeksi tabular cabang beat (blok yang sama dengan template
  rekaman per beat).
- Revisi 15: agregat window juga memakai E2 + E4 (E2E4_NAMES, 20) -> WINDOW_NAMES 242.
- Per window (WINDOW_NAMES, 162 pada Revisi 14): min/max/mean/nilai pada beat paling prematur dari 39 fitur beat +
  6 hitungan beat prematur; masuk expert tabular window (blok template rekaman), sama dengan proksi.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
if str(FOLDER / "fase_e_ns") not in sys.path:
    sys.path.insert(0, str(FOLDER / "fase_e_ns"))
import fitur_ns as NS  # noqa: E402

BEAT_NAMES = [
    "e1c_rr_relative_b5", "e1c_rr_relative_b10", "e1c_rr_relative_b20", "e1c_rr_change_ratio",
    "e1c_rr_ratio_previous", "e1c_local_rr_median_10", "e1c_local_rr_std_10", "e1c_local_rr_mad_10",
    "e1c_local_rr_cv_10", "e1c_rr_minus_local_median_over_mad",
    *[f"e3c7_rr_rel_{k:+d}" for k in range(-6, 0)],
    *[f"e3c7_p_dev_{k:+d}" for k in range(-6, 1)],
    *[f"e3c7_qrs_sim_{k:+d}" for k in range(-6, 1)],
    "e3c7_rr_trend", "e3c7_rr_cv", *[f"e3c7_drr_sign_{k}" for k in range(6)], "e3c7_short_long_alternation",
]
# Revisi 15 (8 Nov 2026, pilot): E2 (deviasi gelombang P) dan E4 (gelombang T / aktivitas atrial tersembunyi).
# Gagal di ablasi beat N/S, tetapi pada proksi window lima kelas B+E13+E2+E4 menaikkan F1 S
# 0,691 -> 0,703 (naik 4/5 fold, Macro naik 5/5, tidak ada kelas turun; fase_e_ns/hasil/proksi_window_ekstra/).
# Sesuai proksi, E2/E4 hanya masuk agregat window; cabang beat tetap BEAT_NAMES (Revisi 14).
E2E4_NAMES = [
    "e2_p_template_corr", "e2_p_deviation_dp", "e2_p_alignment_lag_s", "e2_p_amplitude_deviation",
    "e2_p_duration_deviation_s", "e2_p_area_deviation", "e2_pr_relative", "e2_p_polarity", "e2_p_polarity_change",
    "e2_p_confidence_relative", "e2_p_template_available", "e2_p_template_beats",
    "e4_t_morphology_corr", "e4_t_residual_energy", "e4_t_slope_deviation", "e4_t_area_deviation",
    "e4_pre_candidate_corr", "e4_pre_candidate_residual_energy", "e4_transition_residual_fraction",
    "e4_t_template_available",
]
SOURCE_NAMES = BEAT_NAMES + E2E4_NAMES
COUNT_NAMES = ["n_premature_085", "frac_premature_085", "n_premature_normal_qrs", "n_premature_abnormal_qrs",
               "min_rr_relative_b10", "median_rr_relative_b10"]
WINDOW_NAMES = [f"{stat}_{name}" for stat in ["min", "max", "mean", "most_premature"] for name in SOURCE_NAMES] + COUNT_NAMES
_REL, _QRS = SOURCE_NAMES.index("e1c_rr_relative_b10"), SOURCE_NAMES.index("e3c7_qrs_sim_+0")


def record_features(args):
    """Fitur per R-peak terdeteksi satu rekaman (tanpa label)."""
    database, record, channel, fs = args
    rec = NS.load_record(database, record, channel, fs, annotations=False)
    _, aux0 = NS.fitur_e0_baseline(rec)
    e1, aux1 = NS.fitur_e1_rr_lokal(rec)
    qrs_local = NS._qrs_local(rec["signal"], rec["peaks"], fs)
    stable = NS.beat_stabil(qrs_local, aux1["rr_rel10"], aux1["rr_next"] / aux1["base10"])
    e2 = NS.fitur_e2_deviasi_p(rec, aux0, aux1, stable)
    e3 = NS.fitur_e3_konteks(aux1, e2, qrs_local)
    e4 = NS.fitur_e4_gelombang_t(rec, stable)
    table = pd.concat([e1, e3, e2, e4], axis=1)[SOURCE_NAMES].to_numpy(np.float32)
    return database, record, rec["peaks"], table


def window_features(beats: np.ndarray) -> np.ndarray:
    out = np.full(len(WINDOW_NAMES), np.nan, dtype=np.float32)
    if len(beats) == 0:
        return out
    x = beats.astype(np.float64); n = len(SOURCE_NAMES)
    with np.errstate(all="ignore"):
        rel, qrs = x[:, _REL], x[:, _QRS]
        out[:n] = np.nanmin(x, 0); out[n:2 * n] = np.nanmax(x, 0); out[2 * n:3 * n] = np.nanmean(x, 0)
        if np.isfinite(rel).any():
            out[3 * n:4 * n] = x[int(np.nanargmin(np.where(np.isfinite(rel), rel, np.inf)))]
        premature = rel < 0.85
        out[4 * n:] = [premature.sum(), premature.mean(), (premature & (qrs >= 0.9)).sum(), (premature & (qrs < 0.8)).sum(),
                       np.nanmin(rel) if np.isfinite(rel).any() else np.nan,
                       np.nanmedian(rel) if np.isfinite(rel).any() else np.nan]
    return out


def _source_hash(protocol, windows, local) -> str:
    payload = hashlib.sha256()
    for path in [Path(__file__), Path(NS.__file__)]:
        payload.update(path.read_bytes())
    payload.update(str(protocol.hash).encode())
    payload.update(pd.util.hash_pandas_object(windows[["window_id", "database", "record", "channel"]], index=False).values.tobytes())
    payload.update(pd.util.hash_pandas_object(local[["parent_window_id", "rpeak_sample"]], index=False).values.tobytes())
    return payload.hexdigest()


def build_local_rhythm_features(protocol, workers: int | None = None):
    """(fitur window per window_id, fitur per beat per window_id), urutan beat = urutan R-peak local manifest."""
    cfg = protocol.cfg["data"]
    if (float(cfg["fs"]), cfg["bandpass"]["low_hz"], cfg["bandpass"]["high_hz"], cfg["detector_offset"]) != \
            (250.0, NS.BANDPASS["low_hz"], NS.BANDPASS["high_hz"], 8):
        raise ValueError("Preprocessing/detektor V4 berbeda dengan fitur_ns.")
    windows = pd.read_csv(protocol.prepared / "global_manifest.csv", dtype={"window_id": str, "record": str})
    local = pd.read_csv(protocol.prepared / "local_manifest.csv", dtype={"parent_window_id": str})
    local = local[local.parent_window_id.isin(set(windows.window_id))].sort_values(["parent_window_id", "rpeak_sample"])
    source = _source_hash(protocol, windows, local)
    folder = protocol.out / "feature_store" / f"local_rhythm_{source[:16]}"
    marker = folder / "complete.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        values = np.load(folder / "local_rhythm.npz")
        if (meta["source_hash"] != source or list(values["names"]) != WINDOW_NAMES
                or list(values["beat_names"]) != SOURCE_NAMES):
            raise ValueError("Integritas cache fitur ritme lokal tidak sesuai.")
        ids, offsets, beat_values = values["window_ids"].astype(str), values["beat_offsets"], values["beat_features"]
        return (dict(zip(ids, values["features"])),
                {wid: beat_values[offsets[i]:offsets[i + 1], :len(BEAT_NAMES)] for i, wid in enumerate(ids)})
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [(db, rec, ch, float(cfg["fs"])) for (db, rec, ch), _ in windows.groupby(["database", "record", "channel"], sort=False)]
    workers = workers or int(os.environ.get("V4_RHYTHM_WORKERS", 0)) or max(1, min(10, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(workers) as pool:
        per_record = {(db, rec): (peaks, table) for db, rec, peaks, table in pool.map(record_features, jobs)}
    peaks_by_window = {key: frame.rpeak_sample.to_numpy(np.int64) for key, frame in local.groupby("parent_window_id")}
    window_result, beat_result, unmatched = {}, {}, 0
    for row in windows.itertuples(index=False):
        record_peaks, table = per_record[(row.database, row.record)]
        peaks = peaks_by_window.get(row.window_id, np.empty(0, np.int64))
        position = np.clip(np.searchsorted(record_peaks, peaks), 0, len(record_peaks) - 1)
        match = record_peaks[position] == peaks
        beats = np.full((len(peaks), len(SOURCE_NAMES)), np.nan, dtype=np.float32)
        beats[match] = table[position[match]]
        unmatched += int((~match).sum())
        beat_result[row.window_id] = beats
        window_result[row.window_id] = window_features(beats[match])
    ids = windows.window_id.tolist()
    with (folder / "local_rhythm.tmp").open("wb") as stream:
        np.savez_compressed(stream, window_ids=np.asarray(ids), names=np.asarray(WINDOW_NAMES),
                            features=np.stack([window_result[i] for i in ids]).astype(np.float32),
                            beat_names=np.asarray(SOURCE_NAMES),
                            beat_features=np.concatenate([beat_result[i] for i in ids]).astype(np.float32),
                            beat_offsets=np.cumsum([0] + [len(beat_result[i]) for i in ids]).astype(np.int64))
    os.replace(folder / "local_rhythm.tmp", folder / "local_rhythm.npz")
    marker.write_text(json.dumps({"source_hash": source, "windows": len(ids), "beats": int(len(local)),
                                  "beats_not_matched_to_record_peaks": unmatched,
                                  "label_columns_read_for_features": False}, indent=2))
    return window_result, {wid: b[:, :len(BEAT_NAMES)] for wid, b in beat_result.items()}
