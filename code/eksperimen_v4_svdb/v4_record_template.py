"""Fitur morfologi relatif terhadap template beat tingkat rekaman (Revisi 6).

Fitur morfologi warisan (a5_physiology, a6_wave_delineation) membandingkan beat
dengan template median *di dalam window 10 detik itu sendiri*. Pada window yang
banyak berisi beat prematur, template window ikut berubah sehingga beat S tidak
tampak menyimpang. A2 (baseline RR tingkat rekaman) sudah menerapkan acuan
tingkat rekaman untuk RR; modul ini menerapkan acuan yang sama untuk morfologi.

Revisi 7: fitur yang sama juga diberikan per beat ke cabang beat (keputusan
N/S/V), sesuai definisi klinis beat S: beat prematur dengan gelombang P yang
berbeda dari P normal pasien itu sendiri.

Template = median segmen QRS dan segmen pra-QRS (area gelombang P) yang telah
dinormalisasi-z dari seluruh beat pada window terpilih rekaman/kanal yang sama.
Label beat, label window, split, dan fold tidak dibaca. Satu rekaman selalu
berada di satu sisi split pasien, sehingga tidak ada informasi lintas pasien.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RECORD_TEMPLATE_NAMES = [
    "rec_qrs_distance_median",       # median (1 - korelasi QRS vs template rekaman)
    "rec_qrs_distance_max",
    "rec_qrs_abnormal_fraction",     # porsi beat dengan korelasi QRS < 0,9
    "rec_p_correlation_median",      # korelasi area P vs template rekaman
    "rec_p_correlation_min",
    "rec_p_abnormal_fraction",       # porsi beat dengan korelasi P < 0,5
    "rec_p_correlation_most_premature",
    "rec_qrs_correlation_most_premature",
    "rec_p_correlation_premature_mean",  # rata-rata korelasi P pada beat RR < 0,85 x median rekaman
]
# Fitur per beat untuk cabang beat (Revisi 7).
BEAT_RECORD_TEMPLATE_NAMES = [
    "beat_rec_qrs_correlation",      # korelasi QRS beat vs template rekaman
    "beat_rec_p_correlation",        # korelasi area P beat vs template rekaman
    "beat_rr_previous_over_record_median",
    "beat_rec_p_correlation_previous_beat",
]
# Revisi 10: ketidakteraturan RR antar-beat dominan (beat yang bentuk QRS-nya mirip
# template). Pada window AF yang berisi banyak beat V, ketidakteraturan RR seluruh
# beat bercampur dengan beat V; RR antar-beat dominan memisahkan irama dasar
# (AF tidak teratur) dari ektopi. Dua definisi beat dominan: korelasi QRS >= 0,9
# terhadap template window dan terhadap template rekaman.
DOMINANT_RR_NAMES = [
    f"{prefix}_{name}" for prefix in ["dom_window", "dom_record"]
    for name in ["fraction", "rr_cv", "rr_mean_abs_diff_over_median", "rr_abs_diff_gt_50ms_fraction"]
]
_DOMINANT = 0.9
_QRS = (0.12, 0.16)
_PRE = (0.28, 0.06)
_ABNORMAL_QRS, _ABNORMAL_P, _PREMATURE = 0.9, 0.5, 0.85


def _znorm(segment: np.ndarray):
    deviation = segment.std()
    return (segment - segment.mean()) / deviation if deviation > 1e-9 else None


def _correlation(left, right) -> float:
    return float(np.mean(left * right)) if left is not None and right is not None else np.nan


def _fraction_below(values: np.ndarray, threshold: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite < threshold)) if len(finite) else np.nan


def _window_features(beats: np.ndarray) -> np.ndarray:
    """beats: kolom [rr_prev / median rekaman, korelasi QRS, korelasi P]."""
    output = np.full(len(RECORD_TEMPLATE_NAMES), np.nan, dtype=np.float32)
    if not len(beats):
        return output
    ratio, qrs, p_wave = beats[:, 0], beats[:, 1], beats[:, 2]
    with np.errstate(all="ignore"):
        if np.isfinite(qrs).any():
            output[0] = np.nanmedian(1 - qrs); output[1] = np.nanmax(1 - qrs)
        output[2] = _fraction_below(qrs, _ABNORMAL_QRS)
        if np.isfinite(p_wave).any():
            output[3] = np.nanmedian(p_wave); output[4] = np.nanmin(p_wave)
        output[5] = _fraction_below(p_wave, _ABNORMAL_P)
        if np.isfinite(ratio).any():
            index = int(np.nanargmin(ratio))
            output[6] = p_wave[index]; output[7] = qrs[index]
            premature = np.isfinite(ratio) & (ratio < _PREMATURE)
            if premature.any() and np.isfinite(p_wave[premature]).any():
                output[8] = np.nanmean(p_wave[premature])
    return output


def _source_hash(protocol, windows: pd.DataFrame, local: pd.DataFrame) -> str:
    payload = hashlib.sha256()
    payload.update(Path(__file__).read_bytes())
    payload.update(str(protocol.hash).encode())
    payload.update(pd.util.hash_pandas_object(
        windows[["window_id", "database", "record", "channel", "index", "start_target_fs"]], index=False).values.tobytes())
    payload.update(pd.util.hash_pandas_object(local[["parent_window_id", "rpeak_sample"]], index=False).values.tobytes())
    return payload.hexdigest()


def _beat_features(beats: np.ndarray) -> np.ndarray:
    """beats: kolom [rr_prev / median rekaman, korelasi QRS, korelasi P] berurutan menurut R-peak."""
    output = np.full((len(beats), len(BEAT_RECORD_TEMPLATE_NAMES)), np.nan, dtype=np.float32)
    if len(beats):
        output[:, 0] = beats[:, 1]; output[:, 1] = beats[:, 2]; output[:, 2] = beats[:, 0]
        output[1:, 3] = beats[:-1, 2]
    return output


def build_record_template_features(protocol, fs: float):
    """Hitung (atau muat dari cache) fitur template rekaman.

    Mengembalikan (fitur per window, fitur per beat). Urutan beat di dalam window
    mengikuti R-peak, sama dengan urutan fitur lokal feature store.
    """
    windows = pd.read_csv(protocol.prepared / "global_manifest.csv", dtype={"window_id": str, "record": str})
    local = pd.read_csv(protocol.prepared / "local_manifest.csv", dtype={"parent_window_id": str})
    local = local[local.parent_window_id.isin(set(windows.window_id))].sort_values(["parent_window_id", "rpeak_sample"])
    source = _source_hash(protocol, windows, local)
    folder = protocol.out / "feature_store" / f"record_template_{source[:16]}"
    marker = folder / "complete.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        values = np.load(folder / "record_template.npz")
        if (meta["source_hash"] != source or list(values["names"]) != RECORD_TEMPLATE_NAMES
                or list(values["beat_names"]) != BEAT_RECORD_TEMPLATE_NAMES):
            raise ValueError("Integritas cache fitur template rekaman tidak sesuai.")
        ids, offsets, beat_values = values["window_ids"].astype(str), values["beat_offsets"], values["beat_features"]
        return (dict(zip(ids, values["features"])),
                {wid: beat_values[offsets[i]:offsets[i + 1]] for i, wid in enumerate(ids)})
    folder.mkdir(parents=True, exist_ok=True)
    qrs_left, qrs_right = int(round(_QRS[0] * fs)), int(round(_QRS[1] * fs))
    pre_left, pre_right = int(round(_PRE[0] * fs)), int(round(_PRE[1] * fs))
    peaks_by_window = {key: frame.rpeak_sample.to_numpy(np.int64) for key, frame in local.groupby("parent_window_id")}
    result: dict[str, np.ndarray] = {}
    beat_result: dict[str, np.ndarray] = {}
    for (database, record, _channel), rows in windows.groupby(["database", "record", "channel"], sort=False):
        folder_record = protocol.prepared / "records" / str(database) / str(record)
        waveforms = np.load(folder_record / "waveforms.npy", mmap_mode="r")
        position = {int(value): index for index, value in enumerate(np.load(folder_record / "window_indices.npy"))}
        beats = []  # (window_id, rr_prev_s, qrs_z, pre_z)
        for row in rows.itertuples(index=False):
            signal = np.asarray(waveforms[position[int(row.index)]], dtype=np.float64)
            peaks = peaks_by_window.get(row.window_id, np.empty(0, np.int64)) - int(row.start_target_fs)
            for order, peak in enumerate(peaks):
                qrs = signal[peak - qrs_left:peak + qrs_right + 1] if peak - qrs_left >= 0 and peak + qrs_right < len(signal) else None
                pre = signal[peak - pre_left:peak - pre_right + 1] if peak - pre_left >= 0 and peak - pre_right < len(signal) else None
                rr = (peak - peaks[order - 1]) / fs if order > 0 else np.nan
                beats.append((row.window_id, rr, None if qrs is None else _znorm(qrs), None if pre is None else _znorm(pre)))
        qrs_all = [beat[2] for beat in beats if beat[2] is not None]
        pre_all = [beat[3] for beat in beats if beat[3] is not None]
        qrs_template = _znorm(np.median(np.stack(qrs_all), 0)) if qrs_all else None
        pre_template = _znorm(np.median(np.stack(pre_all), 0)) if pre_all else None
        rr_all = np.array([beat[1] for beat in beats], dtype=np.float64)
        rr_median = float(np.nanmedian(rr_all)) if np.isfinite(rr_all).any() else np.nan
        per_window: dict[str, list] = {}
        for window_id, rr, qrs, pre in beats:
            per_window.setdefault(window_id, []).append((
                rr / rr_median if np.isfinite(rr_median) and rr_median > 0 else np.nan,
                _correlation(qrs, qrs_template), _correlation(pre, pre_template)))
        for window_id in rows.window_id:
            values = np.asarray(per_window.get(window_id, []), dtype=np.float64).reshape(-1, 3)
            result[window_id] = _window_features(values)
            beat_result[window_id] = _beat_features(values)
    ids = windows.window_id.tolist()
    with (folder / "record_template.tmp").open("wb") as stream:
        np.savez_compressed(stream, window_ids=np.asarray(ids), names=np.asarray(RECORD_TEMPLATE_NAMES),
                            features=np.stack([result[i] for i in ids]).astype(np.float32),
                            beat_names=np.asarray(BEAT_RECORD_TEMPLATE_NAMES),
                            beat_features=np.concatenate([beat_result[i] for i in ids]).astype(np.float32),
                            beat_offsets=np.cumsum([0] + [len(beat_result[i]) for i in ids]).astype(np.int64))
    os.replace(folder / "record_template.tmp", folder / "record_template.npz")
    marker.write_text(json.dumps({"source_hash": source, "windows": len(ids), "label_columns_read_for_features": False}, indent=2))
    return result, beat_result


def dominant_rr_features(window_correlation: np.ndarray, record_correlation: np.ndarray, rr_previous: np.ndarray) -> np.ndarray:
    """Fitur DOMINANT_RR_NAMES satu window dari fitur per beat yang label-free.

    window_correlation: korelasi QRS beat vs template window (a5 qrs_template_correlation);
    record_correlation: korelasi QRS beat vs template rekaman; rr_previous: RR sebelumnya (detik).
    RR dihitung hanya bila beat ini dan beat sebelumnya sama-sama dominan.
    """
    output = np.full(len(DOMINANT_RR_NAMES), np.nan, dtype=np.float32)
    if not len(rr_previous):
        return output
    with np.errstate(all="ignore"):
        for index, correlation in enumerate([window_correlation, record_correlation]):
            dominant = correlation >= _DOMINANT
            pair = dominant & np.r_[False, dominant[:-1]] & np.isfinite(rr_previous)
            values = rr_previous[pair]
            output[4 * index] = dominant.mean()
            if len(values) >= 3:
                output[4 * index + 1] = values.std() / values.mean()
                output[4 * index + 2] = np.mean(np.abs(np.diff(values))) / np.median(values)
                output[4 * index + 3] = np.mean(np.abs(np.diff(values)) > 0.05)
    return output
