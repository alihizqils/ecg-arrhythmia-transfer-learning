"""Fitur delineasi P/PR/QRS untuk pilot A6 pada development set saja."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend, savgol_filter

from ecgtl.hierarchical.data import read_manifest
from ecgtl.hierarchical.protocol import atomic_json, digest


LOCAL_WAVE_NAMES = [
    "p_presence_confidence",
    "p_present",
    "p_peak_amplitude_over_scale",
    "p_width_s",
    "pr_interval_s",
    "p_polarity",
    "p_template_correlation",
    "qrs_onset_to_r_s",
    "qrs_r_to_offset_s",
    "qrs_onset_offset_duration_s",
    "qrs_positive_amplitude_over_scale",
    "qrs_negative_amplitude_over_scale",
    "qrs_positive_negative_balance",
]

GLOBAL_WAVE_NAMES = [
    "p_detected_ratio",
    "p_confidence_median",
    "p_width_median_s",
    "pr_interval_median_s",
    "pr_interval_iqr_s",
    "p_template_correlation_median",
    "p_polarity_consistency",
    "qrs_duration_median_s",
    "qrs_duration_iqr_s",
    "qrs_balance_iqr",
]


def _robust_scale(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        q25, q75 = np.percentile(signal, [25, 75])
        scale = (q75 - q25) / 1.349
    return float(scale) if np.isfinite(scale) and scale >= 1e-6 else np.nan


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64) - np.mean(left)
    right = np.asarray(right, dtype=np.float64) - np.mean(right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else np.nan


def _smooth(signal: np.ndarray, fs: float) -> np.ndarray:
    maximum = len(signal) if len(signal) % 2 else len(signal) - 1
    window = min(maximum, max(5, int(round(0.044 * fs)) | 1))
    if window < 5:
        return np.asarray(signal, dtype=np.float64)
    return savgol_filter(np.asarray(signal, dtype=np.float64), window, 3, mode="interp")


def _crossing_bounds(values: np.ndarray, center: int, threshold: float) -> tuple[int, int]:
    """Cari batas aktivitas dengan tiga sampel tenang berturut-turut."""
    active = np.abs(values) >= threshold
    left = int(center)
    quiet = 0
    while left > 0:
        left -= 1
        quiet = quiet + 1 if not active[left] else 0
        if quiet >= 3:
            left += 3
            break
    right = int(center)
    quiet = 0
    while right < len(values) - 1:
        right += 1
        quiet = quiet + 1 if not active[right] else 0
        if quiet >= 3:
            right -= 3
            break
    return max(0, left), min(len(values) - 1, right)


def extract_wave_features(
    signal: np.ndarray,
    peaks: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Delineasi deterministik tanpa anotasi kelas atau fiducial ground truth."""
    signal = np.asarray(signal, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.int64)
    local = np.full((len(peaks), len(LOCAL_WAVE_NAMES)), np.nan, dtype=np.float32)
    scale = _robust_scale(signal)
    if not np.isfinite(scale):
        return np.full(len(GLOBAL_WAVE_NAMES), np.nan, dtype=np.float32), local
    smoothed = _smooth(signal, fs)
    p_segments: list[np.ndarray | None] = [None] * len(peaks)

    for index, peak in enumerate(peaks):
        # QRS onset/offset: ambang slope robust dalam jendela sekitar R-peak.
        q0 = max(0, int(peak - round(0.12 * fs)))
        q1 = min(len(signal), int(peak + round(0.16 * fs)) + 1)
        if q1 - q0 >= 7:
            qrs = smoothed[q0:q1]
            derivative = np.gradient(qrs)
            center = int(np.clip(peak - q0, 0, len(qrs) - 1))
            search0 = max(0, center - int(round(0.06 * fs)))
            search1 = min(len(qrs), center + int(round(0.08 * fs)) + 1)
            slope_center = search0 + int(np.argmax(np.abs(derivative[search0:search1])))
            noise_parts = np.r_[derivative[: max(1, search0)], derivative[min(len(derivative), search1) :]]
            noise_mad = 1.4826 * np.median(np.abs(noise_parts - np.median(noise_parts))) if len(noise_parts) else 0.0
            threshold = max(3.0 * noise_mad, 0.08 * float(np.max(np.abs(derivative))))
            if np.isfinite(threshold) and threshold > 1e-12:
                onset_local, offset_local = _crossing_bounds(derivative, slope_center, threshold)
                onset, offset = q0 + onset_local, q0 + offset_local
                duration = (offset - onset) / fs
                if 0.025 <= duration <= 0.24 and onset <= peak <= offset:
                    local[index, 7] = (peak - onset) / fs
                    local[index, 8] = (offset - peak) / fs
                    local[index, 9] = duration
            centered = qrs - np.median(qrs)
            positive = float(np.max(centered))
            negative = float(np.min(centered))
            local[index, 10] = positive / scale
            local[index, 11] = negative / scale
            denominator = abs(positive) + abs(negative)
            local[index, 12] = (positive - abs(negative)) / denominator if denominator > 1e-12 else np.nan

        # P-wave: kandidat paling menonjol 300--60 ms sebelum R.
        p0 = int(peak - round(0.30 * fs))
        p1 = int(peak - round(0.06 * fs))
        if p0 < 0 or p1 > len(signal) or p1 - p0 < 9:
            continue
        segment = detrend(smoothed[p0:p1], type="linear")
        candidate = int(np.argmax(np.abs(segment)))
        exclusion = int(round(0.06 * fs))
        noise_values = np.r_[
            segment[: max(0, candidate - exclusion)],
            segment[min(len(segment), candidate + exclusion + 1) :],
        ]
        if len(noise_values) < 5:
            noise_values = segment
        baseline_mad = 1.4826 * np.median(
            np.abs(noise_values - np.median(noise_values))
        )
        if not np.isfinite(baseline_mad) or baseline_mad <= 1e-12:
            continue
        amplitude = float(segment[candidate])
        confidence = abs(amplitude) / baseline_mad
        local[index, 0] = confidence
        present = confidence >= 3.0
        local[index, 1] = float(present)
        if not present:
            continue
        threshold = max(1.5 * baseline_mad, 0.10 * abs(amplitude))
        onset_local, offset_local = _crossing_bounds(segment, candidate, threshold)
        width = (offset_local - onset_local) / fs
        qrs_onset = peak - int(round(float(local[index, 7]) * fs)) if np.isfinite(local[index, 7]) else peak - int(round(0.05 * fs))
        pr_interval = (qrs_onset - (p0 + onset_local)) / fs
        if not 0.020 <= width <= 0.18 or not 0.06 <= pr_interval <= 0.36:
            local[index, 1] = 0.0
            continue
        local[index, 2] = amplitude / scale
        local[index, 3] = width
        local[index, 4] = pr_interval
        local[index, 5] = float(np.sign(amplitude))
        half = int(round(0.08 * fs))
        absolute_candidate = p0 + candidate
        start, stop = absolute_candidate - half, absolute_candidate + half + 1
        if start >= 0 and stop <= len(signal):
            p_segments[index] = smoothed[start:stop].astype(np.float64)

    complete = [part for part in p_segments if part is not None]
    if complete:
        template = np.median(np.stack(complete), axis=0)
        for index, part in enumerate(p_segments):
            if part is not None:
                local[index, 6] = _safe_correlation(part, template)

    global_features = summarize_wave_features(local)
    return global_features, local


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _median(values: np.ndarray) -> float:
    values = _finite(values)
    return float(np.median(values)) if len(values) else np.nan


def _iqr(values: np.ndarray) -> float:
    values = _finite(values)
    return float(np.percentile(values, 75) - np.percentile(values, 25)) if len(values) >= 2 else np.nan


def summarize_wave_features(local: np.ndarray) -> np.ndarray:
    output = np.full(len(GLOBAL_WAVE_NAMES), np.nan, dtype=np.float32)
    if not len(local):
        return output
    present = _finite(local[:, 1])
    output[0] = float(np.mean(present)) if len(present) else np.nan
    output[1] = _median(local[:, 0])
    output[2] = _median(local[:, 3])
    output[3] = _median(local[:, 4])
    output[4] = _iqr(local[:, 4])
    output[5] = _median(local[:, 6])
    polarity = _finite(local[:, 5])
    output[6] = float(abs(np.mean(polarity))) if len(polarity) else np.nan
    output[7] = _median(local[:, 9])
    output[8] = _iqr(local[:, 9])
    output[9] = _iqr(local[:, 12])
    return output


def _source_hash(protocol, development: pd.DataFrame, local: pd.DataFrame) -> str:
    payload = {
        "protocol_hash": protocol.hash,
        "implementation_sha256": digest(Path(__file__)),
        "development_ids": hashlib.sha256("\n".join(development.window_id.astype(str)).encode()).hexdigest(),
        "local_crop_ids": hashlib.sha256("\n".join(local.crop_id.astype(str)).encode()).hexdigest(),
        "features_local": LOCAL_WAVE_NAMES,
        "features_global": GLOBAL_WAVE_NAMES,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_wave_cache(protocol, force: bool = False) -> Path:
    """Bangun cache A6 hanya dari waveform dan R-peak development."""
    protocol.verify()
    global_manifest = read_manifest(protocol.prepared / "global_manifest.csv")
    development = global_manifest[global_manifest.split == "development"].copy()
    development["window_id"] = development.window_id.astype(str)
    local = read_manifest(protocol.prepared / "local_manifest.csv")
    local = local[local.split == "development"].copy()
    local["parent_window_id"] = local.parent_window_id.astype(str)
    local = local[local.parent_window_id.isin(set(development.window_id))]
    source_hash = _source_hash(protocol, development, local)
    folder = protocol.out / "ablation_rr" / "a6_wave_cache" / source_hash[:16]
    complete = folder / "complete.json"
    if complete.exists() and not force:
        metadata = json.loads(complete.read_text())
        if metadata["source_hash"] != source_hash or digest(folder / "features.npz") != metadata["features_sha256"]:
            raise ValueError("Integritas cache delineasi A6 tidak sesuai.")
        return folder
    folder.mkdir(parents=True, exist_ok=True)

    local_groups = {
        key: frame.sort_values("rpeak_sample").reset_index(drop=True)
        for key, frame in local.groupby("parent_window_id", sort=False)
    }
    window_ids, global_values, local_values = [], [], []
    offsets = [0]
    fs = float(protocol.cfg["data"]["fs"])
    for (database, record), windows in development.groupby(["database", "record"], sort=False):
        record_folder = protocol.prepared / "records" / str(database) / str(record)
        waveforms = np.load(record_folder / "waveforms.npy", mmap_mode="r")
        indices = np.load(record_folder / "window_indices.npy")
        index_to_row = {int(value): row for row, value in enumerate(indices)}
        for row in windows.itertuples(index=False):
            window_id = str(row.window_id)
            crop_rows = local_groups[window_id]
            waveform = np.asarray(waveforms[index_to_row[int(row.index)]], dtype=np.float64)
            peaks = crop_rows.rpeak_sample.to_numpy(dtype=np.int64) - int(row.start_target_fs)
            if np.any(peaks < 0) or np.any(peaks >= len(waveform)):
                raise ValueError(f"Koordinat R-peak A6 di luar window pada {window_id}.")
            global_feature, beat = extract_wave_features(waveform, peaks, fs)
            fallback = crop_rows.fallback_peak.astype(bool).to_numpy()
            if fallback.any():
                beat[fallback] = np.nan
                global_feature = summarize_wave_features(beat)
            window_ids.append(window_id)
            global_values.append(global_feature)
            local_values.append(beat)
            offsets.append(offsets[-1] + len(beat))

    if window_ids != development.window_id.astype(str).tolist():
        raise ValueError("Urutan cache A6 tidak sama dengan development manifest.")
    local_array = np.concatenate(local_values, axis=0).astype(np.float32)
    global_array = np.stack(global_values).astype(np.float32)
    np.savez_compressed(
        folder / "features.npz",
        window_ids=np.asarray(window_ids),
        global_features=global_array,
        local_features=local_array,
        local_offsets=np.asarray(offsets, dtype=np.int64),
    )
    atomic_json(complete, {
        "source_hash": source_hash,
        "protocol_hash": protocol.hash,
        "implementation_sha256": digest(Path(__file__)),
        "development_windows": len(window_ids),
        "local_beats": len(local_array),
        "global_feature_names": GLOBAL_WAVE_NAMES,
        "local_feature_names": LOCAL_WAVE_NAMES,
        "holdout_accessed": False,
        "features_sha256": digest(folder / "features.npz"),
    })
    return folder


def audit_wave_cache(protocol, folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = json.loads((folder / "complete.json").read_text())
    if metadata["protocol_hash"] != protocol.hash or metadata["holdout_accessed"]:
        raise ValueError("Cache A6 berbeda protokol atau mengakses holdout.")
    if digest(folder / "features.npz") != metadata["features_sha256"]:
        raise ValueError("Checksum cache A6 tidak sesuai.")
    values = np.load(folder / "features.npz")
    rows = []
    for scope, names, array in [
        ("wave_global", GLOBAL_WAVE_NAMES, values["global_features"]),
        ("wave_local", LOCAL_WAVE_NAMES, values["local_features"]),
    ]:
        for index, name in enumerate(names):
            column = array[:, index]
            finite = column[np.isfinite(column)]
            rows.append({
                "scope": scope,
                "feature": name,
                "available": len(finite),
                "missing": int(len(column) - len(finite)),
                "availability": float(len(finite) / len(column)),
                "p01": float(np.percentile(finite, 1)) if len(finite) else np.nan,
                "median": float(np.median(finite)) if len(finite) else np.nan,
                "p99": float(np.percentile(finite, 99)) if len(finite) else np.nan,
            })
    summary = pd.DataFrame([{
        "development_windows": len(values["window_ids"]),
        "local_beats": len(values["local_features"]),
        "global_wave_features": len(GLOBAL_WAVE_NAMES),
        "local_wave_features": len(LOCAL_WAVE_NAMES),
        "holdout_accessed": metadata["holdout_accessed"],
        "checksum": "passed",
    }])
    return summary, pd.DataFrame(rows)
