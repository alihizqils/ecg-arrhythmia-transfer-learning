"""Ekstraksi fitur fisiologis A5 dari waveform development tanpa label."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecgtl.hierarchical.data import read_manifest
from ecgtl.hierarchical.protocol import atomic_json, digest


LOCAL_PHYSIOLOGY_NAMES = [
    "qrs_energy_width_s",
    "qrs_peak_to_peak_over_scale",
    "qrs_abs_area_over_scale",
    "qrs_energy_over_scale2",
    "qrs_max_abs_slope_over_scale",
    "qrs_template_correlation",
    "qrs_template_nrmse",
    "pre_qrs_energy_ratio",
    "pre_qrs_peak_ratio",
    "pre_qrs_max_abs_slope_over_scale",
    "atrial_candidate_to_r_s",
    "atrial_candidate_prominence",
]

GLOBAL_PHYSIOLOGY_NAMES = [
    "rr_permutation_entropy",
    "rr_turning_point_ratio",
    "drr_std_over_median",
    "pnn20",
    "pnn50",
    "rr_lag1_autocorrelation",
    "rr_alternation_ratio",
    "qrs_template_distance_median",
    "qrs_template_distance_max",
    "qrs_width_coefficient_of_variation",
    "pre_qrs_energy_ratio_median",
    "atrial_candidate_timing_std_s",
]


def _robust_scale(signal: np.ndarray) -> float:
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        q25, q75 = np.percentile(signal, [25, 75])
        scale = (q75 - q25) / 1.349
    return float(scale) if np.isfinite(scale) and scale >= 1e-6 else np.nan


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else np.nan


def _permutation_entropy(values: np.ndarray, order: int = 3) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < order + 1 or not np.isfinite(values).all():
        return np.nan
    patterns = [tuple(np.argsort(values[index : index + order], kind="stable"))
                for index in range(len(values) - order + 1)]
    counts = pd.Series(patterns).value_counts(normalize=True).to_numpy(dtype=np.float64)
    entropy = -np.sum(counts * np.log(counts))
    maximum = np.log(np.math.factorial(order)) if hasattr(np, "math") else np.log(6.0)
    return float(entropy / maximum) if maximum > 0 else np.nan


def _global_rr_features(peaks: np.ndarray, fs: float) -> np.ndarray:
    rr = np.diff(peaks.astype(np.float64)) / fs
    rr = rr[np.isfinite(rr) & (rr > 0)]
    output = np.full(len(GLOBAL_PHYSIOLOGY_NAMES), np.nan, dtype=np.float32)
    if len(rr) < 2:
        return output
    median = float(np.median(rr))
    drr = np.diff(rr)
    output[0] = _permutation_entropy(rr)
    if len(rr) >= 3:
        middle = rr[1:-1]
        turning = ((middle > rr[:-2]) & (middle > rr[2:])) | ((middle < rr[:-2]) & (middle < rr[2:]))
        output[1] = float(np.mean(turning))
    if len(drr):
        output[2] = float(np.std(drr, ddof=1) / median) if len(drr) > 1 and median > 0 else np.nan
        output[3] = float(np.mean(np.abs(drr) > 0.020))
        output[4] = float(np.mean(np.abs(drr) > 0.050))
    if len(rr) >= 3 and np.std(rr[:-1]) > 1e-12 and np.std(rr[1:]) > 1e-12:
        output[5] = _safe_correlation(rr[:-1], rr[1:])
    if len(drr) >= 2:
        output[6] = float(np.mean(np.sign(drr[:-1]) * np.sign(drr[1:]) < 0))
    return output


def _beat_features(signal: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
    """Fitur tetap terhadap R-peak; tidak memakai label beat/window."""
    output = np.full((len(peaks), len(LOCAL_PHYSIOLOGY_NAMES)), np.nan, dtype=np.float32)
    scale = _robust_scale(signal)
    if not np.isfinite(scale):
        return output

    qrs_left, qrs_right = int(round(0.10 * fs)), int(round(0.14 * fs))
    template_left, template_right = int(round(0.12 * fs)), int(round(0.16 * fs))
    pre_left, pre_right = int(round(0.28 * fs)), int(round(0.06 * fs))

    template_segments: list[np.ndarray | None] = []
    for peak in peaks:
        start, stop = int(peak) - template_left, int(peak) + template_right + 1
        template_segments.append(signal[start:stop].astype(np.float64) if start >= 0 and stop <= len(signal) else None)
    complete = [segment for segment in template_segments if segment is not None]
    median_template = np.median(np.stack(complete), axis=0) if complete else None

    for index, peak_value in enumerate(peaks):
        peak = int(peak_value)
        qrs_start, qrs_stop = peak - qrs_left, peak + qrs_right + 1
        pre_start, pre_stop = peak - pre_left, peak - pre_right + 1
        if qrs_start < 0 or qrs_stop > len(signal) or pre_start < 0 or pre_stop > len(signal):
            continue
        qrs = signal[qrs_start:qrs_stop].astype(np.float64)
        pre = signal[pre_start:pre_stop].astype(np.float64)
        qrs_centered = qrs - np.median(qrs)
        pre_centered = pre - np.median(pre)
        derivative = np.diff(qrs_centered)
        derivative_energy = derivative**2
        total_derivative_energy = float(derivative_energy.sum())
        if total_derivative_energy > 1e-12:
            cumulative = np.cumsum(derivative_energy) / total_derivative_energy
            left = int(np.searchsorted(cumulative, 0.05))
            right = int(np.searchsorted(cumulative, 0.95))
            output[index, 0] = max(1, right - left) / fs
        qrs_peak_to_peak = float(np.ptp(qrs_centered))
        output[index, 1] = qrs_peak_to_peak / scale
        output[index, 2] = float(np.sum(np.abs(qrs_centered)) / (scale * fs))
        output[index, 3] = float(np.mean(qrs_centered**2) / (scale**2))
        output[index, 4] = float(np.max(np.abs(derivative)) / scale) if len(derivative) else np.nan

        segment = template_segments[index]
        if segment is not None and median_template is not None:
            correlation = _safe_correlation(segment, median_template)
            output[index, 5] = correlation
            denominator = np.std(median_template)
            output[index, 6] = (
                float(np.sqrt(np.mean((segment - median_template) ** 2)) / denominator)
                if denominator > 1e-12 else np.nan
            )

        pre_energy = float(np.mean(pre_centered**2))
        qrs_energy = float(np.mean(qrs_centered**2))
        output[index, 7] = pre_energy / qrs_energy if qrs_energy > 1e-12 else np.nan
        output[index, 8] = float(np.max(np.abs(pre_centered)) / qrs_peak_to_peak) if qrs_peak_to_peak > 1e-12 else np.nan
        output[index, 9] = float(np.max(np.abs(np.diff(pre_centered))) / scale) if len(pre_centered) > 1 else np.nan
        candidate = int(np.argmax(np.abs(pre_centered)))
        output[index, 10] = (pre_left - candidate) / fs
        baseline_mad = np.median(np.abs(pre_centered - np.median(pre_centered)))
        output[index, 11] = (
            float(np.abs(pre_centered[candidate]) / (1.4826 * baseline_mad))
            if baseline_mad > 1e-12 else np.nan
        )
    return output


def _append_global_morphology(global_features: np.ndarray, local_features: np.ndarray) -> np.ndarray:
    output = global_features.copy()
    correlation = local_features[:, 5]
    distance = 1.0 - correlation[np.isfinite(correlation)]
    if len(distance):
        output[7] = float(np.median(distance))
        output[8] = float(np.max(distance))
    width = local_features[:, 0]
    width = width[np.isfinite(width)]
    if len(width) > 1 and np.mean(width) > 1e-12:
        output[9] = float(np.std(width, ddof=1) / np.mean(width))
    pre_energy = local_features[:, 7]
    pre_energy = pre_energy[np.isfinite(pre_energy)]
    if len(pre_energy):
        output[10] = float(np.median(pre_energy))
    timing = local_features[:, 10]
    timing = timing[np.isfinite(timing)]
    if len(timing) > 1:
        output[11] = float(np.std(timing, ddof=1))
    return output


def _source_hash(protocol, development: pd.DataFrame, local: pd.DataFrame) -> str:
    payload = {
        "protocol_hash": protocol.hash,
        "implementation_sha256": digest(Path(__file__)),
        "development_ids": hashlib.sha256("\n".join(development.window_id.astype(str)).encode()).hexdigest(),
        "local_crop_ids": hashlib.sha256("\n".join(local.crop_id.astype(str)).encode()).hexdigest(),
        "features_local": LOCAL_PHYSIOLOGY_NAMES,
        "features_global": GLOBAL_PHYSIOLOGY_NAMES,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_physiology_cache(protocol, force: bool = False) -> Path:
    """Bangun cache hanya untuk development set; holdout tidak dimuat ke output."""
    protocol.verify()
    global_manifest = read_manifest(protocol.prepared / "global_manifest.csv")
    development = global_manifest[global_manifest.split == "development"].copy()
    development["window_id"] = development.window_id.astype(str)
    local = read_manifest(protocol.prepared / "local_manifest.csv")
    local = local[local.split == "development"].copy()
    local["parent_window_id"] = local.parent_window_id.astype(str)
    local = local[local.parent_window_id.isin(set(development.window_id))]
    source_hash = _source_hash(protocol, development, local)
    folder = protocol.out / "ablation_rr" / "a5_physiology_cache" / source_hash[:16]
    complete = folder / "complete.json"
    if complete.exists() and not force:
        metadata = json.loads(complete.read_text())
        if metadata["source_hash"] != source_hash:
            raise ValueError("Identitas cache fisiologi berbeda.")
        return folder
    folder.mkdir(parents=True, exist_ok=True)

    local_groups = {key: frame.sort_values("rpeak_sample").reset_index(drop=True)
                    for key, frame in local.groupby("parent_window_id", sort=False)}
    window_ids: list[str] = []
    global_values: list[np.ndarray] = []
    local_values: list[np.ndarray] = []
    offsets = [0]
    fs = float(protocol.cfg["data"]["fs"])

    for (database, record), windows in development.groupby(["database", "record"], sort=False):
        record_folder = protocol.prepared / "records" / str(database) / str(record)
        waveform_path = record_folder / "waveforms.npy"
        indices_path = record_folder / "window_indices.npy"
        if not waveform_path.exists() or not indices_path.exists():
            raise FileNotFoundError(f"Waveform prepared tidak tersedia: {record_folder}")
        waveforms = np.load(waveform_path, mmap_mode="r")
        indices = np.load(indices_path)
        index_to_row = {int(value): row for row, value in enumerate(indices)}
        for row in windows.itertuples(index=False):
            window_id = str(row.window_id)
            if window_id not in local_groups:
                raise ValueError(f"Local manifest tidak tersedia untuk {window_id}.")
            crop_rows = local_groups[window_id]
            waveform = np.asarray(waveforms[index_to_row[int(row.index)]], dtype=np.float64)
            peaks_absolute = crop_rows.rpeak_sample.to_numpy(dtype=np.int64)
            peaks = peaks_absolute - int(row.start_target_fs)
            if np.any(peaks < 0) or np.any(peaks >= len(waveform)):
                raise ValueError(f"Koordinat R-peak di luar window pada {window_id}.")
            beat = _beat_features(waveform, peaks, fs)
            global_feature = _append_global_morphology(_global_rr_features(peaks, fs), beat)
            window_ids.append(window_id)
            global_values.append(global_feature)
            local_values.append(beat)
            offsets.append(offsets[-1] + len(beat))

    if window_ids != development.window_id.astype(str).tolist():
        raise ValueError("Urutan cache fisiologi tidak sama dengan development manifest.")
    local_array = np.concatenate(local_values, axis=0).astype(np.float32)
    global_array = np.stack(global_values).astype(np.float32)
    np.savez_compressed(
        folder / "features.npz",
        window_ids=np.asarray(window_ids),
        global_features=global_array,
        local_features=local_array,
        local_offsets=np.asarray(offsets, dtype=np.int64),
    )
    atomic_json(
        complete,
        {
            "source_hash": source_hash,
            "protocol_hash": protocol.hash,
            "implementation_sha256": digest(Path(__file__)),
            "development_windows": len(window_ids),
            "local_beats": len(local_array),
            "global_feature_names": GLOBAL_PHYSIOLOGY_NAMES,
            "local_feature_names": LOCAL_PHYSIOLOGY_NAMES,
            "holdout_accessed": False,
            "features_sha256": digest(folder / "features.npz"),
        },
    )
    return folder


def audit_physiology_cache(protocol, folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = json.loads((folder / "complete.json").read_text())
    if metadata["holdout_accessed"]:
        raise ValueError("Cache fisiologi tidak boleh mengakses holdout.")
    if digest(folder / "features.npz") != metadata["features_sha256"]:
        raise ValueError("Checksum cache fisiologi tidak sesuai.")
    values = np.load(folder / "features.npz")
    rows = []
    for scope, names, array in [
        ("global", GLOBAL_PHYSIOLOGY_NAMES, values["global_features"]),
        ("local", LOCAL_PHYSIOLOGY_NAMES, values["local_features"]),
    ]:
        for index, name in enumerate(names):
            column = array[:, index]
            finite = column[np.isfinite(column)]
            rows.append(
                {
                    "scope": scope,
                    "feature": name,
                    "available": len(finite),
                    "missing": int(len(column) - len(finite)),
                    "availability": float(len(finite) / len(column)),
                    "p01": float(np.percentile(finite, 1)) if len(finite) else np.nan,
                    "median": float(np.median(finite)) if len(finite) else np.nan,
                    "p99": float(np.percentile(finite, 99)) if len(finite) else np.nan,
                }
            )
    summary = pd.DataFrame(
        [{
            "development_windows": len(values["window_ids"]),
            "local_beats": len(values["local_features"]),
            "global_features": len(GLOBAL_PHYSIOLOGY_NAMES),
            "local_features": len(LOCAL_PHYSIOLOGY_NAMES),
            "holdout_accessed": metadata["holdout_accessed"],
            "checksum": "passed",
        }]
    )
    return summary, pd.DataFrame(rows)
