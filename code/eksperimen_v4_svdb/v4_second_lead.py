"""Fitur template rekaman pada lead EKG kedua (Revisi 11).

Seluruh rekaman MITDB, SVDB, AFDB, dan VFDB berisi dua sinyal (Holter dua lead).
Citra global dan potongan beat tetap dari lead utama (kanal manifest). Modul ini
hanya menghitung 9 fitur tingkat window yang sama dengan RECORD_TEMPLATE_NAMES
(korelasi QRS dan area gelombang P terhadap template rekaman) pada lead lainnya,
memakai preprocessing kanonik dan posisi R-peak lead utama. Gelombang P sering
lebih jelas pada lead kedua (MITDB umumnya V1), dan gelombang P adalah ciri utama
beat S. Label, split, dan fold tidak dibaca.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

import v4_record_template as template
from ecgtl.data.segmentation import preprocess_record

SECOND_LEAD_NAMES = [f"lead2_{name}" for name in template.RECORD_TEMPLATE_NAMES]


def _source_hash(protocol, windows: pd.DataFrame, local: pd.DataFrame) -> str:
    payload = hashlib.sha256()
    for path in [Path(__file__), Path(template.__file__)]:
        payload.update(path.read_bytes())
    payload.update(str(protocol.hash).encode())
    payload.update(pd.util.hash_pandas_object(
        windows[["window_id", "database", "record", "channel", "start_target_fs"]], index=False).values.tobytes())
    payload.update(pd.util.hash_pandas_object(local[["parent_window_id", "rpeak_sample"]], index=False).values.tobytes())
    return payload.hexdigest()


def build_second_lead_features(protocol) -> dict[str, np.ndarray]:
    """Hitung (atau muat dari cache) fitur SECOND_LEAD_NAMES per window."""
    cfg = protocol.cfg["data"]
    fs = float(cfg["fs"])
    windows = pd.read_csv(protocol.prepared / "global_manifest.csv", dtype={"window_id": str, "record": str})
    local = pd.read_csv(protocol.prepared / "local_manifest.csv", dtype={"parent_window_id": str})
    local = local[local.parent_window_id.isin(set(windows.window_id))].sort_values(["parent_window_id", "rpeak_sample"])
    source = _source_hash(protocol, windows, local)
    folder = protocol.out / "feature_store" / f"second_lead_{source[:16]}"
    marker = folder / "complete.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        values = np.load(folder / "second_lead.npz")
        if meta["source_hash"] != source or list(values["names"]) != SECOND_LEAD_NAMES:
            raise ValueError("Integritas cache fitur lead kedua tidak sesuai.")
        return dict(zip(values["window_ids"].astype(str), values["features"]))
    folder.mkdir(parents=True, exist_ok=True)
    qrs_left, qrs_right = int(round(template._QRS[0] * fs)), int(round(template._QRS[1] * fs))
    pre_left, pre_right = int(round(template._PRE[0] * fs)), int(round(template._PRE[1] * fs))
    peaks_by_window = {key: frame.rpeak_sample.to_numpy(np.int64) for key, frame in local.groupby("parent_window_id")}
    result, leads = {}, []
    for (database, record, channel), rows in windows.groupby(["database", "record", "channel"], sort=False):
        source_record = wfdb.rdrecord(str(protocol.raw / str(database) / str(record)))
        names = list(source_record.sig_name)
        if len(names) != 2:
            raise ValueError(f"Rekaman {database}/{record} tidak berisi tepat dua lead.")
        primary = names.index(channel)
        other = 1 - primary
        leads.append({"database": database, "record": record, "primary": channel, "second": names[other]})
        signal = preprocess_record(source_record.p_signal[:, other].astype(np.float64), source_record.fs,
                                   cfg["fs"], cfg["bandpass"]).signal
        beats = []
        for row in rows.itertuples(index=False):
            start = int(row.start_target_fs)
            segment = np.nan_to_num(signal[start:start + cfg["window_samples"]].astype(np.float64))
            peaks = peaks_by_window.get(row.window_id, np.empty(0, np.int64)) - start
            for order, peak in enumerate(peaks):
                qrs = segment[peak - qrs_left:peak + qrs_right + 1] if peak - qrs_left >= 0 and peak + qrs_right < len(segment) else None
                pre = segment[peak - pre_left:peak - pre_right + 1] if peak - pre_left >= 0 and peak - pre_right < len(segment) else None
                beats.append((row.window_id, (peak - peaks[order - 1]) / fs if order > 0 else np.nan,
                              None if qrs is None else template._znorm(qrs), None if pre is None else template._znorm(pre)))
        qrs_all = [beat[2] for beat in beats if beat[2] is not None]
        pre_all = [beat[3] for beat in beats if beat[3] is not None]
        qrs_template = template._znorm(np.median(np.stack(qrs_all), 0)) if qrs_all else None
        pre_template = template._znorm(np.median(np.stack(pre_all), 0)) if pre_all else None
        rr_all = np.array([beat[1] for beat in beats], dtype=np.float64)
        rr_median = float(np.nanmedian(rr_all)) if np.isfinite(rr_all).any() else np.nan
        per_window: dict[str, list] = {}
        for window_id, rr, qrs, pre in beats:
            per_window.setdefault(window_id, []).append((
                rr / rr_median if np.isfinite(rr_median) and rr_median > 0 else np.nan,
                template._correlation(qrs, qrs_template), template._correlation(pre, pre_template)))
        for window_id in rows.window_id:
            result[window_id] = template._window_features(np.asarray(per_window.get(window_id, []), dtype=np.float64).reshape(-1, 3))
    ids = windows.window_id.tolist()
    with (folder / "second_lead.tmp").open("wb") as stream:
        np.savez_compressed(stream, window_ids=np.asarray(ids), names=np.asarray(SECOND_LEAD_NAMES),
                            features=np.stack([result[i] for i in ids]).astype(np.float32))
    os.replace(folder / "second_lead.tmp", folder / "second_lead.npz")
    pd.DataFrame(leads).to_csv(folder / "lead_pairs.csv", index=False)
    marker.write_text(json.dumps({"source_hash": source, "windows": len(ids), "label_columns_read_for_features": False}, indent=2))
    return result
