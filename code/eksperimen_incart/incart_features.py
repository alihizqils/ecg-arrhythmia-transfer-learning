"""Fitur beat INCART dengan fungsi studi beat yang sama persis (eksperimen_v4_svdb/fase_e_ns/fitur_ns.py).

    python eksperimen_incart/incart_features.py [--workers 4]

Uji lintas database (8 Nov 2026): data, manifest, dan output V4 tidak diubah.
Satu-satunya penyesuaian: INCART punya 12 lead, sedangkan loader studi beat mengasumsikan 2 kanal. Lead utama = II
(setara MLII), lead kedua = V1 (lead kedua MITDB umumnya V1). Preprocessing (resample 257 -> 250 Hz, bandpass
0,5-40 Hz), detektor GQRS + offset 8, dan semua fungsi fitur identik.
Keluaran: eksperimen_incart/fitur/ (meta.csv, fitur.npz); tidak menimpa.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eksperimen_v4_svdb" / "fase_e_ns"))
import fitur_ns as NS  # noqa: E402
from wfdb.processing import gqrs_detect  # noqa: E402

DB, LEAD, LEAD2 = "incartdb", "II", "V1"


def load_record_incart(database, record, channel, fs, annotations=True):
    raw, fs_raw, used = NS.load_channel_signal(NS.RAW / database, record, [channel])
    assert used == channel
    signal = NS.preprocess_record(raw, fs_raw, fs, NS.BANDPASS).signal.astype(np.float64)
    full = wfdb.rdrecord(str(NS.RAW / database / record))
    other = list(full.sig_name).index(LEAD2)
    second = NS.preprocess_record(np.nan_to_num(full.p_signal[:, other].astype(np.float64)), fs_raw, fs, NS.BANDPASS).signal
    offset = int(round(8 * fs / 250))
    peaks = gqrs_detect(sig=signal, fs=fs) + offset
    peaks = np.unique(peaks[(peaks >= 0) & (peaks < len(signal))]).astype(np.int64)
    ann = wfdb.rdann(str(NS.RAW / database / record), "atr")
    keep = [i for i, s in enumerate(ann.symbol) if s in NS.BEAT_SYMBOLS]
    ref = np.rint(np.asarray(ann.sample)[keep] * fs / fs_raw).astype(np.int64)
    symbols = np.asarray(ann.symbol)[keep]
    matches = NS.match_peaks(ref, peaks, int(round(38 * fs / 250)))
    symbol = np.full(len(peaks), "unmatched", dtype=object); distance_ms = np.full(len(peaks), np.nan)
    for j, (i, dist) in matches.items():
        symbol[j] = symbols[i]; distance_ms[j] = dist * 1000 / fs
    return {"signal": signal, "second": np.asarray(second, np.float64), "peaks": peaks, "fs": float(fs),
            "symbol": symbol, "distance_ms": distance_ms, "n_reference": len(ref), "n_matched": len(matches)}


NS.load_record = load_record_incart  # berlaku juga di proses worker (modul ini diimpor ulang saat spawn)


def process(args):
    return NS.process(args)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = HERE / "fitur"
    if (out / "info.json").exists():
        sys.exit(f"{out} sudah ada; tidak ditimpa.")
    out.mkdir(parents=True, exist_ok=True)
    recs = pd.read_csv(HERE / "data" / "incart_records.csv", dtype={"record": str})
    rows = [{"database": DB, "record": r.record, "patient_id": r.patient_id, "split": "development", "fold": r.fold,
             "channel": LEAD} for r in recs.itertuples()]
    t = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        res = list(pool.map(process, [(r, 250) for r in rows]))
    meta = pd.concat([r[0] for r in res], ignore_index=True); feats = pd.concat([r[1] for r in res], ignore_index=True)
    meta.to_csv(out / "meta.csv", index=False)
    np.savez_compressed(out / "fitur.npz", X=feats.to_numpy(np.float32), names=np.asarray(feats.columns))
    pd.DataFrame([r[2] for r in res]).to_csv(out / "qa_detektor.csv", index=False)
    info = {"rekaman": len(rows), "beat": len(meta), "fitur": feats.shape[1], "lead": [LEAD, LEAD2],
            "label": meta.label.value_counts().to_dict(), "detik": round(time.time() - t, 1)}
    (out / "info.json").write_text(json.dumps(info, indent=2)); print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
