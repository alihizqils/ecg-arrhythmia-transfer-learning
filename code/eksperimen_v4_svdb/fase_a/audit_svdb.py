"""Fase A: audit MIT-BIH Supraventricular Arrhythmia Database (SVDB).

Audit ini hanya membaca berkas mentah dan tidak mengubah dataset atau output
eksperimen apa pun. Pelabelan jendela memakai kebijakan kanonik proyek
(`configs/base.yaml` + `ecgtl.data.labeling`) agar hitungan window SVDB dapat
dibandingkan langsung dengan mitdb. Sebagai kontrol, simulasi yang sama
dijalankan pada mitdb dan dicocokkan dengan manifest hierarkis yang ada.

Keluaran ditulis ke folder `hasil/` di samping skrip ini.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from ecgtl.config import load_config  # noqa: E402
from ecgtl.data.annotations import (  # noqa: E402
    NON_BEAT_SYMBOLS,
    load_record_annotations,
    parse_rhythm_label,
)
from ecgtl.data.build_dataset import load_channel_signal, select_capped_windows  # noqa: E402
from ecgtl.data.labeling import LabelingPolicy, label_record  # noqa: E402
from ecgtl.data.segmentation import preprocess_record  # noqa: E402
from ecgtl.hierarchical.data import detected_peaks, match_peaks  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = Path(__file__).resolve().parent / "hasil"
CFG = load_config(ROOT / "configs" / "base.yaml")
POLICY = LabelingPolicy.from_config(CFG)
BEAT_AAMI = CFG.require("labeling.beat_aami")
S_SYMBOLS = list(BEAT_AAMI["S"])
TARGET_FS = float(CFG.require("preprocessing.target_fs"))
BANDPASS = CFG.require("preprocessing.bandpass")
WINDOW_S = float(CFG.require("windowing.duration_s"))
OVERLAP = float(CFG.require("windowing.overlap"))
CHANNELS = CFG.require("windowing.channel_preference")
CAP_CANONICAL = int(CFG.require("sampling.max_windows_per_record_per_class"))
DETECTOR = {"fs": int(TARGET_FS), "detector_offset": 8}
TOLERANCE = 38  # sampel pada 250 Hz, sama dengan gerbang mutu detektor proyek
CLASSES = ["N", "S", "V", "AF", "VT"]


def annotation_audit(directory: Path, record: str) -> dict:
    header = wfdb.rdheader(str(directory / record))
    ann = wfdb.rdann(str(directory / record), "atr")
    symbols = Counter(str(s) for s in ann.symbol)
    rhythms = Counter(
        label for label in (parse_rhythm_label(n) for n in ann.aux_note) if label
    )
    known = {s for group in BEAT_AAMI.values() for s in group}
    unknown = {s: n for s, n in symbols.items() if s not in known and s not in NON_BEAT_SYMBOLS}
    row = {
        "record": record,
        "fs": float(header.fs),
        "duration_min": header.sig_len / header.fs / 60,
        "channels": "|".join(header.sig_name),
        "comments": " / ".join(header.comments or []),
        "rhythm_labels": "|".join(f"{k}:{v}" for k, v in sorted(rhythms.items())),
        "unknown_symbols": json.dumps(unknown) if unknown else "",
    }
    for aami, group in BEAT_AAMI.items():
        row[f"beats_{aami}"] = sum(symbols.get(s, 0) for s in group)
    for s in S_SYMBOLS:
        row[f"beats_S_{s}"] = symbols.get(s, 0)
    return row, symbols


def window_audit(database: str, directory: Path, record: str, has_beats: bool) -> list[dict]:
    rec = load_record_annotations(
        database, directory, record, beat_aami=BEAT_AAMI,
        interruption_labels=CFG.require("labeling.rhythm_interruption"),
        read_beats=has_beats,
    )
    report = label_record(rec, WINDOW_S, OVERLAP, POLICY)
    rows = []
    for w in report.windows:
        beats = rec.beats_in(w.start, w.end)
        s_counts = Counter(b.symbol for b in beats if b.aami == "S")
        rows.append({
            "database": database, "record": record, "start": w.start, "label": w.label,
            "n_beats": w.n_beats, "n_S_beats": sum(s_counts.values()),
            "dominant_S_subtype": s_counts.most_common(1)[0][0] if s_counts else "",
            **{f"S_{s}": s_counts.get(s, 0) for s in S_SYMBOLS},
        })
    rejections = Counter(alasan for (_, alasan) in report.rejections.elements())
    return rows, report.n_candidates, rejections


def detector_audit(directory: Path, record: str) -> dict:
    """Mutu GQRS setelah harmonisasi 128 -> 250 Hz, dicek terhadap anotasi beat."""
    raw, fs, channel = load_channel_signal(directory, record, CHANNELS)
    nan_fraction = float(np.isnan(raw).mean())
    signal = preprocess_record(raw, fs, TARGET_FS, BANDPASS).signal
    peaks = detected_peaks(signal, DETECTOR)
    ann = wfdb.rdann(str(directory / record), "atr")
    symbol_map = {s: c for c, group in BEAT_AAMI.items() for s in group}
    selected = [(s, symbol_map[c]) for s, c in zip(ann.sample, ann.symbol) if c in symbol_map]
    reference = np.rint(np.array([s for s, _ in selected]) * TARGET_FS / fs).astype(int)
    labels = [c for _, c in selected]
    matches = match_peaks(reference, peaks, TOLERANCE)
    hit = {i for i, _ in matches.values()}
    row = {
        "record": record, "channel": channel, "nan_fraction": nan_fraction,
        "raw_std_mV": float(np.nanstd(raw)),
        "reference": len(reference), "detected": len(peaks), "matched": len(matches),
        "detector_f1": 2 * len(matches) / (len(reference) + len(peaks)) if len(reference) + len(peaks) else np.nan,
    }
    for c in ["N", "S", "V"]:
        n = labels.count(c)
        row[f"reference_{c}"] = n
        row[f"recall_{c}"] = sum(labels[i] == c for i in hit) / n if n else np.nan
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    svdb = RAW / "svdb"
    records = sorted(p.stem for p in svdb.glob("*.hea"))

    ann_rows, symbol_total = [], Counter()
    for r in records:
        row, symbols = annotation_audit(svdb, r)
        ann_rows.append(row)
        symbol_total.update(symbols)
    annotations = pd.DataFrame(ann_rows)
    annotations.to_csv(OUT / "svdb_annotasi_per_rekaman.csv", index=False)
    pd.Series(symbol_total).sort_values(ascending=False).rename("count").to_csv(OUT / "svdb_simbol_total.csv")

    windows, rejection_total, candidates = [], Counter(), 0
    for r in records:
        rows, n, rejections = window_audit("svdb", svdb, r, True)
        windows += rows
        candidates += n
        rejection_total.update(rejections)
    windows = pd.DataFrame(windows)
    windows.to_csv(OUT / "svdb_window_berlabel.csv", index=False)

    detector = pd.DataFrame([detector_audit(svdb, r) for r in records])
    detector.to_csv(OUT / "svdb_detektor_kualitas.csv", index=False)

    per_record = (
        windows.pivot_table(index="record", columns="label", values="start", aggfunc="size", fill_value=0)
        .reindex(index=records, columns=CLASSES, fill_value=0)
        .add_prefix("windows_")
    )
    per_record = annotations.set_index("record").join(per_record).join(
        detector.set_index("record")[["channel", "nan_fraction", "detector_f1", "recall_N", "recall_S", "recall_V"]]
    )
    per_record.to_csv(OUT / "svdb_ringkasan_per_rekaman.csv")

    # Kontrol: simulasi yang sama pada mitdb harus mereproduksi manifest kanonik.
    mit_windows = []
    for r in sorted(p.stem for p in (RAW / "mitdb").glob("*.hea")):
        rows, _, _ = window_audit("mitdb", RAW / "mitdb", r, True)
        mit_windows += rows
    mit = pd.DataFrame(mit_windows)
    manifest = pd.read_csv(ROOT / "data/processed/hierarchical_10s_local_v1/global_manifest.csv", dtype={"record": str})
    from ecgtl.data.labeling import LabeledWindow, Sumber  # noqa: E402
    capped = select_capped_windows(
        [LabeledWindow("mitdb", r.record, r.start, r.start + 1, r.label, Sumber.DETAK, 0.0, r.n_beats)
         for r in mit.itertuples()],
        CAP_CANONICAL,
    )
    simulated = Counter(w.label for w in capped)
    actual = manifest[manifest.database == "mitdb"].label.value_counts().to_dict()
    control = {c: {"simulasi": simulated.get(c, 0), "manifest": int(actual.get(c, 0))} for c in CLASSES}

    s = windows[windows.label == "S"]
    summary = {
        "records": len(records),
        "fs": sorted(annotations.fs.unique().tolist()),
        "channels": annotations.channels.value_counts().to_dict(),
        "beats_total": {k: int(annotations[f"beats_{k}"].sum()) for k in BEAT_AAMI},
        "beats_S_subtype": {k: int(annotations[f"beats_S_{k}"].sum()) for k in S_SYMBOLS},
        "records_with_S_beats": int((annotations.beats_S > 0).sum()),
        "window_candidates": candidates,
        "windows_labeled_precap": windows.label.value_counts().to_dict(),
        "records_with_S_windows": int(s.record.nunique()),
        "S_windows_by_dominant_subtype": s.dominant_S_subtype.value_counts().to_dict(),
        "rejections": dict(rejection_total),
        "detector_f1_overall": float(2 * detector.matched.sum() / (detector.reference.sum() + detector.detected.sum())),
        "detector_records_f1_below_0.95": detector.loc[detector.detector_f1 < 0.95, "record"].tolist(),
        "kontrol_mitdb_cap200": control,
    }
    (OUT / "ringkasan_fase_a.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
