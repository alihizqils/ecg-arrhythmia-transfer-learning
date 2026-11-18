"""Protokol dan persiapan data hierarkis V4 (mitdb, afdb, vfdb, SVDB).

Modul ini memakai ulang pipeline hierarkis tanpa mengubah kode lama. Perbedaan
terhadap `ecgtl.hierarchical` hanya:

1. Jalur konfigurasi, data, dan output menunjuk ke namespace V4.
2. Citra global dirender dari `data/processed/v4_svdb/label_manifest.csv`
   (Fase B), dengan fungsi preprocessing dan renderer kanonik.
3. Label beat lokal development diambil dari semua database dengan anotasi beat
   bertipe (`beat_label_databases`: mitdb dan svdb), bukan hanya mitdb.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import yaml
from tqdm.auto import tqdm

FOLDER = Path(__file__).resolve().parent
CODE_ROOT = FOLDER.parent
for path in [CODE_ROOT / "src", CODE_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ecgtl.config import load_config  # noqa: E402
from ecgtl.data.build_dataset import DatasetSpec, load_channel_signal  # noqa: E402
from ecgtl.data.rendering import render_waveform  # noqa: E402
from ecgtl.data.segmentation import is_flat, minmax_normalize, preprocess_record  # noqa: E402
from ecgtl.hierarchical.data import (  # noqa: E402
    BEATS, detected_peaks, detector_report, match_peaks, read_manifest,
    validate_inheritance, window_views,
)
from ecgtl.hierarchical.protocol import CLASSES, Protocol, atomic_json, digest, environment  # noqa: E402

LABEL_MANIFEST = CODE_ROOT / "data/processed/v4_svdb/label_manifest.csv"
PROTOCOL_LOCK_B = CODE_ROOT / "data/processed/v4_svdb/protocol_v4_lock.json"
DATA_CONFIG = load_config(CODE_ROOT / "configs/base.yaml", [FOLDER / "protokol_v4.yaml"])
OLD_SOURCE = CODE_ROOT / "data/processed/w10s_448x224_ov50"


class ProtocolV4(Protocol):
    """Antarmuka sama dengan `Protocol`, tetapi seluruh jalur di namespace V4."""

    def __init__(self, code_root=CODE_ROOT, output_root=None, data_root=None):
        self.code = Path(code_root).resolve()
        self.folder = FOLDER
        self.cfg = yaml.safe_load((self.folder / "hierarchical_v4.yaml").read_text())
        self.root = Path(data_root).resolve() if data_root else self.code
        self.out = Path(output_root).resolve() if output_root else self.folder / "outputs"
        self.prepared = self.root / self.cfg["data"]["prepared"]
        self.source = self.root / self.cfg["data"]["source"]
        self.raw = self.root / self.cfg["data"]["raw"]
        self.selection = self.code / "pilot_durasi_lokal/selected_local_window.json"
        locked = json.loads(self.selection.read_text())
        if (locked["implemented_crop_samples"], locked["detector_offset_samples"],
                locked["holdout_used_for_selection"]) != (312, 8, False):
            raise ValueError("Konfigurasi pilot tidak sesuai protokol.")
        if self.cfg["classes"] != CLASSES:
            raise ValueError("Urutan kelas tidak konsisten.")
        self.out.mkdir(parents=True, exist_ok=True)

    def lock(self):
        files = [self.folder / "hierarchical_v4.yaml", self.folder / "protokol_v4.yaml",
                 Path(__file__).resolve(), self.code / "configs/base.yaml", self.selection]
        files += sorted((self.code / "src/ecgtl/hierarchical").glob("*.py"))
        files += [self.code / "src/ecgtl/data" / n for n in
                  ["segmentation.py", "rendering.py", "build_dataset.py", "inventory.py",
                   "labeling.py", "annotations.py"]]
        hashes = {p.relative_to(self.code).as_posix(): digest(p) for p in files}
        for root_name, root, names in [
            ("source", self.source, ["manifest.csv", "patient_assignment.csv"]),
            ("prepared", self.prepared, ["global_manifest.csv", "local_manifest.csv",
                                         "data_checksums.json", "distribution.csv"]),
        ]:
            hashes.update({f"{root_name}/{n}": digest(root / n) for n in names})
        hashes["fase_b/protocol_v4_lock.json"] = digest(PROTOCOL_LOCK_B)
        hash_value = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        path = self.out / "protocol_lock.json"
        if path.exists():
            if json.loads(path.read_text())["protocol_hash"] != hash_value:
                raise ValueError("Protokol atau artefak V4 berubah; gunakan direktori hasil baru.")
            self.hash = hash_value
            return hash_value
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.code,
                             capture_output=True, text=True, check=False)
        atomic_json(path, {"protocol_hash": hash_value, "files": hashes,
                           "run_id": self.cfg["run_id"], "environment": environment(),
                           "commit": git.stdout.strip() or None})
        self.hash = hash_value
        return hash_value


def render_global(p: ProtocolV4) -> dict:
    """Render citra global 10 detik untuk seluruh window manifest label V4."""
    spec = DatasetSpec.from_config(DATA_CONFIG)
    channels = DATA_CONFIG.require("windowing.channel_preference")
    bandpass = DATA_CONFIG.require("preprocessing.bandpass")
    labels = read_manifest(LABEL_MANIFEST).sort_values("index").reset_index(drop=True)
    if not np.array_equal(labels["index"], np.arange(len(labels))):
        raise ValueError("Indeks manifest label V4 tidak berurutan.")
    p.source.mkdir(parents=True, exist_ok=True)
    marker = p.source / "complete.json"
    if marker.exists():
        receipt = json.loads(marker.read_text())
        if receipt["label_manifest_sha256"] != digest(LABEL_MANIFEST) or \
                receipt["images_sha256"] != digest(p.source / "images.npy"):
            raise ValueError("Citra global V4 tidak sesuai manifest label; jangan ditimpa.")
        return receipt

    images = np.lib.format.open_memmap(p.source / "images.tmp.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(labels), spec.height, spec.width))
    rows = []
    for (database, record), group in tqdm(labels.groupby(["database", "record"], sort=True),
                                          desc="Render global V4"):
        signal, fs, channel = load_channel_signal(p.raw / database, record, channels)
        pre = preprocess_record(signal, fs, spec.target_fs, bandpass, database, record, channel)
        for row in group.itertuples(index=False):
            start, end = pre.window_bounds_at_target_fs(int(row.start_original), spec.window_seconds)
            segment = pre.extract(start, end)
            if segment is None:
                raise ValueError(f"Window di luar rekaman: {database}/{record} {row.start_original}")
            images[int(row.index)] = render_waveform(
                minmax_normalize(segment), width=spec.width, height=spec.height,
                line_width=spec.line_width, background=spec.background, foreground=spec.foreground)
            rows.append({"index": int(row.index), "channel": channel, "start_target_fs": start,
                         "flat": bool(is_flat(segment))})
    images.flush()
    del images
    (p.source / "images.tmp.npy").replace(p.source / "images.npy")

    rendered = pd.DataFrame(rows).set_index("index").sort_index()
    manifest = labels.join(rendered, on="index")
    columns = ["index", "database", "record", "patient_id", "channel", "label", "label_idx", "source",
               "start_original", "start_target_fs", "n_beats", "rhythm_coverage", "flat"]
    manifest[columns].to_csv(p.source / "manifest.csv", index=False)
    assignment = manifest.drop_duplicates("patient_id")[["patient_id", "split", "fold"]]
    assignment = assignment.assign(
        partisi=assignment.split.map({"development": "latih", "holdout": "uji"}),
        lipatan_validasi=np.where(assignment.fold >= 0, assignment.fold + 1, ""),
    )[["patient_id", "partisi", "lipatan_validasi"]].sort_values("patient_id")
    assignment.to_csv(p.source / "patient_assignment.csv", index=False)
    receipt = {"windows": len(manifest), "label_manifest_sha256": digest(LABEL_MANIFEST),
               "images_sha256": digest(p.source / "images.npy"),
               "manifest_sha256": digest(p.source / "manifest.csv"),
               "flat_windows": int(manifest.flat.sum())}
    atomic_json(marker, receipt)
    return receipt


def verify_old_images(p: ProtocolV4) -> dict:
    """Citra mitdb/afdb/vfdb V4 yang juga ada di dataset kanonik lama harus identik piksel demi piksel.

    Kelas tanpa batas kelas tambahan (`sampling.cap_per_class`) harus seluruhnya ada;
    kelas S/V dipilih ulang dengan batas lebih kecil sehingga hanya irisannya yang dibandingkan.
    """
    keys = ["database", "record", "start_original"]
    new = read_manifest(p.source / "manifest.csv")
    old = read_manifest(OLD_SOURCE / "manifest.csv")
    pairs = new.merge(old, on=keys, suffixes=("_v4", "_old"))
    uncapped = set(CLASSES) - set(DATA_CONFIG.get("sampling.cap_per_class", {}))
    if (pairs.label_old.isin(uncapped)).sum() != old.label.isin(uncapped).sum():
        raise ValueError("Tidak semua window lama kelas tanpa batas tambahan ditemukan di V4.")
    for column in ["label", "channel", "start_target_fs", "flat"]:
        if not (pairs[f"{column}_v4"] == pairs[f"{column}_old"]).all():
            raise ValueError(f"Kolom {column} window lama berbeda.")
    new_images = np.load(p.source / "images.npy", mmap_mode="r")
    old_images = np.load(OLD_SOURCE / "images.npy", mmap_mode="r")
    order = pairs.sort_values("index_v4")
    for start in tqdm(range(0, len(order), 2048), desc="Verifikasi citra lama"):
        chunk = order.iloc[start:start + 2048]
        if not np.array_equal(new_images[chunk.index_v4.to_numpy()], old_images[chunk.index_old.to_numpy()]):
            raise ValueError("Citra window lama berbeda dari dataset kanonik.")
    return {"old_windows_compared": len(pairs), "uncapped_classes_complete": True, "pixel_identical": True}


def write_global_manifest(p: ProtocolV4) -> pd.DataFrame:
    """Gabungkan manifest render dengan split Fase B dan validasi jumlah protokol."""
    cfg = p.cfg["data"]
    source = read_manifest(p.source / "manifest.csv")
    labels = read_manifest(LABEL_MANIFEST).set_index("index")
    df = source.join(labels[["split", "fold", "window_id"]], on="index")
    for part in ["development", "holdout"]:
        sub = df[df.split == part]
        if len(sub) != cfg[f"expected_{part}_windows"] or sub.patient_id.nunique() != cfg[f"expected_{part}_patients"]:
            raise ValueError(f"Jumlah {part} tidak sesuai protokol.")
    if len(df) != cfg["expected_windows"] or df.patient_id.nunique() != cfg["expected_patients"]:
        raise ValueError("Jumlah dataset global tidak sesuai protokol.")
    if not np.array_equal(df.label_idx, df.label.map({c: i for i, c in enumerate(CLASSES)})):
        raise ValueError("Indeks kelas tidak sesuai.")
    p.prepared.mkdir(parents=True, exist_ok=True)
    path = p.prepared / "global_manifest.csv"
    if path.exists() and not read_manifest(path).equals(df):
        raise ValueError("Global manifest V4 tersimpan berbeda; tidak ditimpa.")
    df.to_csv(path, index=False)
    distribution = df.groupby(["split", "fold", "label"]).agg(
        windows=("window_id", "size"), patients=("patient_id", "nunique")).reset_index()
    distribution.to_csv(p.prepared / "distribution.csv", index=False)
    return df


def prepare_record(p: ProtocolV4, rows: pd.DataFrame):
    """Salinan `ecgtl.hierarchical.data.prepare_record` dengan label beat dari `beat_label_databases`."""
    cfg = p.cfg["data"]
    first = rows.iloc[0]
    db, record, part = first.database, first.record, first.split
    folder = p.prepared / "records" / db / record
    folder.mkdir(parents=True, exist_ok=True)
    source_files = sorted((p.raw / db).glob(f"{record}.*"))
    source_hashes = {f.name: digest(f) for f in source_files}
    source_key = {"sources": source_hashes, "manifest": digest(p.prepared / "global_manifest.csv"),
                  "config": cfg}
    if (folder / "done.json").exists():
        done = json.loads((folder / "done.json").read_text())
        if done["source_key"] != source_key:
            raise ValueError(f"Cache sumber berubah: {db}/{record}")
        for name, expected in done["hashes"].items():
            if digest(folder / name) != expected:
                raise ValueError(f"Cache rusak: {folder / name}")
        return read_manifest(folder / "crops.csv"), done["qa"]
    raw, fs, channel = load_channel_signal(p.raw / db, record, [first.channel])
    if channel != first.channel:
        raise ValueError("Kanal sumber berbeda dari manifest.")
    signal = preprocess_record(raw, fs, cfg["fs"], cfg["bandpass"]).signal
    peaks = detected_peaks(signal, cfg)
    ref, labels, matches = np.array([], dtype=int), [], {}
    qa = None
    # Label detak hanya dimuat untuk development dari database beranotasi beat bertipe.
    if db in cfg["beat_label_databases"] and part == "development":
        ann = wfdb.rdann(str(p.raw / db / record), "atr")
        selected = [(s, BEATS[c]) for s, c in zip(ann.sample, ann.symbol) if c in BEATS]
        ref = np.rint(np.array([s for s, _ in selected]) * cfg["fs"] / fs).astype(int)
        labels = [c for _, c in selected]
        matches = match_peaks(ref, peaks, cfg["tolerance_samples"])
        hit = {i for i, _ in matches.values()}
        qa = {"record": record, "database": db, "patient_id": first.patient_id, "fold": int(first.fold),
              "reference": len(ref), "detected": len(peaks), "matched": len(matches),
              "errors_ms": [float((peaks[j] - ref[i]) * 1000 / cfg["fs"]) for j, (i, _) in matches.items()]}
        for c in CLASSES[:3]:
            qa[f"reference_{c}"] = labels.count(c)
            qa[f"matched_{c}"] = sum(labels[i] == c for i in hit)
    local, waveform, entries = [], [], []
    original = np.load(p.source / "images.npy", mmap_mode="r")
    offset = 0
    for row in rows.itertuples():
        start = int(row.start_target_fs)
        segment = signal[start:start + cfg["window_samples"]]
        if len(segment) != cfg["window_samples"]:
            raise ValueError("Panjang parent window tidak lengkap.")
        indices = np.flatnonzero((peaks >= start) & (peaks < start + len(segment)))
        global_image, crops, centers, fallback = window_views(segment, peaks[indices] - start, cfg)
        if not np.array_equal(global_image, original[row.index]):
            raise ValueError(f"Renderer global tidak sama dengan sumber: {row.window_id}")
        waveform.append(segment)
        for k, center in enumerate(centers):
            pair = None if fallback else matches.get(int(indices[k]))
            ann_index = pair[0] if pair else None
            label = labels[ann_index] if pair else "unmatched"
            entries.append({"crop_id": f"{row.window_id}_r{k:03d}", "parent_window_id": row.window_id,
                "patient_id": row.patient_id, "record_id": record, "database": db, "channel": channel,
                "rpeak_sample": int(start + center), "annotation_sample": int(ref[ann_index]) if pair else -1,
                "match_distance_ms": pair[1] * 1000 / cfg["fs"] if pair else -1,
                "beat_label": label, "split": part, "fold": int(row.fold),
                "fallback_peak": fallback, "edge_padding": bool(center < 156 or center + 156 > len(segment)),
                "shard": f"records/{db}/{record}/local.npy", "shard_index": offset + k})
        offset += len(crops)
        local.append(crops)
    np.save(folder / "local.npy", np.concatenate(local))
    np.save(folder / "waveforms.npy", np.stack(waveform))
    np.save(folder / "window_indices.npy", rows["index"].to_numpy())
    result = pd.DataFrame(entries)
    result.to_csv(folder / "crops.csv", index=False)
    names = ["local.npy", "waveforms.npy", "window_indices.npy", "crops.csv"]
    atomic_json(folder / "done.json", {"source_key": source_key, "qa": qa,
                                      "hashes": {n: digest(folder / n) for n in names}})
    return result, qa


def prepare_data(p: ProtocolV4) -> pd.DataFrame:
    """Alur `ecgtl.hierarchical.data.prepare_data` dengan split Fase B dan label beat V4."""
    df = write_global_manifest(p)
    all_crops, audits = [], []
    for part in ["development", "holdout"]:
        groups = list(df[df.split == part].groupby(["database", "record"], sort=True))
        for _, rows in tqdm(groups, desc=f"Persiapan {part}"):
            crops, qa = prepare_record(p, rows)
            all_crops.append(crops)
            if qa is not None:
                audits.append(qa)
        if part == "development":
            detector_report(p, audits)
            pd.DataFrame([{k: v for k, v in a.items() if k != "errors_ms"} for a in audits]).to_csv(
                p.out / "detector_qc" / "records_v4.csv", index=False)
    local = pd.concat(all_crops, ignore_index=True)
    validate_inheritance(df, local)
    local.to_csv(p.prepared / "local_manifest.csv", index=False)
    files = list((p.prepared / "records").rglob("*.npy"))
    atomic_json(p.prepared / "data_checksums.json",
                {"source_images": digest(p.source / "images.npy"),
                 "prepared": {f.relative_to(p.prepared).as_posix(): digest(f) for f in files}})
    p.lock()
    return pd.DataFrame([{"windows": len(df), "crops": len(local), "patients": df.patient_id.nunique(),
                          "status": "lengkap"}])
