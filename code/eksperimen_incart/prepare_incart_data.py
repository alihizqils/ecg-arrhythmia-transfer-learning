"""Data V4 + INCART untuk uji lintas database (8 Nov 2026).

    python eksperimen_incart/prepare_incart_data.py

Folder baru: eksperimen_incart/data_v4_incart/{global,prepared}. Data V4 asli tidak diubah:
- berkas V4 yang dipakai ulang dibuat sebagai hard link (isi identik, tanpa ruang disk tambahan);
- citra global = salinan citra V4 + citra INCART di belakangnya (indeks V4 tetap);
- window INCART dilabeli dengan aturan V4 (label_record, kebijakan base.yaml + protokol_v4.yaml) dan batas
  20 window per kelas per rekaman (aturan SVDB); lead utama II, citra dan crop beat dengan fungsi kanonik V4;
  hasilnya harus sama dengan daftar acuan data/incart_windows.csv.
- INCART dicatat sebagai split development dengan fold -1, di luar lima fold CV dan hold-out, agar loader V4 dapat
  membacanya. Model final tidak dilatih ulang: uji B2 hanya melakukan inferensi, dan scaler development V4 dihitung
  ulang tanpa INCART lalu dicocokkan dengan checkpoint final.
- Label beat lokal INCART diambil dari anotasi beat INCART (sama seperti mitdb/svdb).
Embedding EfficientNet-B0 beku V4 di-hard-link ke cache baru agar hanya window INCART yang di-encode.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE / "eksperimen_v4_svdb"
sys.path[:0] = [str(V4), str(V4 / "fase_b"), str(CODE / "src")]
import v4_data as D  # noqa: E402
import buat_protokol as BP  # noqa: E402
from ecgtl.data.annotations import load_record_annotations  # noqa: E402
from ecgtl.data.build_dataset import DatasetSpec, load_channel_signal, select_capped_windows  # noqa: E402
from ecgtl.data.labeling import label_record  # noqa: E402
from ecgtl.data.rendering import render_waveform  # noqa: E402
from ecgtl.data.segmentation import is_flat, minmax_normalize, preprocess_record  # noqa: E402
from ecgtl.hierarchical.data import read_manifest, validate_inheritance  # noqa: E402
from ecgtl.hierarchical.protocol import CLASSES, atomic_json, digest  # noqa: E402

DATA = HERE / "data_v4_incart"
DB, LEAD, CAP = "incartdb", "II", 20


def protocol(output_root=None):
    """ProtocolV4 dengan jalur data INCART; raw tetap code/data/raw (hanya dibaca)."""
    p = D.ProtocolV4(output_root=output_root or (HERE / "outputs_incart"))
    p.source, p.prepared, p.raw = DATA / "global", DATA / "prepared", CODE / "data" / "raw"
    p.cfg["data"]["beat_label_databases"] = [*p.cfg["data"]["beat_label_databases"], DB]
    return p


def incart_labels(first_index: int) -> pd.DataFrame:
    spec = DatasetSpec.from_config(BP.CFG); pol = BP.LabelingPolicy.from_config(BP.CFG)
    recs = pd.read_csv(HERE / "data" / "incart_records.csv", dtype={"record": str})
    ws = []
    for r in recs.record:
        rec = load_record_annotations(DB, CODE / "data/raw" / DB, r, beat_aami=BP.CFG.require("labeling.beat_aami"),
                                      interruption_labels=BP.CFG.require("labeling.rhythm_interruption"), read_beats=True)
        ws += label_record(rec, spec.window_seconds, spec.overlap, pol).windows
    ws = sorted(select_capped_windows(ws, CAP), key=lambda w: (w.record, w.start))
    pid = dict(zip(recs.record, recs.patient_id))
    df = pd.DataFrame([{"index": first_index + i, "database": DB, "record": w.record, "patient_id": pid[w.record],
                        "label": w.label, "label_idx": CLASSES.index(w.label), "source": w.source.value,
                        "start_original": w.start, "n_beats": w.n_beats,
                        "rhythm_coverage": round(float(w.rhythm_coverage), 4)} for i, w in enumerate(ws)])
    # Kontrol: window harus sama dengan daftar acuan yang dirilis (data/incart_windows.csv).
    ref = pd.read_csv(HERE / "data" / "incart_windows.csv", dtype={"record": str})
    key = ["record", "label", "start_original"]
    if not df[key].sort_values(key).reset_index(drop=True).equals(ref[key].sort_values(key).reset_index(drop=True)):
        raise ValueError("Window INCART berbeda dengan daftar acuan.")
    return df


def render(labels: pd.DataFrame, images, offset: int) -> pd.DataFrame:
    spec = DatasetSpec.from_config(D.DATA_CONFIG); bandpass = D.DATA_CONFIG.require("preprocessing.bandpass")
    rows = []
    for record, group in tqdm(labels.groupby("record", sort=True), desc="Render INCART"):
        signal, fs, channel = load_channel_signal(CODE / "data/raw" / DB, record, [LEAD])
        assert channel == LEAD
        pre = preprocess_record(signal, fs, spec.target_fs, bandpass, DB, record, channel)
        for row in group.itertuples(index=False):
            start, end = pre.window_bounds_at_target_fs(int(row.start_original), spec.window_seconds)
            segment = pre.extract(start, end)
            if segment is None:
                raise ValueError(f"Window di luar rekaman: {record} {row.start_original}")
            images[int(row.index)] = render_waveform(
                minmax_normalize(segment), width=spec.width, height=spec.height,
                line_width=spec.line_width, background=spec.background, foreground=spec.foreground)
            rows.append({"index": int(row.index), "channel": channel, "start_target_fs": start,
                         "flat": bool(is_flat(segment))})
    return labels.join(pd.DataFrame(rows).set_index("index"), on="index")


def link_tree(source: Path, target: Path) -> int:
    n = 0
    for f in source.rglob("*"):
        if f.is_file():
            t = target / f.relative_to(source); t.parent.mkdir(parents=True, exist_ok=True)
            if not t.exists():
                os.link(f, t)
            n += 1
    return n


def main():
    if (DATA / "complete.json").exists():
        sys.exit(f"{DATA} sudah lengkap; tidak ditimpa.")
    v4p = D.ProtocolV4()  # hanya membaca jalur data V4
    p = protocol()
    p.source.mkdir(parents=True, exist_ok=True); p.prepared.mkdir(parents=True, exist_ok=True)
    src_manifest = read_manifest(v4p.source / "manifest.csv")
    g4 = read_manifest(v4p.prepared / "global_manifest.csv")
    n4 = len(src_manifest)
    labels = incart_labels(n4)
    print("window INCART:", labels.label.value_counts().to_dict(), flush=True)

    # 1. Citra global: salinan V4 + INCART.
    old = np.load(v4p.source / "images.npy", mmap_mode="r")
    images = np.lib.format.open_memmap(p.source / "images.tmp.npy", mode="w+", dtype=np.uint8,
                                       shape=(n4 + len(labels), *old.shape[1:]))
    for s in tqdm(range(0, n4, 2048), desc="Salin citra V4"):
        images[s:min(s + 2048, n4)] = old[s:min(s + 2048, n4)]
    inc = render(labels, images, n4)
    images.flush(); del images
    (p.source / "images.tmp.npy").replace(p.source / "images.npy")
    columns = list(src_manifest.columns)
    pd.concat([src_manifest, inc[columns]], ignore_index=True).to_csv(p.source / "manifest.csv", index=False)
    old_assign = pd.read_csv(v4p.source / "patient_assignment.csv", dtype=str)
    new_assign = pd.DataFrame({"patient_id": sorted(inc.patient_id.unique()), "partisi": "latih_tambahan_incart",
                               "lipatan_validasi": ""})
    pd.concat([old_assign, new_assign], ignore_index=True).to_csv(p.source / "patient_assignment.csv", index=False)

    # 2. Global manifest.
    inc_g = inc.assign(split="development", fold=-1, window_id=[f"inc{i:05d}" for i in range(len(inc))])
    g = pd.concat([g4, inc_g[list(g4.columns)]], ignore_index=True)
    g.to_csv(p.prepared / "global_manifest.csv", index=False)
    g.groupby(["split", "fold", "label"]).agg(windows=("window_id", "size"), patients=("patient_id", "nunique")) \
        .reset_index().to_csv(p.prepared / "distribution.csv", index=False)

    # 3. Data turunan per rekaman: V4 via hard link, INCART dibuat dengan prepare_record kanonik.
    linked = link_tree(v4p.prepared / "records", p.prepared / "records")
    crops, qa = [], []
    for _, rows in tqdm(list(inc_g.groupby(["database", "record"], sort=True)), desc="Persiapan INCART"):
        c, q = D.prepare_record(p, rows)
        crops.append(c); qa.append({k: v for k, v in q.items() if k != "errors_ms"})
    pd.DataFrame(qa).to_csv(DATA / "qa_detektor_incart.csv", index=False)
    local = pd.concat([read_manifest(v4p.prepared / "local_manifest.csv"), *crops], ignore_index=True)
    validate_inheritance(g, local)
    local.to_csv(p.prepared / "local_manifest.csv", index=False)
    checks = json.loads((v4p.prepared / "data_checksums.json").read_text())
    checks["source_images"] = digest(p.source / "images.npy")
    for f in (p.prepared / "records" / DB).rglob("*.npy"):
        checks["prepared"][f.relative_to(p.prepared).as_posix()] = digest(f)
    atomic_json(p.prepared / "data_checksums.json", checks)
    p.lock()

    # 4. Embedding beku V4 (EfficientNet-B0) dipakai ulang lewat hard link; receipt baru dengan hash protokol baru.
    old_cache = v4p.prepared / "features" / "efficientnet_b0" / json.loads(
        (V4 / "outputs_colab_r15/protocol_lock.json").read_text())["protocol_hash"][:16]
    new_cache = p.prepared / "features" / "efficientnet_b0" / p.hash[:16]; new_cache.mkdir(parents=True, exist_ok=True)
    reused = 0
    for f in tqdm(sorted(old_cache.glob("*.npz")), desc="Link embedding"):
        receipt = json.loads(f.with_suffix(".json").read_text())
        if not (new_cache / f.name).exists():
            os.link(f, new_cache / f.name)
        atomic_json((new_cache / f.name).with_suffix(".json"), {"protocol_hash": p.hash, "sha256": receipt["sha256"]})
        reused += 1
    info = {"windows_v4": n4, "windows_incart": len(inc), "incart_label": inc.label.value_counts().to_dict(),
            "incart_patients": int(inc.patient_id.nunique()), "crops_incart": int(sum(len(c) for c in crops)),
            "record_files_linked": linked, "embeddings_reused": reused, "protocol_hash": p.hash,
            "detector_match_ratio_mean": float(np.mean([q["matched"] / q["reference"] for q in qa]))}
    atomic_json(DATA / "complete.json", info); print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
