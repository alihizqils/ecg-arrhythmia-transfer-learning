"""Fase B: manifest label V4, split pasien, dan lock protokol.

Langkah:
1. Pilih rekaman SVDB menurut `protokol_v4.yaml` (aturan Fase A).
2. Labeli seluruh database dengan kebijakan kanonik, lalu terapkan batas per
   database per rekaman per kelas dengan `select_capped_windows`.
3. Kontrol: window database lama harus identik dengan manifest kanonik lama.
4. Hold-out 20% pasien: partisi 0 dari alokasi serakah seluruh pasien, dengan
   syarat minimal pasien per kelas dan per sumber S (strata kelas + sumber S).
5. Lima fold development dengan algoritme alokasi serakah protokol hierarkis.
6. Tulis manifest, assignment, distribusi, dan lock berisi checksum.

Split dibuat sekali dengan seed tetap. Bila berkas sudah ada, skrip hanya
memverifikasi bahwa hasil deterministiknya sama dan tidak menimpa apa pun.
Tidak ada citra yang dirender di sini (itu Fase C).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from ecgtl.config import load_config  # noqa: E402
from ecgtl.data.annotations import load_record_annotations  # noqa: E402
from ecgtl.data.build_dataset import DatasetSpec, collect_labeled_windows, select_capped_windows  # noqa: E402
from ecgtl.data.labeling import LabelingPolicy, label_record  # noqa: E402
from ecgtl.data.splits import patient_id  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "v4_svdb"
BASE_CONFIG = ROOT / "configs" / "base.yaml"
V4_CONFIG = EXPERIMENT / "protokol_v4.yaml"
OLD_MANIFEST = ROOT / "data/processed/w10s_448x224_ov50/manifest.csv"
CFG = load_config(BASE_CONFIG, [V4_CONFIG])
CLASSES = list(CFG.require("classes"))
STRATA = ["N", "S_mitdb", "S_svdb", "V", "AF", "VT"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_svdb(policy: LabelingPolicy, spec: DatasetSpec) -> pd.DataFrame:
    rule = CFG.require("svdb_selection")
    excluded = {str(r) for r in rule["excluded_records"]}
    rows = []
    for record in sorted(p.stem for p in (RAW / "svdb").glob("*.hea")):
        rec = load_record_annotations(
            "svdb", RAW / "svdb", record, beat_aami=CFG.require("labeling.beat_aami"),
            interruption_labels=CFG.require("labeling.rhythm_interruption"), read_beats=True,
        )
        counts = Counter(w.label for w in label_record(rec, spec.window_seconds, spec.overlap, policy).windows)
        selected = counts.get("S", 0) >= int(rule["min_s_windows"]) and record not in excluded
        reason = "dipilih" if selected else ("dikecualikan_rr_tidak_teratur" if record in excluded else "window_S_kurang")
        rows.append({"record": record, **{f"windows_{c}": counts.get(c, 0) for c in CLASSES},
                     "selected": selected, "reason": reason})
    return pd.DataFrame(rows)


def labeled_manifest(policy: LabelingPolicy, spec: DatasetSpec, svdb_records: list[str]) -> pd.DataFrame:
    windows, _, _, _ = collect_labeled_windows(CFG, RAW, spec, policy, {"svdb": svdb_records})
    caps = CFG.require("sampling.cap_per_database")
    class_caps = CFG.get("sampling.cap_per_class", {})
    chosen = []
    for database, cap in caps.items():
        for label in CLASSES:
            effective = min(int(cap), int(class_caps.get(label, cap)))
            chosen += select_capped_windows(
                [w for w in windows if w.database == database and w.label == label], effective)
    chosen.sort(key=lambda w: (w.database, w.record, w.start))
    aliases = CFG.get("split.patient_aliases", {})
    manifest = pd.DataFrame([{
        "index": i, "database": w.database, "record": w.record,
        "patient_id": patient_id(w.database, w.record, aliases),
        "label": w.label, "label_idx": CLASSES.index(w.label), "source": w.source.value,
        "start_original": w.start, "n_beats": w.n_beats,
        "rhythm_coverage": round(float(w.rhythm_coverage), 4),
    } for i, w in enumerate(chosen)])
    manifest["stratum"] = np.where(manifest.label == "S", "S_" + manifest.database, manifest.label)
    return manifest


def control_old_databases(manifest: pd.DataFrame) -> dict:
    """Kelas tanpa batas kelas tambahan harus identik dengan dataset kanonik lama.

    Kelas S/V dipilih ulang secara merata dengan batas lebih kecil dari seluruh
    window berlabel kanonik (pelabelan sama), sehingga tidak harus subset dari
    pilihan batas 200.
    """
    keys = ["database", "record", "start_original", "label"]
    limited = set(CFG.get("sampling.cap_per_class", {}))
    old = pd.read_csv(OLD_MANIFEST, dtype={"record": str})[keys]
    new = manifest[manifest.database != "svdb"][keys]
    for label in sorted(set(CLASSES) - limited):
        a = old[old.label == label].sort_values(keys).reset_index(drop=True)
        b = new[new.label == label].sort_values(keys).reset_index(drop=True)
        if not a.equals(b):
            raise ValueError(f"Window kelas {label} database lama berbeda dari manifest kanonik.")
    return {"old_windows_canonical": len(old), "old_windows_v4": len(new),
            "identical_classes": sorted(set(CLASSES) - limited),
            "capped_classes": {c: [int((old.label == c).sum()), int((new.label == c).sum())] for c in sorted(limited)}}


def patient_support(frame: pd.DataFrame, parts: np.ndarray, n_parts: int):
    """Jumlah window per kelas dan jumlah pasien per stratum untuk tiap partisi."""
    counts = pd.crosstab(parts, frame.label.to_numpy()).reindex(index=range(n_parts), columns=CLASSES, fill_value=0)
    unique = frame.assign(part=parts).drop_duplicates(["part", "stratum", "patient_id"])
    patients = pd.crosstab(unique.part, unique.stratum).reindex(index=range(n_parts), columns=STRATA, fill_value=0)
    return counts, patients


def holdout_split(manifest: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Hold-out = partisi 0 dari alokasi serakah seluruh pasien ke 1/test_size partisi.

    Syarat diterapkan pada partisi 0: minimal pasien per kelas dan per sumber S.
    """
    rule = CFG.require("split")
    n_parts = int(round(1.0 / float(rule["test_size"])))

    def accept(counts, patients):
        holdout = patients.loc[0]
        per_class = holdout.drop(["S_mitdb", "S_svdb"]).tolist() + [holdout.S_mitdb + holdout.S_svdb]
        return (
            not (counts == 0).any().any()
            and min(per_class) >= int(rule["min_holdout_patients_per_class"])
            and min(holdout.S_mitdb, holdout.S_svdb) >= int(rule["min_holdout_s_patients_per_source"])
        )

    parts, audit = allocate(manifest, n_parts, accept)
    split = pd.Series(np.where(parts == 0, "holdout", "development"), index=manifest.index)
    if set(manifest.patient_id[split == "holdout"]) & set(manifest.patient_id[split == "development"]):
        raise ValueError("Pasien bocor antara development dan holdout.")
    return split, audit


def assign_folds(dev: pd.DataFrame) -> tuple[np.ndarray, dict]:
    rule = CFG.require("split")

    def accept(counts, patients):
        s_patients = patients["S_mitdb"] + patients["S_svdb"]
        return (
            not (counts == 0).any().any()
            and (s_patients >= int(rule["min_s_patients_per_fold"])).all()
            and (patients[["S_mitdb", "S_svdb"]] >= int(rule["min_s_patients_per_source_per_fold"])).all().all()
        )

    return allocate(dev, int(rule["n_folds"]), accept)


def allocate(frame: pd.DataFrame, n_folds: int, accept) -> tuple[np.ndarray, dict]:
    """Alokasi serakah dari `ecgtl.hierarchical.data.assign_folds`, strata kelas + sumber S."""
    rule = CFG.require("split")
    dev = frame
    patient_counts = pd.crosstab(dev.patient_id, dev.stratum).reindex(columns=STRATA, fill_value=0)
    wc = patient_counts.to_numpy(dtype=float)
    pc = (wc > 0).astype(float)
    normalized_w = wc / wc.sum(0)
    normalized_p = pc / pc.sum(0)
    best = None
    for candidate in range(int(rule["split_candidates"])):
        rng = np.random.default_rng(int(CFG.require("seed")) + candidate)
        priority = np.maximum(normalized_w.max(1), normalized_p.max(1)) * rng.uniform(.7, 1.3, len(wc))
        allocations = np.empty(len(wc), dtype=int)
        fw, fp = np.zeros((n_folds, len(STRATA))), np.zeros((n_folds, len(STRATA)))
        size = np.zeros(n_folds)
        for patient in np.argsort(-priority):
            costs = []
            for fold in range(n_folds):
                tw, tp, ts = fw.copy(), fp.copy(), size.copy()
                tw[fold] += normalized_w[patient]
                tp[fold] += normalized_p[patient]
                ts[fold] += 1 / len(wc)
                costs.append(np.square(tw).sum() + 2 * np.square(tp).sum() + np.square(ts).sum())
            fold = int(np.argmin(np.asarray(costs) + rng.uniform(0, 1e-10, n_folds)))
            allocations[patient] = fold
            fw[fold] += normalized_w[patient]
            fp[fold] += normalized_p[patient]
            size[fold] += 1 / len(wc)
        folds = dev.patient_id.map(dict(zip(patient_counts.index, allocations))).to_numpy()
        counts, patients = patient_support(dev, folds, n_folds)
        if not accept(counts, patients):
            continue
        score = (counts / counts.sum()).std().mean() + (patients / patients.sum()).std().mean()
        if best is None or score < best[0]:
            best = (float(score), folds.copy(), candidate)
    if best is None:
        raise ValueError("Tidak ditemukan alokasi yang memenuhi syarat dukungan pasien.")
    return best[1], {"candidate": best[2], "score": best[0]}


def write_or_verify(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        stored = pd.read_csv(path, dtype={"record": str})
        if not stored.equals(pd.read_csv(io.StringIO(frame.to_csv(index=False)), dtype={"record": str})):
            raise ValueError(f"{path.name} tersimpan berbeda dari hasil deterministik; tidak ditimpa.")
        return
    frame.to_csv(path, index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = DatasetSpec.from_config(CFG)
    policy = LabelingPolicy.from_config(CFG)

    selection = select_svdb(policy, spec)
    svdb_records = selection.loc[selection.selected, "record"].tolist()
    manifest = labeled_manifest(policy, spec, svdb_records)
    control = control_old_databases(manifest)

    manifest["split"], holdout_audit = holdout_split(manifest)
    manifest["fold"] = -1
    dev_index = manifest.index[manifest.split == "development"]
    folds, fold_audit = assign_folds(manifest.loc[dev_index])
    manifest.loc[dev_index, "fold"] = folds
    manifest["window_id"] = manifest["index"].map(lambda i: f"w{i:06d}")

    holdout = manifest[manifest.split == "holdout"]
    holdout_s = holdout[holdout.label == "S"].groupby("database").patient_id.nunique()
    minimum = int(CFG.require("split.min_holdout_s_patients_per_source"))
    if missing := [d for d in ["mitdb", "svdb"] if holdout_s.get(d, 0) < minimum]:
        raise ValueError(f"Hold-out kekurangan pasien S dari {missing}.")
    if set(CLASSES) - set(holdout.label):
        raise ValueError("Hold-out tidak memuat seluruh kelas.")

    assignment = (manifest.groupby("patient_id")
                  .agg(database=("database", "first"), split=("split", "first"), fold=("fold", "first"),
                       windows=("index", "size"), classes=("label", lambda x: "|".join(sorted(set(x)))))
                  .reset_index())
    distribution = (manifest.groupby(["split", "fold", "label"])
                    .agg(windows=("window_id", "size"), patients=("patient_id", "nunique")).reset_index())
    s_by_source = (manifest[manifest.label == "S"].groupby(["split", "fold", "database"])
                   .agg(windows=("window_id", "size"), patients=("patient_id", "nunique")).reset_index())

    write_or_verify(OUT / "svdb_selection.csv", selection)
    write_or_verify(OUT / "label_manifest.csv", manifest)
    write_or_verify(OUT / "patient_assignment.csv", assignment)
    distribution.to_csv(OUT / "distribution.csv", index=False)
    s_by_source.to_csv(OUT / "distribution_s_by_source.csv", index=False)

    svdb_files = sorted((RAW / "svdb").glob("*"))
    lock = {
        "version": CFG.require("version"),
        "seed": int(CFG.require("seed")),
        "classes": CLASSES,
        "windows": len(manifest),
        "patients": int(manifest.patient_id.nunique()),
        "per_class": manifest.label.value_counts().reindex(CLASSES).astype(int).to_dict(),
        "per_split": manifest.split.value_counts().to_dict(),
        "svdb_selected_records": svdb_records,
        "control_old_databases": control,
        "holdout_selection": holdout_audit,
        "fold_selection": fold_audit,
        "holdout_s_patients_by_source": holdout_s.astype(int).to_dict(),
        "sha256": {
            "configs/base.yaml": sha256(BASE_CONFIG),
            "eksperimen_v4_svdb/protokol_v4.yaml": sha256(V4_CONFIG),
            "eksperimen_v4_svdb/fase_b/buat_protokol.py": sha256(Path(__file__)),
            "label_manifest.csv": sha256(OUT / "label_manifest.csv"),
            "patient_assignment.csv": sha256(OUT / "patient_assignment.csv"),
            "svdb_raw_combined": hashlib.sha256(b"".join(sha256(f).encode() for f in svdb_files)).hexdigest(),
        },
        "holdout_used_for_selection": False,
    }
    lock_path = OUT / "protocol_v4_lock.json"
    if lock_path.exists():
        stored = json.loads(lock_path.read_text())
        if stored["sha256"]["label_manifest.csv"] != lock["sha256"]["label_manifest.csv"] or \
           stored["sha256"]["patient_assignment.csv"] != lock["sha256"]["patient_assignment.csv"]:
            raise ValueError("Lock protokol V4 berbeda; gunakan versi baru, jangan menimpa.")
    else:
        lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in lock.items() if k != "sha256"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
