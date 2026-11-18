"""Evaluasi holdout V4 dan perbaikan validasi run selesai (di luar hash metode training).

1. `validate_completed_run_label_valid`: sama dengan `v4_pipeline.validate_completed_run`,
   kecuali cakupan train IDs dibandingkan dengan development setelah `label_valid`
   (Revisi 9). Versi di v4_pipeline membandingkan dengan development tanpa saringan
   sehingga menolak run yang sah. Perbaikan hanya memeriksa integritas (tidak memengaruhi
   training/prediksi); ditaruh di modul ini agar hash metode training tidak berubah dan
   run Revisi 12 yang sudah ada tetap sah. `install()` memasangnya ke v4_pipeline.
2. `evaluate_holdout`: model final satu skenario (S1/S2/S3) dievaluasi satu kali pada
   holdout (label_valid), dengan kunci metode saat holdout pertama dibuka dan CI bootstrap
   pasien. S3 (ditambahkan 5 Nov 2026) memuat FullFineTuneNet dari checkpoint final S3.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import v4_pipeline as v4
import final_pipeline as base
from ecgtl.hierarchical.data import read_manifest
from ecgtl.hierarchical.metrics import export_metrics
from ecgtl.hierarchical.protocol import CLASSES, atomic_json, digest
from ecgtl.hierarchical.training import load_checkpoint

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261105


def validate_completed_run_label_valid(protocol, arch, scenario, fold, full=False):
    folder = protocol.out / ("final_models" if full else "cv") / arch / scenario / ("final" if full else f"fold_{int(fold)}")
    result_path, checkpoint_path = folder / "result.json", folder / "best.pt"
    if not result_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Run belum lengkap: {folder}")
    result = json.loads(result_path.read_text())
    expected = {"arch": arch, "scenario": scenario, "fold": -1 if full else int(fold), "full": bool(full),
                "protocol_hash": protocol.hash, "method_hash": v4.method_hash()}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Identitas result berbeda pada {folder}: {key}")
    if digest(checkpoint_path) != result.get("checkpoint_sha256"):
        raise ValueError(f"Checksum checkpoint berbeda: {folder}")
    checkpoint = load_checkpoint(checkpoint_path)
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Identitas checkpoint berbeda pada {folder}: {key}")
    manifest = read_manifest(protocol.prepared / "global_manifest.csv")
    development = v4.label_valid(manifest[manifest.split == "development"].reset_index(drop=True))
    expected_ids = set(development.window_id.astype(str) if full else development[development.fold != int(fold)].window_id.astype(str))
    train_ids = list(map(str, checkpoint.get("train_ids", [])))
    if len(train_ids) != len(set(train_ids)) or set(train_ids) != expected_ids:
        raise ValueError(f"Cakupan train IDs berbeda: {folder}")
    if full and not (folder / "initialization_anchor.npz").exists():
        raise FileNotFoundError(f"Initialization anchor tidak tersedia: {folder}")
    if not full:
        for name in ["metrics.json", "per_class.csv", "probabilities.npy", "labels.npy", "confusion.npy", "epoch_probabilities/labels.npy"]:
            if not (folder / name).exists():
                raise FileNotFoundError(folder / name)
    return result


def install():
    v4.validate_completed_run = validate_completed_run_label_valid


def patient_bootstrap(test: pd.DataFrame, y, prob, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """CI 95% persentil dengan resampling pasien (klaster) holdout."""
    pred = prob.argmax(1); patients = test.patient_id.astype(str).to_numpy()
    index = {p: np.flatnonzero(patients == p) for p in np.unique(patients)}
    keys = list(index); rng = np.random.default_rng(seed); draws = []
    for _ in range(replicates):
        chosen = np.concatenate([index[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        per = f1_score(y[chosen], pred[chosen], labels=range(5), average=None, zero_division=0)
        present = np.isin(range(5), y[chosen])
        draws.append([per[present].mean(), *np.where(present, per, np.nan)])
    draws = np.asarray(draws)
    names = ["macro_f1", *[f"f1_{c}" for c in CLASSES]]
    return {n: {"low": float(np.nanpercentile(draws[:, i], 2.5)), "high": float(np.nanpercentile(draws[:, i], 97.5))}
            for i, n in enumerate(names)} | {"replicates": replicates, "seed": seed, "patients": len(keys)}


def _opening_lock(protocol, arch, scenario, checkpoint_sha):
    path = protocol.out / "holdout" / "opening_lock.json"
    import v4_s3
    current = {"method_hash": v4.method_hash(), "s3_method_hash": v4_s3.s3_method_hash(),
               "data_protocol_hash": protocol.hash}
    if path.exists():
        lock = json.loads(path.read_text())
        if {k: lock[k] for k in current} != current:
            raise ValueError("Metode berubah setelah holdout pertama dibuka (lock_method_at_first_opening).")
    else:
        lock = {**current, "first_opened": {"arch": arch, "scenario": scenario, "checkpoint_sha256": checkpoint_sha,
                                            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}}
        atomic_json(path, lock)
    return lock


def build_final_model(experiment, scenario, checkpoint):
    """Model raw-image (encoder + head) dari checkpoint final; S1/S2 encoder ImageNet beku."""
    if scenario == "S3":
        import v4_s3
        model = v4_s3.FullFineTuneNet(experiment.arch, experiment.p.cfg, experiment._head())
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        head = experiment._head(); head.load_state_dict(checkpoint["model"], strict=True)
        model = base.EncoderHeadNet(experiment.arch, experiment.p.cfg, head, 0)
    return model.to(experiment.device).eval()


def evaluate_holdout(protocol, arch, scenario):
    if scenario not in (*v4.FROZEN, "S3"):
        raise ValueError("Skenario holdout V4: S1/S2/S3.")
    import v4_s3
    v4.initialize(protocol)  # verifikasi protokol data dan kunci metode
    if scenario == "S3":
        v4_s3.initialize_s3(protocol)
    final_result = validate_completed_run_label_valid(protocol, arch, scenario, -1, True)
    output = protocol.out / "holdout" / arch / scenario; marker = output / "complete.json"
    if marker.exists():  # evaluate_once_per_model: hanya membaca ulang hasil terverifikasi
        receipt = json.loads(marker.read_text())
        if receipt["checkpoint_sha256"] != final_result["checkpoint_sha256"] or receipt["method_hash"] != v4.method_hash():
            raise ValueError("Receipt holdout berbeda dari model final.")
        for name, sha in receipt["files"].items():
            if digest(output / name) != sha:
                raise ValueError(f"Artefak holdout rusak: {name}")
        return json.loads((output / "summary.json").read_text())
    _opening_lock(protocol, arch, scenario, final_result["checkpoint_sha256"])
    experiment = v4_s3.V4S3Experiment(protocol, arch) if scenario == "S3" else v4.V4Experiment(protocol, arch, scenario)
    folder = experiment.run_dir(-1, True); checkpoint = load_checkpoint(folder / "best.pt")
    if scenario == "S3" and checkpoint.get("s3_method_hash") != v4_s3.s3_method_hash():
        raise ValueError("Hash metode S3 checkpoint final berbeda.")
    manifest = read_manifest(protocol.prepared / "global_manifest.csv")
    test = v4.label_valid(manifest[manifest.split == "holdout"].reset_index(drop=True))
    if set(test.patient_id) & set(experiment.dev.patient_id):
        raise ValueError("Overlap pasien holdout-development.")
    scaler = v4.V4Scaler(experiment.store, experiment.dev.window_id, experiment.record_template, experiment.beat_record_template)
    if scaler.state_dict() != checkpoint["rr_scaler"]:
        raise ValueError("Scaler final berbeda.")
    model = build_final_model(experiment, scenario, checkpoint)
    audit, _ = experiment._audit_raw_anchor(model, scaler, folder)  # encoder mentah = jalur cache
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "encoder_initialization_audit.json", audit)
    prediction = experiment.predict(model, test, scaler, False)
    score = export_metrics(output, prediction["y"], prediction["prob"], test, prediction["global_prob"],
                           prediction["beat_y"], prediction["beat_prob"])
    np.save(output / "probabilities.npy", prediction["prob"].astype(np.float32))
    np.save(output / "labels.npy", prediction["y"])
    pred = prediction["prob"].argmax(1)
    s = test.assign(correct=pred == CLASSES.index("S"))[test.label == "S"]
    s_patients = s.groupby(["database", "patient_id"]).correct.agg(["size", "mean"]).reset_index()
    s_patients.to_csv(output / "s_patients.csv", index=False)
    summary = {"arch": arch, "scenario": scenario, "windows": len(test), "patients": int(test.patient_id.nunique()),
               "final_epochs": final_result["epochs_run"], "checkpoint_sha256": final_result["checkpoint_sha256"],
               "macro_f1": score["macro_f1"],
               "f1": {c: float(v) for c, v in zip(CLASSES, f1_score(prediction["y"], pred, labels=range(5), average=None, zero_division=0))},
               "ci95_patient_bootstrap": patient_bootstrap(test, prediction["y"], prediction["prob"]),
               "s_patient_recall_median": float(s_patients["mean"].median()),
               "s_patients_recall_below_0_2": int((s_patients["mean"] < 0.2).sum()), "s_patients": len(s_patients),
               "method_hash": v4.method_hash(), "protocol_hash": protocol.hash}
    atomic_json(output / "summary.json", summary)
    atomic_json(marker, {"checkpoint_sha256": final_result["checkpoint_sha256"], "method_hash": v4.method_hash(),
                         "protocol_hash": protocol.hash,
                         "files": {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "complete.json"}})
    return summary
