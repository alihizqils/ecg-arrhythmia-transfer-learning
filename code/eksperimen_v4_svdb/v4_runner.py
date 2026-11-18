"""Langkah eksperimen V4 untuk notebook (di luar hash metode training).

Satu skenario = CV 5 fold -> epoch final bersama E* -> model final -> holdout.
Semua langkah dapat dijalankan ulang: fold/model yang lengkap diverifikasi lalu dipakai
ulang, run yang terputus dilanjutkan dari `last.pt`, holdout yang sudah dievaluasi hanya
dibaca ulang dari receipt-nya.

    import v4_runner as R
    R.cv("efficientnet_b0", "S1")          # 5 fold
    R.final_epoch("efficientnet_b0", "S1")  # E*
    R.final_model("efficientnet_b0", "S1")
    R.holdout("efficientnet_b0", "S1")
"""
from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

import v4_holdout
import v4_pipeline as v4
import v4_s3
from v4_data import ProtocolV4

ARCHS = ("efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0")
SCENARIOS = ("S1", "S2", "S3")
FOLDS = int(v4.METHOD["cv"]["folds"])
_EXPERIMENTS: dict = {}
_CACHE_VERIFIED: set = set()
_PROTOCOL = None


def _hash_metode_lama_lintas_os():
    """13 Nov 2026: final_pipeline.method_hash memakai str(path) sehingga pemisah path ikut di-hash ('\\' di Windows,
    '/' di Colab/Linux). Isi berkas yang di-hash tidak berubah; hasil Colab (final_method_lock e4512d69...) hanya bisa
    diverifikasi di Windows bila pemisah dinormalkan ke '/'. final_pipeline.py termasuk hash metode V4 sehingga tidak
    diubah; normalisasi dilakukan saat runtime. Di Linux tidak ada efek."""
    import os
    if os.sep == "/" or getattr(v4.base, "_hash_posix", False):
        return
    asli = v4.base.method_hash

    def method_hash_posix():
        # str() di dalam final_pipeline.method_hash hanya dipakai untuk path relatif; selama hash dihitung,
        # str(path) diganti path.as_posix() lewat global modul (bukan builtins), lalu dikembalikan.
        import builtins, pathlib
        v4.base.str = lambda x: x.as_posix() if isinstance(x, pathlib.PurePath) else builtins.str(x)
        try:
            return asli()
        finally:
            del v4.base.str
    v4.base.method_hash = method_hash_posix; v4.base._hash_posix = True


def protocol() -> ProtocolV4:
    global _PROTOCOL
    if _PROTOCOL is None:
        v4_holdout.install()  # validasi run selesai dengan cakupan label_valid (Revisi 9)
        _hash_metode_lama_lintas_os()
        # V4_OUTPUT_ROOT (13 Nov 2026): baca hasil dari folder lain, mis. outputs_colab_r15 (rekap/benchmark).
        _PROTOCOL = ProtocolV4(output_root=__import__("os").environ.get("V4_OUTPUT_ROOT") or None)
        v4.initialize(_PROTOCOL); v4_s3.initialize_s3(_PROTOCOL)
    return _PROTOCOL


# Audit inisialisasi S3 (di luar hash metode; tidak memengaruhi training, seleksi, maupun evaluasi).
# Riwayat:
#   6 Nov 2026: prediksi validasi S2 tersimpan dihitung dari embedding cache (encoder batch 32), sedangkan S3
#     meng-encode citra mentah (batch 4). Selisih numerik cuDNN antar ukuran batch: 0,01007 pada 1/3.261 window
#     (argmax identik). Toleransi runtime dinaikkan 0,01 -> 0,02 + syarat argmax identik 100%.
#   10 Nov 2026 (R15, fase_e_ns/hasil/pilot_s1_s3_r15/diagnostik_audit_s3): pada R15 fold 1 jalur mentah memberi
#     selisih maks 0,036 dan argmax 3.369/3.370, sedangkan jalur cache + head S2 mereproduksi probabilitas
#     tersimpan dengan selisih 0,0. Penyebabnya derau numerik jalur encoder, bukan bobot. Keputusan:
#       1. jalur cache + head S2 (= head S3 sebelum fine-tuning) wajib mereproduksi probabilitas S2 tersimpan
#          (selisih <= S3_CACHE_TOLERANCE, presisi float32; argmax identik 100%);
#       2. bobot head S3 = state_dict checkpoint S2 dan bobot encoder = bobot ImageNet, identik bit per bit;
#       3. jalur citra mentah: toleransi derau numerik S3_AUDIT_TOLERANCE dan argmax >= S3_ARGMAX_MIN.
#     Yaml dan hash metode S3 tidak berubah; toleransi hanya diganti saat runtime.
S3_AUDIT_TOLERANCE = 0.05
S3_ARGMAX_MIN = 0.999
S3_CACHE_TOLERANCE = 1e-5
v4_s3.S3["initialization_probability_tolerance"] = S3_AUDIT_TOLERANCE


def _same_state(a: dict, b: dict) -> bool:
    import torch
    return a.keys() == b.keys() and all(torch.equal(a[k].detach().cpu(), b[k].detach().cpu()) for k in a)


class _S3Experiment(v4_s3.V4S3Experiment):
    def _model_scaler(self, train, fold, full, last_exists):
        import numpy as np
        import torch
        from ecgtl.hierarchical.model import backbone
        result = super()._model_scaler(train, fold, full, last_exists)  # audit jalur mentah (toleransi runtime)
        if full or last_exists:
            return result
        model, scaler, _ = result
        folder, source = self._validate_source(fold, full)
        audit_path = self.run_dir(fold, full) / "initialization_audit.json"
        raw = json.loads(audit_path.read_text())
        valid = self.dev[self.dev.fold == int(fold)].reset_index(drop=True)
        expected = np.load(folder / "probabilities.npy")
        head = self._head(); head.load_state_dict(source["model"], strict=True)
        with torch.no_grad():
            cache = self.predict(head, valid, scaler, True)
        reference, _ = backbone(self.arch, pretrained=True)
        audit = {"jalur_mentah": raw,
                 "jalur_cache_selisih_maks": float(np.max(np.abs(cache["prob"] - expected))),
                 "jalur_cache_argmax_sama": float((cache["prob"].argmax(1) == expected.argmax(1)).mean()),
                 "jalur_cache_toleransi": S3_CACHE_TOLERANCE,
                 "bobot_head_sama_checkpoint_S2": _same_state(model.head.state_dict(), source["model"]),
                 "bobot_encoder_sama_imagenet": _same_state(model.encoder.state_dict(), reference.state_dict()),
                 "jalur_mentah_toleransi": S3_AUDIT_TOLERANCE, "jalur_mentah_argmax_min": S3_ARGMAX_MIN}
        audit["lolos"] = bool(raw["labels_identical"] and raw["maximum_probability_difference"] <= S3_AUDIT_TOLERANCE
                              and raw["argmax_agreement"] >= S3_ARGMAX_MIN
                              and audit["jalur_cache_selisih_maks"] <= S3_CACHE_TOLERANCE
                              and audit["jalur_cache_argmax_sama"] == 1.0
                              and audit["bobot_head_sama_checkpoint_S2"] and audit["bobot_encoder_sama_imagenet"])
        v4_s3.atomic_json(self.run_dir(fold, full) / "initialization_audit_r15.json", audit)
        if not audit["lolos"]:
            raise ValueError(f"Audit inisialisasi S3 gagal: {audit}")
        return result


def _check(arch, scenario):
    if arch not in ARCHS or scenario not in SCENARIOS:
        raise ValueError(f"Backbone/skenario tidak dikenal: {arch} {scenario}")


def experiment(arch: str, scenario: str):
    """Objek eksperimen (dibuat sekali per kernel; konstruksi memuat feature store)."""
    _check(arch, scenario)
    key = (arch, scenario)
    if key not in _EXPERIMENTS:
        p = protocol()
        _EXPERIMENTS[key] = _S3Experiment(p, arch) if scenario == "S3" else v4.V4Experiment(p, arch, scenario)
    return _EXPERIMENTS[key]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text); stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def logged(arch: str, scenario: str):
    """Salin keluaran print ke outputs/logs/<arch>_<skenario>.log (progres tetap terbaca bila notebook macet)."""
    folder = protocol().out / "logs"; folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{arch}_{scenario}.log").open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        original = sys.stdout; sys.stdout = _Tee(original, handle)
        try:
            yield
        finally:
            sys.stdout = original


def _require_source(arch, scenario, fold=None, full=False):
    """S3 diinisialisasi dari S2 fold/final yang sama."""
    if scenario != "S3":
        return
    v4.validate_completed_run(protocol(), arch, "S2", -1 if full else fold, full)


def cv_fold(arch: str, scenario: str, fold: int) -> dict:
    _require_source(arch, scenario, fold)
    with logged(arch, scenario):
        print(time.strftime("%H:%M"), f"{arch} {scenario} fold {fold}", flush=True)
        if scenario in v4.FROZEN and arch not in _CACHE_VERIFIED:
            experiment(arch, scenario).base.cache_features()  # buat atau verifikasi cache embedding beku
            _CACHE_VERIFIED.add(arch)
        return experiment(arch, scenario).fit(int(fold))


def cv(arch: str, scenario: str, folds=None) -> pd.DataFrame:
    rows = [cv_fold(arch, scenario, fold) for fold in (range(FOLDS) if folds is None else folds)]
    keys = ["fold", "epochs_run", "best_epoch", "macro_f1", "f1_S", "accuracy", "macro_auc", "training_s"]
    return pd.DataFrame([{k: r.get(k) for k in keys} for r in rows])


def final_epoch(arch: str, scenario: str) -> dict:
    return experiment(arch, scenario).select_final_epoch()


def cv_summary(arch: str, scenario: str) -> dict:
    return v4.cv_summary(protocol(), arch, scenario)


def final_model(arch: str, scenario: str) -> dict:
    _require_source(arch, scenario, full=True)
    selected = int(final_epoch(arch, scenario)["selected_epoch"])
    with logged(arch, scenario):
        print(time.strftime("%H:%M"), f"{arch} {scenario} model final, {selected} epoch", flush=True)
        return experiment(arch, scenario).fit(-1, True, selected)


def holdout(arch: str, scenario: str) -> dict:
    with logged(arch, scenario):
        print(time.strftime("%H:%M"), f"{arch} {scenario} holdout", flush=True)
        return v4_holdout.evaluate_holdout(protocol(), arch, scenario)


def run_scenario(arch: str, scenario: str) -> dict:
    """Alur lengkap satu skenario (dipakai bila ingin satu sel saja)."""
    cv(arch, scenario)
    final_epoch(arch, scenario)  # tulis selection/{scenario}_final_epoch.json sebelum ringkasan CV
    summary = cv_summary(arch, scenario)
    final_model(arch, scenario)
    return {"cv": summary, "holdout": holdout(arch, scenario)}


def status(arch: str | None = None) -> pd.DataFrame:
    """Tabel progres tanpa memuat eksperimen (hanya membaca berkas)."""
    out = ProtocolV4().out; rows = []
    for a in ([arch] if arch else ARCHS):
        for s in SCENARIOS:
            row = {"arch": a, "skenario": s}
            for fold in range(FOLDS):
                folder = out / "cv" / a / s / f"fold_{fold}"
                if (folder / "result.json").exists():
                    row[f"fold_{fold}"] = "selesai"
                elif (folder / "last.pt").exists():
                    row[f"fold_{fold}"] = f"berjalan (epoch {len(pd.read_csv(folder / 'history.csv'))})" if (folder / "history.csv").exists() else "berjalan"
                else:
                    row[f"fold_{fold}"] = "-"
            selection = out / "selection" / a / f"{s}_final_epoch.json"
            row["E*"] = json.loads(selection.read_text())["selected_epoch"] if selection.exists() else "-"
            final = out / "final_models" / a / s / "final"
            row["final"] = "selesai" if (final / "result.json").exists() else ("berjalan" if (final / "last.pt").exists() else "-")
            summary = out / "holdout" / a / s / "summary.json"
            row["holdout_macro_f1"] = round(json.loads(summary.read_text())["macro_f1"], 4) if (out / "holdout" / a / s / "complete.json").exists() else "-"
            rows.append(row)
    return pd.DataFrame(rows)
