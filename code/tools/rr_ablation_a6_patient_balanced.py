"""Pilot A6-PB: A6 dengan patient-balanced sampling pada subtrain."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from tools.rr_ablation_a6_standalone import (
    A6Dataset,
    A6StandaloneExperiment,
    collate_a6,
)


class A6PatientBalancedExperiment(A6StandaloneExperiment):
    """A6 dengan kontribusi pasien setara dalam ekspektasi per epoch."""

    variant = "A6_PATIENT_BALANCED"
    implementation_file = Path(__file__)

    @staticmethod
    def sampling_weights(rows):
        if rows.empty:
            raise ValueError("Subtrain tidak boleh kosong.")
        patient = rows.patient_id.astype(str)
        counts = patient.value_counts()
        if (counts <= 0).any():
            raise ValueError("Jumlah window pasien tidak valid.")
        weights = patient.map((1.0 / counts).to_dict()).to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Bobot sampling pasien tidak valid.")
        return weights

    def loader(self, rows, scaler, shuffle=False, seed=42):
        rows = rows.reset_index(drop=True)
        generator = torch.Generator().manual_seed(seed)
        dataset = A6Dataset(self.p, rows, self.cache, self.store, scaler)
        sampler = None
        if shuffle:
            sampler = WeightedRandomSampler(
                weights=torch.tensor(
                    self.sampling_weights(rows).copy(), dtype=torch.double
                ),
                num_samples=len(rows),
                replacement=True,
                generator=generator,
            )
        return DataLoader(
            dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_a6,
            num_workers=self.cfg["num_workers"],
            generator=generator,
            pin_memory=self.device.type == "cuda",
        )

    def sampling_audit(self, rows):
        rows = rows.reset_index(drop=True).copy()
        weights = self.sampling_weights(rows)
        probabilities = weights / weights.sum()
        patient_mass = (
            pd.DataFrame(
                {
                    "patient_id": rows.patient_id.astype(str),
                    "probability": probabilities,
                }
            )
            .groupby("patient_id", as_index=False)
            .probability.sum()
        )
        expected_per_patient = len(rows) / len(patient_mass)
        return {
            "subtrain_windows": int(len(rows)),
            "subtrain_patients": int(len(patient_mass)),
            "minimum_windows_per_patient": int(rows.groupby("patient_id").size().min()),
            "maximum_windows_per_patient": int(rows.groupby("patient_id").size().max()),
            "original_patient_window_ratio": float(
                rows.groupby("patient_id").size().max()
                / rows.groupby("patient_id").size().min()
            ),
            "minimum_expected_patient_mass": float(patient_mass.probability.min()),
            "maximum_expected_patient_mass": float(patient_mass.probability.max()),
            "expected_samples_per_patient": float(expected_per_patient),
            "samples_per_epoch": int(len(rows)),
            "replacement": True,
        }

    def audit(self):
        summary, feature_audit = super().audit()
        summary.update(
            {
                "variant": self.variant,
                "architecture": "identical to A6",
                "new_feature_groups_vs_A6": 0,
                "sampling": "inverse patient window count",
                "samples_per_epoch": "same as A6 subtrain size",
                "class_weighting": "S3 tau=2 unchanged",
                "checkpoint_A6_loaded": False,
                "holdout_accessed": False,
            }
        )
        return summary, feature_audit

    def compare_fold(self, fold=4):
        fold = int(fold)
        folders = {
            "A6": (
                self.p.out / "ablation_rr" / self.arch / "S3_tau2" /
                "A6_WAVE_DELINEATION_STANDALONE" / f"fold_{fold}"
            ),
            "A6_PB": self.run_dir(fold),
        }
        if any(not (folder / "result.json").exists() for folder in folders.values()):
            raise FileNotFoundError("Hasil A6 atau A6-PB fold belum lengkap.")
        rows = []
        for name, folder in folders.items():
            result = json.loads((folder / "result.json").read_text())
            per_class = pd.read_csv(folder / "per_class.csv").set_index("class")
            rows.append(
                {
                    "variant": name,
                    "macro_f1": float(result["macro_f1"]),
                    **{
                        f"f1_{label}": float(per_class.loc[label, "f1"])
                        for label in ["N", "S", "V", "AF", "VT"]
                    },
                }
            )
        comparison = pd.DataFrame(rows).set_index("variant").T
        comparison["delta_A6_PB_vs_A6"] = comparison.A6_PB - comparison.A6
        return comparison.reset_index(names="metric")

    def compare_a5_fold(self, fold=0):
        """Perbandingan lima-fold yang adil terhadap A5 standalone lengkap."""
        fold = int(fold)
        base = self.p.out / "ablation_rr" / self.arch / "S3_tau2"
        folders = {
            "A5": base / "A5_PHYSIOLOGY_STANDALONE" / f"fold_{fold}",
            "A6_PB": self.run_dir(fold),
        }
        if any(not (folder / "result.json").exists() for folder in folders.values()):
            raise FileNotFoundError(f"Hasil A5 atau A6-PB fold {fold} belum lengkap.")
        rows = []
        for name, folder in folders.items():
            result = json.loads((folder / "result.json").read_text())
            per_class = pd.read_csv(folder / "per_class.csv").set_index("class")
            rows.append(
                {
                    "variant": name,
                    "macro_f1": float(result["macro_f1"]),
                    **{
                        f"f1_{label}": float(per_class.loc[label, "f1"])
                        for label in ["N", "S", "V", "AF", "VT"]
                    },
                }
            )
        comparison = pd.DataFrame(rows).set_index("variant").T
        comparison["delta_A6_PB_vs_A5"] = comparison.A6_PB - comparison.A5
        return comparison.reset_index(names="metric")
