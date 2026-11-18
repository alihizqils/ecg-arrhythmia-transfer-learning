"""Pilot A6 mandiri: A5 ditambah delineasi P/PR/QRS tanpa checkpoint pilot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecgtl.hierarchical.training import Experiment
from tools.a5_physiology import GLOBAL_PHYSIOLOGY_NAMES, LOCAL_PHYSIOLOGY_NAMES
from tools.a6_wave_delineation import (
    GLOBAL_WAVE_NAMES,
    LOCAL_WAVE_NAMES,
    audit_wave_cache,
    build_wave_cache,
)
from tools.rr_ablation_a5_physiology import A5Dataset, A5FeatureStore, A5Scaler, collate_a5
from tools.rr_ablation_a5_standalone import A5StandaloneExperiment, A5StandaloneHead


class A6FeatureStore(A5FeatureStore):
    def __init__(self, protocol, allowed_window_ids):
        allowed = list(map(str, allowed_window_ids))
        super().__init__(protocol, allowed)
        self.wave_cache_folder = build_wave_cache(protocol)
        self.wave_cache_summary, self.wave_cache_audit = audit_wave_cache(
            protocol, self.wave_cache_folder
        )
        values = np.load(self.wave_cache_folder / "features.npz")
        ids = values["window_ids"].astype(str)
        positions = {window_id: index for index, window_id in enumerate(ids)}
        if set(allowed) - set(positions):
            raise ValueError("Cache delineasi A6 tidak mencakup development set.")
        global_array = values["global_features"]
        local_array = values["local_features"]
        offsets = values["local_offsets"]
        self.wave_by_window = {}
        for window_id in allowed:
            index = positions[window_id]
            local = local_array[offsets[index] : offsets[index + 1]].astype(np.float32)
            global_features = global_array[index].astype(np.float32)
            if len(local) != len(self.by_window[window_id][1]):
                raise ValueError(f"Jumlah fitur delineasi berbeda pada {window_id}.")
            self.wave_by_window[window_id] = (global_features, local)


class A6Scaler(A5Scaler):
    def __init__(self, store, window_ids):
        super().__init__(store, window_ids)
        global_values = np.stack([
            store.wave_by_window[str(window_id)][0] for window_id in window_ids
        ]).astype(np.float64)
        local_values = np.concatenate([
            store.wave_by_window[str(window_id)][1] for window_id in window_ids
        ], axis=0).astype(np.float64)
        self.wave_global_mean, self.wave_global_std = self._fit(global_values)
        self.wave_local_mean, self.wave_local_std = self._fit(local_values)

    def transform_wave_global(self, values):
        return self._transform(values, self.wave_global_mean, self.wave_global_std)

    def transform_wave_local(self, values):
        return self._transform(values, self.wave_local_mean, self.wave_local_std)

    def state_dict(self):
        state = super().state_dict()
        state.update({
            "wave_global_mean": self.wave_global_mean.tolist(),
            "wave_global_std": self.wave_global_std.tolist(),
            "wave_local_mean": self.wave_local_mean.tolist(),
            "wave_local_std": self.wave_local_std.tolist(),
        })
        return state


class A6Dataset(A5Dataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        window_id = str(item["id"])
        wave_global, wave_local = self.store.wave_by_window[window_id]
        if len(wave_local) != len(item["local"]):
            raise ValueError(f"Dimensi fitur lokal A6 berbeda pada {window_id}.")
        item["wave_global"] = torch.from_numpy(
            self.scaler.transform_wave_global(wave_global)
        )
        item["wave_local"] = torch.from_numpy(
            self.scaler.transform_wave_local(wave_local)
        )
        return item


def collate_a6(items):
    batch = collate_a5(items)
    maximum = batch["local"].shape[1]
    dimension = items[0]["wave_local"].shape[-1]
    wave_local = torch.zeros((len(items), maximum, dimension), dtype=torch.float32)
    for index, item in enumerate(items):
        wave_local[index, : len(item["local"])] = item["wave_local"]
    batch["wave_global"] = torch.stack([item["wave_global"] for item in items])
    batch["wave_local"] = wave_local
    return batch


class A6StandaloneHead(A5StandaloneHead):
    """Head baru yang memfusi A5 dan fitur delineasi secara bersamaan."""

    def __init__(self, dim=1280, dropout=0.2):
        super().__init__(dim=dim, dropout=dropout)
        self.global_rr = nn.Sequential(
            nn.Linear(
                20 + 2 * (len(GLOBAL_PHYSIOLOGY_NAMES) + len(GLOBAL_WAVE_NAMES)),
                32,
            ),
            nn.ReLU(),
            nn.LayerNorm(32),
        )
        self.local_rr = nn.Sequential(
            nn.Linear(
                14 + 2 * (len(LOCAL_PHYSIOLOGY_NAMES) + len(LOCAL_WAVE_NAMES)),
                32,
            ),
            nn.ReLU(),
            nn.LayerNorm(32),
        )

    def forward(
        self,
        global_x,
        local_x,
        mask,
        global_rr,
        local_rr,
        rr_sequence,
        physiology_global,
        physiology_local,
        wave_global,
        wave_local,
    ):
        return super().forward(
            global_x,
            local_x,
            mask,
            global_rr,
            local_rr,
            rr_sequence,
            torch.cat([physiology_global, wave_global], dim=-1),
            torch.cat([physiology_local, wave_local], dim=-1),
        )


class A6StandaloneExperiment(A5StandaloneExperiment):
    variant = "A6_WAVE_DELINEATION_STANDALONE"
    implementation_file = Path(__file__)

    def __init__(self, protocol, arch="efficientnet_b0", device=None, resume=True):
        if arch != "efficientnet_b0":
            raise ValueError("Pilot A6 dikunci pada EfficientNet-B0.")
        # Hindari membangun A5FeatureStore sementara sebelum A6FeatureStore;
        # indeks development berukuran besar cukup dibuat satu kali.
        self.base = Experiment(protocol, arch, device=device, resume=resume)
        self.p, self.arch = protocol, arch
        self.device, self.resume = self.base.device, resume
        self.cfg, self.dev, self.cache = self.base.cfg, self.base.dev, self.base.cache
        self.store = A6FeatureStore(protocol, self.dev.window_id)

    def build_model(self, fold):
        return A6StandaloneHead(
            dim=1280, dropout=self.p.cfg["model"]["dropout"]
        ).to(self.device)

    def make_scaler(self, window_ids):
        return A6Scaler(self.store, window_ids)

    def loader(self, rows, scaler, shuffle=False, seed=42):
        generator = torch.Generator().manual_seed(seed)
        dataset = A6Dataset(self.p, rows, self.cache, self.store, scaler)
        return DataLoader(
            dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=shuffle,
            collate_fn=collate_a6,
            num_workers=self.cfg["num_workers"],
            generator=generator,
            pin_memory=self.device.type == "cuda",
        )

    @staticmethod
    def forward_batch(model, batch):
        return model(
            batch["global"], batch["local"], batch["mask"],
            batch["global_rr"], batch["local_rr"], batch["rr_sequence"],
            batch["physiology_global"], batch["physiology_local"],
            batch["wave_global"], batch["wave_local"],
        )

    def rr_feature_metadata(self):
        metadata = super().rr_feature_metadata()
        metadata.update({
            "wave_global": GLOBAL_WAVE_NAMES,
            "wave_local": LOCAL_WAVE_NAMES,
        })
        return metadata

    def audit(self):
        if not (self.cache / "complete.json").exists():
            raise FileNotFoundError("Feature cache EfficientNet-B0 belum lengkap.")
        summary = {
            "variant": self.variant,
            "development_windows": len(self.dev),
            "patients": int(self.dev.patient_id.nunique()),
            "folds": int(self.dev.fold.nunique()),
            "initialization": "new trainable head; frozen ImageNet feature cache",
            "checkpoint_A2_or_A5_loaded": False,
            "physiology_global_features": len(GLOBAL_PHYSIOLOGY_NAMES),
            "physiology_local_features": len(LOCAL_PHYSIOLOGY_NAMES),
            "wave_global_features": len(GLOBAL_WAVE_NAMES),
            "wave_local_features": len(LOCAL_WAVE_NAMES),
            "holdout_accessed": False,
        }
        audits = [
            self.store.audit(self.dev.window_id),
            self.store.sequence_audit(self.dev.window_id),
            self.store.cache_audit,
            self.store.wave_cache_audit,
        ]
        return summary, pd.concat(audits, ignore_index=True)

    def compare_fold(self, fold=0):
        fold = int(fold)
        folders = {
            "A5": (
                self.p.out / "ablation_rr" / self.arch / "S3_tau2" /
                "A5_PHYSIOLOGY_STANDALONE" / f"fold_{fold}"
            ),
            "A6": self.run_dir(fold),
        }
        if any(not (folder / "result.json").exists() for folder in folders.values()):
            raise FileNotFoundError("Hasil A5 atau A6 fold belum lengkap.")
        rows = []
        for name, folder in folders.items():
            result = json.loads((folder / "result.json").read_text())
            per_class = pd.read_csv(folder / "per_class.csv").set_index("class")
            rows.append({
                "variant": name,
                "macro_f1": result["macro_f1"],
                **{
                    f"f1_{label}": float(per_class.loc[label, "f1"])
                    for label in ["N", "S", "V", "AF", "VT"]
                },
            })
        comparison = pd.DataFrame(rows).set_index("variant").T
        comparison["delta_A6_vs_A5"] = comparison.A6 - comparison.A5
        return comparison.reset_index(names="metric")

