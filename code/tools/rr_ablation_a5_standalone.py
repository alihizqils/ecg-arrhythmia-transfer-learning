"""Pilot A5 mandiri: head citra/RR/fisiologi dilatih bersama tanpa checkpoint A2."""
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from ecgtl.hierarchical.training import Experiment
from tools.a5_physiology import GLOBAL_PHYSIOLOGY_NAMES, LOCAL_PHYSIOLOGY_NAMES
from tools.rr_ablation_a4_sequence import A4SequenceHierarchicalHead
from tools.rr_ablation_a5_physiology import A5FeatureStore, A5PhysiologyExperiment


class A5StandaloneHead(A4SequenceHierarchicalHead):
    """Fusi fisiologi global/lokal dan konteks RR dengan seluruh head trainable."""

    def __init__(self, dim=1280, dropout=0.2):
        super().__init__(dim=dim, dropout=dropout)
        self.global_rr = nn.Sequential(
            nn.Linear(20 + 2 * len(GLOBAL_PHYSIOLOGY_NAMES), 32),
            nn.ReLU(), nn.LayerNorm(32),
        )
        self.local_rr = nn.Sequential(
            nn.Linear(14 + 2 * len(LOCAL_PHYSIOLOGY_NAMES), 32),
            nn.ReLU(), nn.LayerNorm(32),
        )

    def forward(self, global_x, local_x, mask, global_rr, local_rr,
                rr_sequence, physiology_global, physiology_local):
        return super().forward(
            global_x, local_x, mask,
            torch.cat([global_rr, physiology_global], dim=-1),
            torch.cat([local_rr, physiology_local], dim=-1),
            rr_sequence,
        )


class A5StandaloneExperiment(A5PhysiologyExperiment):
    variant = "A5_PHYSIOLOGY_STANDALONE"
    preserve_initial_baseline = False
    implementation_file = Path(__file__)

    def __init__(self, protocol, arch="efficientnet_b0", device=None, resume=True):
        if arch != "efficientnet_b0":
            raise ValueError("Pilot ini dikunci pada EfficientNet-B0.")
        self.base = Experiment(protocol, arch, device=device, resume=resume)
        self.p, self.arch = protocol, arch
        self.device, self.resume = self.base.device, resume
        self.cfg, self.dev, self.cache = self.base.cfg, self.base.dev, self.base.cache
        self.store = A5FeatureStore(protocol, self.dev.window_id)

    def build_model(self, fold):
        return A5StandaloneHead(
            dim=1280, dropout=self.p.cfg["model"]["dropout"]
        ).to(self.device)

    def _fit_image_scaler(self, model, rows):
        self.base.fit_scaler(model, rows)

    def encoder_parameter_count(self):
        # Arsitektur saja; penghitungan metadata tidak memuat bobot/checkpoint ECG.
        from ecgtl.hierarchical.model import HierarchicalNet
        network = HierarchicalNet(self.arch, "S3", self.p.cfg, pretrained=False)
        return sum(parameter.numel() for parameter in network.encoder.parameters())

    def audit(self):
        if not (self.cache / "complete.json").exists():
            raise FileNotFoundError("Feature cache EfficientNet-B0 belum lengkap.")
        return {
            "variant": self.variant,
            "development_windows": len(self.dev),
            "patients": int(self.dev.patient_id.nunique()),
            "folds": int(self.dev.fold.nunique()),
            "initialization": "new trainable head; frozen ImageNet feature cache",
            "checkpoint_A2_loaded": False,
            "physiology_global_features": len(GLOBAL_PHYSIOLOGY_NAMES),
            "physiology_local_features": len(LOCAL_PHYSIOLOGY_NAMES),
            "holdout_accessed": False,
        }, pd.concat([
            self.store.audit(self.dev.window_id),
            self.store.sequence_audit(self.dev.window_id),
            self.store.cache_audit,
        ], ignore_index=True)

    def initialization_audit(self, fold=0):
        raise RuntimeError("A5 mandiri memakai head baru; audit reproduksi A2 tidak berlaku.")
