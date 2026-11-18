"""A5: fusi residual terarah A2 dengan urutan RR dan fitur fisiologis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from ecgtl.hierarchical.training import load_checkpoint
from tools.a5_physiology import (
    GLOBAL_PHYSIOLOGY_NAMES,
    LOCAL_PHYSIOLOGY_NAMES,
    audit_physiology_cache,
    build_physiology_cache,
)
from tools.rr_ablation import RRWindowDataset, collate_rr
from tools.rr_ablation_a2 import A2HierarchicalHead
from tools.rr_ablation_a4_sequence import (
    A4SequenceExperiment,
    A4SequenceFeatureStore,
    A4SequenceScaler,
    SEQUENCE_RR_NAMES,
)


class A5FeatureStore(A4SequenceFeatureStore):
    def __init__(self, protocol, allowed_window_ids):
        allowed = list(map(str, allowed_window_ids))
        super().__init__(protocol, allowed)
        self.cache_folder = build_physiology_cache(protocol)
        summary, audit = audit_physiology_cache(protocol, self.cache_folder)
        self.cache_summary, self.cache_audit = summary, audit
        values = np.load(self.cache_folder / "features.npz")
        ids = values["window_ids"].astype(str)
        global_array = values["global_features"]
        local_array = values["local_features"]
        offsets = values["local_offsets"]
        positions = {window_id: index for index, window_id in enumerate(ids)}
        if set(allowed) - set(positions):
            raise ValueError("Cache fisiologi tidak mencakup seluruh development set.")
        self.physiology_by_window = {}
        for window_id in allowed:
            index = positions[window_id]
            local = local_array[offsets[index] : offsets[index + 1]].astype(np.float32)
            global_features = global_array[index].astype(np.float32)
            if len(local) != len(self.by_window[window_id][1]):
                raise ValueError(f"Jumlah fitur fisiologi berbeda pada {window_id}.")
            self.physiology_by_window[window_id] = (global_features, local)


class A5Scaler(A4SequenceScaler):
    def __init__(self, store, window_ids):
        super().__init__(store, window_ids)
        global_values = np.stack(
            [store.physiology_by_window[str(window_id)][0] for window_id in window_ids]
        ).astype(np.float64)
        local_values = np.concatenate(
            [store.physiology_by_window[str(window_id)][1] for window_id in window_ids], axis=0
        ).astype(np.float64)
        self.physiology_global_mean, self.physiology_global_std = self._fit(global_values)
        self.physiology_local_mean, self.physiology_local_std = self._fit(local_values)

    def transform_physiology_global(self, values):
        return self._transform(values, self.physiology_global_mean, self.physiology_global_std)

    def transform_physiology_local(self, values):
        return self._transform(values, self.physiology_local_mean, self.physiology_local_std)

    def state_dict(self):
        state = super().state_dict()
        state.update(
            {
                "physiology_global_mean": self.physiology_global_mean.tolist(),
                "physiology_global_std": self.physiology_global_std.tolist(),
                "physiology_local_mean": self.physiology_local_mean.tolist(),
                "physiology_local_std": self.physiology_local_std.tolist(),
            }
        )
        return state


class A5Dataset(RRWindowDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        window_id = str(item["id"])
        sequence = self.store.sequence_by_window[window_id]
        physiology_global, physiology_local = self.store.physiology_by_window[window_id]
        if len(sequence) != len(item["local"]) or len(physiology_local) != len(item["local"]):
            raise ValueError(f"Dimensi fitur lokal A5 berbeda pada {window_id}.")
        item["rr_sequence"] = torch.from_numpy(self.scaler.transform_sequence(sequence))
        item["physiology_global"] = torch.from_numpy(
            self.scaler.transform_physiology_global(physiology_global)
        )
        item["physiology_local"] = torch.from_numpy(
            self.scaler.transform_physiology_local(physiology_local)
        )
        return item


def collate_a5(items):
    batch = collate_rr(items)
    maximum = batch["local"].shape[1]
    sequence_dim = items[0]["rr_sequence"].shape[-1]
    physiology_dim = items[0]["physiology_local"].shape[-1]
    sequence = torch.zeros((len(items), maximum, sequence_dim), dtype=torch.float32)
    physiology_local = torch.zeros((len(items), maximum, physiology_dim), dtype=torch.float32)
    for index, item in enumerate(items):
        count = len(item["local"])
        sequence[index, :count] = item["rr_sequence"]
        physiology_local[index, :count] = item["physiology_local"]
    batch["rr_sequence"] = sequence
    batch["physiology_global"] = torch.stack([item["physiology_global"] for item in items])
    batch["physiology_local"] = physiology_local
    return batch


class A5RoutedResidualHead(nn.Module):
    """A2 beku ditambah residual lokal/global yang dibatasi."""

    residual_limit = 0.5

    def __init__(self, base: A2HierarchicalHead, sequence_hidden=16, physiology_hidden=32):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.base.eval()
        self.sequence_encoder = nn.GRU(
            input_size=2 * len(SEQUENCE_RR_NAMES),
            hidden_size=sequence_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.sequence_norm = nn.LayerNorm(2 * sequence_hidden)
        self.global_physiology = nn.Sequential(
            nn.Linear(2 * len(GLOBAL_PHYSIOLOGY_NAMES), physiology_hidden),
            nn.ReLU(),
            nn.LayerNorm(physiology_hidden),
        )
        self.local_physiology = nn.Sequential(
            nn.Linear(2 * len(LOCAL_PHYSIOLOGY_NAMES), physiology_hidden),
            nn.ReLU(),
            nn.LayerNorm(physiology_hidden),
        )
        self.global_residual = nn.Linear(2 * sequence_hidden + physiology_hidden, 3)
        self.local_residual = nn.Linear(2 * sequence_hidden + physiology_hidden, 3)
        nn.init.zeros_(self.global_residual.weight)
        nn.init.zeros_(self.global_residual.bias)
        nn.init.zeros_(self.local_residual.weight)
        nn.init.zeros_(self.local_residual.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

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
    ):
        with torch.no_grad():
            baseline = self.base(global_x, local_x, mask, global_rr, local_rr)
        lengths = mask.sum(1).to(dtype=torch.long)
        packed = pack_padded_sequence(
            rr_sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_context, hidden = self.sequence_encoder(packed)
        local_context, _ = pad_packed_sequence(
            packed_context, batch_first=True, total_length=local_x.shape[1]
        )
        local_context = self.sequence_norm(local_context)
        global_context = self.sequence_norm(torch.cat([hidden[-2], hidden[-1]], dim=-1))

        global_input = torch.cat(
            [global_context, self.global_physiology(physiology_global)], dim=-1
        )
        local_input = torch.cat(
            [local_context, self.local_physiology(physiology_local)], dim=-1
        )
        global_delta = self.residual_limit * torch.tanh(self.global_residual(global_input))
        local_delta = self.residual_limit * torch.tanh(self.local_residual(local_input))
        global_logits = baseline["global_logits"] + global_delta
        beat_logits = baseline["beat_logits"] + local_delta
        attention = baseline["attention"]
        local_logits = (attention * beat_logits).sum(1)
        pg, pl = global_logits.log_softmax(-1), local_logits.log_softmax(-1)
        logp = torch.cat([pg[:, :1] + pl, pg[:, 1:]], dim=-1)
        return {
            "logp": logp,
            "global_logits": global_logits,
            "beat_logits": beat_logits,
            "attention": attention,
            "local_logits": local_logits,
            "global_delta": global_delta,
            "local_delta": local_delta,
        }


class A5PhysiologyExperiment(A4SequenceExperiment):
    variant = "A5_PHYSIOLOGY_ROUTED_RESIDUAL"
    preserve_initial_baseline = True
    implementation_file = Path(__file__)

    def __init__(self, protocol, arch="efficientnet_b0", device=None, resume=True):
        super().__init__(protocol, arch=arch, device=device, resume=resume)
        self.store = A5FeatureStore(protocol, self.dev.window_id)

    def run_dir(self, fold):
        return (
            self.p.out / "ablation_rr" / self.arch / "S3_tau2" /
            self.variant / f"fold_{int(fold)}"
        )

    def source_a2_dir(self, fold):
        return (
            self.p.out / "ablation_rr" / self.arch / "S3_tau2" /
            "A2_RECORD_RR" / f"fold_{int(fold)}"
        )

    def audit(self):
        summary, feature_audit = super().audit()
        summary.update(
            {
                "variant": self.variant,
                "initialization": "A2 checkpoint per fold; residual zero-initialized",
                "residual_limit": A5RoutedResidualHead.residual_limit,
                "physiology_global_features": len(GLOBAL_PHYSIOLOGY_NAMES),
                "physiology_local_features": len(LOCAL_PHYSIOLOGY_NAMES),
                "holdout_accessed": False,
            }
        )
        return summary, pd.concat(
            [feature_audit, self.store.cache_audit], ignore_index=True
        )

    def loader(self, rows, scaler, shuffle=False, seed=42):
        generator = torch.Generator().manual_seed(seed)
        dataset = A5Dataset(self.p, rows, self.cache, self.store, scaler)
        return DataLoader(
            dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=shuffle,
            collate_fn=collate_a5,
            num_workers=self.cfg["num_workers"],
            generator=generator,
            pin_memory=self.device.type == "cuda",
        )

    def make_scaler(self, window_ids):
        return A5Scaler(self.store, window_ids)

    def build_model(self, fold):
        checkpoint = load_checkpoint(self.source_a2_dir(fold) / "best.pt")
        if checkpoint["protocol_hash"] != self.p.hash:
            raise ValueError("Checkpoint A2 berbeda protokol.")
        base = A2HierarchicalHead(
            dim=1280,
            dropout=self.p.cfg["model"]["dropout"],
        ).to(self.device)
        base.load_state_dict(checkpoint["model"])
        return A5RoutedResidualHead(base).to(self.device)

    def _fit_image_scaler(self, model, rows):
        return None

    def forward_batch(self, model, batch):
        return model(
            batch["global"], batch["local"], batch["mask"],
            batch["global_rr"], batch["local_rr"], batch["rr_sequence"],
            batch["physiology_global"], batch["physiology_local"],
        )

    def rr_feature_metadata(self):
        return {
            "global": self.global_rr_names,
            "local": self.local_rr_names,
            "sequence": SEQUENCE_RR_NAMES,
            "physiology_global": GLOBAL_PHYSIOLOGY_NAMES,
            "physiology_local": LOCAL_PHYSIOLOGY_NAMES,
        }

    @torch.inference_mode()
    def initialization_audit(self, fold=0):
        fold = int(fold)
        train = self.dev[self.dev.fold != fold].reset_index(drop=True)
        valid = self.dev[self.dev.fold == fold].reset_index(drop=True)
        scaler = self.make_scaler(train.window_id)
        model = self.build_model(fold)
        prediction = self.predict(model, valid, scaler)
        source_probability = np.load(self.source_a2_dir(fold) / "probabilities.npy")
        source_labels = np.load(self.source_a2_dir(fold) / "labels.npy")
        maximum = float(np.max(np.abs(prediction["prob"] - source_probability)))
        return {
            "fold": fold,
            "labels_identical": bool(np.array_equal(prediction["y"], source_labels)),
            "maximum_probability_difference": maximum,
            "predicted_classes_identical": bool(np.array_equal(
                prediction["prob"].argmax(1), source_probability.argmax(1)
            )),
            "residual_zero": maximum <= 1e-7,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "frozen_a2_parameters": sum(
                parameter.numel() for parameter in model.base.parameters()
            ),
        }

    def compare_fold(self, fold=0):
        fold = int(fold)
        folders = {"A2": self.source_a2_dir(fold), "A5": self.run_dir(fold)}
        if any(not (folder / "result.json").exists() for folder in folders.values()):
            raise FileNotFoundError("Hasil A2 atau A5 fold belum lengkap.")
        rows = []
        for name, folder in folders.items():
            result = json.loads((folder / "result.json").read_text())
            per_class = pd.read_csv(folder / "per_class.csv").set_index("class")
            rows.append({
                "variant": name,
                "macro_f1": result["macro_f1"],
                **{f"f1_{label}": float(per_class.loc[label, "f1"])
                   for label in ["N", "S", "V", "AF", "VT"]},
            })
        comparison = pd.DataFrame(rows).set_index("variant").T
        comparison["delta_A5_vs_A2"] = comparison.A5 - comparison.A2
        return comparison.reset_index(names="metric")
