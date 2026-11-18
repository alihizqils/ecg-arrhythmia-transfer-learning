"""A4: A2 ditambah encoder urutan RR/dRR tanpa mengubah backbone citra."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from tools.rr_ablation import RRScaler, RRWindowDataset, collate_rr
from tools.rr_ablation_a2 import (
    A2AblationExperiment,
    A2FeatureStore,
    A2_GLOBAL_RR_NAMES,
    A2_LOCAL_RR_NAMES,
)


SEQUENCE_RR_NAMES = [
    "rr_previous_over_record_median",
    "rr_next_over_record_median",
    "drr_next_minus_previous_over_record_median",
]


class A4SequenceFeatureStore(A2FeatureStore):
    """Menyimpan urutan RR terurut per window dengan baseline record tanpa label."""

    def __init__(self, protocol, allowed_window_ids):
        super().__init__(protocol, allowed_window_ids)
        self.sequence_by_window = {}
        for window_id in map(str, allowed_window_ids):
            global_rr, local_rr = self.by_window[window_id]
            baseline = float(global_rr[-1])
            previous = local_rr[:, 0].astype(np.float32)
            following = local_rr[:, 1].astype(np.float32)
            if np.isfinite(baseline) and baseline > 0:
                sequence = np.column_stack(
                    [
                        previous / baseline,
                        following / baseline,
                        (following - previous) / baseline,
                    ]
                ).astype(np.float32)
            else:
                sequence = np.full(
                    (len(local_rr), len(SEQUENCE_RR_NAMES)), np.nan, dtype=np.float32
                )
            self.sequence_by_window[window_id] = sequence

    def sequence_audit(self, window_ids):
        values = np.concatenate(
            [self.sequence_by_window[str(window_id)] for window_id in window_ids], axis=0
        )
        rows = []
        for index, name in enumerate(SEQUENCE_RR_NAMES):
            finite = np.isfinite(values[:, index])
            rows.append(
                {
                    "scope": "sequence",
                    "feature": name,
                    "available": int(finite.sum()),
                    "missing": int((~finite).sum()),
                    "availability": float(finite.mean()),
                }
            )
        return pd.DataFrame(rows)


class A4SequenceScaler(RRScaler):
    """Scaler subtrain-only untuk fitur A2 dan urutan RR/dRR."""

    def __init__(self, store, window_ids):
        super().__init__(store, window_ids)
        sequence_values = np.concatenate(
            [store.sequence_by_window[str(window_id)] for window_id in window_ids], axis=0
        ).astype(np.float64)
        self.sequence_mean, self.sequence_std = self._fit(sequence_values)

    def transform_sequence(self, values):
        return self._transform(values, self.sequence_mean, self.sequence_std)

    def state_dict(self):
        state = super().state_dict()
        state.update(
            {
                "sequence_mean": self.sequence_mean.tolist(),
                "sequence_std": self.sequence_std.tolist(),
            }
        )
        return state


class A4SequenceDataset(RRWindowDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        sequence = self.store.sequence_by_window[str(item["id"])]
        if len(sequence) != len(item["local"]):
            raise ValueError(f"Panjang urutan RR berbeda pada {item['id']}.")
        item["rr_sequence"] = torch.from_numpy(self.scaler.transform_sequence(sequence))
        return item


def collate_a4_sequence(items):
    batch = collate_rr(items)
    feature_dim = items[0]["rr_sequence"].shape[-1]
    sequence = torch.zeros(
        (len(items), batch["local"].shape[1], feature_dim), dtype=torch.float32
    )
    for index, item in enumerate(items):
        sequence[index, : len(item["rr_sequence"])] = item["rr_sequence"]
    batch["rr_sequence"] = sequence
    return batch


class A4SequenceHierarchicalHead(nn.Module):
    """Head A2 dengan BiGRU ringan untuk mempertahankan urutan RR/dRR."""

    def __init__(self, dim=1280, dropout=0.2, rr_hidden=32, sequence_hidden=16):
        super().__init__()
        self.dim = dim
        self.frozen = True
        self.global_mean = nn.Buffer(torch.zeros(dim))
        self.global_std = nn.Buffer(torch.ones(dim))
        self.local_mean = nn.Buffer(torch.zeros(dim))
        self.local_std = nn.Buffer(torch.ones(dim))
        self.global_rr = nn.Sequential(
            nn.Linear(2 * len(A2_GLOBAL_RR_NAMES), rr_hidden),
            nn.ReLU(),
            nn.LayerNorm(rr_hidden),
        )
        self.local_rr = nn.Sequential(
            nn.Linear(2 * len(A2_LOCAL_RR_NAMES), rr_hidden),
            nn.ReLU(),
            nn.LayerNorm(rr_hidden),
        )
        self.sequence_encoder = nn.GRU(
            input_size=2 * len(SEQUENCE_RR_NAMES),
            hidden_size=sequence_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.sequence_norm = nn.LayerNorm(2 * sequence_hidden)
        self.dropout = nn.Dropout(dropout)
        fused = dim + rr_hidden + 2 * sequence_hidden
        self.global_head = nn.Linear(fused, 3)
        self.beat_head = nn.Linear(fused, 3)
        self.attention = nn.Linear(fused, 3)

    def forward(self, global_x, local_x, mask, global_rr, local_rr, rr_sequence):
        if not mask.any(1).all():
            raise ValueError("Setiap window harus memiliki setidaknya satu crop.")
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

        global_x = (global_x - self.global_mean) / self.global_std
        local_x = (local_x - self.local_mean) / self.local_std
        global_fused = torch.cat(
            [global_x, self.global_rr(global_rr), global_context], dim=-1
        )
        local_fused = torch.cat(
            [local_x, self.local_rr(local_rr), local_context], dim=-1
        )
        global_logits = self.global_head(self.dropout(global_fused))
        beat_logits = self.beat_head(self.dropout(local_fused))
        attention = self.attention(local_fused).masked_fill(
            ~mask[..., None], -torch.inf
        ).softmax(1)
        local_logits = (attention * beat_logits).sum(1)
        pg, pl = global_logits.log_softmax(-1), local_logits.log_softmax(-1)
        logp = torch.cat([pg[:, :1] + pl, pg[:, 1:]], dim=-1)
        return {
            "logp": logp,
            "global_logits": global_logits,
            "beat_logits": beat_logits,
            "attention": attention,
            "local_logits": local_logits,
        }


class A4SequenceExperiment(A2AblationExperiment):
    variant = "A4_RR_SEQUENCE_BIGRU"
    sequence_rr_names = SEQUENCE_RR_NAMES
    implementation_file = Path(__file__)

    def __init__(self, protocol, arch="efficientnet_b0", device=None, resume=True):
        super().__init__(protocol, arch=arch, device=device, resume=resume)
        self.store = A4SequenceFeatureStore(protocol, self.dev.window_id)

    def audit(self):
        summary, feature_audit = super().audit()
        for fold in range(5):
            reference = super().run_dir(fold) / "result.json"
            if not reference.exists():
                raise FileNotFoundError(f"Hasil A2 fold {fold} belum tersedia.")
        summary.update(
            {
                "reference": "A2_RECORD_RR",
                "rr_features_sequence": len(SEQUENCE_RR_NAMES),
                "sequence_encoder": "bidirectional GRU, hidden=16 per direction",
                "backbone": "frozen EfficientNet-B0 feature cache",
            }
        )
        return summary, pd.concat(
            [feature_audit, self.store.sequence_audit(self.dev.window_id)],
            ignore_index=True,
        )

    def run_dir(self, fold):
        return (
            self.p.out
            / "ablation_rr"
            / self.arch
            / "S3_tau2"
            / self.variant
            / f"fold_{int(fold)}"
        )

    def loader(self, rows, scaler, shuffle=False, seed=42):
        generator = torch.Generator().manual_seed(seed)
        dataset = A4SequenceDataset(self.p, rows, self.cache, self.store, scaler)
        return DataLoader(
            dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=shuffle,
            collate_fn=collate_a4_sequence,
            num_workers=self.cfg["num_workers"],
            generator=generator,
            pin_memory=self.device.type == "cuda",
        )

    def make_scaler(self, window_ids):
        return A4SequenceScaler(self.store, window_ids)

    def _model(self):
        return A4SequenceHierarchicalHead(
            dim=1280,
            dropout=self.p.cfg["model"]["dropout"],
        ).to(self.device)

    def forward_batch(self, model, batch):
        return model(
            batch["global"],
            batch["local"],
            batch["mask"],
            batch["global_rr"],
            batch["local_rr"],
            batch["rr_sequence"],
        )

    def rr_feature_metadata(self):
        return {
            "global": self.global_rr_names,
            "local": self.local_rr_names,
            "sequence": self.sequence_rr_names,
        }

    def compare_fold(self, fold=0):
        fold = int(fold)
        a2_folder = super().run_dir(fold)
        a4_folder = self.run_dir(fold)
        if not (a2_folder / "result.json").exists() or not (
            a4_folder / "result.json"
        ).exists():
            raise FileNotFoundError("Hasil A2 atau A4 fold belum lengkap.")
        rows = []
        for class_name in ["N", "S", "V", "AF", "VT"]:
            a2 = pd.read_csv(a2_folder / "per_class.csv").set_index("class")
            a4 = pd.read_csv(a4_folder / "per_class.csv").set_index("class")
            rows.append(
                {
                    "metric": f"f1_{class_name}",
                    "A2": float(a2.loc[class_name, "f1"]),
                    "A4": float(a4.loc[class_name, "f1"]),
                }
            )
        import json

        a2_result = json.loads((a2_folder / "result.json").read_text())
        a4_result = json.loads((a4_folder / "result.json").read_text())
        rows.insert(
            0,
            {
                "metric": "macro_f1",
                "A2": float(a2_result["macro_f1"]),
                "A4": float(a4_result["macro_f1"]),
            },
        )
        result = pd.DataFrame(rows)
        result["delta_A4_vs_A2"] = result.A4 - result.A2
        return result
