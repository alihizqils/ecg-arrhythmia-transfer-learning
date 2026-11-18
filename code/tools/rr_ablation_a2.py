"""A2: A1 ditambah baseline RR tingkat rekaman, tanpa mengubah backbone."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ecgtl.hierarchical.data import read_manifest
from tools.rr_ablation import (
    GLOBAL_RR_NAMES,
    LOCAL_RR_NAMES,
    RRFeatureStore,
    RRAblationExperiment,
)


A2_GLOBAL_RR_NAMES = [*GLOBAL_RR_NAMES, "record_median_rr_s"]
A2_LOCAL_RR_NAMES = [*LOCAL_RR_NAMES, "rr_previous_over_record_median"]


class A2FeatureStore(RRFeatureStore):
    """Menambahkan baseline record yang dihitung hanya dari R-peak GQRS."""

    def __init__(self, protocol, allowed_window_ids):
        allowed = set(map(str, allowed_window_ids))
        super().__init__(protocol, allowed)
        local = read_manifest(protocol.prepared / "local_manifest.csv")
        local = local[local.parent_window_id.astype(str).isin(allowed)].copy()
        local["record_key"] = (
            local.database.astype(str) + "/" + local.record_id.astype(str) + "/" + local.channel.astype(str)
        )
        window_to_record = local.groupby("parent_window_id", sort=False).record_key.first().to_dict()

        record_intervals = {}
        for window_id, rows in local.groupby("parent_window_id", sort=False):
            rows = rows.reset_index(drop=True)
            if rows.fallback_peak.astype(bool).any():
                continue
            peaks = rows.rpeak_sample.to_numpy(dtype=float)
            rr = np.diff(peaks) / self.fs
            rr = rr[np.isfinite(rr) & (rr > 0)]
            if len(rr):
                record_intervals.setdefault(rows.record_key.iloc[0], []).append(rr)
        record_median = {
            key: float(np.median(np.concatenate(parts)))
            for key, parts in record_intervals.items()
            if parts
        }

        for window_id in allowed:
            global_rr, local_rr = self.by_window[window_id]
            baseline = record_median.get(window_to_record[window_id], np.nan)
            previous_ratio = (
                local_rr[:, 0] / baseline
                if np.isfinite(baseline) and baseline > 0
                else np.full(len(local_rr), np.nan, dtype=np.float32)
            )
            self.by_window[window_id] = (
                np.concatenate([global_rr, np.asarray([baseline], dtype=np.float32)]),
                np.column_stack([local_rr, previous_ratio.astype(np.float32)]),
            )

    def audit(self, window_ids):
        global_values = np.stack([self.by_window[str(w)][0] for w in window_ids])
        local_values = np.concatenate([self.by_window[str(w)][1] for w in window_ids])
        rows = []
        for scope, names, values in [
            ("global", A2_GLOBAL_RR_NAMES, global_values),
            ("local", A2_LOCAL_RR_NAMES, local_values),
        ]:
            for index, name in enumerate(names):
                finite = np.isfinite(values[:, index])
                rows.append({
                    "scope": scope,
                    "feature": name,
                    "available": int(finite.sum()),
                    "missing": int((~finite).sum()),
                    "availability": float(finite.mean()),
                })
        return pd.DataFrame(rows)


class A2HierarchicalHead(nn.Module):
    """Head A1 yang dimensinya diperluas untuk dua fitur A2."""

    def __init__(self, dim=1280, dropout=0.2, rr_hidden=32):
        super().__init__()
        self.dim = dim
        self.frozen = True
        self.global_mean = nn.Buffer(torch.zeros(dim))
        self.global_std = nn.Buffer(torch.ones(dim))
        self.local_mean = nn.Buffer(torch.zeros(dim))
        self.local_std = nn.Buffer(torch.ones(dim))
        self.global_rr = nn.Sequential(
            nn.Linear(2 * len(A2_GLOBAL_RR_NAMES), rr_hidden), nn.ReLU(), nn.LayerNorm(rr_hidden)
        )
        self.local_rr = nn.Sequential(
            nn.Linear(2 * len(A2_LOCAL_RR_NAMES), rr_hidden), nn.ReLU(), nn.LayerNorm(rr_hidden)
        )
        self.dropout = nn.Dropout(dropout)
        fused = dim + rr_hidden
        self.global_head = nn.Linear(fused, 3)
        self.beat_head = nn.Linear(fused, 3)
        self.attention = nn.Linear(fused, 3)

    def forward(self, global_x, local_x, mask, global_rr, local_rr):
        if not mask.any(1).all():
            raise ValueError("Setiap window harus memiliki setidaknya satu crop.")
        global_x = (global_x - self.global_mean) / self.global_std
        local_x = (local_x - self.local_mean) / self.local_std
        global_fused = torch.cat([global_x, self.global_rr(global_rr)], dim=-1)
        local_fused = torch.cat([local_x, self.local_rr(local_rr)], dim=-1)
        global_logits = self.global_head(self.dropout(global_fused))
        beat_logits = self.beat_head(self.dropout(local_fused))
        attention = self.attention(local_fused).masked_fill(~mask[..., None], -torch.inf).softmax(1)
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


class A2AblationExperiment(RRAblationExperiment):
    variant = "A2_RECORD_RR"
    global_rr_names = A2_GLOBAL_RR_NAMES
    local_rr_names = A2_LOCAL_RR_NAMES
    implementation_file = Path(__file__)

    def __init__(self, protocol, arch="efficientnet_b0", device=None, resume=True):
        super().__init__(protocol, arch=arch, device=device, resume=resume)
        self.store = A2FeatureStore(protocol, self.dev.window_id)

    def audit(self):
        if not (self.cache / "complete.json").exists():
            raise FileNotFoundError("Feature cache EfficientNet-B0 belum lengkap.")
        for fold in range(5):
            if not (self.base.run_dir("S3", fold, 2) / "result.json").exists():
                raise FileNotFoundError(f"Hasil A0 fold {fold} belum tersedia.")
            if not (self.p.out / "ablation_rr" / self.arch / "S3_tau2" / "A1_LG" / f"fold_{fold}" / "result.json").exists():
                raise FileNotFoundError(f"Hasil A1 fold {fold} belum tersedia.")
        audit = self.store.audit(self.dev.window_id)
        return {
            "development_windows": len(self.dev),
            "patients": int(self.dev.patient_id.nunique()),
            "folds": int(self.dev.fold.nunique()),
            "rr_features_global": len(A2_GLOBAL_RR_NAMES),
            "rr_features_local": len(A2_LOCAL_RR_NAMES),
            "record_baseline_scope": "record/channel, R-peak GQRS, tanpa label",
        }, audit

    def run_dir(self, fold):
        return self.p.out / "ablation_rr" / self.arch / "S3_tau2" / "A2_RECORD_RR" / f"fold_{fold}"

    def _model(self):
        return A2HierarchicalHead(
            dim=1280,
            dropout=self.p.cfg["model"]["dropout"],
        ).to(self.device)

    def compare_fold(self, fold=0):
        fold = int(fold)
        folders = {
            "A0": self.base.run_dir("S3", fold, 2),
            "A1": self.p.out / "ablation_rr" / self.arch / "S3_tau2" / "A1_LG" / f"fold_{fold}",
            "A2": self.run_dir(fold),
        }
        if any(not (folder / "result.json").exists() for folder in folders.values()):
            raise FileNotFoundError("Hasil A0, A1, atau A2 fold belum lengkap.")
        results = {name: __import__("json").loads((folder / "result.json").read_text()) for name, folder in folders.items()}
        classes = {name: pd.read_csv(folder / "per_class.csv").set_index("class") for name, folder in folders.items()}
        rows = [{"metric": "macro_f1", **{name: result["macro_f1"] for name, result in results.items()}}]
        for class_name in ["N", "S", "V", "AF", "VT"]:
            rows.append({"metric": f"f1_{class_name}", **{
                name: float(frame.loc[class_name, "f1"]) for name, frame in classes.items()
            }})
        result = pd.DataFrame(rows)
        result["delta_A2_vs_A1"] = result.A2 - result.A1
        result["delta_A2_vs_A0"] = result.A2 - result.A0
        return result
