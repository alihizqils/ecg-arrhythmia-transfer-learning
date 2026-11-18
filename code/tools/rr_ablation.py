"""Ablation RR pada feature cache eksperimen hierarkis tanpa mengubah protokol A0."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ecgtl.hierarchical.data import WindowDataset, collate, read_manifest
from ecgtl.hierarchical.metrics import calculate, export_metrics
from ecgtl.hierarchical.model import inverse_weights, loss_terms
from ecgtl.hierarchical.protocol import atomic_json, atomic_torch, digest, environment, seed_all
from ecgtl.hierarchical.training import Experiment, load_checkpoint, to_device


LOCAL_RR_NAMES = [
    "rr_previous_s",
    "rr_next_s",
    "rr_previous_over_median",
    "rr_next_over_median",
    "rr_next_over_previous",
    "compensatory_pause_ratio",
]

GLOBAL_RR_NAMES = [
    "detected_beats",
    "median_rr_s",
    "mean_heart_rate_bpm",
    "sdnn_s",
    "rmssd_s",
    "rr_coefficient_of_variation",
    "rr_iqr_s",
    "minimum_rr_over_median",
    "maximum_rr_over_median",
]


class RRFeatureStore:
    """Menghitung RR hanya dari R-peak GQRS yang diwariskan oleh local manifest."""

    def __init__(self, protocol, allowed_window_ids):
        self.protocol = protocol
        self.fs = float(protocol.cfg["data"]["fs"])
        local = read_manifest(protocol.prepared / "local_manifest.csv")
        allowed = set(map(str, allowed_window_ids))
        local = local[local.parent_window_id.astype(str).isin(allowed)]
        if set(local.parent_window_id.astype(str)) != allowed:
            raise ValueError("Local manifest development tidak lengkap untuk ablation RR.")
        self.by_window = {}
        for window_id, rows in local.groupby("parent_window_id", sort=False):
            rows = rows.reset_index(drop=True)
            peaks = rows.rpeak_sample.to_numpy(dtype=float)
            if len(peaks) > 1 and np.any(np.diff(peaks) < 0):
                raise ValueError(f"Urutan R-peak tidak monoton pada {window_id}.")
            fallback = rows.fallback_peak.astype(bool).to_numpy()
            valid_peaks = peaks[~fallback]
            rr = np.diff(valid_peaks) / self.fs if len(valid_peaks) > 1 else np.empty(0)
            median = float(np.median(rr)) if len(rr) else np.nan

            local_rr = np.full((len(rows), len(LOCAL_RR_NAMES)), np.nan, dtype=np.float32)
            if not fallback.any() and len(peaks):
                previous = np.r_[np.nan, np.diff(peaks) / self.fs]
                following = np.r_[np.diff(peaks) / self.fs, np.nan]
                with np.errstate(divide="ignore", invalid="ignore"):
                    local_rr[:, 0] = previous
                    local_rr[:, 1] = following
                    local_rr[:, 2] = previous / median
                    local_rr[:, 3] = following / median
                    local_rr[:, 4] = following / previous
                    local_rr[:, 5] = (previous + following) / (2 * median)

            global_rr = np.full(len(GLOBAL_RR_NAMES), np.nan, dtype=np.float32)
            global_rr[0] = len(valid_peaks)
            if len(rr):
                mean_rr = float(np.mean(rr))
                global_rr[1] = median
                global_rr[2] = 60.0 / mean_rr if mean_rr > 0 else np.nan
                global_rr[3] = float(np.std(rr, ddof=1)) if len(rr) > 1 else np.nan
                global_rr[4] = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else np.nan
                global_rr[5] = float(np.std(rr) / mean_rr) if mean_rr > 0 else np.nan
                q25, q75 = np.percentile(rr, [25, 75])
                global_rr[6] = float(q75 - q25)
                global_rr[7] = float(np.min(rr) / median) if median > 0 else np.nan
                global_rr[8] = float(np.max(rr) / median) if median > 0 else np.nan
            self.by_window[str(window_id)] = (global_rr, local_rr)

    def audit(self, window_ids):
        global_values = np.stack([self.by_window[str(w)][0] for w in window_ids])
        local_values = np.concatenate([self.by_window[str(w)][1] for w in window_ids])
        rows = []
        for scope, names, values in [
            ("global", GLOBAL_RR_NAMES, global_values),
            ("local", LOCAL_RR_NAMES, local_values),
        ]:
            for i, name in enumerate(names):
                finite = np.isfinite(values[:, i])
                rows.append({
                    "scope": scope,
                    "feature": name,
                    "available": int(finite.sum()),
                    "missing": int((~finite).sum()),
                    "availability": float(finite.mean()),
                })
        return pd.DataFrame(rows)


class RRScaler:
    """Standardisasi subtrain-only dengan mask eksplisit untuk nilai yang hilang."""

    def __init__(self, store, window_ids):
        global_values = np.stack([store.by_window[str(w)][0] for w in window_ids]).astype(np.float64)
        local_values = np.concatenate([store.by_window[str(w)][1] for w in window_ids]).astype(np.float64)
        self.global_mean, self.global_std = self._fit(global_values)
        self.local_mean, self.local_std = self._fit(local_values)

    @staticmethod
    def _fit(values):
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
        mean[~np.isfinite(mean)] = 0.0
        std[(~np.isfinite(std)) | (std < 1e-6)] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def _transform(values, mean, std):
        values = np.asarray(values, dtype=np.float32)
        valid = np.isfinite(values)
        normalized = np.zeros_like(values, dtype=np.float32)
        normalized[valid] = ((values - mean) / std)[valid]
        normalized = np.clip(normalized, -10.0, 10.0)
        return np.concatenate([normalized, valid.astype(np.float32)], axis=-1)

    def transform_global(self, values):
        return self._transform(values, self.global_mean, self.global_std)

    def transform_local(self, values):
        return self._transform(values, self.local_mean, self.local_std)

    def state_dict(self):
        return {
            "global_mean": self.global_mean.tolist(),
            "global_std": self.global_std.tolist(),
            "local_mean": self.local_mean.tolist(),
            "local_std": self.local_std.tolist(),
        }


class RRWindowDataset(Dataset):
    def __init__(self, protocol, rows, feature_cache, store, scaler):
        self.base = WindowDataset(protocol, rows, feature_cache)
        self.store, self.scaler = store, scaler

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        global_rr, local_rr = self.store.by_window[str(item["id"])]
        if len(local_rr) != len(item["local"]):
            raise ValueError(f"Jumlah fitur RR lokal berbeda pada {item['id']}.")
        item["global_rr"] = torch.from_numpy(self.scaler.transform_global(global_rr))
        item["local_rr"] = torch.from_numpy(self.scaler.transform_local(local_rr))
        return item


def collate_rr(items):
    batch = collate(items)
    local_dim = items[0]["local_rr"].shape[-1]
    local_rr = torch.zeros((len(items), batch["local"].shape[1], local_dim), dtype=torch.float32)
    for i, item in enumerate(items):
        local_rr[i, : len(item["local_rr"])] = item["local_rr"]
    batch["global_rr"] = torch.stack([item["global_rr"] for item in items])
    batch["local_rr"] = local_rr
    return batch


class RRHierarchicalHead(nn.Module):
    """Head S3 dengan fusi RR; probabilitas hierarkis tetap sama dengan A0."""

    def __init__(self, dim, dropout=0.2, rr_hidden=32):
        super().__init__()
        self.dim = dim
        self.frozen = True
        self.global_mean = nn.Buffer(torch.zeros(dim))
        self.global_std = nn.Buffer(torch.ones(dim))
        self.local_mean = nn.Buffer(torch.zeros(dim))
        self.local_std = nn.Buffer(torch.ones(dim))
        self.global_rr = nn.Sequential(
            nn.Linear(2 * len(GLOBAL_RR_NAMES), rr_hidden), nn.ReLU(), nn.LayerNorm(rr_hidden)
        )
        self.local_rr = nn.Sequential(
            nn.Linear(2 * len(LOCAL_RR_NAMES), rr_hidden), nn.ReLU(), nn.LayerNorm(rr_hidden)
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


class RRAblationExperiment:
    class_weight_tau = 2.0
    variant = "A1_LG"
    preserve_initial_baseline = False
    global_rr_names = GLOBAL_RR_NAMES
    local_rr_names = LOCAL_RR_NAMES
    implementation_file = Path(__file__)

    def __init__(self, protocol, arch="efficientnet_b0", device=None, resume=True):
        if arch != "efficientnet_b0":
            raise ValueError("Pilot Tahap 1 dikunci pada EfficientNet-B0.")
        self.base = Experiment(protocol, arch, device=device, resume=resume)
        self.p, self.arch = protocol, arch
        self.device, self.resume = self.base.device, resume
        self.cfg, self.dev, self.cache = self.base.cfg, self.base.dev, self.base.cache
        self.store = RRFeatureStore(protocol, self.dev.window_id)

    def audit(self):
        if not (self.cache / "complete.json").exists():
            raise FileNotFoundError("Feature cache EfficientNet-B0 belum lengkap.")
        a0 = self.base.run_dir("S3", 0, 2) / "result.json"
        if not a0.exists():
            raise FileNotFoundError("Hasil A0 S3 fold 0 belum tersedia.")
        audit = self.store.audit(self.dev.window_id)
        return {
            "development_windows": len(self.dev),
            "patients": int(self.dev.patient_id.nunique()),
            "folds": int(self.dev.fold.nunique()),
            "a0_fold0": str(a0),
            "rr_features_global": len(GLOBAL_RR_NAMES),
            "rr_features_local": len(LOCAL_RR_NAMES),
        }, audit

    def run_dir(self, fold):
        return self.p.out / "ablation_rr" / self.arch / "S3_tau2" / "A1_LG" / f"fold_{fold}"

    def loader(self, rows, scaler, shuffle=False, seed=42):
        generator = torch.Generator().manual_seed(seed)
        dataset = RRWindowDataset(self.p, rows, self.cache, self.store, scaler)
        return DataLoader(
            dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=shuffle,
            collate_fn=collate_rr,
            num_workers=self.cfg["num_workers"],
            generator=generator,
            pin_memory=self.device.type == "cuda",
        )

    def make_scaler(self, window_ids):
        return RRScaler(self.store, window_ids)

    def forward_batch(self, model, batch):
        return model(
            batch["global"],
            batch["local"],
            batch["mask"],
            batch["global_rr"],
            batch["local_rr"],
        )

    def rr_feature_metadata(self):
        return {"global": self.global_rr_names, "local": self.local_rr_names}

    def _model(self):
        return RRHierarchicalHead(
            dim=1280,
            dropout=self.p.cfg["model"]["dropout"],
        ).to(self.device)

    def build_model(self, fold):
        return self._model()

    def _fit_image_scaler(self, model, rows):
        self.base.fit_scaler(model, rows)

    def compute_loss(self, output, batch, weights):
        """Hook loss pilot; default identik dengan eksperimen terdahulu."""
        return loss_terms(output, batch, weights, self.cfg)

    def fold_seed(self, fold):
        """Hook seed agar eksperimen keragaman dapat memakai split yang sama."""
        return int(self.p.cfg["seed"] + int(fold))

    def make_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.cfg["lr_frozen"],
                                 weight_decay=self.cfg["weight_decay"])

    def clip_gradients(self, model):
        return torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

    @torch.inference_mode()
    def predict(self, model, rows, scaler):
        model.eval()
        y, probabilities, global_probabilities, beat_y, beat_probabilities = [], [], [], [], []
        for batch in self.loader(rows, scaler):
            batch = to_device(batch, self.device)
            output = self.forward_batch(model, batch)
            y.extend(batch["y"].cpu().tolist())
            probabilities.append(output["logp"].exp().cpu().numpy())
            global_probabilities.append(output["global_logits"].softmax(-1).cpu().numpy())
            valid = batch["mask"] & (batch["beat"] >= 0) & (batch["y"][:, None] < 3)
            beat_y.extend(batch["beat"][valid].cpu().tolist())
            beat_probabilities.append(output["beat_logits"][valid].softmax(-1).cpu().numpy())
        return {
            "y": np.asarray(y),
            "prob": np.concatenate(probabilities),
            "global_prob": np.concatenate(global_probabilities),
            "beat_y": np.asarray(beat_y),
            "beat_prob": np.concatenate(beat_probabilities) if beat_probabilities else np.empty((0, 3)),
        }

    def fit_fold(self, fold=0, max_epochs=None, stop_after_epoch=None):
        fold = int(fold)
        if fold not in range(5):
            raise ValueError("Fold harus berada pada rentang 0-4.")
        folder = self.run_dir(fold)
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["protocol_hash"] != self.p.hash or digest(folder / "best.pt") != result["checkpoint_sha256"]:
                raise ValueError("Integritas hasil A1 tidak sesuai.")
            return result

        train = self.dev[self.dev.fold != fold].reset_index(drop=True)
        valid = self.dev[self.dev.fold == fold].reset_index(drop=True)
        if set(train.patient_id) & set(valid.patient_id):
            raise ValueError("Pasien subtrain dan validation tumpang tindih.")
        seed = self.fold_seed(fold)
        seed_all(seed, self.cfg["deterministic"])
        rr_scaler = self.make_scaler(train.window_id)
        model = self.build_model(fold)
        self._fit_image_scaler(model, train)
        weights, counts = self.base.weights(train, self.class_weight_tau)
        optimizer = self.make_optimizer(model)
        epochs = int(max_epochs or self.cfg["max_epochs"])
        limit = epochs if stop_after_epoch is None else int(stop_after_epoch)
        if not 1 <= limit <= epochs:
            raise ValueError('stop_after_epoch must be between 1 and max_epochs.')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        amp = torch.amp.GradScaler("cuda", enabled=self.cfg["amp"] and self.device.type == "cuda")
        last_path = folder / "last.pt"
        history, best, best_epoch, start, elapsed = [], -1.0, 0, 0, 0.0
        if last_path.exists():
            if not self.resume:
                raise FileExistsError("RESUME=False membutuhkan direktori A1 baru.")
            checkpoint = load_checkpoint(last_path)
            if checkpoint["protocol_hash"] != self.p.hash or checkpoint["max_epochs"] != epochs:
                raise ValueError("Checkpoint A1 tidak sesuai protokol atau jumlah epoch.")
            if checkpoint["rr_scaler"] != rr_scaler.state_dict():
                raise ValueError("Scaler RR checkpoint berbeda dari subtrain.")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            amp.load_state_dict(checkpoint["amp"])
            history, best = checkpoint["history"], checkpoint["best"]
            best_epoch, start, elapsed = checkpoint["best_epoch"], checkpoint["epoch"], checkpoint["elapsed"]
            torch.set_rng_state(checkpoint["rng_torch"])
            np.random.set_state(checkpoint["rng_numpy"])
            random.setstate(checkpoint["rng_python"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])

        if start == 0 and not history and self.preserve_initial_baseline:
            initial_predictions = self.predict(model, valid, rr_scaler)
            initial_score = calculate(initial_predictions["y"], initial_predictions["prob"])[0]
            best, best_epoch = float(initial_score["macro_f1"]), 0
            history = [{
                "epoch": 0,
                "train_loss": 0.0,
                "val_macro_f1": best,
                "elapsed_s": 0.0,
            }]
            initial_payload = {
                "protocol_hash": self.p.hash,
                "ablation_code_sha256": digest(self.implementation_file),
                "variant": self.variant,
                "arch": self.arch,
                "scenario": "S3",
                "tau": float(self.class_weight_tau),
                "fold": fold,
                "epoch": 0,
                "max_epochs": epochs,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "amp": amp.state_dict(),
                "rr_scaler": rr_scaler.state_dict(),
                "rr_features": self.rr_feature_metadata(),
                "history": history,
                "best": best,
                "best_epoch": best_epoch,
                "elapsed": elapsed,
                "counts": {key: value.tolist() for key, value in counts.items()},
                "weights": {key: value.cpu().tolist() for key, value in weights.items()},
                "train_ids": train.window_id.tolist(),
                "rng_torch": torch.get_rng_state(),
                "rng_numpy": np.random.get_state(),
                "rng_python": random.getstate(),
                "rng_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            }
            atomic_torch(folder / "best.pt", initial_payload)
            atomic_torch(last_path, initial_payload)
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if start > limit:
            raise ValueError('Checkpoint already exceeds requested screening epoch.')
        for epoch in range(start, limit):
            if epoch - best_epoch >= self.cfg["patience"]:
                break
            begin = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loader = self.loader(train, rr_scaler, shuffle=True, seed=seed + epoch)
            losses, n_windows = 0.0, 0
            component_sums = {}
            accumulation = self.cfg["accumulation_steps"]
            for step, batch in enumerate(tqdm(
                loader,
                desc=f"{self.variant} fold {fold} epoch {epoch + 1}",
                leave=False,
            )):
                batch = to_device(batch, self.device)
                group_start = (step // accumulation) * accumulation
                group_windows = min(accumulation * loader.batch_size, len(train) - group_start * loader.batch_size)
                with torch.autocast(self.device.type, enabled=amp.is_enabled()):
                    output = self.forward_batch(model, batch)
                    loss, components = self.compute_loss(output, batch, weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Loss A1 tidak finite.")
                amp.scale(loss * len(batch["y"]) / group_windows).backward()
                losses += float(loss.detach()) * len(batch["y"])
                for key, component in components.items():
                    component_sums[key] = component_sums.get(key, 0.0) + float(component.detach()) * len(batch['y'])
                n_windows += len(batch["y"])
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    amp.unscale_(optimizer)
                    self.clip_gradients(model)
                    amp.step(optimizer)
                    amp.update()
                    optimizer.zero_grad(set_to_none=True)
            predictions = self.predict(model, valid, rr_scaler)
            score = calculate(predictions["y"], predictions["prob"])[0]
            scheduler.step()
            elapsed += time.perf_counter() - begin
            value = score["macro_f1"]
            improved = value > best
            if improved:
                best, best_epoch = value, epoch + 1
            history.append({
                "epoch": epoch + 1,
                "train_loss": losses / n_windows,
                "val_macro_f1": value,
                "elapsed_s": elapsed,
                **{f'train_{key}_loss': total / n_windows for key, total in component_sums.items()},
            })
            payload = {
                "protocol_hash": self.p.hash,
                "ablation_code_sha256": digest(self.implementation_file),
                "variant": self.variant,
                "arch": self.arch,
                "scenario": "S3",
                "tau": float(self.class_weight_tau),
                "fold": fold,
                "epoch": epoch + 1,
                "max_epochs": epochs,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "amp": amp.state_dict(),
                "rr_scaler": rr_scaler.state_dict(),
                "rr_features": self.rr_feature_metadata(),
                "history": history,
                "best": best,
                "best_epoch": best_epoch,
                "elapsed": elapsed,
                "counts": {key: value.tolist() for key, value in counts.items()},
                "weights": {key: value.cpu().tolist() for key, value in weights.items()},
                "train_ids": train.window_id.tolist(),
                "rng_torch": torch.get_rng_state(),
                "rng_numpy": np.random.get_state(),
                "rng_python": random.getstate(),
                "rng_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            }
            if improved:
                atomic_torch(folder / "best.pt", payload)
            atomic_torch(last_path, payload)
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            print(f"{self.variant} | fold {fold} | epoch {epoch + 1} | loss {losses / n_windows:.4f} | macro-F1 {value:.4f}")

        checkpoint = load_checkpoint(folder / "best.pt")
        model.load_state_dict(checkpoint["model"])
        predictions = self.predict(model, valid, rr_scaler)
        completed_epoch = int(history[-1]['epoch'])
        paused = (limit < epochs and completed_epoch - best_epoch < self.cfg['patience'])
        evaluation_folder = folder / f'screen_epoch_{completed_epoch}' if paused else folder
        score = export_metrics(
            evaluation_folder,
            predictions["y"],
            predictions["prob"],
            valid,
            predictions["global_prob"],
            predictions["beat_y"],
            predictions["beat_prob"],
        )
        if paused:
            report = {'status': 'paused', 'variant': self.variant, 'fold': fold,
                      'completed_epoch': completed_epoch, 'best_epoch': best_epoch,
                      'scheduler_horizon': epochs, 'protocol_hash': self.p.hash,
                      'checkpoint_sha256': digest(folder / 'best.pt'), **score}
            atomic_json(evaluation_folder / 'screening.json', report)
            return report
        head_parameters = sum(value.numel() for value in model.parameters())
        if hasattr(self, "encoder_parameter_count"):
            deployed_parameters = int(self.encoder_parameter_count() + head_parameters)
        else:
            a0_reference = json.loads((self.base.run_dir("S3", fold, 2) / "result.json").read_text())
            deployed_parameters = int(a0_reference["total_parameters"] - a0_reference["trainable_parameters"] + head_parameters)
        result = {
            "variant": self.variant,
            "arch": self.arch,
            "scenario": "S3",
            "fold": fold,
            "tau": float(self.class_weight_tau),
            "best_epoch": best_epoch,
            "protocol_hash": self.p.hash,
            "checkpoint_sha256": digest(folder / "best.pt"),
            "training_s": elapsed,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if self.device.type == "cuda" else 0,
            "total_parameters": deployed_parameters,
            "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
            "ablation_code_sha256": digest(self.implementation_file),
            **score,
            "environment": environment(),
            "device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
        }
        atomic_json(result_path, result)
        return result

    def compare_fold(self, fold=0):
        a0_folder = self.base.run_dir("S3", int(fold), 2)
        a1_folder = self.run_dir(int(fold))
        if not (a0_folder / "result.json").exists() or not (a1_folder / "result.json").exists():
            raise FileNotFoundError("Hasil A0 atau A1 fold belum lengkap.")
        a0 = json.loads((a0_folder / "result.json").read_text())
        a1 = json.loads((a1_folder / "result.json").read_text())
        p0 = pd.read_csv(a0_folder / "per_class.csv").set_index("class")
        p1 = pd.read_csv(a1_folder / "per_class.csv").set_index("class")
        rows = [{
            "metric": "macro_f1",
            "A0": a0["macro_f1"],
            "A1": a1["macro_f1"],
            "delta": a1["macro_f1"] - a0["macro_f1"],
        }]
        for class_name in ["N", "S", "V", "AF", "VT"]:
            for metric in ["precision", "sensitivity", "f1"]:
                before, after = float(p0.loc[class_name, metric]), float(p1.loc[class_name, metric])
                rows.append({
                    "metric": f"{metric}_{class_name}",
                    "A0": before,
                    "A1": after,
                    "delta": after - before,
                })
        return pd.DataFrame(rows)
