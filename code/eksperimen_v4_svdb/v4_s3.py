"""S3 V4: full fine-tuning EfficientNet-B0 dari checkpoint S2 fold/final yang sama.

Modul terpisah agar hash metode S1/S2 (`v4_pipeline.method_hash`) tidak berubah.
Lock S3 sendiri (`outputs/v4_s3_method_lock.json`) mencakup hash S1/S2, yaml S3,
dan berkas ini. Loop training, loss, EMA, dan pelaporan diwarisi dari V4Experiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.checkpoint import checkpoint

import v4_pipeline as v4
from ecgtl.hierarchical.model import backbone
from ecgtl.hierarchical.protocol import atomic_json, digest
from ecgtl.hierarchical.training import load_checkpoint

FOLDER = Path(__file__).resolve().parent
S3_PATH = FOLDER / "metode_v4_s3.yaml"
S3 = yaml.safe_load(S3_PATH.read_text())


def s3_method_hash() -> str:
    return hashlib.sha256(v4.method_hash().encode() + S3_PATH.read_bytes() + Path(__file__).read_bytes()).hexdigest()


def initialize_s3(protocol) -> dict:
    payload = {"method_hash": s3_method_hash(), "source_method_hash": v4.method_hash(),
               "data_protocol_hash": protocol.hash, "holdout_used_for_selection": False}
    path = protocol.out / "v4_s3_method_lock.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Metode S3 V4 berubah setelah output dibuat; gunakan direktori output baru.")
    atomic_json(path, payload)
    return payload


class FullFineTuneNet(nn.Module):
    """Seluruh encoder trainable; running statistics BatchNorm beku (kontrak eksplorasi)."""

    def __init__(self, arch: str, cfg: dict, head: nn.Module):
        super().__init__()
        self.encoder, self.dim = backbone(arch, pretrained=True)
        self.head = head
        self.chunk = int(cfg["model"]["feature_chunk"])
        self.checkpointing = bool(cfg["training"]["gradient_checkpointing"])
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def encode(self, images):
        parts = []
        for part in images.split(self.chunk):
            parts.append(checkpoint(self.encoder, part, use_reentrant=False)
                         if self.training and self.checkpointing else self.encoder(part))
        return torch.cat(parts)

    def forward(self, global_x, local_x, mask, *features):
        global_features = self.encode(global_x)
        flat = self.encode(local_x[mask])
        local_features = flat.new_zeros((*mask.shape, self.dim))
        local_features[mask] = flat
        return self.head(global_features, local_features, mask, *features)


class V4S3Experiment(v4.V4Experiment):
    def __init__(self, protocol, arch: str, device=None, resume=True):
        # Data, fitur, tau 2, dan head identik dengan S2; hanya model/optimizer berbeda.
        super().__init__(protocol, arch, "S2", device=device, resume=resume)
        initialize_s3(protocol)
        self.name = "S3"
        self.tau = float(S3["tau"])
        self.min_epochs = int(S3["min_epochs"])

    def source_dir(self, fold, full=False):
        return self.p.out / ("final_models" if full else "cv") / self.arch / "S2" / ("final" if full else f"fold_{int(fold)}")

    def _validate_source(self, fold, full):
        folder = self.source_dir(fold, full)
        result = json.loads((folder / "result.json").read_text())
        source = load_checkpoint(folder / "best.pt")
        if result.get("method_hash") != v4.method_hash() or source.get("method_hash") != v4.method_hash():
            raise ValueError("Method hash checkpoint S2 sumber berbeda.")
        if digest(folder / "best.pt") != result["checkpoint_sha256"] or result.get("protocol_hash") != self.p.hash:
            raise ValueError("Checksum/protokol checkpoint S2 sumber berbeda.")
        expected = set(self.dev.window_id.astype(str) if full else self.dev[self.dev.fold != int(fold)].window_id.astype(str))
        if set(map(str, source["train_ids"])) != expected:
            raise ValueError("Train IDs S2 sumber berbeda dari subtrain S3.")
        return folder, source

    def _model_scaler(self, train, fold, full, last_exists):
        folder, source = self._validate_source(fold, full)
        scaler = v4.V4Scaler(self.store, train.window_id, self.record_template, self.beat_record_template)
        if source["rr_scaler"] != scaler.state_dict():
            raise ValueError("Scaler S2 sumber berbeda dari subtrain S3.")
        head = self._head(); head.load_state_dict(source["model"], strict=True)
        model = FullFineTuneNet(self.arch, self.p.cfg, head).to(self.device)
        audit_path = self.run_dir(fold, full) / "initialization_audit.json"
        if not full and not last_exists:
            valid = self.dev[self.dev.fold == int(fold)].reset_index(drop=True)
            prediction = self.predict(model, valid, scaler, False)
            expected = np.load(folder / "probabilities.npy")
            audit = {"source": str(folder), "source_checkpoint_sha256": digest(folder / "best.pt"),
                     "windows": len(valid),
                     "labels_identical": bool(np.array_equal(prediction["y"], np.load(folder / "labels.npy"))),
                     "argmax_agreement": float((prediction["prob"].argmax(1) == expected.argmax(1)).mean()),
                     "maximum_probability_difference": float(np.max(np.abs(prediction["prob"] - expected))),
                     "tolerance": float(S3["initialization_probability_tolerance"])}
            atomic_json(audit_path, audit)
            if not audit["labels_identical"] or audit["maximum_probability_difference"] > audit["tolerance"]:
                raise ValueError(f"Inisialisasi S3 tidak mereproduksi S2: {audit}")
        return model, scaler, False

    def _optimizer(self, model):
        return torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": float(S3["encoder_learning_rate"])},
            {"params": model.head.parameters(), "lr": float(S3["head_learning_rate"])}],
            weight_decay=float(S3["weight_decay"]))

    def _payload(self, *args):
        payload = super()._payload(*args)
        payload["s3_method_hash"] = s3_method_hash(); payload["source_scenario"] = "S2"
        return payload
