"""Baseline transfer learning murni (citra saja) dengan protokol MA1.

Tujuan (10 Nov 2026): pembanding yang adil bagi MA1, untuk mengukur kontribusi fitur tabular dan
Beat-Level Specialist terhadap transfer learning biasa.

SAMA dengan CNN MA1 (diwarisi tanpa perubahan dari v4_pipeline.V4Experiment):
  data, label (Revisi 9), fold pasien, sampler, citra window 10 detik + potongan citra per beat, backbone
  EfficientNet-B0 ImageNet beku (embedding cache yang sama), struktur hierarkis (cabang global
  beat/AF/VT + cabang beat N/S/V dengan attention), loss utama + loss bantu global dan per beat, optimizer, LR,
  scheduler, EMA, early stopping, minimal 20 epoch, maksimal 60, penyimpanan probabilitas per epoch, aturan E*.
BERBEDA (satu-satunya perbedaan):
  - tanpa SEMUA fitur tabular: RR, fisiologi, gelombang P/T, template rekaman, lead kedua, ritme lokal;
  - tanpa GRU urutan RR (bila diisi nol pun, GRU masih membawa jumlah beat = informasi laju jantung, sehingga
    cabang ini dihapus, bukan dinolkan);
  - tanpa expert tabular window (dan tanpa loss expert);
  - tanpa class weighting (tau = 1, sama dengan S1);
  - tanpa specialist dan fusi (tahap di luar CNN).
Head citra-saja: lapisan global_head, beat_head (ResidualFusion seperti NonlinearHead) dan attention sama bentuknya
dengan head MA1, tetapi masukannya hanya embedding citra (1.280 dimensi), bukan citra + 64 dimensi tabular.

Kode MA1 tidak diubah: modul terpisah, hash metode V4 tetap. Output: <output_root>/cv/<arch>/TL_murni/fold_k.
Kunci implementasi: berkas ini + hash metode V4 dicatat di folder run (tl_murni_lock.json).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

import v4_pipeline as v4
from ecgtl.hierarchical.model import loss_terms
from tools.a6_pb_nonlinear_fusion import ResidualFusion

NAME = "TL_murni"


def tl_hash() -> str:
    return hashlib.sha256(v4.method_hash().encode() + Path(__file__).read_bytes()).hexdigest()


class ImageOnlyHierarchicalHead(nn.Module):
    """Head hierarkis MA1 tanpa jalur tabular: hanya embedding citra global dan per beat."""

    def __init__(self, dim=1280, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.frozen = True
        self.global_mean = nn.Buffer(torch.zeros(dim))
        self.global_std = nn.Buffer(torch.ones(dim))
        self.local_mean = nn.Buffer(torch.zeros(dim))
        self.local_std = nn.Buffer(torch.ones(dim))
        self.dropout = nn.Dropout(dropout)
        self.global_head = nn.Linear(dim, 3)
        self.beat_head = ResidualFusion(dim)
        self.attention = nn.Linear(dim, 3)

    def forward(self, global_x, local_x, mask, *tabular_ignored):
        if not mask.any(1).all():
            raise ValueError("Setiap window harus memiliki setidaknya satu crop.")
        global_x = (global_x - self.global_mean) / self.global_std
        local_x = (local_x - self.local_mean) / self.local_std
        global_logits = self.global_head(self.dropout(global_x))
        beat_logits = self.beat_head(self.dropout(local_x))
        attention = self.attention(local_x).masked_fill(~mask[..., None], -torch.inf).softmax(1)
        local_logits = (attention * beat_logits).sum(1)
        pg, pl = global_logits.log_softmax(-1), local_logits.log_softmax(-1)
        logp = torch.cat([pg[:, :1] + pl, pg[:, 1:]], dim=-1)
        return {"logp": logp, "global_logits": global_logits, "beat_logits": beat_logits,
                "attention": attention, "local_logits": local_logits}


class TLMurniExperiment(v4.V4Experiment):
    def __init__(self, protocol, arch="efficientnet_b0", device=None, resume=True):
        super().__init__(protocol, arch, "S1", device=device, resume=resume)   # tau 1, min 20 epoch, beku
        self.name = NAME

    def _head(self):
        return ImageOnlyHierarchicalHead(v4.base.DIMENSIONS[self.arch], self.p.cfg["model"]["dropout"]).to(self.device)

    def loss(self, output, batch, weights):
        return loss_terms(output, batch, weights, self.cfg)

    def fit(self, fold, full=False, epochs=None):
        folder = self.run_dir(fold, full); folder.mkdir(parents=True, exist_ok=True)
        lock = folder / "tl_murni_lock.json"; payload = {"tl_hash": tl_hash(), "v4_method_hash": v4.method_hash()}
        if lock.exists() and json.loads(lock.read_text()) != payload:
            raise ValueError("Kode TL murni berubah sejak run dimulai; gunakan folder baru.")
        lock.write_text(json.dumps(payload, indent=2))
        return super().fit(fold, full, epochs)
