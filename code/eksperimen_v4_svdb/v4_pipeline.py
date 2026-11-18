"""Mesin eksperimen V4: S1/S2 per skenario CV -> final model -> holdout.

Arsitektur, fitur, dataset, loss, optimizer, sampling, dan scheduler diwarisi dari
`eksperimen_final_hierarkis_10s_lokal/final_pipeline.py` tanpa perubahan. Aturan
baru V4 (lihat `metode_v4.yaml`):

1. Early stopping CV baru aktif setelah `min_epochs`.
2. Probabilitas validasi setiap epoch disimpan; metrik CV utama dihitung pada
   epoch final bersama E*, bukan pada epoch terbaik masing-masing fold.
3. Holdout per skenario, satu kali per final model; metode seluruh skenario
   dikunci ketika holdout pertama dibuka. Evaluasi holdout dan S3 berada di
   modul terpisah agar hash metode S1/S2 tidak berubah ketika keduanya ditambahkan.

Penamaan skenario V4: S1 = backbone beku tanpa class weighting (tau=1), S2 = backbone
beku dengan class weighting (tau=2), S3 = full fine-tuning dari S2. Kode warisan
memakai nama eksplorasi (S2/S3/S4); pemetaan `LEGACY` hanya dipakai untuk memilih
perilaku kelas induk, sedangkan seluruh direktori, checkpoint, dan tabel memakai
nama V4.

Head: NonlinearHead warisan + expert tabular tingkat window (`v4_head.py`, revisi
development 4 Nov 2026) dengan loss klasifikasi expert berbobot `head.lambda_expert`.

S3 full fine-tuning ditambahkan setelah pilot S2 dinilai.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

FOLDER = Path(__file__).resolve().parent
CODE_ROOT = FOLDER.parent
FINAL_FOLDER = CODE_ROOT / "eksperimen_final_hierarkis_10s_lokal"
for path in [CODE_ROOT / "src", CODE_ROOT, FINAL_FOLDER, FOLDER]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import final_pipeline as base  # noqa: E402
from ecgtl.hierarchical.data import read_manifest  # noqa: E402
from ecgtl.hierarchical.metrics import calculate, export_metrics  # noqa: E402
from ecgtl.hierarchical.model import loss_terms  # noqa: E402
from ecgtl.hierarchical.protocol import CLASSES, atomic_json, atomic_torch, digest, environment, seed_all  # noqa: E402
from ecgtl.hierarchical.training import load_checkpoint, to_device  # noqa: E402
from v4_data import ProtocolV4  # noqa: E402
from v4_head import WindowTabularExpertHead  # noqa: E402
from v4_record_template import build_record_template_features, dominant_rr_features  # noqa: E402
from v4_second_lead import build_second_lead_features  # noqa: E402
from v4_local_rhythm import build_local_rhythm_features  # noqa: E402
from tools.a5_physiology import LOCAL_PHYSIOLOGY_NAMES  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402
from tools.rr_ablation_a6_standalone import collate_a6  # noqa: E402


class V4Scaler(base.FinalScaler):
    """FinalScaler + standardisasi fitur template rekaman (dihitung dari data latih saja)."""

    def __init__(self, store, window_ids, record_template, beat_record_template):
        super().__init__(store, window_ids)
        self.record_template, self.beat_record_template = record_template, beat_record_template
        values = np.stack([record_template[str(w)] for w in window_ids]).astype(np.float64)
        self.record_mean, self.record_std = self._fit(values)
        beats = np.concatenate([beat_record_template[str(w)] for w in window_ids]).astype(np.float64)
        self.beat_record_mean, self.beat_record_std = self._fit(beats)

    def transform_record(self, window_id):
        return self._transform(self.record_template[str(window_id)], self.record_mean, self.record_std)

    def transform_beat_record(self, window_id):
        return self._transform(self.beat_record_template[str(window_id)], self.beat_record_mean, self.beat_record_std)

    def state_dict(self):
        state = super().state_dict()
        state["record_template_mean"] = self.record_mean.tolist()
        state["record_template_std"] = self.record_std.tolist()
        state["beat_record_template_mean"] = self.beat_record_mean.tolist()
        state["beat_record_template_std"] = self.beat_record_std.tolist()
        return state


class V4Dataset(base.FinalDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["record_template"] = torch.from_numpy(self.scaler.transform_record(item["id"]))
        beat = self.scaler.transform_beat_record(item["id"])
        if len(beat) != len(item["local"]):
            raise ValueError(f"Jumlah fitur template per beat berbeda pada {item['id']}.")
        item["beat_record_template"] = torch.from_numpy(beat)
        return item


def collate_v4(items):
    batch = collate_a6(items)
    batch["record_template"] = torch.stack([item["record_template"] for item in items])
    beat = torch.zeros((len(items), batch["local"].shape[1], items[0]["beat_record_template"].shape[-1]), dtype=torch.float32)
    for index, item in enumerate(items):
        beat[index, :len(item["local"])] = item["beat_record_template"]
    batch["beat_record_template"] = beat
    return batch

METHOD_PATH = FOLDER / "metode_v4.yaml"
METHOD = yaml.safe_load(METHOD_PATH.read_text())
FROZEN = ("S1", "S2")
LEGACY = {"S1": "S2", "S2": "S3", "S3": "S4"}


def update_ema(ema_model, model, decay: float) -> None:
    """Revisi 12: rata-rata bergerak eksponensial bobot (tensor non-float disalin)."""
    with torch.no_grad():
        source = model.state_dict()
        for name, value in ema_model.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[name].detach(), alpha=1.0 - decay)
            else:
                value.copy_(source[name])


def label_valid(rows: pd.DataFrame) -> pd.DataFrame:
    """Revisi 9 (metode_v4.yaml, label_validity): kelas yang didefinisikan anotasi beat
    (N) hanya diambil dari database beranotasi jenis beat. AFDB/VFDB tetap menyumbang AF/VT."""
    rule = METHOD["label_validity"]
    invalid = rows.label.isin(rule["classes_requiring_beat_annotation"]) & ~rows.database.isin(rule["beat_annotated_databases"])
    return rows[~invalid].reset_index(drop=True)


def method_hash() -> str:
    files = [
        METHOD_PATH, FOLDER / "hierarchical_v4.yaml", Path(__file__).resolve(), FOLDER / "v4_data.py",
        FOLDER / "v4_head.py", FOLDER / "v4_record_template.py", FOLDER / "v4_second_lead.py",
        FOLDER / "v4_local_rhythm.py", FOLDER / "fase_e_ns" / "fitur_ns.py",  # Revisi 14
        FINAL_FOLDER / "final_pipeline.py",
        *(CODE_ROOT / "src/ecgtl/hierarchical" / n for n in ["data.py", "metrics.py", "model.py", "protocol.py", "training.py"]),
        *(CODE_ROOT / "tools" / n for n in [
            "a5_physiology.py", "a6_pb_nonlinear_fusion.py", "a6_wave_delineation.py",
            "rr_ablation_a4_sequence.py", "rr_ablation_a6_standalone.py"]),
    ]
    return hashlib.sha256(b"".join(
        p.relative_to(CODE_ROOT).as_posix().encode() + p.read_bytes() for p in files)).hexdigest()


def initialize(protocol: ProtocolV4) -> dict:
    protocol.verify() if (protocol.out / "protocol_lock.json").exists() else protocol.lock()
    payload = {"method_hash": method_hash(), "data_protocol_hash": protocol.hash,
               "holdout_used_for_selection": False}
    path = protocol.out / "v4_method_lock.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Metode V4 berubah setelah output dibuat; gunakan direktori output baru.")
    atomic_json(path, payload)
    return payload


class V4Experiment(base.FinalExperiment):
    """FinalExperiment dengan minimal epoch CV, probabilitas per epoch, dan namespace V4."""

    def __init__(self, protocol: ProtocolV4, arch: str, scenario: str, device=None, resume=True):
        if scenario not in FROZEN:
            raise ValueError("Pipeline V4 saat ini hanya untuk S1/S2; S3 ditambahkan setelah pilot.")
        initialize(protocol)
        super().__init__(protocol, arch, LEGACY[scenario], device=device, resume=resume)
        self.name = scenario
        self.tau = float(METHOD["tau"][scenario])
        self.min_epochs = int(METHOD["cv"]["min_epochs"][scenario])
        # Feature store/cache encoder tetap memakai seluruh window (self.base.dev);
        # hanya baris training/validasi yang disaring.
        self.dev = label_valid(self.dev)
        self.record_template, self.beat_record_template = build_record_template_features(
            self.p, float(self.p.cfg["data"]["fs"]))
        # Revisi 10: fitur RR antar-beat dominan digabung ke blok fitur template rekaman
        # tingkat window (masukan expert tabular), dihitung dari fitur per beat label-free.
        correlation = LOCAL_PHYSIOLOGY_NAMES.index("qrs_template_correlation")
        for window_id in list(self.record_template):
            _, physiology_local = self.store.physiology_by_window[window_id]
            _, local_rr = self.store.by_window[window_id]
            dominant = dominant_rr_features(physiology_local[:, correlation].astype(np.float64),
                                            self.beat_record_template[window_id][:, 0].astype(np.float64),
                                            local_rr[:, 0].astype(np.float64))
            self.record_template[window_id] = np.concatenate([self.record_template[window_id], dominant])
        # Revisi 11: fitur template rekaman lead EKG kedua (v4_second_lead.py), blok yang sama.
        second_lead = build_second_lead_features(self.p)
        for window_id in list(self.record_template):
            self.record_template[window_id] = np.concatenate([self.record_template[window_id], second_lead[window_id]])
        # Revisi 14: fitur ritme lokal E1 + konteks 7 beat E3 (v4_local_rhythm.py). Agregat window masuk
        # blok template rekaman (expert tabular); fitur per beat masuk blok template per beat (cabang beat).
        rhythm_window, rhythm_beat = build_local_rhythm_features(self.p)
        for window_id in list(self.record_template):
            beat = self.beat_record_template[window_id]
            if len(beat) != len(rhythm_beat[window_id]):
                raise ValueError(f"Jumlah beat fitur ritme lokal berbeda pada {window_id}.")
            self.record_template[window_id] = np.concatenate([self.record_template[window_id], rhythm_window[window_id]])
            self.beat_record_template[window_id] = np.concatenate([beat, rhythm_beat[window_id]], axis=1)

    @staticmethod
    def sampling_weights(rows):
        """Porsi kelas = proporsi asli data; di dalam kelas, setiap pasien berporsi sama.

        Sampler pasien-seimbang warisan memberi porsi sama per pasien, sehingga
        kelas dengan banyak pasien berwindow sedikit (S dari SVDB) ikut terdongkrak
        dan porsi AF/V tergeser. Di sini keseimbangan pasien hanya berlaku di dalam
        kelas; penanganan ketidakseimbangan kelas tetap hanya lewat tau.
        """
        label = rows.label_idx.to_numpy()
        group = rows.groupby(["label_idx", "patient_id"])["window_id"].transform("size").to_numpy()
        patients = rows.label_idx.map(rows.groupby("label_idx").patient_id.nunique()).to_numpy()
        prior = rows.label_idx.map(rows.label_idx.value_counts(normalize=True)).to_numpy()
        weights = prior / (patients * group)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Distribusi sampling tidak valid.")
        return (weights / weights.sum()).astype(np.float64)

    def _model_scaler(self, train, fold, full, last_exists):
        model, _, cached = super()._model_scaler(train, fold, full, last_exists)
        return model, V4Scaler(self.store, train.window_id, self.record_template, self.beat_record_template), cached

    def loader(self, rows, scaler, cached, shuffle=False, seed=42):
        rows = rows.reset_index(drop=True); generator = torch.Generator().manual_seed(seed)
        dataset = V4Dataset(self.p, rows, self.cache if cached else None, self.store, scaler)
        sampler = None
        if shuffle:
            sampler = WeightedRandomSampler(torch.tensor(self.sampling_weights(rows), dtype=torch.double),
                                            num_samples=len(rows), replacement=True, generator=generator)
        return DataLoader(dataset, batch_size=self.cfg["batch_size"], shuffle=False, sampler=sampler,
                          collate_fn=collate_v4, num_workers=self.cfg["num_workers"], generator=generator,
                          pin_memory=self.device.type == "cuda")

    @staticmethod
    def forward(model, batch):
        return model(batch["global"], batch["local"], batch["mask"], batch["global_rr"], batch["local_rr"],
                     batch["rr_sequence"], batch["physiology_global"], batch["physiology_local"],
                     batch["wave_global"], batch["wave_local"], batch["record_template"],
                     batch["beat_record_template"])

    def _head(self):
        return WindowTabularExpertHead(base.DIMENSIONS[self.arch], self.p.cfg["model"]["dropout"],
                                       int(METHOD["head"]["expert_hidden"]),
                                       float(METHOD["head"]["expert_weight"]),
                                       float(METHOD["head"]["vt_gate_power"])).to(self.device)

    def loss(self, output, batch, weights):
        """Loss warisan + loss klasifikasi expert tabular (bobot kelas sama dengan loss akhir)."""
        total, parts = loss_terms(output, batch, weights, self.cfg)
        expert = F.cross_entropy(output["expert_logits"], batch["y"], weight=weights["final"])
        return total + float(METHOD["head"]["lambda_expert"]) * expert, {**parts, "expert": expert.detach()}

    def run_dir(self, fold, full=False):
        return self.p.out / ("final_models" if full else "cv") / self.arch / self.name / ("final" if full else f"fold_{int(fold)}")

    def fit(self, fold, full=False, epochs=None):
        folder = self.run_dir(fold, full); folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / "result.json"
        if result_path.exists():
            return validate_completed_run(self.p, self.arch, self.name, fold, full)
        train = self.dev if full else self.dev[self.dev.fold != int(fold)].reset_index(drop=True)
        valid = None if full else self.dev[self.dev.fold == int(fold)].reset_index(drop=True)
        if valid is not None and set(train.patient_id) & set(valid.patient_id):
            raise ValueError("Overlap pasien.")
        seed = int(self.p.cfg["seed"] + (100 if full else int(fold))); seed_all(seed, self.cfg["deterministic"])
        last_path = folder / "last.pt"
        model, scaler, cached = self._model_scaler(train, fold, full, last_path.exists())
        weights, counts = self.base.weights(train, self.tau)
        self._active_weights = {key: value.detach().cpu().tolist() for key, value in weights.items()}
        optimizer = self._optimizer(model)
        # Revisi 12: EMA bobot. Prediksi validasi, checkpoint "model", dan model final
        # memakai bobot EMA; bobot training mentah disimpan terpisah untuk resume.
        ema_model = copy.deepcopy(model); ema_model.requires_grad_(False); ema_updates = 0
        max_epochs = int(self.cfg["max_epochs"] if epochs is None else epochs)
        if max_epochs < 1:
            raise ValueError("Jumlah epoch minimal 1.")
        # Horizon cosine sama dengan CV agar epoch yang sama memiliki LR yang sama.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(self.cfg["max_epochs"])))
        amp = torch.amp.GradScaler("cuda", enabled=self.cfg["amp"] and self.device.type == "cuda")
        history, best, best_epoch, start, elapsed = [], -1.0, 0, 0, 0.0
        epoch_folder = folder / "epoch_probabilities"
        if last_path.exists():
            ck = load_checkpoint(last_path)
            if ck["method_hash"] != method_hash() or ck["max_epochs"] != max_epochs:
                raise ValueError("Checkpoint resume berbeda.")
            model.load_state_dict(ck["train_model"]); ema_model.load_state_dict(ck["model"]); ema_updates = ck["ema_updates"]
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"]); amp.load_state_dict(ck["amp"])
            history, best, best_epoch, start, elapsed = ck["history"], ck["best"], ck["best_epoch"], ck["epoch"], ck["elapsed"]
            torch.set_rng_state(ck["rng_torch"]); np.random.set_state(ck["rng_numpy"]); random.setstate(ck["rng_python"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all(ck["rng_cuda"])
        if valid is not None:
            epoch_folder.mkdir(exist_ok=True)
            np.save(epoch_folder / "labels.npy", valid.label_idx.to_numpy())
        for epoch in range(start, max_epochs):
            if valid is not None and epoch >= self.min_epochs and epoch - best_epoch >= self.cfg["patience"]:
                break
            begin = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
            loader = self.loader(train, scaler, cached, True, seed + epoch); losses = 0.0; windows = 0
            optimizer_steps = -(-len(loader) // self.cfg["accumulation_steps"])
            ema_decay = 0.5 ** (1.0 / (float(METHOD["ema"]["half_life_epochs"]) * optimizer_steps))
            for step, batch in enumerate(tqdm(loader, desc=f"{self.arch} {self.name} fold {fold} epoch {epoch + 1}", leave=False)):
                batch = to_device(batch, self.device)
                group_start = (step // self.cfg["accumulation_steps"]) * self.cfg["accumulation_steps"]
                group_windows = min(self.cfg["accumulation_steps"] * loader.batch_size, len(train) - group_start * loader.batch_size)
                with torch.autocast(self.device.type, enabled=amp.is_enabled()):
                    output = self.forward(model, batch); loss, _ = self.loss(output, batch, weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Loss tidak finite.")
                amp.scale(loss * len(batch["y"]) / group_windows).backward()
                losses += float(loss.detach()) * len(batch["y"]); windows += len(batch["y"])
                if (step + 1) % self.cfg["accumulation_steps"] == 0 or step + 1 == len(loader):
                    amp.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    amp.step(optimizer); amp.update(); optimizer.zero_grad(set_to_none=True)
                    ema_updates += 1; update_ema(ema_model, model, min(ema_decay, (1.0 + ema_updates) / (10.0 + ema_updates)))
            prediction = None if valid is None else self.predict(ema_model, valid, scaler, cached)
            score = None if prediction is None else calculate(prediction["y"], prediction["prob"])[0]
            if prediction is not None:
                np.save(epoch_folder / f"epoch_{epoch + 1:03d}.npy", prediction["prob"].astype(np.float32))
            scheduler.step(); elapsed += time.perf_counter() - begin
            value = float(epoch + 1) if score is None else score["macro_f1"]; improved = value > best
            if improved:
                best, best_epoch = value, epoch + 1
            history.append({"epoch": epoch + 1, "train_loss": losses / windows,
                            "val_macro_f1": None if score is None else value,
                            "val_f1_S": None if score is None else score["f1_S"], "elapsed_s": elapsed})
            payload = self._payload(ema_model, optimizer, scheduler, amp, scaler, train, fold, full, epoch + 1,
                                    max_epochs, history, best, best_epoch, elapsed, counts)
            payload["train_model"] = model.state_dict(); payload["ema_updates"] = ema_updates
            if improved:
                atomic_torch(folder / "best.pt", payload)
            atomic_torch(last_path, payload); pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            print(f"{self.name} | fold {fold} | epoch {epoch + 1} | macro-F1 {value:.4f}"
                  if valid is not None else f"{self.name} final | epoch {epoch + 1}/{max_epochs}")
        checkpoint = load_checkpoint(folder / "best.pt"); model.load_state_dict(checkpoint["model"]); score = {}
        if valid is not None:
            prediction = self.predict(model, valid, scaler, cached)
            score = export_metrics(folder, prediction["y"], prediction["prob"], valid, prediction["global_prob"],
                                   prediction["beat_y"], prediction["beat_prob"])
        else:
            anchor_rows = self.dev.iloc[:base.INITIALIZATION_ANCHOR_WINDOWS].reset_index(drop=True)
            anchor = self.predict(model, anchor_rows, scaler, cached)
            temporary = folder / "initialization_anchor.tmp"
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, window_ids=np.asarray(anchor_rows.window_id.astype(str).tolist(), dtype=np.str_),
                                    labels=anchor["y"], probabilities=anchor["prob"])
            os.replace(temporary, folder / "initialization_anchor.npz")
        result = {"arch": self.arch, "scenario": self.name, "fold": int(fold), "full": bool(full), "tau": self.tau,
                  "min_epochs": self.min_epochs, "epochs_run": len(history),
                  "best_epoch": int(checkpoint["best_epoch"]), "protocol_hash": self.p.hash, "method_hash": method_hash(),
                  "checkpoint_sha256": digest(folder / "best.pt"), "training_s": elapsed,
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), **score,
                  "environment": environment(),
                  "device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"}
        atomic_json(result_path, result); return result

    def _payload(self, *args):
        payload = super()._payload(*args)
        payload["method_hash"] = method_hash(); payload["tau"] = self.tau; payload["scenario"] = self.name
        return payload

    def select_final_epoch(self):
        """Argmax mean validation Macro-F1 pada epoch yang tersedia di kelima fold."""
        histories, fold_checkpoints = [], {}
        for fold in range(int(METHOD["cv"]["folds"])):
            result = validate_completed_run(self.p, self.arch, self.name, fold, False)
            history = pd.read_csv(self.run_dir(fold) / "history.csv")
            history = history[np.isfinite(history.val_macro_f1)].copy(); history["epoch"] = history.epoch.astype(int)
            histories.append(history.set_index("epoch").val_macro_f1)
            fold_checkpoints[str(fold)] = result["checkpoint_sha256"]
        common = sorted(set.intersection(*[set(h.index) for h in histories]))
        if not common:
            raise ValueError("Tidak ada epoch bersama pada kelima fold.")
        curve = pd.DataFrame([{"epoch": e, "mean_macro_f1": float(np.mean([h.loc[e] for h in histories])),
                               "std_macro_f1": float(np.std([h.loc[e] for h in histories], ddof=1)),
                               **{f"fold_{k}_macro_f1": float(h.loc[e]) for k, h in enumerate(histories)}} for e in common])
        tolerance = float(METHOD["final_epoch"]["tie_tolerance"])
        selected = int(curve[curve.mean_macro_f1 >= curve.mean_macro_f1.max() - tolerance].epoch.min())
        receipt = {"arch": self.arch, "scenario": self.name, "rule": METHOD["final_epoch"]["rule"],
                   "tie_tolerance": tolerance, "selected_epoch": selected,
                   "selected_mean_macro_f1": float(curve.loc[curve.epoch == selected, "mean_macro_f1"].iloc[0]),
                   "common_epochs": common, "fold_checkpoint_sha256": fold_checkpoints,
                   "protocol_hash": self.p.hash, "method_hash": method_hash()}
        folder = self.p.out / "selection" / self.arch; folder.mkdir(parents=True, exist_ok=True)
        curve.to_csv(folder / f"{self.name}_final_epoch_curve.csv", index=False)
        atomic_json(folder / f"{self.name}_final_epoch.json", receipt)
        return receipt

    def final(self):
        self.cv()
        return self.fit(-1, True, int(self.select_final_epoch()["selected_epoch"]))


def validate_completed_run(protocol, arch, scenario, fold, full=False):
    folder = protocol.out / ("final_models" if full else "cv") / arch / scenario / ("final" if full else f"fold_{int(fold)}")
    result_path, checkpoint_path = folder / "result.json", folder / "best.pt"
    if not result_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Run belum lengkap: {folder}")
    result = json.loads(result_path.read_text())
    expected = {"arch": arch, "scenario": scenario, "fold": -1 if full else int(fold), "full": bool(full),
                "protocol_hash": protocol.hash, "method_hash": method_hash()}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Identitas result berbeda pada {folder}: {key}")
    if digest(checkpoint_path) != result.get("checkpoint_sha256"):
        raise ValueError(f"Checksum checkpoint berbeda: {folder}")
    checkpoint = load_checkpoint(checkpoint_path)
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Identitas checkpoint berbeda pada {folder}: {key}")
    manifest = read_manifest(protocol.prepared / "global_manifest.csv")
    development = manifest[manifest.split == "development"]
    expected_ids = set(development.window_id.astype(str) if full else development[development.fold != int(fold)].window_id.astype(str))
    train_ids = list(map(str, checkpoint.get("train_ids", [])))
    if len(train_ids) != len(set(train_ids)) or set(train_ids) != expected_ids:
        raise ValueError(f"Cakupan train IDs berbeda: {folder}")
    if full and not (folder / "initialization_anchor.npz").exists():
        raise FileNotFoundError(f"Initialization anchor tidak tersedia: {folder}")
    if not full:
        for name in ["metrics.json", "per_class.csv", "probabilities.npy", "labels.npy", "confusion.npy", "epoch_probabilities/labels.npy"]:
            if not (folder / name).exists():
                raise FileNotFoundError(folder / name)
    return result


def cv_summary(protocol, arch, scenario) -> dict:
    """Metrik CV pada epoch final bersama E* (utama) dan epoch terbaik per fold (pelengkap)."""
    selection = json.loads((protocol.out / "selection" / arch / f"{scenario}_final_epoch.json").read_text())
    epoch = int(selection["selected_epoch"])
    manifest = read_manifest(protocol.prepared / "global_manifest.csv")
    development = label_valid(manifest[manifest.split == "development"].reset_index(drop=True))
    rows, pooled = [], []
    for fold in range(int(METHOD["cv"]["folds"])):
        result = validate_completed_run(protocol, arch, scenario, fold, False)
        folder = protocol.out / "cv" / arch / scenario / f"fold_{fold}"
        valid = development[development.fold == fold].reset_index(drop=True)
        labels = np.load(folder / "epoch_probabilities/labels.npy")
        probabilities = np.load(folder / "epoch_probabilities" / f"epoch_{epoch:03d}.npy").astype(np.float64)
        probabilities /= probabilities.sum(1, keepdims=True)
        if not np.array_equal(labels, valid.label_idx.to_numpy()):
            raise ValueError(f"Urutan label validasi berbeda: fold {fold}")
        score, per_class, _ = calculate(labels, probabilities)
        rows.append({"fold": fold, "epoch": epoch, "macro_f1_at_E": score["macro_f1"], "f1_S_at_E": score["f1_S"],
                     "accuracy_at_E": score["accuracy"], **{f"f1_{r['class']}_at_E": r["f1"] for r in per_class.to_dict("records")},
                     "best_epoch": result["best_epoch"], "macro_f1_at_fold_best": result["macro_f1"], "epochs_run": result["epochs_run"]})
        pooled.append(valid.assign(pred=probabilities.argmax(1), **{f"p_{c}": probabilities[:, i] for i, c in enumerate(CLASSES)}))
    folds = pd.DataFrame(rows)
    pooled = pd.concat(pooled, ignore_index=True)
    s = pooled[pooled.label == "S"]
    s_patients = s.assign(correct=s.pred == CLASSES.index("S")).groupby(["database", "patient_id"]).correct.agg(["size", "mean"]).reset_index()
    s_source = {}
    for database, rows_db in pooled.groupby("database"):
        if (rows_db.label == "S").any():
            score = calculate(rows_db.label_idx.to_numpy(), rows_db[[f"p_{c}" for c in CLASSES]].to_numpy())[0]
            s_source[database] = {"f1_S": score["f1_S"], "s_windows": int((rows_db.label == "S").sum())}
    summary = {"arch": arch, "scenario": scenario, "selected_epoch": epoch,
               "macro_f1_at_E_mean": float(folds.macro_f1_at_E.mean()), "macro_f1_at_E_std": float(folds.macro_f1_at_E.std(ddof=1)),
               "f1_S_at_E_mean": float(folds.f1_S_at_E.mean()), "f1_S_at_E_std": float(folds.f1_S_at_E.std(ddof=1)),
               "macro_f1_at_fold_best_mean": float(folds.macro_f1_at_fold_best.mean()),
               "selection_bias_gap": float(folds.macro_f1_at_fold_best.mean() - folds.macro_f1_at_E.mean()),
               "f1_S_by_source_pooled": s_source,
               "s_patient_recall_median": float(s_patients["mean"].median()),
               "s_patients_recall_below_0_2": int((s_patients["mean"] < 0.2).sum()), "s_patients": len(s_patients)}
    out = protocol.out / "tables" / arch; out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out / f"{scenario}_cv_folds.csv", index=False)
    s_patients.to_csv(out / f"{scenario}_cv_s_patients.csv", index=False)
    atomic_json(out / f"{scenario}_cv_summary.json", summary)
    return summary


def run_development(protocol, arch, scenario) -> dict:
    """Bagian development satu skenario: 5-fold CV -> pemilihan E* -> final model."""
    experiment = V4Experiment(protocol, arch, scenario)
    experiment.base.cache_features()
    experiment.final()
    return cv_summary(protocol, arch, scenario)
