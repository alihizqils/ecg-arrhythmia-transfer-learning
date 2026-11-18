"""XAI pasca-pelatihan V4 (di luar hash metode training; tidak mengubah bobot model).

Tiga lapis analisis pada model final yang sudah terkunci:

1. SHAP citra - expected gradients (algoritme `shap.GradientExplainer`) terhadap citra
   global dan crop lokal, fitur tabular ditahan pada nilai sampel itu sendiri.
   Diimplementasikan langsung agar crop lokal yang jumlahnya bervariasi dan memori GPU
   6 GB dapat ditangani; kesetaraannya dengan pustaka SHAP diuji di
   tests/test_v4_notebook_support.py. Ringkasan pendukung: porsi massa |SHAP| pada segmen
   P, QRS, dan ST-T (perkiraan) dibanding porsi luas segmen tersebut pada citra.
2. SHAP fitur tabular berkelompok - expected gradients terhadap input tabular dengan
   embedding citra sampel ditahan; dijumlahkan per kelompok fitur klinis.
3. Kontribusi modalitas - knock-out embedding citra (rerata data latih) dan/atau fitur
   tabular (0 = rerata data latih) pada seluruh holdout; Shapley dua pemain.

Target penjelasan: logit kelas sebenarnya, yaitu log-probabilitas akhir dikurangi rerata
log-probabilitas kelima kelas (logit terpusat; identik untuk semua parameterisasi softmax).
Log-probabilitas sendiri jenuh (gradien ~0) bila model sangat yakin.
Sampel dibekukan (`freeze_samples`) sebelum model mana pun dijelaskan dan dipilih tanpa
melihat prediksi model. Hasil bersifat kualitatif; bukan validasi klinis.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import v4_holdout
import v4_pipeline as v4
import v4_runner as R
from ecgtl.hierarchical.data import read_manifest
from ecgtl.hierarchical.protocol import CLASSES, atomic_json, digest
from ecgtl.hierarchical.training import load_checkpoint, to_device
from tools.a5_physiology import GLOBAL_PHYSIOLOGY_NAMES, LOCAL_PHYSIOLOGY_NAMES
from tools.a6_wave_delineation import GLOBAL_WAVE_NAMES, LOCAL_WAVE_NAMES
from v4_record_template import BEAT_RECORD_TEMPLATE_NAMES, DOMINANT_RR_NAMES, RECORD_TEMPLATE_NAMES
from v4_second_lead import SECOND_LEAD_NAMES
from v4_local_rhythm import BEAT_NAMES as LOCAL_RHYTHM_BEAT_NAMES, WINDOW_NAMES as LOCAL_RHYTHM_WINDOW_NAMES

SEED = 20261105
GALLERY_PER_CLASS = 2
SUMMARY_PER_CLASS = 10
BACKGROUND_PER_CLASS = 4
NSAMPLES_IMAGE = 200
NSAMPLES_TABULAR = 200
FS = 250
WINDOW = 2500
CROP = 312
GLOBAL_WIDTH, LOCAL_WIDTH = 448, 224
FEATURES = ["global_rr", "local_rr", "rr_sequence", "physiology_global", "physiology_local",
            "wave_global", "wave_local", "record_template", "beat_record_template"]
SEGMENTS = ("P", "QRS", "ST-T")
GROUPS = ["Ritme RR window", "RR lokal & sekuens", "Fisiologi", "Gelombang P", "Kompleks QRS",
          "Template morfologi rekaman", "Lead kedua",
          "Ritme lokal & konteks"]  # fitur ritme lokal E1-E4 (Revisi 14); ditambahkan 10 Nov 2026


def out_dir() -> Path:
    return R.protocol().out / "xai"


# --------------------------------------------------------------------------- sampel

def _round_robin(rows: pd.DataFrame, count: int, rng) -> list[str]:
    """Pasien diacak, lalu satu window per pasien per putaran (keragaman pasien maksimum)."""
    patients = sorted(map(str, rows.patient_id.unique())); rng.shuffle(patients)
    pools = {p: list(rng.permutation(rows[rows.patient_id.astype(str) == p].window_id.astype(str).to_numpy())) for p in patients}
    chosen = []
    while len(chosen) < count and any(pools.values()):
        for p in patients:
            if pools[p] and len(chosen) < count:
                chosen.append(pools[p].pop(0))
    return chosen


def freeze_samples() -> dict:
    """Pilih dan bekukan sampel XAI (holdout) dan background (development). Idempoten."""
    path = out_dir() / "sampel_xai.json"
    p = R.protocol()
    if path.exists():
        payload = json.loads(path.read_text())
        body = {k: v for k, v in payload.items() if k != "sha256"}
        if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != payload["sha256"]:
            raise ValueError("sampel_xai.json berubah setelah dibekukan.")
        if payload["protocol_hash"] != p.hash:
            raise ValueError("Protokol data berbeda dari saat sampel XAI dibekukan.")
        return payload
    manifest = read_manifest(p.prepared / "global_manifest.csv")
    holdout = v4.label_valid(manifest[manifest.split == "holdout"].reset_index(drop=True))
    development = v4.label_valid(manifest[manifest.split == "development"].reset_index(drop=True))
    rng = np.random.default_rng(SEED); explain, background, gallery = {}, {}, {}
    for label in CLASSES:
        explain[label] = _round_robin(holdout[holdout.label == label], SUMMARY_PER_CLASS, rng)
        gallery[label] = explain[label][:GALLERY_PER_CLASS]
        background[label] = _round_robin(development[development.label == label], BACKGROUND_PER_CLASS, rng)
    body = {"seed": SEED, "protocol_hash": p.hash, "selection": "round_robin_patient_tanpa_prediksi_model",
            "explain": explain, "gallery": gallery, "background": background}
    payload = {**body, "sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}
    atomic_json(path, payload)
    return payload


# --------------------------------------------------------------------------- model

def load_final(arch: str, scenario: str):
    """Eksperimen, model raw-image final (eval), dan scaler final (fit pada seluruh development)."""
    R.protocol()
    v4_holdout.validate_completed_run_label_valid(R.protocol(), arch, scenario, -1, True)
    exp = R.experiment(arch, scenario)
    checkpoint = load_checkpoint(exp.run_dir(-1, True) / "best.pt")
    scaler = v4.V4Scaler(exp.store, exp.dev.window_id, exp.record_template, exp.beat_record_template)
    if scaler.state_dict() != checkpoint["rr_scaler"]:
        raise ValueError("Scaler final berbeda dari checkpoint.")
    model = v4_holdout.build_final_model(exp, scenario, checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return exp, model, scaler, checkpoint


def batch_for(exp, scaler, window_ids) -> dict:
    rows = exp.all_rows.set_index(exp.all_rows.window_id.astype(str)).loc[list(map(str, window_ids))].reset_index(drop=True)
    dataset = exp.loader(rows, scaler, False).dataset
    items = [dataset[i] for i in range(len(rows))]
    return to_device(v4.collate_v4(items), exp.device), rows


def _cycle(tensor: torch.Tensor, length: int) -> torch.Tensor:
    """Ulangi crop/beat sepanjang dimensi 1 agar background berjumlah crop sama dengan sampel."""
    index = torch.arange(length, device=tensor.device) % tensor.shape[1]
    return tensor[:, index]


def _single(batch, i):
    count = int(batch["mask"][i].sum())
    single = {"global": batch["global"][i:i + 1], "local": batch["local"][i:i + 1, :count],
              "mask": batch["mask"][i:i + 1, :count], "y": batch["y"][i:i + 1]}
    for name in FEATURES:
        value = batch[name][i:i + 1]
        single[name] = value[:, :count] if value.dim() == 3 else value
    return single


def _background(bg, length):
    """Background per window dengan jumlah crop disamakan ke sampel (crop valid saja yang diulang)."""
    rows = []
    for k in range(len(bg["y"])):
        one = _single(bg, k)
        rows.append({key: (_cycle(v, length) if key in ("local", "mask") or (key in FEATURES and v.dim() == 3) else v)
                     for key, v in one.items()})
    return {key: torch.cat([r[key] for r in rows]) for key in rows[0]}


def _logit(logp):
    return logp - logp.mean(-1, keepdim=True)


def _forward(model, g, l, mask, feats):
    """Logit terpusat lima kelas."""
    return _logit(model(g, l, mask, *[feats[name] for name in FEATURES])["logp"])


# --------------------------------------------------------------------------- SHAP citra

def expected_gradients_image(model, sample, background, target, nsamples=NSAMPLES_IMAGE, chunk=2, seed=SEED):
    g, l = sample["global"], sample["local"]; L = l.shape[1]
    bg_g, bg_l = background["global"], background["local"]
    mask = sample["mask"]; feats = {name: sample[name] for name in FEATURES}
    rng = np.random.default_rng(seed)
    attr_g = torch.zeros_like(g[0]); attr_l = torch.zeros_like(l[0]); done = 0
    while done < nsamples:
        m = min(chunk, nsamples - done)
        index = torch.as_tensor(rng.integers(len(bg_g), size=m), device=g.device)
        alpha = torch.as_tensor(rng.random(m), dtype=g.dtype, device=g.device)
        xg = (bg_g[index] + alpha.view(-1, *[1] * (g.dim() - 1)) * (g - bg_g[index])).requires_grad_(True)
        xl = (bg_l[index] + alpha.view(-1, *[1] * (l.dim() - 1)) * (l - bg_l[index])).requires_grad_(True)
        expanded = {k: v.expand(m, *v.shape[1:]) for k, v in feats.items()}
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):  # backward GRU dalam mode eval
            out = _forward(model, xg, xl, mask.expand(m, L), expanded)[:, target].sum()
            grad_g, grad_l = torch.autograd.grad(out, [xg, xl])
        attr_g += ((g - bg_g[index]) * grad_g).sum(0).detach()
        attr_l += ((l - bg_l[index]) * grad_l).sum(0).detach()
        done += m
    attr_g /= nsamples; attr_l /= nsamples
    with torch.no_grad():
        fx = float(_forward(model, g, l, mask, feats)[0, target])
        fbg = []
        for k in range(len(bg_g)):
            fbg.append(float(_forward(model, bg_g[k:k + 1], bg_l[k:k + 1], mask, feats)[0, target]))
    total = float(attr_g.sum() + attr_l.sum())
    return attr_g.cpu().numpy(), attr_l.cpu().numpy(), {"f_x": fx, "f_background_mean": float(np.mean(fbg)),
                                                       "attribution_sum": total,
                                                       "additivity_residual": fx - float(np.mean(fbg)) - total}


def beat_segments(exp, window_id) -> list[dict]:
    """Segmen P/QRS/ST-T per beat (sampel relatif window) dari delineasi feature store (nilai mentah)."""
    local = exp._xai_local_manifest if hasattr(exp, "_xai_local_manifest") else None
    if local is None:
        local = read_manifest(exp.p.prepared / "local_manifest.csv")
        exp._xai_local_manifest = local
    crops = local[local.parent_window_id == str(window_id)]
    start = int(exp.all_rows.loc[exp.all_rows.window_id.astype(str) == str(window_id), "start_target_fs"].iloc[0])
    _, wave = exp.store.wave_by_window[str(window_id)]
    if len(wave) != len(crops):
        raise ValueError(f"Jumlah delineasi berbeda dengan crop pada {window_id}.")
    peaks = crops.rpeak_sample.to_numpy(np.float64) - start; beats = []  # relatif awal window
    for j, r in enumerate(peaks):
        on, off = wave[j, LOCAL_WAVE_NAMES.index("qrs_onset_to_r_s")], wave[j, LOCAL_WAVE_NAMES.index("qrs_r_to_offset_s")]
        estimated = not (np.isfinite(on) and np.isfinite(off))
        qrs = (r - (0.05 if estimated else on) * FS, r + (0.05 if estimated else off) * FS)
        segment = {"r": r, "QRS": qrs, "qrs_estimated": estimated, "P": None}
        pr, width = wave[j, LOCAL_WAVE_NAMES.index("pr_interval_s")], wave[j, LOCAL_WAVE_NAMES.index("p_width_s")]
        if wave[j, LOCAL_WAVE_NAMES.index("p_present")] == 1 and np.isfinite(pr) and np.isfinite(width):
            start = qrs[0] - pr * FS; segment["P"] = (start, start + width * FS)
        stop = r + 0.40 * FS
        if j + 1 < len(peaks):
            stop = min(stop, peaks[j + 1] - 0.20 * FS)
        segment["ST-T"] = (qrs[1] + 0.04 * FS, stop) if stop > qrs[1] + 0.04 * FS else None
        beats.append(segment)
    return beats


def _column_mask(beats, name, samples):
    mask = np.zeros(len(samples), dtype=bool)
    for beat in beats:
        if beat[name] is not None:
            mask |= (samples >= beat[name][0]) & (samples < beat[name][1])
    return mask


def segment_mass(attr_global, attr_local, beats) -> dict:
    """Porsi massa |SHAP| per segmen vs porsi luas kolom; rasio > 1 = atribusi terkonsentrasi."""
    result = {}
    centres = (np.arange(GLOBAL_WIDTH) + 0.5) * WINDOW / GLOBAL_WIDTH
    profile = np.abs(attr_global).sum(axis=tuple(range(attr_global.ndim - 1)))
    for name in SEGMENTS:
        mask = _column_mask(beats, name, centres)
        mass = float(profile[mask].sum() / max(profile.sum(), 1e-12)); area = float(mask.mean())
        result[f"global_{name}_massa"] = mass; result[f"global_{name}_luas"] = area
        result[f"global_{name}_rasio"] = mass / area if area > 0 else np.nan
    masses = {name: [0.0, 0.0] for name in SEGMENTS}; total_mass = 0.0
    for j, beat in enumerate(beats):
        centres_local = beat["r"] - CROP // 2 + (np.arange(LOCAL_WIDTH) + 0.5) * CROP / LOCAL_WIDTH
        profile_local = np.abs(attr_local[j]).sum(axis=tuple(range(attr_local[j].ndim - 1)))
        total_mass += profile_local.sum()
        for name in SEGMENTS:
            mask = _column_mask(beats, name, centres_local)
            masses[name][0] += profile_local[mask].sum(); masses[name][1] += mask.mean()
    for name in SEGMENTS:
        mass = float(masses[name][0] / max(total_mass, 1e-12)); area = float(masses[name][1] / max(len(beats), 1))
        result[f"lokal_{name}_massa"] = mass; result[f"lokal_{name}_luas"] = area
        result[f"lokal_{name}_rasio"] = mass / area if area > 0 else np.nan
    return result


# --------------------------------------------------------------------------- SHAP tabular

def feature_groups() -> dict[str, np.ndarray]:
    """Indeks kelompok per tensor fitur (tata letak scaler: [nilai..., penanda ada...])."""
    def doubled(names, group_of):
        groups = [group_of(n) for n in names]
        return np.asarray(groups + groups)
    wave = lambda n: "Kompleks QRS" if n.startswith("qrs") else "Gelombang P"
    record = ([("Template morfologi rekaman")] * len(RECORD_TEMPLATE_NAMES) + ["Ritme RR window"] * len(DOMINANT_RR_NAMES)
              + ["Lead kedua"] * len(SECOND_LEAD_NAMES) + ["Ritme lokal & konteks"] * len(LOCAL_RHYTHM_WINDOW_NAMES))
    return {"global_rr": np.asarray(["Ritme RR window"] * 20), "local_rr": None, "rr_sequence": None,
            "physiology_global": doubled(GLOBAL_PHYSIOLOGY_NAMES, lambda n: "Fisiologi"),
            "physiology_local": doubled(LOCAL_PHYSIOLOGY_NAMES, lambda n: "Fisiologi"),
            "wave_global": doubled(GLOBAL_WAVE_NAMES, wave), "wave_local": doubled(LOCAL_WAVE_NAMES, wave),
            "record_template": np.asarray(record + record),
            "beat_record_template": doubled(BEAT_RECORD_TEMPLATE_NAMES + LOCAL_RHYTHM_BEAT_NAMES,
                                            lambda n: "Ritme lokal & konteks" if n.startswith(("e1c_", "e3c7_"))
                                            else "Template morfologi rekaman")}


def _group_of(name, width):
    groups = feature_groups()[name]
    if groups is None:
        return np.asarray(["RR lokal & sekuens"] * width)
    if len(groups) != width:
        raise ValueError(f"Lebar fitur {name} = {width}, peta kelompok = {len(groups)}.")
    return groups


def expected_gradients_tabular(model, sample, background, target, nsamples=NSAMPLES_TABULAR, chunk=50, seed=SEED):
    """Atribusi per kelompok fitur; embedding citra sampel ditahan (dihitung sekali)."""
    with torch.no_grad():
        gf = model.encode(sample["global"]); lf = model.encode(sample["local"][sample["mask"]])[None]
    mask = sample["mask"]; L = mask.shape[1]; rng = np.random.default_rng(seed)
    x = {n: sample[n] for n in FEATURES}; bg = {n: background[n] for n in FEATURES}
    attr = {n: torch.zeros_like(x[n][0]) for n in FEATURES}; done = 0
    while done < nsamples:
        m = min(chunk, nsamples - done)
        index = torch.as_tensor(rng.integers(len(bg["global_rr"]), size=m), device=mask.device)
        alpha = torch.as_tensor(rng.random(m), dtype=gf.dtype, device=mask.device)
        inputs = {n: (bg[n][index] + alpha.view(-1, *[1] * (x[n].dim() - 1)) * (x[n] - bg[n][index])).requires_grad_(True)
                  for n in FEATURES}
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):  # backward GRU dalam mode eval
            out = model.head(gf.expand(m, -1), lf.expand(m, -1, -1), mask.expand(m, L),
                             *[inputs[n] for n in FEATURES])["logp"]
            out = _logit(out)[:, target].sum()
            grads = torch.autograd.grad(out, [inputs[n] for n in FEATURES])
        for n, grad in zip(FEATURES, grads):
            attr[n] += ((x[n] - bg[n][index]) * grad).sum(0).detach()
        done += m
    result = {group: 0.0 for group in GROUPS}
    for n in FEATURES:
        values = (attr[n] / nsamples).cpu().numpy()
        values = values.sum(0) if values.ndim == 2 else values  # jumlah atas beat
        for group, value in zip(_group_of(n, values.shape[-1]), values):
            result[group] += float(value)
    return result


# --------------------------------------------------------------------------- kontribusi modalitas

@torch.inference_mode()
def modality_knockout(exp, model, scaler, rows) -> dict:
    conditions = {"gabungan": (True, True), "tanpa_citra": (False, True),
                  "tanpa_tabular": (True, False), "tanpa_keduanya": (False, False)}
    head = model.head; probs = {k: [] for k in conditions}; ys = []
    for batch in exp.loader(rows, scaler, False):
        batch = to_device(batch, exp.device); ys.append(batch["y"].cpu().numpy())
        gf = model.encode(batch["global"]); flat = model.encode(batch["local"][batch["mask"]])
        lf = flat.new_zeros((*batch["mask"].shape, flat.shape[-1])); lf[batch["mask"]] = flat
        for key, (image, tabular) in conditions.items():
            g = gf if image else head.global_mean.expand_as(gf)
            l = lf if image else head.local_mean.expand_as(lf)
            feats = [batch[n] if tabular else torch.zeros_like(batch[n]) for n in FEATURES]
            probs[key].append(head(g, l, batch["mask"], *feats)["logp"].exp().cpu().numpy())
    y = np.concatenate(ys); scores = {}
    for key in conditions:
        prob = np.concatenate(probs[key]); per = f1_score(y, prob.argmax(1), labels=range(5), average=None, zero_division=0)
        scores[key] = {"macro": float(per.mean()), **{c: float(v) for c, v in zip(CLASSES, per)}}
        if key == "gabungan":
            scores[key]["_prob"] = prob
    shapley = {}
    for metric in ["macro", *CLASSES]:
        f = {k: scores[k][metric] for k in conditions}
        image = 0.5 * ((f["gabungan"] - f["tanpa_citra"]) + (f["tanpa_tabular"] - f["tanpa_keduanya"]))
        tabular = 0.5 * ((f["gabungan"] - f["tanpa_tabular"]) + (f["tanpa_citra"] - f["tanpa_keduanya"]))
        shapley[metric] = {"citra": image, "tabular": tabular,
                           "persen_citra": 100 * image / (image + tabular) if image + tabular > 0 else None}
    return {"skor": scores, "shapley": shapley, "y": y}


# --------------------------------------------------------------------------- orkestrasi

def explain(arch: str, scenario: str, nsamples_image=NSAMPLES_IMAGE, nsamples_tabular=NSAMPLES_TABULAR,
            image_chunk=2, limit=None, output=None) -> dict:
    """Jalankan ketiga lapis XAI untuk satu model final. Idempoten (receipt per model)."""
    samples = freeze_samples()
    folder = Path(output) if output else out_dir() / arch / scenario  # output: hanya untuk uji asap
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "complete.json"
    exp, model, scaler, checkpoint = load_final(arch, scenario)
    sha = digest(exp.run_dir(-1, True) / "best.pt")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["checkpoint_sha256"] != sha or receipt["samples_sha256"] != samples["sha256"]:
            raise ValueError("Receipt XAI berbeda dari model final/sampel.")
        return receipt
    gallery = {w for ids in samples["gallery"].values() for w in ids}
    ids = [w for label in CLASSES for w in samples["explain"][label]][:limit]
    bg_ids = [w for label in CLASSES for w in samples["background"][label]]
    background, _ = batch_for(exp, scaler, bg_ids)
    image_rows, tabular_rows = [], []
    (folder / "citra").mkdir(exist_ok=True)
    # Lanjut per sampel (10 Nov 2026, bundel Colab R15): tiap sampel yang selesai disimpan atomik bersama sha
    # checkpoint/sampel/nsamples; bila runtime terputus, sampel itu dibaca ulang dan tidak dihitung lagi.
    per_sample = folder / "per_sampel"; per_sample.mkdir(exist_ok=True)
    key = {"checkpoint_sha256": sha, "samples_sha256": samples["sha256"], "nsamples_image": nsamples_image,
           "nsamples_tabular": nsamples_tabular}
    for n, window_id in enumerate(ids):
        done = per_sample / f"{window_id}.json"
        if done.exists():
            saved = json.loads(done.read_text())
            if saved["key"] != key:
                raise ValueError(f"Hasil XAI tersimpan untuk {window_id} berasal dari model/sampel lain; tidak ditimpa.")
            image_rows.append(saved["citra"]); tabular_rows.append(saved["tabular"])
            print(f"{arch} {scenario} XAI {n + 1}/{len(ids)} {window_id} (tersimpan)", flush=True)
            continue
        batch, rows = batch_for(exp, scaler, [window_id]); sample = _single(batch, 0)
        target = int(sample["y"][0]); L = sample["mask"].shape[1]; bg = _background(background, L)
        with torch.no_grad():
            prob = _forward(model, sample["global"], sample["local"], sample["mask"], {k: sample[k] for k in FEATURES}).softmax(-1)[0].cpu().numpy()
        attr_g, attr_l, check = expected_gradients_image(model, sample, bg, target, nsamples_image, image_chunk)
        beats = beat_segments(exp, window_id)
        row = {"window_id": window_id, "label": CLASSES[target], "pred": CLASSES[int(prob.argmax())],
               "p_target": float(prob[target]), "patient_id": rows.patient_id.iloc[0], "database": rows.database.iloc[0],
               "crops": L, **check, **segment_mass(attr_g, attr_l, beats)}
        image_rows.append(row)
        if window_id in gallery:
            np.savez_compressed(folder / "citra" / f"{window_id}.npz", global_image=sample["global"][0].cpu().numpy(),
                                local_images=sample["local"][0].cpu().numpy(), attr_global=attr_g, attr_local=attr_l,
                                probabilities=prob, target=target,
                                segments=json.dumps([{k: (list(v) if isinstance(v, tuple) else v) for k, v in b.items()} for b in beats]))
        tabular_rows.append({"window_id": window_id, "label": CLASSES[target], "pred": row["pred"],
                             **expected_gradients_tabular(model, sample, bg, target, nsamples_tabular)})
        # NaN sah di sini (mis. massa segmen tanpa gelombang), sama seperti di CSV; ditulis atomik tanpa allow_nan=False.
        tmp = done.with_name(done.name + ".tmp")
        tmp.write_text(json.dumps({"key": key, "citra": image_rows[-1], "tabular": tabular_rows[-1]},
                                  default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")
        tmp.replace(done)
        image_rows[-1] = json.loads(done.read_text())["citra"]; tabular_rows[-1] = json.loads(done.read_text())["tabular"]
        print(f"{arch} {scenario} XAI {n + 1}/{len(ids)} {window_id} {CLASSES[target]}->{row['pred']}", flush=True)
    pd.DataFrame(image_rows).to_csv(folder / "shap_citra_ringkasan.csv", index=False)
    pd.DataFrame(tabular_rows).to_csv(folder / "shap_tabular_kelompok.csv", index=False)
    manifest = read_manifest(R.protocol().prepared / "global_manifest.csv")
    holdout = v4.label_valid(manifest[manifest.split == "holdout"].reset_index(drop=True))
    knockout = modality_knockout(exp, model, scaler, holdout if limit is None else holdout.iloc[:limit])
    reference = np.load(R.protocol().out / "holdout" / arch / scenario / "probabilities.npy") if limit is None else None
    combined = knockout["skor"]["gabungan"].pop("_prob")
    knockout["maks_selisih_vs_holdout"] = None if reference is None else float(np.abs(combined - reference).max())
    knockout.pop("y")
    atomic_json(folder / "modalitas_holdout.json", knockout)
    receipt = {"arch": arch, "scenario": scenario, "checkpoint_sha256": sha, "samples_sha256": samples["sha256"],
               "nsamples_image": nsamples_image, "nsamples_tabular": nsamples_tabular, "windows": len(ids),
               "limit": limit, "additivity_residual_abs_median": float(pd.DataFrame(image_rows).additivity_residual.abs().median())}
    if limit is None:
        atomic_json(receipt_path, receipt)
    return receipt
