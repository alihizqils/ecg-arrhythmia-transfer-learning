"""Inferensi hold-out (template bebas label) untuk checkpoint uji Colab G0L4/U2/U3/U4/U5/U6.

Tidak ada training. Untuk setiap uji:
  1. checkpoint final dicek sha256 = selesai.json;
  2. eksperimen resmi dibangun (protokol, feature store, template, cache resmi; V4_OUTPUT_ROOT=outputs_colab_r15),
     dengan baris development sama seperti saat training uji (U2 tanpa 12 pasien, U6 subset pasien);
  3. scaler dihitung ulang dari baris training itu dan harus sama dengan rr_scaler di checkpoint;
  4. model dibangun dari checkpoint (U5: encoder acak dari encoder_acak.pt), lalu audit anchor: prediksi encoder
     mentah pada window anchor development harus mereproduksi anchor yang disimpan saat training;
  5. hold-out diprediksi dengan template rekaman bebas label (sama dengan semua hasil hold-out paper).
     U2 juga diprediksi pada window 12 pasien yang dikeluarkan (template bebas label).
Kontrol reproduksi: model final resmi S1/S2/S3 ketiga backbone lewat jalur yang sama harus sama dengan
main_results/holdout_labelfree/<arch>/<sc>/probabilities_labelfree.npy.

Keluaran baru: outputs_colab_r15/additional_tests/uji_colab/<id uji>/{prediksi.npz, complete.json}
Pakai: python evaluate_tests.py <folder hasil uji tambahan> [filter id ...]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
os.environ.setdefault("V4_OUTPUT_ROOT", str(HERE / "outputs_colab_r15"))
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "colab"))
import torch  # noqa: E402

import v4_holdout  # noqa: E402
import v4_pipeline as v4  # noqa: E402
import v4_runner as R  # noqa: E402
from ecgtl.hierarchical.training import load_checkpoint  # noqa: E402
from holdout_labelfree import load_new, patch  # noqa: E402
import colab_tests as U  # noqa: E402
from check_downloads import urutan  # noqa: E402

OUT = HERE / "outputs_colab_r15"
DEST = OUT / "additional_tests" / "uji_colab"
ARCHS = ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Lingkungan:
    """Eksperimen resmi per (arch, jenis), dibangun sekali; template asli disimpan untuk scaler."""

    def __init__(self):
        v4_holdout.install(); self.p = R.protocol(); self.cache = {}; self.new = load_new()
        hm = pd.read_csv(OUT / "holdout" / "efficientnet_b0" / "S2" / "manifest.csv", dtype={"window_id": str})
        self.hold_ids = list(hm.window_id)

    def exp(self, arch, jenis):
        key = (arch, jenis)
        if key not in self.cache:
            for k in [k for k in self.cache if k[0] != arch]:
                del self.cache[k]
            if jenis == "S3":
                import v4_s3
                e = v4_s3.V4S3Experiment(self.p, arch)
            elif jenis == "U3":
                import v4_tl_murni as T

                class TLMurniS2(T.TLMurniExperiment):
                    def __init__(self, protocol, arch):
                        v4.V4Experiment.__init__(self, protocol, arch, "S2")
                        self.name = U.U3_NAMA
                e = TLMurniS2(self.p, arch)
            else:
                e = v4.V4Experiment(self.p, arch, jenis)
            e._dev0 = e.dev.copy(); e._rec0, e._beat0 = e.record_template, e.beat_record_template
            self.cache[key] = e
        e = self.cache[key]
        e.dev = e._dev0.copy(); e.record_template, e.beat_record_template = e._rec0, e._beat0
        return e

    def prediksi(self, arch, jenis, ckpt_path, dev_filter=None, encoder_path=None, ekstra_ids=None):
        e = self.exp(arch, jenis)
        if dev_filter is not None:
            e.dev = dev_filter(e.dev).reset_index(drop=True)
        ck = load_checkpoint(ckpt_path)
        if set(map(str, ck["train_ids"])) != set(e.dev.window_id.astype(str)):
            raise ValueError("train_ids checkpoint berbeda dari baris development yang direkonstruksi.")
        scaler = v4.V4Scaler(e.store, e.dev.window_id, e._rec0, e._beat0)
        if scaler.state_dict() != ck["rr_scaler"]:
            raise ValueError("Scaler hasil rekonstruksi berbeda dari rr_scaler checkpoint.")
        sc = "S3" if jenis == "S3" else "S2" if jenis == "U3" else jenis
        model = v4_holdout.build_final_model(e, sc, ck)
        if encoder_path is not None:
            from ecgtl.hierarchical.model import backbone
            enc, _ = backbone(arch, False)
            enc.load_state_dict(torch.load(encoder_path, map_location="cpu"), strict=True)
            for prm in enc.parameters():
                prm.requires_grad_(False)
            model.encoder = enc.to(e.device).eval()
        try:
            audit, _ = e._audit_raw_anchor(model, scaler, Path(ckpt_path).parent)
        except ValueError as err:
            # Hanya untuk selisih numerik kecil (kelas anchor identik): dicek ulang lewat jalur cache embedding
            # resmi (masukan yang sama dengan saat training) pada seluruh hold-out, lalu dicatat.
            a = json.loads(str(err).split(": ", 1)[1].replace("'", '"').replace("True", "true").replace("False", "false"))
            if not (a["labels_identical"] and a["predicted_classes_identical"] and a["maximum_probability_difference"] < 5e-3
                    and encoder_path is None):
                raise
            audit = {**a, "toleransi_dilonggarkan": True}
            # cache embedding hanya memuat window development: bandingkan jalur cache vs encoder mentah pada
            # 2.000 window development acak (template asli, sama dengan saat training)
            dr = e.dev.sample(n=min(2000, len(e.dev)), random_state=0).reset_index(drop=True)
            pm = e.predict(model, dr, scaler, False)
            pc = e.predict(model.head, dr, scaler, True)
            audit["dev2000_cache_vs_mentah"] = {
                "argmax_sama": float((pc["prob"].argmax(1) == pm["prob"].argmax(1)).mean()),
                "selisih_prob_maks": float(np.abs(pc["prob"] - pm["prob"]).max())}
            print("  cek jalur cache (2.000 window dev):", audit["dev2000_cache_vs_mentah"], flush=True)
            if audit["dev2000_cache_vs_mentah"]["argmax_sama"] < 0.99:
                raise ValueError(f"Jalur cache dan encoder mentah tidak sepakat: {audit}")
        ids = self.hold_ids + list(ekstra_ids or [])
        rec, beat = patch(e, ids, self.new)
        e.record_template, e.beat_record_template = rec, beat
        scaler.record_template, scaler.beat_record_template = rec, beat
        rows = e.all_rows.set_index(e.all_rows.window_id.astype(str)).loc[ids].reset_index(drop=True)
        pr = e.predict(model, rows, scaler, False)
        del model; torch.cuda.empty_cache()
        return {"window_id": np.array(ids), "prob": pr["prob"].astype(np.float32), "y": pr["y"],
                "pasien": (rows.database + "/" + rows.record.astype(str)).to_numpy().astype(str),
                "n_holdout": len(self.hold_ids)}, audit


def simpan(dest, hasil, info):
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / "prediksi.npz", **{k: v for k, v in hasil.items() if k != "n_holdout"},
             n_holdout=hasil["n_holdout"])
    info = json.loads(json.dumps(info, default=float))
    (dest / "complete.json").write_text(json.dumps(info, indent=2))


def kontrol(L, filt):
    for arch in ARCHS:
        for sc in ("S1", "S2", "S3"):
            uid = f"RESMI_{arch}_{sc}"; dest = DEST / uid
            if (dest / "complete.json").exists() or (filt and not any(f in uid for f in filt)):
                continue
            t = time.time()
            hasil, audit = L.prediksi(arch, sc, OUT / "final_models" / arch / sc / "final" / "best.pt")
            ref = np.load(OUT / "main_results" / "holdout_labelfree" / arch / sc / "probabilities_labelfree.npy")
            m = pd.read_csv(OUT / "holdout" / arch / sc / "manifest.csv", dtype={"window_id": str})
            assert list(m.window_id) == L.hold_ids
            rep = {"selisih_maks_prob": float(np.abs(hasil["prob"] - ref).max()),
                   "argmax_sama": float((hasil["prob"].argmax(1) == ref.argmax(1)).mean())}
            print(uid, rep, f"{time.time() - t:.0f} dtk", flush=True)
            if rep["argmax_sama"] < 1 or rep["selisih_maks_prob"] > 1e-4:
                raise ValueError(f"Jalur evaluasi tidak mereproduksi hold-out bebas label resmi {uid}: {rep}")
            simpan(dest, hasil, {"uji": "RESMI", "arch": arch, "skenario": sc, "audit_anchor": audit,
                                 "reproduksi_labelfree_resmi": rep})


def main(base: Path, filt):
    DEST.mkdir(parents=True, exist_ok=True)
    L = Lingkungan()
    kontrol(L, filt)
    for arch in ARCHS:
        for uji, sc, seed, fr, ul in urutan(arch):
            uid = U.uji_id(uji, arch, sc, seed, False, fr, ul); dest = DEST / uid
            if (dest / "complete.json").exists() or (filt and not any(f in uid for f in filt)):
                continue
            r = base / arch / "TA_R15_uji_tambahan_hasil" / uid
            s = json.loads((r / "selesai.json").read_text()); u = json.loads((r / "uji.json").read_text())
            sk = U.U3_NAMA if uji == "U3" else sc
            ck = r / "final_models" / arch / sk / "final" / "best.pt"
            if sha(ck) != s["checkpoint_sha256"]:
                raise ValueError(f"{uid}: sha checkpoint != selesai.json")
            jenis = "U3" if uji == "U3" else sc
            dev_filter, ekstra, enc = None, None, None
            if uji == "U2":
                keluar = set(u["pasien_dikeluarkan"])
                dev_filter = lambda d, k=keluar: d[~d.patient_id.isin(k)]
                e0 = L.exp(arch, jenis)
                ekstra = list(e0._dev0[e0._dev0.patient_id.isin(keluar)].window_id.astype(str))
                if len(keluar) != U.U2_PASIEN or len(ekstra) != U.U2_WINDOW:
                    raise ValueError(f"U2: {len(keluar)} pasien, {len(ekstra)} window")
            if uji == "U6":
                latih = set(u["pasien_latih"])
                dev_filter = lambda d, k=latih: d[d.patient_id.isin(k)]
            if uji == "U5":
                enc = r / "encoder_acak.pt"
                if sha(enc) != s["encoder_acak_sha256"]:
                    raise ValueError(f"{uid}: encoder_acak.pt != selesai.json")
            t = time.time()
            hasil, audit = L.prediksi(arch, jenis, ck, dev_filter, enc, ekstra)
            print(uid, "anchor", {k: audit[k] for k in ("windows", "predicted_classes_identical",
                                                       "maximum_probability_difference")},
                  f"{time.time() - t:.0f} dtk", flush=True)
            simpan(dest, hasil, {"uji": uji, "arch": arch, "skenario": sk, "seed_final": s["seed_final"], "E*": s["E*"],
                                 "checkpoint_sha256": s["checkpoint_sha256"], "audit_anchor": audit,
                                 "n_window_ekstra_u2": len(ekstra or []),
                                 "kode_sha256": sha(Path(__file__)), "holdout_dipakai_untuk_seleksi": False})


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
