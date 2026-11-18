"""Uji tambahan yang membutuhkan training CNN (dijalankan di Colab L4, satu proses per uji).

    python colab_tests.py --uji G0L4 --arch efficientnet_b0 --base <folder hasil> --sumber-fitur <feature_store resmi>
    python colab_tests.py --uji U2   --arch ... --base ... --sumber-fitur ...
    python colab_tests.py --uji U4   --arch ... --skenario S1 --seed 1042 --base ... --sumber-fitur ...
    python colab_tests.py --uji U3   --arch ... --base ... --sumber-fitur ...
    python colab_tests.py --uji U5   --arch ... --base ... --sumber-fitur ... [--cache-lokal <folder disk lokal>]
    python colab_tests.py --uji U4   --arch ... --skenario S3 --seed 1042 ...   (setelah U4 S2 seed yang sama)
    python colab_tests.py --uji U6   --arch efficientnet_b0 --fraksi 0.25 --ulang 0 ...
    --asap: uji asap (1 epoch per fold, subset kecil; hanya memeriksa kode, hasil tidak dipakai)

Protokol uji dikunci sebelum run. Semua uji memakai kode, data, cache embedding,
dan hash metode resmi V4 (diperiksa); kode training V4 tidak diubah. Setiap uji punya folder keluaran sendiri
(<base>/<id uji>), dibuat langsung lewat ProtocolV4(output_root=...), sehingga tidak pernah menulis ke outputs resmi
(v4_runner di bundel 10 Nov belum mengenal V4_OUTPUT_ROOT, jadi R.protocol() sengaja tidak dipakai). Feature store resmi
akun yang sama disalin ke folder uji (bukan dibangun ulang) agar masukan tabular identik bit per bit dengan run resmi.

  G0L4  model final S2 resmi dilatih ulang di L4 (seed dan E* resmi): apakah lingkungan Colab sekarang sama dengan run resmi.
  U2    leave-source-out: model final S2 tanpa 12 pasien development yang memiliki AF di luar AFDB atau VT di luar VFDB
        (seluruh window pasien itu dikeluarkan), E* resmi, seed resmi. Dinilai secara lokal pada 12 pasien itu.
  U3    image-only: head citra-saja v4_tl_murni (tanpa fitur tabular, GRU RR, dan expert tabular) dengan protokol S2
        (beku, tau 2), sejajar dengan baseline tabular-only B4. CV 5 fold, E* dengan aturan resmi, model final.
  U4    seed: model final S1/S2 resmi dengan seed final lain (1042, 2042), E* resmi. S3 seed k = fine-tuning penuh resep
        S3 resmi (E* S3 resmi) dari model final U4 S2 seed k, sehingga rantai S2 -> S3 diulang utuh per seed.
  U5    backbone acak beku: encoder diinisialisasi acak (seed 42) lalu statistik BatchNorm dikalibrasi dari 512 window
        development tanpa label (forward saja, tanpa gradien); tanpa kalibrasi, aktivasi encoder acak dengan statistik
        BatchNorm bawaan meluruh ke ~1e-14 sehingga jalur citra mati (kontrol tidak adil). Cache embedding baru, lalu
        protokol S2 penuh (CV 5 fold, E* aturan resmi, model final). Encoder acak disimpan untuk inferensi hold-out.
  U6    learning curve pasien: model final S2 EfficientNet-B0 pada 25/50/75 % pasien development (berstrata per kelas
        penentu pasien, 3 ulangan bersarang), E* dan seed resmi.
Hold-out tidak dibaca di sini; semua evaluasi dilakukan secara lokal dari checkpoint yang disimpan.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = {
    "data_protocol_hash": "9db60710fd60433663bab15e1bbe3f61ba0b97cdf0ef59f0fc6531d5d510735d",
    "method_hash": "d7f73001bb35e08774efdbfb9bdb487fadf4dff0a7b3c5bcdc6df48966c71569",
}
# E* resmi (outputs_colab_r15/selection/<arch>/<sc>_final_epoch.json) dan sha256 checkpoint final resmi
RESMI = {
    ("efficientnet_b0", "S1"): (5, "1aef8391"), ("efficientnet_b0", "S2"): (6, "8b5f9fa0"),
    ("mobilenet_v2", "S1"): (18, "d7da1682"), ("mobilenet_v2", "S2"): (20, "728a5bde"),
    ("shufflenet_v2_x1_0", "S1"): (6, "76c15692"), ("shufflenet_v2_x1_0", "S2"): (9, "1bc5073f"),
}
RESMI_S3 = {"efficientnet_b0": (4, "2981ecba"), "mobilenet_v2": (2, "d30bd855"), "shufflenet_v2_x1_0": (1, "4f7050eb")}
S3_HASH = "44315f01cd185a8c4d39e12aeffe3b7184b1727d2d1deff147fb4c3d9a2386af"
SEEDS_U4 = (1042, 2042)
U5_KALIBRASI, U5_SEED = 512, 42
U6_FRAKSI, U6_ULANG = (0.25, 0.5, 0.75), (0, 1, 2)
KELAS = ["N", "S", "V", "AF", "VT"]
U2_PASIEN, U2_WINDOW = 12, 3134
U3_NAMA = "TL_murni_S2"


def sha_berkas(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def uji_id(uji, arch, skenario="S2", seed=None, asap=False, fraksi=None, ulang=None):
    if uji == "U4":
        nama = f"U4_{arch}_{skenario}_seed{seed}"
    elif uji == "U6":
        nama = f"U6_{arch}_p{round(fraksi * 100)}_r{ulang}"
    else:
        nama = f"{uji}_{arch}"
    return nama + ("_asap" if asap else "")


def salin_feature_store(sumber: Path, root: Path) -> int:
    """Salin feature store resmi (read-only) ke folder uji; berhenti bila isi yang sudah ada berbeda."""
    sumber, tujuan, n = Path(sumber), root / "feature_store", 0
    berkas = sorted(f for f in sumber.rglob("*") if f.is_file())
    if not any(f.name == "complete.json" for f in berkas):
        raise FileNotFoundError(f"Feature store resmi tidak ditemukan atau belum lengkap di {sumber}")
    for f in berkas:
        t = tujuan / f.relative_to(sumber)
        if t.exists():
            if t.stat().st_size != f.stat().st_size or sha_berkas(t) != sha_berkas(f):
                raise FileExistsError(f"{t} sudah ada dengan isi berbeda dari feature store resmi; tidak ditimpa.")
            continue
        t.parent.mkdir(parents=True, exist_ok=True); tmp = t.with_name(t.name + ".tmp")
        shutil.copy2(f, tmp); os.replace(tmp, t); n += 1
    return n


def sidik_feature_store(root: Path) -> dict:
    folder = root / "feature_store"
    return {f.relative_to(folder).as_posix(): sha_berkas(f) for f in sorted(folder.rglob("*")) if f.is_file()}


def tulis_receipt(root: Path, isi: dict):
    path, kunci = root / "uji.json", ("uji", "arch", "skenario", "seed", "E*", "asap", "kode_sha256")
    if path.exists():
        lama = json.loads(path.read_text())
        beda = {k: (lama.get(k), isi.get(k)) for k in kunci if lama.get(k) != isi.get(k)}
        if beda:
            raise ValueError(f"Folder {root} berisi uji lain atau kode berubah: {beda}; gunakan folder baru.")
        return
    path.write_text(json.dumps(isi, indent=2))


def pasien_u2(dev):
    af = dev[(dev.label == "AF") & (dev.database != "afdb")].patient_id
    vt = dev[(dev.label == "VT") & (dev.database != "vfdb")].patient_id
    pasien = sorted(set(af) | set(vt))
    n = int(dev.patient_id.isin(pasien).sum())
    if len(pasien) != U2_PASIEN or n != U2_WINDOW:
        raise ValueError(f"Pasien U2 berbeda dari rencana: {len(pasien)} pasien, {n} window")
    return pasien, n


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text); s.flush()
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def dicatat(root: Path, nama: str):
    folder = root / "logs"; folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{nama}.log").open("a", encoding="utf-8") as h:
        h.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        asli = sys.stdout; sys.stdout = _Tee(asli, h)
        try:
            yield
        finally:
            sys.stdout = asli


def protokol(root: Path):
    """ProtocolV4 ke folder uji, dengan inisialisasi yang sama seperti v4_runner.protocol() (tanpa S3)."""
    import v4_holdout
    import v4_pipeline as v4
    import v4_runner as R
    from v4_data import ProtocolV4
    v4_holdout.install()                         # validasi run selesai dengan cakupan label_valid (Revisi 9)
    if hasattr(R, "_hash_metode_lama_lintas_os"):
        R._hash_metode_lama_lintas_os()          # hanya berefek di Windows (uji asap lokal)
    p = ProtocolV4(output_root=root); v4.initialize(p)
    got = {"data_protocol_hash": p.hash, "method_hash": v4.method_hash()}
    if got != EXPECTED:
        raise ValueError(f"Hash berbeda dari run resmi: {got}")
    if Path(p.out).resolve() != root:
        raise ValueError(f"Folder keluaran {p.out} bukan {root}")
    return p, v4, got


def pasien_u6(dev, fraksi, ulang):
    """Subset pasien berstrata: kelas penentu pasien = kelas miliknya yang paling sedikit pasiennya di development.
    Satu permutasi per ulangan (seed = ulangan), sehingga subset 25 % < 50 % < 75 % bersarang dalam ulangan yang sama."""
    import numpy as np
    n_pasien = dev.groupby("label").patient_id.nunique()
    urut = sorted(KELAS, key=lambda c: (n_pasien.get(c, 0), KELAS.index(c)))
    milik = dev.groupby("patient_id").label.agg(set)
    penentu = milik.map(lambda ks: next(c for c in urut if c in ks))
    rng, pilih = np.random.default_rng(int(ulang)), []
    for c in urut:
        grup = sorted(penentu[penentu == c].index)
        if grup:
            perm = [grup[i] for i in rng.permutation(len(grup))]
            pilih += perm[:max(1, int(np.ceil(fraksi * len(grup))))]
    sub = dev[dev.patient_id.isin(pilih)]
    if set(sub.label) != set(dev.label):
        raise ValueError("Subset U6 kehilangan kelas.")
    return sorted(pilih)


def encoder_acak(exp, root: Path):
    """Encoder acak (seed U5_SEED) dengan statistik BatchNorm dikalibrasi dari window development tanpa label."""
    import torch
    from torch import nn
    from ecgtl.hierarchical.model import backbone
    from ecgtl.hierarchical.protocol import atomic_torch, seed_all
    from ecgtl.hierarchical.training import to_device
    path = root / "encoder_acak.pt"
    if path.exists():
        encoder, dim = backbone(exp.arch, False)
        encoder.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        return encoder.to(exp.device).eval(), dim, json.loads((root / "encoder_acak.json").read_text())
    seed_all(U5_SEED, True)
    encoder, dim = backbone(exp.arch, False)
    encoder = encoder.to(exp.device).eval()
    for m in encoder.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats(); m.momentum = None; m.train()   # rerata kumulatif, hanya BatchNorm yang train
    rows = exp.base.dev.sample(n=min(U5_KALIBRASI, len(exp.base.dev)), random_state=U5_SEED).reset_index(drop=True)
    chunk = int(exp.p.cfg["model"]["feature_chunk"])
    with torch.no_grad():
        for batch in exp.base.loader(rows, batch_size=exp.cfg["feature_batch_size"]):
            batch = to_device(batch, exp.device)
            for images in (batch["global"], batch["local"][batch["mask"]]):
                for part in images.split(chunk):
                    encoder(part)
    encoder.eval()
    atomic_torch(path, encoder.state_dict())
    info = {"seed": U5_SEED, "kalibrasi_window": int(len(rows)),
            "kalibrasi_window_ids_sha256": hashlib.sha256("\n".join(rows.window_id.astype(str)).encode()).hexdigest(),
            "label_dibaca": False, "holdout_dibaca": False, "encoder_sha256": sha_berkas(path)}
    (root / "encoder_acak.json").write_text(json.dumps(info, indent=2))
    return encoder, dim, info


def pasang_cache_acak(exp, encoder, dim, cache: Path, arsip: Path):
    """Cache embedding U5 di disk lokal (bukan Drive) + arsip tar di folder uji; model cache memakai encoder acak."""
    import copy
    import tarfile
    from ecgtl.hierarchical.model import HierarchicalNet
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "complete.json").exists() and arsip.exists():
        with tarfile.open(arsip) as tar:
            tar.extractall(cache, filter="data")
    exp.base.cache = exp.cache = cache
    exp.base.model = lambda scenario, pretrained=True: HierarchicalNet(
        exp.arch, scenario, exp.p.cfg, encoder=copy.deepcopy(encoder), feature_dim=dim).to(exp.device)
    exp.base.cache_features()
    if not arsip.exists():
        tmp = arsip.with_name(arsip.name + ".tmp")
        with tarfile.open(tmp, "w") as tar:
            for f in sorted(cache.iterdir()):
                tar.add(f, arcname=f.name)
        os.replace(tmp, arsip)


def salin_sumber_s3(base: Path, arch, seed, asap, root: Path, v4):
    """Salin model final U4 S2 seed yang sama (terverifikasi) sebagai sumber S3 di folder uji S3."""
    sumber = base / uji_id("U4", arch, "S2", seed, asap)
    if not (sumber / "selesai.json").exists():
        raise FileNotFoundError(f"U4 S2 seed {seed} belum selesai ({sumber}); jalankan dulu.")
    selesai = json.loads((sumber / "selesai.json").read_text())
    src, dst = sumber / "final_models" / arch / "S2" / "final", root / "final_models" / arch / "S2" / "final"
    if v4.digest(src / "best.pt") != selesai["checkpoint_sha256"]:
        raise ValueError("Checkpoint U4 S2 sumber tidak cocok dengan selesai.json.")
    if dst.exists():
        if v4.digest(dst / "best.pt") != selesai["checkpoint_sha256"]:
            raise FileExistsError(f"{dst} berisi checkpoint lain.")
    else:
        shutil.copytree(src, dst)
    return selesai["checkpoint_sha256"]


def validator_asap(v4, dev):
    """Hanya untuk --asap: validator resmi menuntut train_ids = seluruh development; di sini cakupan yang dicek
    adalah subset asap. Pemeriksaan identitas, hash, dan checksum validator resmi tetap dijalankan lebih dulu."""
    from ecgtl.hierarchical.training import load_checkpoint
    asli = v4.validate_completed_run

    def validasi(protocol, arch, scenario, fold, full=False):
        try:
            return asli(protocol, arch, scenario, fold, full)
        except ValueError as error:
            if "Cakupan train IDs" not in str(error):
                raise
        folder = protocol.out / ("final_models" if full else "cv") / arch / scenario / ("final" if full else f"fold_{int(fold)}")
        harap = set((dev if full else dev[dev.fold != int(fold)]).window_id.astype(str))
        if set(map(str, load_checkpoint(folder / "best.pt")["train_ids"])) != harap:
            raise ValueError(f"Cakupan train IDs asap berbeda: {folder}")
        return json.loads((folder / "result.json").read_text())
    v4.validate_completed_run = validasi


def eksperimen(uji, p, v4, arch, skenario):
    if skenario == "S3":
        import v4_s3
        if v4_s3.s3_method_hash() != S3_HASH:
            raise ValueError("Hash metode S3 berbeda dari run resmi.")
        return v4_s3.V4S3Experiment(p, arch)
    if uji != "U3":
        return v4.V4Experiment(p, arch, skenario)
    import v4_tl_murni as T

    class TLMurniS2(T.TLMurniExperiment):
        """Head citra-saja v4_tl_murni dengan protokol S2 (tau 2, minimal epoch S2); nama run terpisah."""
        def __init__(self, protocol, arch):
            v4.V4Experiment.__init__(self, protocol, arch, "S2")
            self.name = U3_NAMA

    return TLMurniS2(p, arch)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--uji", required=True, choices=["G0L4", "U2", "U3", "U4", "U5", "U6"])
    ap.add_argument("--arch", required=True, choices=["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"])
    ap.add_argument("--skenario", default="S2", choices=["S1", "S2", "S3"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--sumber-fitur", required=True, type=Path)
    ap.add_argument("--fraksi", type=float, default=None)
    ap.add_argument("--ulang", type=int, default=None)
    ap.add_argument("--cache-lokal", type=Path, default=None, help="U5: folder cache embedding acak di disk lokal")
    ap.add_argument("--asap", action="store_true")
    ap.add_argument("--wajib-gpu", default="L4", help="'' untuk uji asap tanpa syarat GPU")
    a = ap.parse_args(argv)
    if a.uji != "U4" and a.skenario != "S2":
        raise ValueError("Hanya U4 yang memakai S1/S3.")
    if (a.uji == "U6") != (a.fraksi is not None and a.ulang is not None):
        raise ValueError("--fraksi dan --ulang wajib (dan hanya) untuk U6.")
    if a.uji == "U6" and (a.fraksi not in U6_FRAKSI or a.ulang not in U6_ULANG or a.arch != "efficientnet_b0"):
        raise ValueError(f"U6 terkunci: EfficientNet-B0, fraksi {U6_FRAKSI}, ulangan {U6_ULANG}.")
    if a.uji == "U4" and a.seed not in SEEDS_U4:
        raise ValueError(f"Seed U4 terkunci: {SEEDS_U4}")
    if a.uji != "U4" and a.seed is not None:
        raise ValueError("--seed hanya untuk U4.")
    root = (a.base / uji_id(a.uji, a.arch, a.skenario, a.seed, a.asap, a.fraksi, a.ulang)).resolve()
    if (root / "selesai.json").exists():
        print("Sudah selesai:", (root / "selesai.json").read_text()); return json.loads((root / "selesai.json").read_text())
    root.mkdir(parents=True, exist_ok=True)
    disalin = salin_feature_store(a.sumber_fitur, root)
    sidik_awal = sidik_feature_store(root)

    import torch
    if a.wajib_gpu:
        nama = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "tanpa GPU"
        if a.wajib_gpu not in nama:
            raise RuntimeError(f"GPU wajib {a.wajib_gpu}; didapat {nama}.")
    p, v4, got = protokol(root)
    estar, sha_resmi = RESMI_S3[a.arch] if a.skenario == "S3" else RESMI[(a.arch, a.skenario)]
    sumber_s3 = salin_sumber_s3(a.base.resolve(), a.arch, a.seed, a.asap, root, v4) if a.skenario == "S3" else None
    if a.seed is not None:
        p.cfg["seed"] = a.seed - 100   # fit memakai cfg seed + 100 untuk model final -> seed final = a.seed
    seed_final = int(p.cfg["seed"]) + 100
    exp = eksperimen(a.uji, p, v4, a.arch, a.skenario)
    kode = {n: sha_berkas(HERE / n) for n in ["colab_tests.py", "v4_tl_murni.py"] if (HERE / n).exists()}
    cv_penuh = a.uji in ("U3", "U5")
    isi = {"uji": a.uji, "arch": a.arch, "skenario": U3_NAMA if a.uji == "U3" else a.skenario, "seed": a.seed,
           "seed_final": seed_final, "E*": None if cv_penuh else estar, "fraksi": a.fraksi, "ulang": a.ulang, "asap": a.asap, "hash": got,
           "kode_sha256": kode, "feature_store_disalin_dari": str(a.sumber_fitur), "feature_store_berkas_baru": disalin,
           "dibuat": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "torch": torch.__version__, "holdout_dibaca": False}
    if sumber_s3:
        isi["sumber_S2_sha256"] = sumber_s3
    if a.uji == "U6":
        pasien = pasien_u6(exp.dev, a.fraksi, a.ulang)
        exp.dev = exp.dev[exp.dev.patient_id.isin(pasien)].reset_index(drop=True)
        isi.update({"pasien_latih": pasien, "n_pasien_latih": len(pasien), "window_latih": int(len(exp.dev))})
    if a.uji == "U2":
        pasien, n = pasien_u2(exp.dev)
        exp.dev = exp.dev[~exp.dev.patient_id.isin(pasien)].reset_index(drop=True)
        isi.update({"pasien_dikeluarkan": pasien, "window_dikeluarkan": n, "window_latih": int(len(exp.dev))})
    if a.asap:   # subset kecil per kelas dan per fold agar uji asap cepat; hasil tidak dipakai
        exp.dev = exp.dev.groupby(["fold", "label_idx"]).head(12).reset_index(drop=True)
        isi["window_latih_asap"] = int(len(exp.dev))
        exp.base.dev = exp.base.dev[exp.base.dev.window_id.isin(exp.dev.window_id)].reset_index(drop=True)
        validator_asap(v4, exp.dev)
    if a.uji == "U5":
        encoder, dim, info = encoder_acak(exp, root)
        isi["encoder_acak"] = {k: info[k] for k in ("seed", "kalibrasi_window", "kalibrasi_window_ids_sha256")}
    tulis_receipt(root, isi)
    print(json.dumps(isi, indent=1), flush=True)
    if a.uji == "U5":
        pasang_cache_acak(exp, encoder, dim, (a.cache_lokal or root / "cache_embedding_acak").resolve(),
                          root / "cache_embedding_acak.tar")
    elif a.skenario != "S3":
        exp.base.cache_features()   # verifikasi sha256 cache embedding beku (dipulihkan dari arsip resmi)

    nama_log = f"{a.arch}_{isi['skenario']}"
    with dicatat(root, nama_log):
        if cv_penuh:
            for k in range(int(v4.METHOD["cv"]["folds"])):
                print(time.strftime("%H:%M"), f"{a.uji} {a.arch} {exp.name} fold {k}", flush=True)
                exp.fit(k, False, 1 if a.asap else None)
            e_final = int(exp.select_final_epoch()["selected_epoch"])   # aturan E* resmi
            print(time.strftime("%H:%M"), f"{a.uji} {a.arch} {exp.name} final, E* = {e_final}", flush=True)
        else:
            e_final = 1 if a.asap else estar
            print(time.strftime("%H:%M"), f"{a.uji} {a.arch} {a.skenario} final, {e_final} epoch, seed final {seed_final}",
                  flush=True)
        folder = exp.run_dir(-1, True)
        if (folder / "result.json").exists():   # proses sebelumnya selesai training tetapi terputus sebelum selesai.json
            res = json.loads((folder / "result.json").read_text())
            if v4.digest(folder / "best.pt") != res["checkpoint_sha256"] or res["epochs_run"] != e_final:
                raise ValueError(f"result.json di {folder} tidak cocok dengan best.pt atau E*.")
        else:
            res = exp.fit(-1, True, e_final)
    if sidik_feature_store(root) != sidik_awal:
        raise ValueError("Feature store berubah selama run; masukan tabular tidak lagi identik dengan run resmi.")
    selesai = {"E*": e_final, "checkpoint_sha256": res["checkpoint_sha256"], "training_s": res["training_s"],
               "device": res["device"], "epochs_run": res["epochs_run"], "seed_final": seed_final}
    if a.uji == "U5":
        selesai["encoder_acak_sha256"] = sha_berkas(root / "encoder_acak.pt")
    if a.uji == "G0L4" and not a.asap:
        selesai["sha_sama_dengan_resmi"] = res["checkpoint_sha256"].startswith(sha_resmi)
    (root / "selesai.json").write_text(json.dumps(selesai, indent=2))
    print(json.dumps(selesai, indent=1), flush=True)
    return selesai


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
