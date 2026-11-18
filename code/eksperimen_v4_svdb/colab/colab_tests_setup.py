"""Persiapan runtime Colab untuk uji tambahan CNN (G0L4, U2, U3, U4), di atas bundel R15 yang sudah ada.

Dipanggil dari notebook di folder TA_R15_UJI_TAMBAHAN (penanda BUNDEL_UJI_TAMBAHAN.json):

    import colab_setup as C, colab_tests_setup as CR
    C.pasang_paket()
    INFO = CR.siapkan("efficientnet_b0", BUNDEL, ADDON)
    CR.jalankan("G0L4", "efficientnet_b0")

Aturan:
1. Bundel R15 (TA_R15_COLAB) disalin dan diverifikasi sha256 dengan colab_setup.salin_bundel, sama seperti run resmi.
2. Folder hasil resmi MyDrive/TA_R15_hasil hanya DIBACA: arsip cache embedding beku dan feature store resmi akun ini.
   outputs/ eksperimen TIDAK ditautkan ke hasil resmi.
3. Dua berkas kode tambahan (colab_tests.py, v4_tl_murni.py) ditempatkan di folder eksperimen setelah sha256-nya
   dicocokkan dengan BUNDEL_UJI_TAMBAHAN.json; berkas bundel tidak pernah ditimpa.
4. Setiap uji ditulis ke MyDrive/TA_R15_uji_tambahan_hasil/<id uji>/ (folder baru, tersimpan per epoch, lanjut otomatis).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import colab_setup as C

LOKAL = C.LOKAL
HASIL_RESMI = C.HASIL                                       # MyDrive/TA_R15_hasil (hanya dibaca)
HASIL_UJI = Path("/content/drive/MyDrive/TA_R15_uji_tambahan_hasil")
PENANDA = "BUNDEL_UJI_TAMBAHAN.json"
EKSPERIMEN = LOKAL / "code" / "eksperimen_v4_svdb"
FEATURE_STORE = ["5ef71c976478fd22", "local_rhythm_7f0e07c14ef8087d", "record_template_897108b3876e7550",
                 "second_lead_bcb32c8feb415a10"]
CACHE_U5 = Path("/content/cache_u5")                         # disk lokal runtime (cepat); arsip tar di folder uji


def urutan(arch):
    """Urutan uji per backbone (prioritas lebih dulu): (uji, skenario, seed, fraksi, ulangan)."""
    u = [("G0L4", None, None, None, None), ("U2", None, None, None, None), ("U3", None, None, None, None),
         ("U5", None, None, None, None)]
    u += [("U4", sc, seed, None, None) for seed in (1042, 2042) for sc in ("S1", "S2")]
    u += [("U4", "S3", seed, None, None) for seed in (1042, 2042)]
    if arch == "efficientnet_b0":
        u += [("U6", None, None, f, r) for r in (0, 1, 2) for f in (0.25, 0.5, 0.75)]
    return u


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cari(penanda, drive=Path("/content/drive/MyDrive")):
    """Cari folder berisi berkas penanda di My Drive atau target pintasan (kedalaman 3)."""
    drive = Path(drive)
    for root in [drive, drive.parent / ".shortcut-targets-by-id"]:
        for depth in range(4 if root == drive else 3):
            found = sorted(root.glob("/".join(["*"] * depth + [penanda])))
            if found:
                if len(found) > 1:
                    print("Peringatan: lebih dari satu folder, dipakai:", found[0].parent)
                return found[0].parent
    raise FileNotFoundError(f"Folder dengan {penanda} tidak ditemukan di {drive}.")


def _pasang_kode(addon: Path, spek: dict, tujuan: Path = EKSPERIMEN):
    for nama, sha in spek["kode"].items():
        src, dst = addon / "kode" / nama, tujuan / nama
        if _sha(src) != sha:
            raise ValueError(f"{src} rusak/terpotong (sha256 berbeda dari {PENANDA}); unggah ulang folder TA_R15_UJI_TAMBAHAN.")
        if dst.exists():
            if _sha(dst) != sha:
                raise FileExistsError(f"{dst} sudah ada dengan isi lain; tidak ditimpa. Mulai ulang runtime (Runtime > "
                                      "Putuskan sambungan dan hapus runtime), lalu Run all.")
            continue
        shutil.copy2(src, dst)


def siapkan(arch, bundel, addon, lokal=LOKAL, hasil_resmi=HASIL_RESMI, hasil_uji=HASIL_UJI, gpu="L4"):
    bundel, addon, hasil_resmi, hasil_uji = Path(bundel), Path(addon), Path(hasil_resmi), Path(hasil_uji)
    spek = json.loads((addon / PENANDA).read_text(encoding="utf-8"))
    nama_gpu = C.cek_gpu(gpu) if gpu else None
    manifest_bundel = json.loads((bundel / C.MANIFEST).read_text(encoding="utf-8"))
    if (manifest_bundel["created"], manifest_bundel["data_protocol_hash"]) != (spek["bundel_r15_created"],
                                                                               spek["data_protocol_hash"]):
        raise ValueError("Folder TA_R15_COLAB bukan bundel R15 yang dipakai run resmi (tanggal/hash berbeda).")
    hasil_s2 = hasil_resmi / "outputs" / "final_models" / arch / "S2" / "final" / "result.json"
    if not hasil_s2.exists():
        ada = sorted(p.name for p in (hasil_resmi / "outputs" / "final_models").glob("*"))
        raise FileNotFoundError(f"Akun ini tidak melatih {arch} (TA_R15_hasil berisi {ada}). Jalankan notebook ini di akun "
                                f"yang dulu menjalankan notebook R15 {arch}.")
    resmi = json.loads(hasil_s2.read_text())
    if not resmi["checkpoint_sha256"].startswith(spek["sha_final_s2"][arch]):
        raise ValueError(f"Model final S2 {arch} di {hasil_resmi} bukan model resmi paper; akun salah?")
    fitur = hasil_resmi / "outputs" / "feature_store"
    kurang = [n for n in FEATURE_STORE if not (fitur / n / "complete.json").exists()]
    tar = C._cache_tar(hasil_resmi, arch, spek["data_protocol_hash"])
    if kurang or not tar.exists():
        raise FileNotFoundError(f"Hasil resmi akun ini tidak lengkap di {hasil_resmi}: feature store kurang {kurang}, "
                                f"arsip cache {tar.name} {'ada' if tar.exists() else 'TIDAK ADA'}. Pastikan notebook ini "
                                f"dijalankan di akun yang dulu melatih {arch}.")
    C.cek_ruang(bundel, lokal, hasil_uji, drive_min_gb=6.0)   # arsip cache U5 ~1,6 GB + checkpoint S3
    C.salin_bundel(bundel, lokal)
    _pasang_kode(addon, spek, Path(lokal) / "code" / "eksperimen_v4_svdb")
    C.pulihkan_cache(arch, spek["data_protocol_hash"], lokal, hasil_resmi)   # hanya membaca arsip tar resmi
    hasil_uji.mkdir(parents=True, exist_ok=True)
    os.chdir(Path(lokal) / "code" / "eksperimen_v4_svdb")
    info = {"arch": arch, "gpu": nama_gpu, "bundel": str(bundel), "addon": str(addon), "addon_dibuat": spek["created"],
            "feature_store_resmi": str(fitur), "hasil_uji": str(hasil_uji)}
    print(json.dumps(info, indent=1))
    return info


def _id(uji, arch, skenario, seed, fraksi=None, ulang=None):
    if str(EKSPERIMEN) not in sys.path:
        sys.path.insert(0, str(EKSPERIMEN))
    import colab_tests as U
    return U.uji_id(uji, arch, skenario or "S2", seed, False, fraksi, ulang)


def jalankan(uji, arch, skenario=None, seed=None, fraksi=None, ulang=None, hasil_resmi=HASIL_RESMI, hasil_uji=HASIL_UJI):
    """Jalankan satu uji sebagai proses terpisah (satu protokol per proses); aman dijalankan ulang."""
    cmd = [sys.executable, "colab_tests.py", "--uji", uji, "--arch", arch, "--base", str(hasil_uji),
           "--sumber-fitur", str(Path(hasil_resmi) / "outputs" / "feature_store")]
    if skenario:
        cmd += ["--skenario", skenario]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if fraksi is not None:
        cmd += ["--fraksi", str(fraksi), "--ulang", str(ulang)]
    if uji == "U5":
        cmd += ["--cache-lokal", str(CACHE_U5 / arch)]
    env = {**os.environ, "V4_RHYTHM_WORKERS": "4", "PYTHONUNBUFFERED": "1"}
    t = time.time()
    proc = subprocess.Popen(cmd, cwd=EKSPERIMEN, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    for line in proc.stdout:
        if "it/s]" in line or "s/it]" in line:   # bilah tqdm per batch
            continue
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"Uji {uji} gagal (kode {proc.returncode}). Baca pesan di atas; jangan hapus foldernya. "
                           f"Bila runtime terputus, Run all lagi: training dilanjutkan dari last.pt.")
    root = Path(hasil_uji) / _id(uji, arch, skenario, seed, fraksi, ulang)
    selesai = json.loads((root / "selesai.json").read_text())
    print(f"{uji} selesai dalam {(time.time() - t) / 60:.1f} menit -> {root}")
    if uji == "G0L4" and not selesai.get("sha_sama_dengan_resmi"):
        print("CATATAN: checkpoint G0L4 tidak identik bit per bit dengan checkpoint resmi. Lanjutkan; kesetaraan dinilai di "
              "mesin lokal dengan kriteria G0 (argmax >= 99 %, |selisih macro-F1| <= 0,005).")
    return selesai


def jalankan_semua(arch):
    """Semua uji backbone ini berurutan; yang selesai dilewati, yang terputus dilanjutkan."""
    for uji, sc, seed, fraksi, ulang in urutan(arch):
        print(f"\n===== {_id(uji, arch, sc, seed, fraksi, ulang)} =====", flush=True)
        jalankan(uji, arch, sc, seed, fraksi, ulang)


def status(arch, hasil_uji=HASIL_UJI):
    rows = []
    for uji, sc, seed, fraksi, ulang in urutan(arch):
        root = Path(hasil_uji) / _id(uji, arch, sc, seed, fraksi, ulang)
        s = json.loads((root / "selesai.json").read_text()) if (root / "selesai.json").exists() else {}
        rows.append({"uji": root.name, "status": "selesai" if s else ("berjalan/terputus" if root.exists() else "belum"),
                     "E*": s.get("E*"), "menit": round(s["training_s"] / 60, 1) if s else None,
                     "sha": s.get("checkpoint_sha256", "")[:8], "identik_resmi": s.get("sha_sama_dengan_resmi")})
    return rows
