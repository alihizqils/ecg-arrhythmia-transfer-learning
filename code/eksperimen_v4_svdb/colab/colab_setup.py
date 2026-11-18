"""Persiapan runtime Colab untuk eksperimen V4, paket model akhir R15 (di luar hash metode).

10 Nov 2026 (bundel R15): penanda folder BUNDEL_R15.json dan folder hasil baru MyDrive/TA_R15_hasil, agar
run Colab R15 dimulai dari awal dan tidak tercampur dengan bundel/hasil 8 Nov (BUNDEL_V4.json, TA_V4_hasil).

Dipanggil dari sel pertama notebook Colab:

    import colab_setup as C
    C.pasang_paket()
    INFO = C.siapkan("efficientnet_b0", BUNDEL)

Alur `siapkan`:
1. cek GPU (wajib L4 agar seluruh backbone dilatih di satu jenis GPU);
2. salin kode + data dari bundel Drive ke disk lokal runtime (`/content/ta`) sambil
   memverifikasi sha256 setiap berkas terhadap `MANIFEST_BUNDEL.json` (data dibaca acak
   dari memmap, sehingga membaca langsung dari Drive sangat lambat);
3. `outputs/` eksperimen ditautkan (symlink) ke `MyDrive/TA_R15_hasil/outputs` agar
   checkpoint per epoch tetap ada bila runtime terputus;
4. cache embedding beku yang pernah dibuat dipulihkan dari arsip tar di Drive.

Semua langkah idempoten: menjalankan ulang notebook (Run all) setelah runtime terputus
akan melanjutkan dari checkpoint terakhir.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PAKET = {  # versi lingkungan lokal (eksperimen_final_hierarkis_10s_lokal/requirements-local.txt)
    "torch": "2.6.0", "torchvision": "0.21.0", "wfdb": "4.3.1", "numpy": "2.5.2", "scipy": "1.18.0",
    "pandas": "3.0.5", "scikit-learn": "1.9.0", "shap": "0.52.0", "matplotlib": "3.11.1",
    "PyYAML": "6.0.3", "tqdm": "4.70.0", "psutil": "7.2.2",
}
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
LOKAL = Path("/content/ta")
HASIL = Path("/content/drive/MyDrive/TA_R15_hasil")
PENANDA = "BUNDEL_R15.json"
MANIFEST = "MANIFEST_BUNDEL.json"
CHUNK = 16 * 2**20


def _versi(nama):
    try:
        return metadata.version(nama).split("+")[0]
    except metadata.PackageNotFoundError:
        return None


def pasang_paket(paket=PAKET):
    """Pasang versi paket yang sama dengan lingkungan lokal (hanya bila berbeda)."""
    kurang = {n: v for n, v in paket.items() if _versi(n) != v}
    if not kurang:
        print("Paket sudah sesuai.")
        return {}
    dimuat = [n for n in ("torch", "numpy", "pandas", "scipy", "sklearn", "matplotlib") if n in sys.modules]
    torch_pkgs = [f"{n}=={kurang.pop(n)}" for n in ("torch", "torchvision") if n in kurang]
    t = time.time()
    if torch_pkgs:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *torch_pkgs, "--index-url", TORCH_INDEX], check=True)
    if kurang:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *(f"{n}=={v}" for n, v in kurang.items())], check=True)
    print(f"Paket dipasang ({time.time() - t:.0f} dtk).")
    if dimuat:
        # Modul lama sudah termuat di kernel: runtime harus dimulai ulang agar versi baru dipakai.
        print("RUNTIME DIMULAI ULANG agar versi paket baru dipakai. Setelah tersambung lagi, klik Runtime > Run all.")
        sys.stdout.flush(); time.sleep(2)
        os.kill(os.getpid(), 9)
    return kurang


def cari_bundel(drive=Path("/content/drive/MyDrive")):
    """Cari folder bundel (berisi BUNDEL_R15.json) hingga kedalaman 3 di My Drive (termasuk pintasan).

    Akun 2 dan 3 memakai pintasan ke folder bersama; Colab memetakan target pintasan ke
    /content/drive/.shortcut-targets-by-id/<id>/<nama>, yang ikut dicari bila My Drive tidak memuatnya."""
    drive = Path(drive)
    roots = [drive, drive.parent / ".shortcut-targets-by-id"]
    for root in roots:
        for depth in range(4 if root == drive else 3):
            found = sorted(root.glob("/".join(["*"] * depth + [PENANDA])))
            if found:
                if len(found) > 1:
                    print("Peringatan: lebih dari satu bundel, dipakai:", found[0].parent)
                return found[0].parent
    raise FileNotFoundError(f"Folder bundel ({PENANDA}) tidak ditemukan di {drive}. Unggah folder TA_R15_COLAB ke My Drive "
                            "atau tambahkan pintasannya (klik kanan folder bersama > Atur > Tambahkan pintasan ke Drive).")


def cek_gpu(wajib="L4"):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU tidak aktif. Pilih Runtime > Ubah jenis runtime > GPU L4.")
    nama = torch.cuda.get_device_name(0)
    if wajib and wajib not in nama:
        raise RuntimeError(f"GPU yang didapat {nama}, bukan {wajib}. Agar semua backbone dilatih di jenis GPU yang sama, "
                           f"pilih Runtime > Ubah jenis runtime > {wajib}, lalu Run all lagi.")
    return nama


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _salin_berkas(src, dst, sha, percobaan=5):
    """Salin dengan percobaan ulang: baca Drive dari folder bersama kadang gagal sementara (Errno 5)."""
    for i in range(percobaan):
        try:
            return _salin_sekali(src, dst, sha)
        except OSError as error:
            if i == percobaan - 1:
                raise OSError(f"Gagal membaca {src} dari Drive setelah {percobaan} percobaan ({error}). Biasanya batas baca "
                              "Google Drive untuk berkas bersama sedang tercapai: tunggu 1-2 jam lalu Run all lagi (berkas "
                              "yang sudah tersalin tidak diulang). Bila terus terjadi, unggah folder bundel langsung ke "
                              "Drive akun ini.") from error
            tunggu = 30 * (i + 1)
            print(f"  baca Drive gagal ({error}); coba lagi dalam {tunggu} dtk", flush=True); time.sleep(tunggu)


def _salin_sekali(src, dst, sha):
    """Salin + hitung sha256 dalam satu kali baca. Hardlink bila satu volume (uji lokal)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        os.link(src, tmp)
        got = _sha(tmp)
    except OSError:
        h = hashlib.sha256()
        with open(src, "rb") as fi, open(tmp, "wb") as fo:
            for block in iter(lambda: fi.read(CHUNK), b""):
                h.update(block); fo.write(block)
        got = h.hexdigest()
    if got != sha:
        tmp.unlink()
        raise ValueError(f"Checksum berbeda untuk {src}. Berkas di Drive rusak/terpotong saat diunggah; unggah ulang berkas ini.")
    os.replace(tmp, dst)


def salin_bundel(bundel, lokal=LOKAL):
    """Salin isi bundel ke disk lokal runtime dengan verifikasi sha256 (dilewati bila sudah terverifikasi di sesi ini)."""
    bundel, lokal = Path(bundel), Path(lokal)
    manifest = json.loads((bundel / MANIFEST).read_text(encoding="utf-8"))
    tanda = lokal / ".terverifikasi.json"
    sudah = json.loads(tanda.read_text()) if tanda.exists() else {}
    files = manifest["files"]
    total = sum(f["size"] for f in files.values()); selesai = 0; t = time.time(); disalin = 0
    for rel, info in files.items():
        dst = lokal / rel
        if sudah.get(rel) == info["sha256"] and dst.exists() and dst.stat().st_size == info["size"]:
            selesai += info["size"]; continue
        _salin_berkas(bundel / rel, dst, info["sha256"])
        sudah[rel] = info["sha256"]; selesai += info["size"]; disalin += 1
        if disalin % 25 == 0 or info["size"] > 2**28:
            tanda.write_text(json.dumps(sudah))
            print(f"  salin {selesai / 2**30:5.1f}/{total / 2**30:.1f} GB  ({time.time() - t:.0f} dtk)", flush=True)
    tanda.parent.mkdir(parents=True, exist_ok=True); tanda.write_text(json.dumps(sudah))
    print(f"Bundel siap: {len(files)} berkas, {total / 2**30:.1f} GB terverifikasi sha256 ({time.time() - t:.0f} dtk, {disalin} disalin).")
    return manifest


def _tautkan(link, target):
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve():
            return
        raise FileExistsError(f"{link} sudah ada dan tidak menunjuk ke {target}; tidak ditimpa.")
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi  # uji lokal di Windows: junction tidak butuh hak admin
        _winapi.CreateJunction(str(target), str(link))


def _cache_dir(lokal, arch, protocol_hash):
    return Path(lokal) / "code/data/processed/v4_svdb/prepared/features" / arch / protocol_hash[:16]


def _cache_tar(hasil, arch, protocol_hash):
    return Path(hasil) / f"cache_fitur_{arch}_{protocol_hash[:16]}.tar"


def pulihkan_cache(arch, protocol_hash, lokal=LOKAL, hasil=HASIL):
    folder, arsip = _cache_dir(lokal, arch, protocol_hash), _cache_tar(hasil, arch, protocol_hash)
    if (folder / "complete.json").exists() or not arsip.exists():
        return False
    t = time.time(); folder.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(arsip) as tar:
        tar.extractall(folder.parent, filter="data")
    print(f"Cache embedding {arch} dipulihkan dari Drive ({time.time() - t:.0f} dtk); checksum diverifikasi saat dipakai.")
    return True


def simpan_cache(arch, protocol_hash, lokal=LOKAL, hasil=HASIL):
    """Arsipkan cache embedding beku ke Drive (sekali) agar sesi berikutnya tidak membangun ulang."""
    folder, arsip = _cache_dir(lokal, arch, protocol_hash), _cache_tar(hasil, arch, protocol_hash)
    if arsip.exists():
        return arsip
    if not (folder / "complete.json").exists():
        raise FileNotFoundError("Cache embedding belum lengkap.")
    t = time.time(); tmp = arsip.with_name(arsip.name + ".tmp"); arsip.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tmp, "w") as tar:
        tar.add(folder, arcname=folder.name)
    os.replace(tmp, arsip)
    print(f"Cache embedding {arch} diarsipkan ke {arsip} ({time.time() - t:.0f} dtk).")
    return arsip


def cek_ruang(bundel, lokal, hasil, drive_min_gb=4.0):
    """Disk runtime harus muat bundel + cache; Drive akun ini harus muat hasil (checkpoint + arsip cache, ~2-3 GB)."""
    manifest = json.loads((Path(bundel) / MANIFEST).read_text(encoding="utf-8"))
    perlu = sum(f["size"] for f in manifest["files"].values()) / 2**30 + 10
    Path(lokal).mkdir(parents=True, exist_ok=True)
    bebas = shutil.disk_usage(lokal).free / 2**30
    sudah = sum(f.stat().st_size for f in Path(lokal).rglob("*") if f.is_file()) / 2**30 if any(Path(lokal).iterdir()) else 0
    if bebas + sudah < perlu:
        raise OSError(f"Disk runtime kurang: bebas {bebas:.0f} GB, perlu sekitar {perlu:.0f} GB.")
    try:
        drive_bebas = shutil.disk_usage(Path(hasil).parent).free / 2**30
    except OSError:
        return
    if drive_bebas < drive_min_gb:
        raise OSError(f"Ruang Google Drive akun ini tinggal {drive_bebas:.1f} GB; hasil butuh sekitar 2-3 GB "
                      f"(arsip cache embedding + checkpoint). Kosongkan Drive dulu agar hasil tidak gagal tersimpan.")
    print(f"Ruang: disk runtime bebas {bebas:.0f} GB, Drive akun ini bebas {drive_bebas:.1f} GB.")


def siapkan(arch, bundel, lokal=LOKAL, hasil=HASIL, gpu="L4"):
    """Siapkan runtime: GPU, salinan data terverifikasi, outputs di Drive, cache embedding."""
    bundel, lokal, hasil = Path(bundel), Path(lokal), Path(hasil)
    nama_gpu = cek_gpu(gpu) if gpu is not None else None
    cek_ruang(bundel, lokal, hasil)
    manifest = salin_bundel(bundel, lokal)
    eksperimen = lokal / "code" / "eksperimen_v4_svdb"
    _tautkan(eksperimen / "outputs", hasil / "outputs")
    pulihkan_cache(arch, manifest["data_protocol_hash"], lokal, hasil)
    os.chdir(eksperimen)
    if str(eksperimen) not in sys.path:
        sys.path.insert(0, str(eksperimen))
    info = {"arch": arch, "gpu": nama_gpu, "bundel": str(bundel), "lokal": str(lokal), "outputs": str(hasil / "outputs"),
            "bundel_dibuat": manifest["created"], "data_protocol_hash": manifest["data_protocol_hash"]}
    print(json.dumps(info, indent=1))
    return info
