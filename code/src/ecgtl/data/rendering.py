"""Perenderan sinyal EKG 1D menjadi citra 2D grayscale.

Citra yang dihasilkan adalah plot garis biasa, tanpa transformasi spektral,
sesuai rancangan proposal. Gaya gambar dibuat identik untuk seluruh basis data:
garis hitam di latar putih, tanpa aksis, kisi, maupun margin. Keseragaman ini
penting karena setiap perbedaan gaya antar sumber akan menjadi penanda asal
rekaman yang dapat dipelajari model sebagai pintasan.

Perenderan menggambar langsung ke array piksel, bukan lewat matplotlib.
Alasannya kecepatan: satu konfigurasi dataset memuat lebih dari dua puluh ribu
citra, dan membuat lalu menyimpan gambar matplotlib satu per satu memakan waktu
berjam-jam untuk hasil yang sama.

Pada jendela 10 detik dengan laju 250 Hz dan lebar citra 224 piksel, terdapat
sekitar sebelas sampel per kolom piksel. Karena itu setiap kolom digambar sebagai
rentang vertikal dari nilai terendah ke tertinggi pada kolom tersebut, yang
merupakan representasi selubung sinyal yang benar pada resolusi itu.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BACKGROUND = 255
DEFAULT_FOREGROUND = 0


def _samples_to_columns(n_samples: int, width: int) -> np.ndarray:
    """Indeks kolom piksel untuk setiap sampel, menaik dan mencakup semua kolom."""
    indeks = np.arange(n_samples, dtype=np.int64)
    kolom = (indeks * width) // n_samples
    return np.clip(kolom, 0, width - 1)


def _resample_to_width(x: np.ndarray, width: int) -> np.ndarray:
    """Interpolasi linear agar jumlah titik minimal sebanyak lebar citra.

    Hanya dipakai bila jendela memuat lebih sedikit sampel daripada kolom piksel,
    misalnya jendela sangat pendek pada citra lebar. Tanpa ini akan ada kolom
    kosong yang membuat garis tampak putus-putus.
    """
    if x.size >= width:
        return x
    posisi_lama = np.linspace(0.0, 1.0, num=x.size)
    posisi_baru = np.linspace(0.0, 1.0, num=width)
    return np.interp(posisi_baru, posisi_lama, x)


def render_waveform(
    normalized: np.ndarray,
    width: int = 224,
    height: int = 224,
    line_width: int = 1,
    background: int = DEFAULT_BACKGROUND,
    foreground: int = DEFAULT_FOREGROUND,
) -> np.ndarray:
    """Merender sinyal ternormalisasi [0, 1] menjadi citra grayscale 8-bit.

    Args:
        normalized: sinyal satu dimensi bernilai di [0, 1].
        width, height: ukuran citra dalam piksel.
        line_width: ketebalan garis dalam piksel.
        background, foreground: nilai piksel latar dan garis.

    Returns:
        Array uint8 berbentuk (height, width). Nilai amplitudo 1,0 digambar di
        baris paling atas, mengikuti konvensi tampilan EKG.
    """
    if width <= 0 or height <= 0:
        raise ValueError("ukuran citra harus positif")
    if line_width < 1:
        raise ValueError("ketebalan garis minimal 1 piksel")

    x = np.asarray(normalized, dtype=np.float64).ravel()
    if x.size == 0:
        return np.full((height, width), background, dtype=np.uint8)

    x = np.clip(x, 0.0, 1.0)
    x = _resample_to_width(x, width)

    # Amplitudo 1,0 dipetakan ke baris 0 agar puncak berada di atas.
    y = np.rint((1.0 - x) * (height - 1)).astype(np.int64)
    y = np.clip(y, 0, height - 1)

    kolom = _samples_to_columns(x.size, width)

    # Karena `kolom` menaik dan setiap kolom terisi, batas tiap kolom dapat
    # dicari sekali lalu dipakai untuk agregasi tanpa perulangan Python.
    batas_awal = np.searchsorted(kolom, np.arange(width), side="left")
    y_min = np.minimum.reduceat(y, batas_awal)
    y_maks = np.maximum.reduceat(y, batas_awal)

    # Sambungkan tiap kolom dengan sampel terakhir kolom sebelumnya, supaya
    # lonjakan tajam tidak terputus antar kolom.
    batas_akhir = np.searchsorted(kolom, np.arange(width), side="right")
    y_terakhir = y[batas_akhir - 1]
    y_sebelumnya = np.empty(width, dtype=np.int64)
    y_sebelumnya[0] = y_min[0]
    y_sebelumnya[1:] = y_terakhir[:-1]

    bawah = np.minimum(y_min, y_sebelumnya)
    atas = np.maximum(y_maks, y_sebelumnya)

    baris = np.arange(height, dtype=np.int64)[:, None]
    mask = (baris >= bawah[None, :]) & (baris <= atas[None, :])

    if line_width > 1:
        mask = _thicken(mask, line_width)

    citra = np.full((height, width), background, dtype=np.uint8)
    citra[mask] = foreground
    return citra


def _thicken(mask: np.ndarray, line_width: int) -> np.ndarray:
    """Menebalkan garis dengan dilasi kotak sederhana."""
    radius = line_width // 2
    hasil = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            hasil |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return hasil


def render_from_config(normalized: np.ndarray, image_cfg: dict) -> np.ndarray:
    """Merender memakai bagian `image` pada konfigurasi."""
    return render_waveform(
        normalized,
        width=int(image_cfg.get("width", 224)),
        height=int(image_cfg.get("height", 224)),
        line_width=int(image_cfg.get("line_width", 1)),
        background=int(image_cfg.get("background", DEFAULT_BACKGROUND)),
        foreground=int(image_cfg.get("foreground", DEFAULT_FOREGROUND)),
    )


def pixels_per_second(width: int, window_seconds: float) -> float:
    """Kerapatan horizontal citra.

    Angka ini menentukan apakah ciri morfologi masih terlihat. Kompleks QRS
    berdurasi sekitar 100 ms, sehingga pada 22,4 piksel per detik ia hanya
    selebar dua piksel. Fungsi ini dipakai untuk melaporkan keterbatasan
    tersebut secara eksplisit pada studi durasi jendela.
    """
    if window_seconds <= 0:
        raise ValueError("durasi jendela harus positif")
    return width / window_seconds


def qrs_width_in_pixels(
    width: int, window_seconds: float, qrs_ms: float = 100.0
) -> float:
    """Perkiraan lebar kompleks QRS dalam piksel pada konfigurasi tertentu."""
    return pixels_per_second(width, window_seconds) * (qrs_ms / 1000.0)
