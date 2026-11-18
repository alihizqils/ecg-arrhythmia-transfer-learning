"""Pemrosesan sinyal: penyeragaman laju sampling, penapisan, dan normalisasi.

Urutan pemrosesan sengaja dipisah menjadi dua tingkat:

    per rekaman   resample ke laju seragam, lalu band-pass
    per jendela   pemotongan, lalu normalisasi min-max

Penapisan dilakukan sekali per rekaman, bukan per jendela. Alasannya bukan
sekadar kecepatan: penapis rekursif memerlukan riwayat sinyal, sehingga
menerapkannya pada potongan 10 detik menimbulkan artefak transien di kedua ujung
setiap jendela. Artefak itu akan tergambar sebagai lonjakan palsu di citra dan
berpotensi dipelajari model sebagai ciri.

Normalisasi sebaliknya harus per jendela, sesuai Persamaan 2.5 proposal, agar
setiap citra memakai seluruh rentang dinamis yang tersedia.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal as sp_signal

LOGGER = logging.getLogger(__name__)


def resample_signal(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Mengubah laju sampling memakai penapis polifase.

    `resample_poly` dipilih daripada interpolasi sederhana karena ia menyertakan
    penapis anti-alias, sehingga penurunan laju dari 360 Hz ke 250 Hz tidak
    melipat komponen frekuensi tinggi ke dalam pita yang diamati.
    """
    x = np.asarray(x, dtype=np.float64)
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("laju sampling harus positif")
    if abs(fs_in - fs_out) < 1e-9:
        return x

    rasio = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
    return sp_signal.resample_poly(x, up=rasio.numerator, down=rasio.denominator)


def design_bandpass(
    fs: float, low_hz: float, high_hz: float, order: int = 4
) -> np.ndarray:
    """Merancang penapis band-pass Butterworth dalam bentuk second-order sections.

    Bentuk SOS dipakai daripada koefisien b/a karena jauh lebih stabil secara
    numerik pada orde tinggi.
    """
    nyquist = fs / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"pita tapis tidak valid: butuh 0 < {low_hz} < {high_hz} < {nyquist}"
        )
    return sp_signal.butter(
        order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos"
    )


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """Menerapkan band-pass Butterworth.

    Mode zero_phase menerapkan penapis maju lalu mundur, sehingga tidak ada
    pergeseran fasa. Ini penting karena pergeseran fasa akan menggeser posisi
    kompleks QRS terhadap anotasi detak yang menjadi dasar pelabelan.
    """
    x = np.asarray(x, dtype=np.float64)
    sos = design_bandpass(fs, low_hz, high_hz, order)

    # sosfiltfilt memerlukan sinyal lebih panjang dari padding internalnya.
    minimum = 3 * (2 * order + 1)
    if zero_phase and x.size > minimum:
        return sp_signal.sosfiltfilt(sos, x)
    if zero_phase:
        LOGGER.debug("sinyal terlalu pendek (%d sampel) untuk zero-phase", x.size)
    return sp_signal.sosfilt(sos, x)


def minmax_normalize(x: np.ndarray, konstan_ke: float = 0.5) -> np.ndarray:
    """Menskalakan sinyal ke rentang [0, 1] sesuai Persamaan 2.5 proposal.

    Segmen yang seluruh amplitudonya sama, misalnya potongan datar hasil blok
    data yang hilang pada tiga rekaman afdb, tidak dapat dinormalisasi karena
    pembaginya nol. Segmen seperti itu dikembalikan sebagai nilai konstan
    di tengah rentang, dan pemanggil dapat memeriksanya lewat `is_flat`.
    """
    x = np.asarray(x, dtype=np.float64)
    minimum = float(np.min(x))
    maksimum = float(np.max(x))
    rentang = maksimum - minimum
    if rentang <= 0:
        return np.full_like(x, konstan_ke)
    return (x - minimum) / rentang


def is_flat(x: np.ndarray, tolerance: float = 1e-12) -> bool:
    """Apakah segmen tidak memiliki variasi amplitudo sama sekali."""
    x = np.asarray(x)
    return bool(x.size == 0 or (float(np.max(x)) - float(np.min(x))) <= tolerance)


@dataclass
class PreprocessedRecord:
    """Sinyal satu rekaman yang sudah diseragamkan laju dan ditapis."""

    database: str
    record: str
    signal: np.ndarray
    fs: float
    fs_original: float
    channel_name: str

    @property
    def n_samples(self) -> int:
        return int(self.signal.size)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs if self.fs else 0.0

    def window_bounds_at_target_fs(
        self, start_original: int, window_seconds: float
    ) -> tuple[int, int]:
        """Memetakan awal jendela dari domain sampel asli ke domain target.

        Pemetaan dilakukan lewat satuan waktu, lalu panjang jendela dihitung dari
        laju target. Cara ini menjaga panjang setiap jendela tepat sama; bila
        kedua batas dipetakan lalu dibulatkan sendiri-sendiri, pembulatan bisa
        menghasilkan jendela 2.499 dan 2.500 sampel secara bergantian.
        """
        panjang = int(round(window_seconds * self.fs))
        mulai = int(round(start_original / self.fs_original * self.fs))
        return mulai, mulai + panjang

    def extract(self, start: int, end: int) -> np.ndarray | None:
        """Mengambil potongan [start, end); None bila keluar batas rekaman."""
        if start < 0 or end > self.n_samples or end <= start:
            return None
        return self.signal[start:end]


def preprocess_record(
    raw_signal: np.ndarray,
    fs_original: float,
    target_fs: float,
    bandpass: dict | None,
    database: str = "",
    record: str = "",
    channel_name: str = "",
) -> PreprocessedRecord:
    """Menyeragamkan laju sampling lalu menapis satu rekaman utuh."""
    sinyal = resample_signal(raw_signal, fs_original, target_fs)

    if bandpass and bandpass.get("enabled", True):
        sinyal = bandpass_filter(
            sinyal,
            target_fs,
            low_hz=float(bandpass["low_hz"]),
            high_hz=float(bandpass["high_hz"]),
            order=int(bandpass.get("order", 4)),
            zero_phase=bool(bandpass.get("zero_phase", True)),
        )

    return PreprocessedRecord(
        database=database,
        record=record,
        signal=np.asarray(sinyal, dtype=np.float32),
        fs=float(target_fs),
        fs_original=float(fs_original),
        channel_name=channel_name,
    )
