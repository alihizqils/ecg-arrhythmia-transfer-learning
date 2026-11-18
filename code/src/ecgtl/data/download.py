"""Pengunduhan basis data PhysioNet.

Tiga basis data dipakai:
    mitdb  MIT-BIH Arrhythmia Database          48 rekaman, 360 Hz, anotasi per-detak
    afdb   MIT-BIH Atrial Fibrillation Database 23 rekaman bersinyal, 250 Hz, anotasi ritme
    vfdb   MIT-BIH Malignant Ventricular Ectopy 22 rekaman, 250 Hz, anotasi ritme

Unduhan bersifat dapat dilanjutkan: berkas yang sudah ada dilewati, sehingga
proses yang terputus dapat diulang tanpa mengunduh ulang dari awal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import wfdb

LOGGER = logging.getLogger(__name__)

# Rekaman afdb yang terdaftar tetapi tidak memiliki berkas sinyal (.dat);
# keduanya hanya berisi anotasi sehingga tidak dapat dipakai.
AFDB_TANPA_SINYAL = frozenset({"00735", "03665"})


@dataclass
class DownloadResult:
    """Hasil pengunduhan satu basis data."""

    database: str
    directory: Path
    records_expected: int | None = None
    records_with_header: int = 0
    records_with_signal: int = 0
    missing_signal: list[str] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)
    downloaded: bool = False

    @property
    def complete(self) -> bool:
        if self.records_expected is None:
            return self.records_with_signal > 0
        return self.records_with_signal >= self.records_expected


def remote_record_list(database: str) -> list[str]:
    """Daftar nama rekaman menurut server PhysioNet."""
    return [str(name).strip("/") for name in wfdb.get_record_list(database)]


def local_record_names(directory: Path) -> list[str]:
    """Daftar rekaman yang berkas headernya sudah ada secara lokal."""
    return sorted(path.stem for path in directory.glob("*.hea"))


def has_signal_files(directory: Path, record: str) -> bool:
    """Apakah berkas sinyal rekaman tersedia.

    Sebagian rekaman WFDB memakai satu `.dat`, sebagian memakai berkas per
    kanal, sehingga pencocokan dilakukan dengan pola.
    """
    if (directory / f"{record}.dat").is_file():
        return True
    return any(directory.glob(f"{record}*.dat"))


# Jumlah bit per sampel untuk format penyimpanan WFDB yang relevan. Format 212
# memampatkan dua sampel ke dalam tiga byte, sehingga rata-ratanya 12 bit.
BITS_PER_SAMPLE: dict[str, float] = {
    "8": 8,
    "80": 8,
    "16": 16,
    "24": 24,
    "32": 32,
    "61": 16,
    "160": 16,
    "212": 12,
    "310": 10 + 2 / 3,
    "311": 10 + 2 / 3,
}


def expected_signal_bytes(header) -> int | None:
    """Ukuran berkas sinyal yang seharusnya, dihitung dari header.

    Mengembalikan None bila format penyimpanan tidak dikenal atau rekaman
    memakai lebih dari satu berkas sinyal, karena pada kasus itu perbandingan
    ukuran tidak dapat dilakukan secara sederhana.
    """
    sig_len = int(getattr(header, "sig_len", 0) or 0)
    fmt = list(getattr(header, "fmt", None) or [])
    file_names = list(getattr(header, "file_name", None) or [])
    if sig_len <= 0 or not fmt:
        return None
    if len(set(file_names)) > 1:
        return None
    if any(f not in BITS_PER_SAMPLE for f in fmt):
        return None

    total_bits = sum(BITS_PER_SAMPLE[f] for f in fmt) * sig_len
    return int(total_bits // 8)


def record_is_complete(directory: Path, record: str) -> bool:
    """Apakah rekaman sudah terunduh utuh.

    Keberadaan berkas header saja tidak cukup: unduhan yang terputus bisa
    menyisakan `.hea` tanpa `.dat`, atau `.dat` yang terpotong. Keduanya baru
    terdeteksi jauh di dalam pipeline sebagai galat yang membingungkan, jadi
    lebih baik diperiksa di sini dengan membandingkan ukuran berkas sinyal
    terhadap panjang yang dinyatakan header.
    """
    directory = Path(directory)
    header_path = directory / f"{record}.hea"
    if not header_path.is_file():
        return False

    try:
        header = wfdb.rdheader(str(directory / record))
    except Exception:  # noqa: BLE001 - header rusak berarti perlu unduh ulang
        return False

    # Rekaman yang memang tidak menyertakan sinyal, seperti afdb 00735 dan
    # 03665, dianggap selesai agar tidak diunduh berulang tanpa hasil.
    if int(getattr(header, "n_sig", 0) or 0) == 0:
        return True
    if record in AFDB_TANPA_SINYAL:
        return True

    if not has_signal_files(directory, record):
        return False

    diharapkan = expected_signal_bytes(header)
    if diharapkan is None:
        return True

    dat_path = directory / f"{record}.dat"
    if not dat_path.is_file():
        return True
    return dat_path.stat().st_size >= diharapkan


def download_database(
    database: str,
    dest_root: Path,
    records: list[str] | None = None,
    overwrite: bool = False,
) -> DownloadResult:
    """Mengunduh satu basis data ke `dest_root/<database>`.

    Args:
        database: nama basis data di PhysioNet, mis. "mitdb".
        dest_root: direktori induk, biasanya `data/raw`.
        records: bila diisi, hanya rekaman tersebut yang diunduh. Berguna untuk
            mode uji asap yang cukup memakai beberapa rekaman.
        overwrite: bila True, berkas yang sudah ada diunduh ulang.
    """
    directory = Path(dest_root) / database
    directory.mkdir(parents=True, exist_ok=True)

    result = DownloadResult(database=database, directory=directory)

    if records is None:
        try:
            records = remote_record_list(database)
        except Exception as exc:  # noqa: BLE001 - jaringan bisa gagal, laporkan jelas
            LOGGER.warning("gagal mengambil daftar rekaman %s: %s", database, exc)
            records = None

    if records is not None:
        perlu_diunduh = [
            r for r in records if overwrite or not record_is_complete(directory, r)
        ]
        tidak_utuh = [
            r
            for r in perlu_diunduh
            if (directory / f"{r}.hea").is_file() and not overwrite
        ]
        if tidak_utuh:
            LOGGER.info(
                "%s: %d rekaman terunduh tidak utuh dan akan diulang: %s",
                database,
                len(tidak_utuh),
                ", ".join(tidak_utuh),
            )
    else:
        perlu_diunduh = None

    if perlu_diunduh is None or perlu_diunduh:
        jumlah = "semua" if perlu_diunduh is None else str(len(perlu_diunduh))
        LOGGER.info("mengunduh %s: %s rekaman", database, jumlah)
        wfdb.dl_database(
            database,
            str(directory),
            records=perlu_diunduh if perlu_diunduh is not None else "all",
            overwrite=overwrite,
        )
        result.downloaded = True
    else:
        LOGGER.info("%s: seluruh rekaman sudah ada, unduhan dilewati", database)

    lokal = local_record_names(directory)
    result.records_with_header = len(lokal)
    bersinyal = [r for r in lokal if has_signal_files(directory, r)]
    result.records_with_signal = len(bersinyal)
    result.missing_signal = sorted(
        r for r in set(lokal) - set(bersinyal) if r not in AFDB_TANPA_SINYAL
    )
    result.incomplete = sorted(r for r in lokal if not record_is_complete(directory, r))
    return result


def download_all(
    dest_root: Path,
    databases: dict[str, dict],
    records_per_db: dict[str, list[str]] | None = None,
    overwrite: bool = False,
) -> list[DownloadResult]:
    """Mengunduh seluruh basis data yang tercantum di konfigurasi."""
    hasil: list[DownloadResult] = []
    for name, spec in databases.items():
        records = (records_per_db or {}).get(name)
        result = download_database(name, dest_root, records=records, overwrite=overwrite)
        result.records_expected = spec.get("expected_records")
        hasil.append(result)
    return hasil
