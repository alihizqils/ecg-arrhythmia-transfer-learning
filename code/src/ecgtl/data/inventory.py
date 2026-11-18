"""Inventaris rekaman: laju sampling, kanal, durasi, dan ketersediaan anotasi.

Inventaris ini bukan sekadar catatan teknis. Ia menjadi bukti tertulis untuk dua
klaim di proposal yang perlu disesuaikan: laju sampling yang berbeda antar
basis data, dan klaim bahwa sinyal diambil dari kanal MLII, padahal hanya mitdb
yang memakai penamaan kanal tersebut.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import wfdb

LOGGER = logging.getLogger(__name__)

# Ekstensi berkas anotasi yang relevan pada ketiga basis data.
#   atr   anotasi acuan (per-detak di mitdb, perubahan ritme di afdb dan vfdb)
#   qrs   anotasi detak hasil detektor otomatis, belum dikoreksi manual (afdb)
#   qrsc  anotasi detak afdb yang sudah dikoreksi manual, bila tersedia
ANNOTATION_EXTS: tuple[str, ...] = ("atr", "qrs", "qrsc")


@dataclass
class ChannelChoice:
    """Kanal yang dipilih untuk satu rekaman."""

    index: int
    name: str
    from_preference: bool

    @property
    def note(self) -> str:
        return "preferensi" if self.from_preference else "fallback kanal pertama"


def _normalize(name: str) -> str:
    return str(name).strip().upper().replace(" ", "").replace("-", "")


def select_channel(sig_names: Sequence[str], preference: Iterable[str]) -> ChannelChoice:
    """Memilih kanal berdasarkan daftar preferensi, dengan fallback kanal pertama.

    Pencocokan tidak peka huruf besar-kecil dan mengabaikan spasi serta tanda
    hubung, sehingga "MLII", "ML II", dan "mlii" dianggap sama.
    """
    if not sig_names:
        raise ValueError("rekaman tidak memiliki kanal sinyal")

    dinormalisasi = [_normalize(n) for n in sig_names]
    for pref in preference:
        target = _normalize(pref)
        if target in dinormalisasi:
            idx = dinormalisasi.index(target)
            return ChannelChoice(index=idx, name=str(sig_names[idx]), from_preference=True)

    return ChannelChoice(index=0, name=str(sig_names[0]), from_preference=False)


@dataclass
class RecordInfo:
    """Satu baris inventaris."""

    database: str
    record: str
    fs: float | None
    n_sig: int
    sig_names: str
    sig_len: int
    duration_s: float
    selected_channel_index: int | None
    selected_channel_name: str | None
    channel_selection: str
    has_signal: bool
    annotations: str
    catatan: str = ""


def _annotation_exts_present(directory: Path, record: str) -> list[str]:
    return [ext for ext in ANNOTATION_EXTS if (directory / f"{record}.{ext}").is_file()]


def inspect_record(
    database: str,
    directory: Path,
    record: str,
    channel_preference: Sequence[str],
) -> RecordInfo:
    """Membaca header satu rekaman dan menyusun barisnya untuk inventaris."""
    anotasi = _annotation_exts_present(directory, record)
    has_signal = (directory / f"{record}.dat").is_file() or any(
        directory.glob(f"{record}*.dat")
    )

    try:
        header = wfdb.rdheader(str(directory / record))
    except Exception as exc:  # noqa: BLE001 - header rusak harus terlihat di inventaris
        LOGGER.warning("%s/%s: gagal membaca header (%s)", database, record, exc)
        return RecordInfo(
            database=database,
            record=record,
            fs=None,
            n_sig=0,
            sig_names="",
            sig_len=0,
            duration_s=0.0,
            selected_channel_index=None,
            selected_channel_name=None,
            channel_selection="gagal baca header",
            has_signal=has_signal,
            annotations=",".join(anotasi),
            catatan=str(exc),
        )

    sig_names: list[str] = list(header.sig_name or [])
    fs = float(header.fs) if header.fs else None
    sig_len = int(header.sig_len or 0)
    duration = sig_len / fs if fs else 0.0

    if sig_names:
        pilihan = select_channel(sig_names, channel_preference)
        idx: int | None = pilihan.index
        nama: str | None = pilihan.name
        keterangan = pilihan.note
    else:
        idx, nama, keterangan = None, None, "tidak ada kanal"

    catatan = ""
    if not has_signal:
        catatan = "tanpa berkas sinyal, hanya anotasi"

    return RecordInfo(
        database=database,
        record=record,
        fs=fs,
        n_sig=int(header.n_sig or 0),
        sig_names="|".join(sig_names),
        sig_len=sig_len,
        duration_s=round(duration, 3),
        selected_channel_index=idx,
        selected_channel_name=nama,
        channel_selection=keterangan,
        has_signal=has_signal,
        annotations=",".join(anotasi),
        catatan=catatan,
    )


def build_inventory(
    raw_root: Path,
    databases: Iterable[str],
    channel_preference: Sequence[str],
) -> pd.DataFrame:
    """Menyusun inventaris seluruh rekaman yang tersedia secara lokal."""
    baris: list[dict[str, Any]] = []
    for database in databases:
        directory = Path(raw_root) / database
        if not directory.is_dir():
            LOGGER.warning("direktori %s tidak ditemukan, dilewati", directory)
            continue
        for header_path in sorted(directory.glob("*.hea")):
            info = inspect_record(
                database, directory, header_path.stem, channel_preference
            )
            baris.append(asdict(info))

    kolom = [f.name for f in RecordInfo.__dataclass_fields__.values()]  # type: ignore[attr-defined]
    return pd.DataFrame(baris, columns=kolom)


def summarize_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan per basis data: jumlah rekaman, laju sampling, total durasi."""
    if inventory.empty:
        return pd.DataFrame()

    bersinyal = inventory[inventory["has_signal"]]
    ringkas = (
        bersinyal.groupby("database")
        .agg(
            rekaman_bersinyal=("record", "count"),
            fs_unik=("fs", lambda s: "|".join(sorted({str(int(v)) for v in s.dropna()}))),
            durasi_total_jam=("duration_s", lambda s: round(float(s.sum()) / 3600, 2)),
            durasi_per_rekaman_menit=(
                "duration_s",
                lambda s: round(float(s.mean()) / 60, 2),
            ),
            kanal_terpilih=(
                "selected_channel_name",
                lambda s: "|".join(sorted(set(s.dropna().astype(str)))),
            ),
        )
        .reset_index()
    )

    total_header = inventory.groupby("database")["record"].count().rename("rekaman_header")
    return ringkas.merge(total_header, on="database", how="left")


def save_inventory(inventory: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(path, index=False, encoding="utf-8")
    return path
