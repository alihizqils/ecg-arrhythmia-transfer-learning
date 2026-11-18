"""Pelabelan jendela: hierarki ritme di atas detak, dengan ambang kemurnian.

Satu jendela 10 detik memuat belasan detak dan dapat menyentuh lebih dari satu
episode ritme, sedangkan proposal menuntut satu label tunggal per jendela.
Modul ini menerapkan konvensi berbasis tingkat kegentingan: label jendela adalah
temuan paling genting yang ada di dalamnya.

Urutan kegentingan yang dipakai:

    VT  >  AF  >  V  >  S  >  N

Dua kelas pertama berasal dari anotasi ritme, dua berikutnya dari anotasi detak,
dan N menjadi label terakhir yang hanya diberikan bila tidak ada temuan lain.
Penempatan V dan S di bawah AF, tetapi di atas N, penting: jendela di dalam
episode bigemini ventrikel di mitdb bukan berlabel ritme yang dikenal, namun
memuat banyak detak V sehingga harus menjadi kelas V, bukan terbuang.

Kemurnian ritme diukur sebagai proporsi cakupan, bukan pengurungan penuh.
Alasannya empiris: episode VT di mitdb rata-rata hanya 3,5 detik dan flutter
ventrikel di vfdb 15,7 detik, sehingga menuntut jendela terkurung utuh dalam
satu episode akan membuang hampir seluruh kelas VT.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Mapping, Sequence

from .annotations import RecordAnnotations

LOGGER = logging.getLogger(__name__)


class Alasan(str, Enum):
    """Alasan sebuah jendela kandidat dibuang."""

    RITME_DITOLAK = "ritme_ditolak"
    BERSINGGUNGAN_DERAU = "bersinggungan_derau"
    DETAK_TERLALU_SEDIKIT = "detak_terlalu_sedikit"
    DETAK_TIDAK_KONSISTEN = "detak_tidak_konsisten"
    RITME_NORMAL_TAK_TERKONFIRMASI = "ritme_normal_tak_terkonfirmasi"
    TANPA_LABEL = "tanpa_label"


class Sumber(str, Enum):
    """Dari mana label sebuah jendela berasal."""

    RITME = "ritme"
    DETAK = "detak"
    RITME_DAN_DETAK = "ritme+detak"


@dataclass(frozen=True)
class LabelingPolicy:
    """Aturan pelabelan, seluruhnya berasal dari konfigurasi."""

    rhythm_to_class: Mapping[str, str]
    rhythm_reject: frozenset[str]
    # Kelas ritme berurut dari paling genting. N sengaja tidak masuk daftar ini
    # karena harus kalah dari temuan tingkat detak.
    rhythm_priority: tuple[str, ...] = ("VT", "AF")
    normal_class: str = "N"
    # Ambang jumlah detak ektopik per kelas. Nilai 1 berarti satu detak ektopik
    # sudah cukup, yang secara klinis benar untuk kontraksi prematur terisolasi.
    min_ectopic_beats: Mapping[str, int] = field(default_factory=lambda: {"V": 2, "S": 2})
    min_beats_per_window: int = 3
    min_rhythm_coverage: float = 0.5
    drop_noise_overlap: bool = True
    # Bila True, kelas N hanya diberikan saat ritme normal ikut terkonfirmasi.
    # Ini mencegah jendela di dalam episode takiaritmia supraventrikel, blok
    # derajat dua, atau irama junctional dilabeli normal hanya karena seluruh
    # detaknya beranotasi normal.
    require_normal_rhythm: bool = True

    @classmethod
    def from_config(cls, cfg) -> "LabelingPolicy":
        ambang = cfg.require("labeling.min_ectopic_beats")
        if isinstance(ambang, Mapping):
            min_ectopic = {str(k): int(v) for k, v in ambang.items()}
        else:
            min_ectopic = {"V": int(ambang), "S": int(ambang)}

        return cls(
            rhythm_to_class={
                str(k).upper(): str(v) for k, v in cfg.require("labeling.rhythm_to_class").items()
            },
            rhythm_reject=frozenset(str(x).upper() for x in cfg.require("labeling.rhythm_reject")),
            min_ectopic_beats=min_ectopic,
            min_beats_per_window=int(cfg.require("labeling.min_beats_per_window")),
            min_rhythm_coverage=float(cfg.require("labeling.min_rhythm_coverage")),
            drop_noise_overlap=bool(cfg.require("labeling.drop_noise_overlap")),
        )

    def ambang_ektopik(self, kelas: str) -> int:
        return int(self.min_ectopic_beats.get(kelas, 1))


@dataclass(frozen=True)
class LabeledWindow:
    """Satu jendela yang berhasil dilabeli."""

    database: str
    record: str
    start: int
    end: int
    label: str
    source: Sumber
    rhythm_coverage: float
    n_beats: int


@dataclass
class LabelingReport:
    """Rekap hasil pelabelan satu atau beberapa rekaman."""

    windows: list[LabeledWindow] = field(default_factory=list)
    rejections: Counter = field(default_factory=Counter)
    per_class: Counter = field(default_factory=Counter)
    per_source: Counter = field(default_factory=Counter)
    n_candidates: int = 0

    def add_window(self, window: LabeledWindow) -> None:
        self.windows.append(window)
        self.per_class[window.label] += 1
        self.per_source[(window.database, window.label, window.source.value)] += 1

    def add_rejection(self, database: str, alasan: Alasan) -> None:
        self.rejections[(database, alasan.value)] += 1

    def merge(self, other: "LabelingReport") -> None:
        self.windows.extend(other.windows)
        self.rejections.update(other.rejections)
        self.per_class.update(other.per_class)
        self.per_source.update(other.per_source)
        self.n_candidates += other.n_candidates


def iter_window_bounds(
    n_samples: int, window_samples: int, overlap: float
) -> Iterator[tuple[int, int]]:
    """Menghasilkan batas jendela [mulai, selesai) dengan tumpang tindih seragam.

    Tumpang tindih diterapkan sama untuk semua kelas, sehingga ini merupakan
    parameter segmentasi dan bukan teknik penyeimbangan kelas. Jendela terakhir
    yang tidak penuh dibuang, karena citra harus mewakili durasi yang sama.
    """
    if window_samples <= 0:
        raise ValueError("panjang jendela harus positif")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"tumpang tindih harus di [0, 1), diterima {overlap}")

    stride = max(1, int(round(window_samples * (1.0 - overlap))))
    start = 0
    while start + window_samples <= n_samples:
        yield start, start + window_samples
        start += stride


def rhythm_coverage_per_class(
    rec: RecordAnnotations, start: int, end: int, policy: LabelingPolicy
) -> dict[str, float]:
    """Proporsi jendela yang tercakup tiap kelas ritme target."""
    lebar = end - start
    if lebar <= 0:
        return {}

    cakupan: Counter = Counter()
    for episode in rec.rhythms_in(start, end):
        kelas = policy.rhythm_to_class.get(episode.label)
        if kelas:
            cakupan[kelas] += episode.overlap(start, end)

    return {kelas: jumlah / lebar for kelas, jumlah in cakupan.items()}


def label_window(
    rec: RecordAnnotations, start: int, end: int, policy: LabelingPolicy
) -> LabeledWindow | Alasan:
    """Melabeli satu jendela, atau mengembalikan alasan penolakannya."""
    episode_menyentuh = rec.rhythms_in(start, end)

    # 1. Ritme yang secara eksplisit ditolak membuang jendela tanpa evaluasi
    #    lanjutan. Untuk vfdb ini satu-satunya cara mengenali irama pacu jantung
    #    dan asistol, karena vfdb tidak memiliki anotasi detak.
    for episode in episode_menyentuh:
        if episode.label in policy.rhythm_reject:
            return Alasan.RITME_DITOLAK

    # 2. Segmen bertanda derau dibuang agar citra tidak memuat artefak.
    if policy.drop_noise_overlap and rec.noise_overlap(start, end) > 0:
        return Alasan.BERSINGGUNGAN_DERAU

    cakupan = rhythm_coverage_per_class(rec, start, end, policy)

    # 3. Kelas ritme genting menang lebih dulu.
    for kelas in policy.rhythm_priority:
        proporsi = cakupan.get(kelas, 0.0)
        if proporsi >= policy.min_rhythm_coverage:
            return LabeledWindow(
                database=rec.database,
                record=rec.record,
                start=start,
                end=end,
                label=kelas,
                source=Sumber.RITME,
                rhythm_coverage=proporsi,
                n_beats=len(rec.beats_in(start, end)),
            )

    cakupan_normal = cakupan.get(policy.normal_class, 0.0)
    # Rekaman tanpa anotasi ritme sama sekali tidak dapat dimintai konfirmasi
    # ritme normal; pada kasus itu penilaian sepenuhnya bergantung pada detak.
    ritme_tersedia = bool(rec.rhythms)
    normal_terkonfirmasi = (
        cakupan_normal >= policy.min_rhythm_coverage
        or not ritme_tersedia
        or not policy.require_normal_rhythm
    )

    # 4. Tingkat detak, hanya mungkin untuk basis data yang membedakan jenis detak.
    if rec.beats:
        detak = rec.beats_in(start, end)
        if len(detak) < policy.min_beats_per_window:
            return Alasan.DETAK_TERLALU_SEDIKIT

        hitung = Counter(b.aami for b in detak if b.aami)

        for kelas in ("V", "S"):
            if hitung.get(kelas, 0) >= policy.ambang_ektopik(kelas):
                return LabeledWindow(
                    database=rec.database,
                    record=rec.record,
                    start=start,
                    end=end,
                    label=kelas,
                    source=Sumber.DETAK,
                    rhythm_coverage=cakupan_normal,
                    n_beats=len(detak),
                )

        # Kelas N menuntut seluruh detak beranotasi normal. Jendela yang memuat
        # detak fusi, detak pacu, atau detak tak terklasifikasi dibuang karena
        # kelas F dan Q berada di luar lima kelas target.
        if hitung.get(policy.normal_class, 0) == len(detak):
            if not normal_terkonfirmasi:
                return Alasan.RITME_NORMAL_TAK_TERKONFIRMASI
            return LabeledWindow(
                database=rec.database,
                record=rec.record,
                start=start,
                end=end,
                label=policy.normal_class,
                source=Sumber.RITME_DAN_DETAK if ritme_tersedia else Sumber.DETAK,
                rhythm_coverage=cakupan_normal,
                n_beats=len(detak),
            )

        return Alasan.DETAK_TIDAK_KONSISTEN

    # 5. Tanpa anotasi detak, satu-satunya label yang tersisa adalah ritme normal.
    if cakupan_normal >= policy.min_rhythm_coverage:
        return LabeledWindow(
            database=rec.database,
            record=rec.record,
            start=start,
            end=end,
            label=policy.normal_class,
            source=Sumber.RITME,
            rhythm_coverage=cakupan_normal,
            n_beats=0,
        )

    return Alasan.TANPA_LABEL


def label_record(
    rec: RecordAnnotations,
    window_seconds: float,
    overlap: float,
    policy: LabelingPolicy,
) -> LabelingReport:
    """Melabeli seluruh jendela kandidat satu rekaman."""
    report = LabelingReport()
    if not rec.fs:
        return report

    window_samples = int(round(window_seconds * rec.fs))

    for start, end in iter_window_bounds(rec.n_samples, window_samples, overlap):
        report.n_candidates += 1
        hasil = label_window(rec, start, end, policy)
        if isinstance(hasil, LabeledWindow):
            report.add_window(hasil)
        else:
            report.add_rejection(rec.database, hasil)

    return report


def label_records(
    records: Sequence[RecordAnnotations],
    window_seconds: float,
    overlap: float,
    policy: LabelingPolicy,
) -> LabelingReport:
    """Melabeli beberapa rekaman dan menggabungkan rekapnya."""
    total = LabelingReport()
    for rec in records:
        total.merge(label_record(rec, window_seconds, overlap, policy))
    return total
