"""Pengurai anotasi WFDB menjadi episode ritme dan detak berkelas AAMI.

Ketiga basis data memakai struktur anotasi yang berbeda, dan perbedaan itu
menentukan kelas apa yang bisa diambil dari mana:

    mitdb   anotasi per-detak (~110.000) DAN penanda perubahan ritme
    afdb    hanya penanda perubahan ritme: (AFIB, (AFL, (J, (N
    vfdb    hanya penanda perubahan ritme: (VT, (VF, (VFL, (NOISE, dan lain-lain

Konsekuensinya kelas S dan V hanya dapat bersumber dari mitdb.

Dua kehalusan yang ditangani modul ini:

1.  Penanda ritme WFDB diletakkan di awal episode, tanpa penanda akhir. Episode
    karenanya berakhir pada penanda berikutnya atau di ujung rekaman.
2.  Menurut dokumentasi vfdb, ritme sebelumnya berlanjut selama episode (NOISE
    dan baru berakhir pada anotasi setelahnya. Jadi (NOISE adalah interupsi,
    bukan pergantian ritme, dan harus diperlakukan sebagai lapisan terpisah.
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import wfdb

LOGGER = logging.getLogger(__name__)

# Simbol anotasi WFDB yang tidak menandai kompleks QRS. Didaftarkan eksplisit
# agar simbol tak terduga dapat dilaporkan, bukan diam-diam dianggap detak.
#   +  perubahan ritme (label ada di aux_note)
#   ~  perubahan kualitas sinyal
#   |  artefak mirip QRS terisolasi
#   "  komentar (di mitdb memuat MISSB, TS, PSE)
#   x  gelombang P tidak terkonduksi (APB terblokir), tidak ada QRS
#   !  gelombang flutter ventrikel
#   [  awal flutter/fibrilasi ventrikel
#   ]  akhir flutter/fibrilasi ventrikel
#   =  spike pacu jantung, ?  detak belum terklasifikasi saat pembelajaran
NON_BEAT_SYMBOLS: frozenset[str] = frozenset("+~|\"x![]=?")


@dataclass(frozen=True)
class Interval:
    """Selang sampel setengah terbuka [start, end) dengan label."""

    start: int
    end: int
    label: str

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"selang tidak valid: [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap(self, start: int, end: int) -> int:
        """Panjang tumpang tindih dengan selang [start, end)."""
        return max(0, min(self.end, end) - max(self.start, start))


@dataclass(frozen=True)
class Beat:
    """Satu anotasi detak."""

    sample: int
    symbol: str
    aami: str | None


@dataclass
class RecordAnnotations:
    """Seluruh anotasi satu rekaman dalam bentuk siap dipakai pelabelan."""

    database: str
    record: str
    fs: float
    n_samples: int
    rhythms: list[Interval] = field(default_factory=list)
    noise: list[Interval] = field(default_factory=list)
    beats: list[Beat] = field(default_factory=list)
    unknown_symbols: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._beat_samples = [b.sample for b in self.beats]
        # Episode ritme saling tidak tumpang tindih dan terurut, sehingga akhir
        # episode juga terurut dan dapat dicari secara biner. Ini penting karena
        # satu rekaman afdb 10 jam menghasilkan ribuan jendela, dan pencarian
        # linear per jendela membuat pembangunan dataset lambat tanpa alasan.
        self._rhythm_ends = [r.end for r in self.rhythms]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs if self.fs else 0.0

    def beats_in(self, start: int, end: int) -> list[Beat]:
        """Detak pada selang [start, end), memakai pencarian biner."""
        kiri = bisect_left(self._beat_samples, start)
        kanan = bisect_left(self._beat_samples, end)
        return self.beats[kiri:kanan]

    def rhythms_in(self, start: int, end: int) -> list[Interval]:
        """Episode ritme yang bersinggungan dengan selang [start, end)."""
        awal = bisect_right(self._rhythm_ends, start)
        hasil: list[Interval] = []
        for episode in self.rhythms[awal:]:
            if episode.start >= end:
                break
            if episode.overlap(start, end) > 0:
                hasil.append(episode)
        return hasil

    def noise_overlap(self, start: int, end: int) -> int:
        """Total sampel pada [start, end) yang tertandai derau."""
        return sum(n.overlap(start, end) for n in self.noise)


def clean_aux_note(note: object) -> str:
    """Membersihkan aux_note dari byte nul dan spasi.

    wfdb mengembalikan aux_note dengan padding byte nul, dan sebagian anotasi
    memuat teks non-ritme seperti "MISSB" sehingga perlu dibedakan.
    """
    if note is None:
        return ""
    return str(note).replace("\x00", "").strip()


def parse_rhythm_label(note: object) -> str | None:
    """Mengambil label ritme dari aux_note, atau None bila bukan penanda ritme.

    Penanda ritme WFDB selalu dimulai tanda kurung buka, mis. "(AFIB".
    """
    bersih = clean_aux_note(note)
    if not bersih.startswith("("):
        return None
    label = bersih[1:].strip()
    return label.upper() or None


def build_beat_symbol_map(beat_aami: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Membalik pemetaan kelas AAMI menjadi simbol -> kelas.

    Memunculkan ValueError bila satu simbol terdaftar pada lebih dari satu
    kelas, karena kondisi itu membuat pelabelan tidak deterministik.
    """
    hasil: dict[str, str] = {}
    for kelas, simbol_list in beat_aami.items():
        for simbol in simbol_list:
            simbol = str(simbol)
            if simbol in hasil and hasil[simbol] != kelas:
                raise ValueError(
                    f"simbol {simbol!r} terdaftar di kelas {hasil[simbol]!r} dan {kelas!r}"
                )
            hasil[simbol] = str(kelas)
    return hasil


def extract_rhythm_episodes(
    samples: Sequence[int],
    aux_notes: Sequence[object],
    n_samples: int,
    interruption_labels: Iterable[str] = ("NOISE",),
) -> tuple[list[Interval], list[Interval]]:
    """Mengubah penanda perubahan ritme menjadi selang berlabel.

    Mengembalikan pasangan (episode_ritme, episode_derau). Label interupsi
    dikeluarkan dari daftar ritme dan dikembalikan terpisah; ritme sebelumnya
    dilanjutkan melewati interupsi tersebut, sesuai dokumentasi vfdb.

    Episode ritme yang berdampingan dan berlabel sama digabung, sehingga
    interupsi derau di tengah satu ritme tidak memecahnya menjadi dua.
    """
    interupsi = {str(label).upper() for label in interruption_labels}

    penanda: list[tuple[int, str]] = []
    for sample, note in zip(samples, aux_notes or []):
        label = parse_rhythm_label(note)
        if label is not None:
            penanda.append((int(sample), label))

    if not penanda:
        return [], []

    penanda.sort(key=lambda p: p[0])

    mentah: list[Interval] = []
    for idx, (mulai, label) in enumerate(penanda):
        akhir = penanda[idx + 1][0] if idx + 1 < len(penanda) else n_samples
        if akhir <= mulai:
            continue
        mentah.append(Interval(start=mulai, end=akhir, label=label))

    derau = [seg for seg in mentah if seg.label in interupsi]

    # Bangun garis waktu ritme: segmen interupsi mewarisi ritme sebelumnya.
    ritme_terurai: list[Interval] = []
    ritme_terakhir: str | None = None
    for seg in mentah:
        if seg.label in interupsi:
            if ritme_terakhir is None:
                continue  # derau sebelum ritme pertama diketahui, tidak dapat diwarisi
            ritme_terurai.append(Interval(seg.start, seg.end, ritme_terakhir))
        else:
            ritme_terakhir = seg.label
            ritme_terurai.append(seg)

    # Gabungkan segmen berdampingan berlabel sama.
    digabung: list[Interval] = []
    for seg in ritme_terurai:
        if digabung and digabung[-1].label == seg.label and digabung[-1].end == seg.start:
            sebelumnya = digabung.pop()
            digabung.append(Interval(sebelumnya.start, seg.end, seg.label))
        else:
            digabung.append(seg)

    return digabung, derau


def extract_beats(
    samples: Sequence[int],
    symbols: Sequence[str],
    symbol_to_aami: Mapping[str, str],
) -> tuple[list[Beat], dict[str, int]]:
    """Menyaring anotasi detak dan memetakannya ke kelas AAMI.

    Mengembalikan pasangan (daftar_detak, simbol_tak_dikenal). Simbol yang tidak
    terdaftar sebagai detak maupun sebagai non-detak yang dikenal dilaporkan,
    agar tidak ada anotasi yang hilang tanpa jejak.
    """
    detak: list[Beat] = []
    tak_dikenal: dict[str, int] = {}

    for sample, symbol in zip(samples, symbols):
        symbol = str(symbol)
        if symbol in symbol_to_aami:
            detak.append(
                Beat(sample=int(sample), symbol=symbol, aami=symbol_to_aami[symbol])
            )
        elif symbol not in NON_BEAT_SYMBOLS:
            tak_dikenal[symbol] = tak_dikenal.get(symbol, 0) + 1

    return detak, tak_dikenal


def load_record_annotations(
    database: str,
    directory: Path,
    record: str,
    beat_aami: Mapping[str, Sequence[str]],
    annotation_ext: str = "atr",
    interruption_labels: Iterable[str] = ("NOISE",),
    read_beats: bool = True,
) -> RecordAnnotations:
    """Memuat dan mengurai anotasi satu rekaman.

    Args:
        read_beats: bila False, anotasi detak diabaikan. Dipakai untuk afdb yang
            anotasi detaknya tidak dibedakan jenisnya sehingga semuanya tampak
            normal dan akan menyesatkan bila dipakai melabeli kelas S atau V.
    """
    directory = Path(directory)
    path = directory / record

    header = wfdb.rdheader(str(path))
    n_samples = int(header.sig_len or 0)
    fs = float(header.fs or 0.0)

    ann = wfdb.rdann(str(path), annotation_ext)

    ritme, derau = extract_rhythm_episodes(
        ann.sample, ann.aux_note, n_samples, interruption_labels
    )

    if read_beats:
        detak, tak_dikenal = extract_beats(
            ann.sample, ann.symbol, build_beat_symbol_map(beat_aami)
        )
    else:
        detak, tak_dikenal = [], {}

    if tak_dikenal:
        LOGGER.debug(
            "%s/%s: simbol anotasi tak dikenal %s", database, record, tak_dikenal
        )

    return RecordAnnotations(
        database=database,
        record=record,
        fs=fs,
        n_samples=n_samples,
        rhythms=ritme,
        noise=derau,
        beats=detak,
        unknown_symbols=tak_dikenal,
    )


def summarize_rhythms(annotations: Iterable[RecordAnnotations]) -> "pd.DataFrame":
    """Tabel episode ritme per basis data dan label.

    Kolom durasi memakai satuan jam agar perbandingan antar basis data langsung
    terbaca, karena afdb berdurasi ratusan jam sementara mitdb hanya puluhan.
    """
    import pandas as pd

    baris: list[dict[str, object]] = []
    for rec in annotations:
        for episode in rec.rhythms:
            baris.append(
                {
                    "database": rec.database,
                    "record": rec.record,
                    "label": episode.label,
                    "durasi_s": episode.length / rec.fs if rec.fs else 0.0,
                }
            )

    if not baris:
        return pd.DataFrame(
            columns=[
                "database",
                "label",
                "episode",
                "durasi_jam",
                "durasi_s",
                "mean_durasi_s",
                "rekaman",
            ]
        )

    df = pd.DataFrame(baris)
    ringkas = (
        df.groupby(["database", "label"])
        .agg(
            episode=("durasi_s", "size"),
            durasi_jam=("durasi_s", lambda s: round(float(s.sum()) / 3600, 6)),
            durasi_s=("durasi_s", lambda s: round(float(s.sum()), 2)),
            mean_durasi_s=("durasi_s", lambda s: round(float(s.mean()), 2)),
            rekaman=("record", "nunique"),
        )
        .reset_index()
        .sort_values(["database", "durasi_jam"], ascending=[True, False])
    )
    return ringkas


def summarize_beats(annotations: Iterable[RecordAnnotations]) -> "pd.DataFrame":
    """Tabel jumlah detak per basis data, rekaman, dan kelas AAMI."""
    import pandas as pd

    baris: list[dict[str, object]] = []
    for rec in annotations:
        hitung: dict[str, int] = {}
        for beat in rec.beats:
            if beat.aami:
                hitung[beat.aami] = hitung.get(beat.aami, 0) + 1
        if not hitung:
            continue
        baris.append({"database": rec.database, "record": rec.record, **hitung})

    if not baris:
        return pd.DataFrame(columns=["database", "record"])

    df = pd.DataFrame(baris).fillna(0)
    kolom_kelas = [c for c in df.columns if c not in ("database", "record")]
    df[kolom_kelas] = df[kolom_kelas].astype(int)
    return df.sort_values(["database", "record"]).reset_index(drop=True)
