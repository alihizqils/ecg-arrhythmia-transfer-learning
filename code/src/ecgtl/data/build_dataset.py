"""Pembangunan dataset citra: pelabelan, capping, perenderan, dan penyimpanan.

Proses berjalan dalam dua lintasan agar sinyal hanya dibaca sekali:

    Lintasan 1  hanya anotasi. Melabeli seluruh jendela kandidat, menerapkan cap,
                lalu menentukan indeks akhir setiap jendela terpilih.
    Lintasan 2  per rekaman, sinyal dibaca dan diproses satu kali, kemudian
                seluruh jendela terpilih rekaman itu dirender ke memmap.

Anotasi jauh lebih murah dibaca daripada sinyal, sehingga memisahkan keduanya
menghindari pembacaan sinyal ganda tanpa perlu menyimpan seluruh sinyal di
memori.

Keluaran satu konfigurasi dataset:

    images.npy      array uint8 berbentuk (N, tinggi, lebar), ditulis lewat memmap
    manifest.csv    satu baris per citra: sumber, pasien, kelas, posisi jendela
    meta.json       konfigurasi efektif, bentuk array, urutan kelas
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import wfdb

from ..config import Config
from .annotations import RecordAnnotations, load_record_annotations
from .download import has_signal_files
from .inventory import select_channel
from .labeling import LabeledWindow, LabelingPolicy, label_record
from .rendering import render_waveform
from .segmentation import is_flat, minmax_normalize, preprocess_record

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetSpec:
    """Parameter yang menentukan bentuk satu konfigurasi dataset."""

    window_seconds: float
    overlap: float
    width: int
    height: int
    target_fs: float
    line_width: int = 1
    background: int = 255
    foreground: int = 0
    cap_per_record_per_class: int = 200

    @classmethod
    def from_config(cls, cfg: Config) -> "DatasetSpec":
        image = cfg.require("image")
        return cls(
            window_seconds=float(cfg.require("windowing.duration_s")),
            overlap=float(cfg.require("windowing.overlap")),
            width=int(image["width"]),
            height=int(image["height"]),
            target_fs=float(cfg.require("preprocessing.target_fs")),
            line_width=int(image.get("line_width", 1)),
            background=int(image.get("background", 255)),
            foreground=int(image.get("foreground", 0)),
            cap_per_record_per_class=int(
                cfg.require("sampling.max_windows_per_record_per_class")
            ),
        )

    @property
    def slug(self) -> str:
        """Nama direktori yang mendeskripsikan konfigurasi secara ringkas."""
        durasi = f"{self.window_seconds:g}".replace(".", "p")
        return f"w{durasi}s_{self.width}x{self.height}_ov{int(self.overlap * 100)}"


@dataclass
class BuildResult:
    """Rekap pembangunan satu konfigurasi dataset."""

    spec: DatasetSpec
    output_dir: Path
    n_images: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    n_candidates: int = 0
    n_labeled: int = 0
    n_capped_away: int = 0
    n_flat_skipped: int = 0
    rejections: dict[str, int] = field(default_factory=dict)


def select_capped_windows(
    windows: Sequence[LabeledWindow], cap: int
) -> list[LabeledWindow]:
    """Menerapkan batas jumlah jendela per rekaman per kelas.

    Pemilihan menyebar merata sepanjang rekaman, bukan mengambil jendela pertama.
    Mengambil yang pertama akan membatasi cakupan hanya pada menit-menit awal
    rekaman, sehingga variasi morfologi dan derau di bagian lain hilang. Pada
    rekaman afdb berdurasi 10 jam, perbedaan ini besar.

    Pemilihan bersifat deterministik: tidak ada keacakan, sehingga dataset dapat
    dibangun ulang persis sama.
    """
    if cap <= 0:
        raise ValueError("cap harus positif")

    per_kelompok: dict[tuple[str, str, str], list[LabeledWindow]] = defaultdict(list)
    for w in windows:
        per_kelompok[(w.database, w.record, w.label)].append(w)

    terpilih: list[LabeledWindow] = []
    for kunci in sorted(per_kelompok):
        daftar = sorted(per_kelompok[kunci], key=lambda w: w.start)
        if len(daftar) <= cap:
            terpilih.extend(daftar)
            continue
        indeks = np.unique(np.linspace(0, len(daftar) - 1, num=cap).round().astype(int))
        terpilih.extend(daftar[i] for i in indeks)

    return terpilih


def load_channel_signal(
    directory: Path, record: str, channel_preference: Sequence[str]
) -> tuple[np.ndarray, float, str]:
    """Membaca satu kanal sinyal beserta laju sampling dan nama kanalnya."""
    header = wfdb.rdheader(str(directory / record))
    pilihan = select_channel(list(header.sig_name or []), channel_preference)
    rekaman = wfdb.rdrecord(str(directory / record), channels=[pilihan.index])
    sinyal = np.asarray(rekaman.p_signal[:, 0], dtype=np.float64)
    # Sebagian rekaman afdb memuat blok data yang hilang; wfdb mengembalikannya
    # sebagai NaN. Nilai tersebut diganti nol agar penapis tidak menyebarkan NaN
    # ke seluruh rekaman.
    if np.isnan(sinyal).any():
        jumlah = int(np.isnan(sinyal).sum())
        LOGGER.debug("%s: %d sampel NaN diganti nol", record, jumlah)
        sinyal = np.nan_to_num(sinyal, nan=0.0)
    return sinyal, float(header.fs), pilihan.name


def collect_labeled_windows(
    cfg: Config,
    raw_root: Path,
    spec: DatasetSpec,
    policy: LabelingPolicy,
    records_filter: dict[str, Sequence[str]] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[list[LabeledWindow], dict[str, int], int, int]:
    """Lintasan pertama: melabeli seluruh jendela kandidat dari anotasi saja.

    Returns:
        (jendela_berlabel, rekap_penolakan, jumlah_kandidat, jumlah_berlabel)
    """
    log = logger or LOGGER
    beat_aami = cfg.require("labeling.beat_aami")
    interupsi = cfg.require("labeling.rhythm_interruption")

    semua: list[LabeledWindow] = []
    penolakan: dict[str, int] = defaultdict(int)
    kandidat = 0

    for db, db_spec in cfg.require("databases").items():
        direktori = raw_root / db
        if not direktori.is_dir():
            log.warning("%s tidak ditemukan, dilewati", direktori)
            continue

        baca_detak = bool(db_spec.get("has_beat_annotations", False))
        daftar = sorted(p.stem for p in direktori.glob("*.hea"))
        if records_filter and db in records_filter:
            diminta = set(str(r) for r in records_filter[db])
            daftar = [r for r in daftar if r in diminta]

        for record in daftar:
            if not has_signal_files(direktori, record):
                continue
            try:
                anotasi: RecordAnnotations = load_record_annotations(
                    db,
                    direktori,
                    record,
                    beat_aami=beat_aami,
                    annotation_ext=db_spec.get("annotation_ext", "atr"),
                    interruption_labels=interupsi,
                    read_beats=baca_detak,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("%s/%s gagal diurai: %s", db, record, exc)
                continue

            laporan = label_record(anotasi, spec.window_seconds, spec.overlap, policy)
            semua.extend(laporan.windows)
            kandidat += laporan.n_candidates
            for (_, alasan), n in laporan.rejections.items():
                penolakan[alasan] += n

    return semua, dict(penolakan), kandidat, len(semua)


def build_dataset(
    cfg: Config,
    raw_root: Path,
    output_root: Path,
    spec: DatasetSpec | None = None,
    records_filter: dict[str, Sequence[str]] | None = None,
    logger: logging.Logger | None = None,
    show_progress: bool = True,
) -> BuildResult:
    """Membangun satu konfigurasi dataset citra dari data mentah."""
    log = logger or LOGGER
    spec = spec or DatasetSpec.from_config(cfg)
    policy = LabelingPolicy.from_config(cfg)
    classes: list[str] = list(cfg.require("classes"))
    channel_preference = cfg.require("windowing.channel_preference")
    bandpass = cfg.require("preprocessing.bandpass")
    aliases = cfg.get("split.patient_aliases", {})

    output_dir = Path(output_root) / spec.slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Lintasan 1: pelabelan dan capping ---
    log.info("lintasan 1: melabeli jendela kandidat (%s)", spec.slug)
    berlabel, penolakan, kandidat, n_berlabel = collect_labeled_windows(
        cfg, raw_root, spec, policy, records_filter, log
    )
    if not berlabel:
        raise RuntimeError("tidak ada jendela berlabel; dataset tidak dapat dibangun")

    terpilih = select_capped_windows(berlabel, spec.cap_per_record_per_class)
    terpilih.sort(key=lambda w: (w.database, w.record, w.start))
    log.info(
        "kandidat %d, berlabel %d, terpilih setelah cap %d",
        kandidat, n_berlabel, len(terpilih),
    )

    # --- Lintasan 2: perenderan ---
    from .splits import patient_id

    path_images = output_dir / "images.npy"
    bentuk = (len(terpilih), spec.height, spec.width)
    images = np.lib.format.open_memmap(
        path_images, mode="w+", dtype=np.uint8, shape=bentuk
    )

    per_record: dict[tuple[str, str], list[tuple[int, LabeledWindow]]] = defaultdict(list)
    for i, w in enumerate(terpilih):
        per_record[(w.database, w.record)].append((i, w))

    baris_manifest: list[dict] = []
    n_flat = 0
    iterator: Iterable = sorted(per_record)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=f"render {spec.slug}", unit="rekaman")
        except ImportError:  # pragma: no cover
            pass

    for db, record in iterator:
        direktori = raw_root / db
        try:
            sinyal, fs_asli, nama_kanal = load_channel_signal(
                direktori, record, channel_preference
            )
            pre = preprocess_record(
                sinyal, fs_asli, spec.target_fs, bandpass, db, record, nama_kanal
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("%s/%s gagal diproses, seluruh jendelanya dilewati: %s", db, record, exc)
            continue

        for indeks, w in per_record[(db, record)]:
            mulai, akhir = pre.window_bounds_at_target_fs(w.start, spec.window_seconds)
            potongan = pre.extract(mulai, akhir)
            if potongan is None:
                continue
            datar = is_flat(potongan)
            if datar:
                n_flat += 1
            citra = render_waveform(
                minmax_normalize(potongan),
                width=spec.width,
                height=spec.height,
                line_width=spec.line_width,
                background=spec.background,
                foreground=spec.foreground,
            )
            images[indeks] = citra
            baris_manifest.append(
                {
                    "index": indeks,
                    "database": db,
                    "record": record,
                    "patient_id": patient_id(db, record, aliases),
                    "channel": nama_kanal,
                    "label": w.label,
                    "label_idx": classes.index(w.label),
                    "source": w.source.value,
                    "start_original": w.start,
                    "start_target_fs": mulai,
                    "n_beats": w.n_beats,
                    "rhythm_coverage": round(float(w.rhythm_coverage), 4),
                    "flat": bool(datar),
                }
            )

    images.flush()
    del images

    manifest = pd.DataFrame(baris_manifest).sort_values("index").reset_index(drop=True)

    # Jendela yang gagal dirender menyisakan baris kosong di array. Baris tersebut
    # dibuang dan array dipadatkan agar manifest dan array selalu sejajar.
    if len(manifest) != bentuk[0]:
        log.warning(
            "%d dari %d jendela gagal dirender, array dipadatkan",
            bentuk[0] - len(manifest),
            bentuk[0],
        )
        lama = np.load(path_images, mmap_mode="r")
        baru = np.lib.format.open_memmap(
            output_dir / "images_tmp.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(manifest), spec.height, spec.width),
        )
        for baru_i, lama_i in enumerate(manifest["index"].to_numpy()):
            baru[baru_i] = lama[lama_i]
        baru.flush()
        del baru, lama
        path_images.unlink()
        (output_dir / "images_tmp.npy").rename(path_images)
        manifest["index"] = np.arange(len(manifest))

    manifest.to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8")

    per_class = manifest["label"].value_counts().to_dict()
    meta = {
        "spec": asdict(spec),
        "slug": spec.slug,
        "classes": classes,
        "n_images": len(manifest),
        "image_shape": [spec.height, spec.width],
        "dtype": "uint8",
        "per_class": {k: int(per_class.get(k, 0)) for k in classes},
        "n_candidates": kandidat,
        "n_labeled": n_berlabel,
        "rejections": penolakan,
        "preprocessing": cfg.require("preprocessing"),
        "labeling": cfg.require("labeling"),
        "sampling": cfg.require("sampling"),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    return BuildResult(
        spec=spec,
        output_dir=output_dir,
        n_images=len(manifest),
        per_class={k: int(per_class.get(k, 0)) for k in classes},
        n_candidates=kandidat,
        n_labeled=n_berlabel,
        n_capped_away=n_berlabel - len(terpilih),
        n_flat_skipped=n_flat,
        rejections=penolakan,
    )


def load_dataset(output_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Memuat dataset yang sudah dibangun.

    Array dibuka dalam mode memmap baca-saja, sehingga dataset berukuran lebih
    dari satu gigabyte tidak perlu dimuat seluruhnya ke memori.
    """
    output_dir = Path(output_dir)
    images = np.load(output_dir / "images.npy", mmap_mode="r")
    manifest = pd.read_csv(output_dir / "manifest.csv", dtype={"record": str})
    meta = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))

    if len(manifest) != images.shape[0]:
        raise ValueError(
            f"manifest ({len(manifest)}) tidak sejajar dengan array citra ({images.shape[0]})"
        )
    return images, manifest, meta
