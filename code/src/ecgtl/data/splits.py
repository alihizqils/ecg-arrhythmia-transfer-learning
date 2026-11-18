"""Pembagian data pada level pasien.

Dua tingkat pembagian dipakai, sesuai rancangan proposal:

    Hold-Out 80/20              menyisihkan sebagian pasien untuk pengujian akhir
    Cross-validation K=5        di dalam 80 persen data latih

Keduanya memakai `StratifiedGroupKFold`, bukan `GroupKFold` biasa. Alasannya
bukan preferensi: kelas S dan V hanya dapat bersumber dari mitdb dan hanya
tersebar di 24 serta 36 rekaman, sehingga pembagian yang tidak
memperhatikan sebaran kelas berpeluang menghasilkan lipatan tanpa kelas S sama
sekali. Pada lipatan seperti itu sensitivitas dan F1 kelas S tidak terdefinisi,
dan rerata makro menjadi tidak dapat dibandingkan antar lipatan.

Jaminan utama `GroupKFold` yang menjadi alasan proposal memilihnya, yaitu tidak
ada pasien yang terbelah antar partisi, tetap dipertahankan sepenuhnya.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LOGGER = logging.getLogger(__name__)


def patient_id(
    database: str, record: str, aliases: Mapping[str, Mapping[str, str]] | None = None
) -> str:
    """Identitas pasien untuk satu rekaman.

    Secara default satu rekaman dianggap satu pasien. Pemetaan `aliases`
    menggabungkan rekaman yang diketahui berasal dari subjek yang sama.
    """
    if aliases:
        per_db = aliases.get(database) or {}
        alias = per_db.get(str(record))
        if alias:
            return str(alias)
    return f"{database}/{record}"


def add_patient_ids(
    manifest: pd.DataFrame, aliases: Mapping[str, Mapping[str, str]] | None = None
) -> pd.DataFrame:
    """Menambahkan kolom `patient_id` ke manifest."""
    hasil = manifest.copy()
    hasil["patient_id"] = [
        patient_id(db, rec, aliases)
        for db, rec in zip(hasil["database"].astype(str), hasil["record"].astype(str))
    ]
    return hasil


@dataclass
class SplitResult:
    """Hasil pembagian data lengkap."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    n_folds: int = 0

    @property
    def train_patients(self) -> set[str]:
        return set(self._train_patients)

    @property
    def test_patients(self) -> set[str]:
        return set(self._test_patients)

    def attach_patients(self, patients: Sequence[str]) -> None:
        patients = np.asarray(patients)
        self._train_patients = patients[self.train_idx].tolist()
        self._test_patients = patients[self.test_idx].tolist()


def _stratified_group_split(
    labels: Sequence, groups: Sequence, n_splits: int, seed: int, fold: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Satu pembagian dari StratifiedGroupKFold, dipakai sebagai hold-out."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    dummy = np.zeros((len(labels), 1))

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (besar_idx, kecil_idx) in enumerate(splitter.split(dummy, labels, groups)):
        if i == fold:
            return besar_idx, kecil_idx
    raise ValueError(f"lipatan {fold} tidak tersedia dari {n_splits} pembagian")


def make_splits(
    manifest: pd.DataFrame,
    test_size: float = 0.2,
    n_folds: int = 5,
    seed: int = 42,
    label_column: str = "label",
    group_column: str = "patient_id",
) -> SplitResult:
    """Membuat hold-out per pasien lalu lipatan validasi silang di dalamnya.

    Hold-out dihasilkan dengan mengambil satu lipatan dari StratifiedGroupKFold
    berukuran 1/test_size. Cara ini memberi hold-out yang sekaligus menjaga
    keutuhan pasien dan sebaran kelas, tanpa perlu mekanisme terpisah.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size harus di (0, 1), diterima {test_size}")
    if n_folds < 2:
        raise ValueError(f"n_folds minimal 2, diterima {n_folds}")

    labels = manifest[label_column].to_numpy()
    groups = manifest[group_column].to_numpy()

    n_splits_holdout = int(round(1.0 / test_size))
    if n_splits_holdout < 2:
        raise ValueError(f"test_size {test_size} terlalu besar untuk pembagian bergrup")

    train_idx, test_idx = _stratified_group_split(labels, groups, n_splits_holdout, seed)

    # Lipatan validasi silang dibuat hanya di dalam data latih, sehingga data uji
    # tidak pernah tersentuh sampai evaluasi akhir.
    labels_train = labels[train_idx]
    groups_train = groups[train_idx]
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    dummy = np.zeros((len(labels_train), 1))

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for tr_lokal, va_lokal in splitter.split(dummy, labels_train, groups_train):
        # Indeks dipetakan kembali ke penomoran manifest global agar pemanggil
        # tidak perlu mengingat dua sistem penomoran.
        folds.append((train_idx[tr_lokal], train_idx[va_lokal]))

    hasil = SplitResult(train_idx=train_idx, test_idx=test_idx, folds=folds, n_folds=n_folds)
    hasil.attach_patients(groups)
    return hasil


def verify_splits(
    manifest: pd.DataFrame,
    splits: SplitResult,
    classes: Sequence[str],
    label_column: str = "label",
    group_column: str = "patient_id",
) -> list[str]:
    """Memeriksa jaminan yang harus dipenuhi pembagian data.

    Mengembalikan daftar masalah. Daftar kosong berarti seluruh jaminan terpenuhi.
    """
    masalah: list[str] = []
    labels = manifest[label_column].to_numpy()
    groups = manifest[group_column].to_numpy()

    pasien_latih = set(groups[splits.train_idx])
    pasien_uji = set(groups[splits.test_idx])
    bocor = pasien_latih & pasien_uji
    if bocor:
        masalah.append(f"pasien muncul di data latih dan uji: {sorted(bocor)}")

    if len(splits.train_idx) + len(splits.test_idx) != len(manifest):
        masalah.append("gabungan latih dan uji tidak mencakup seluruh manifest")

    if set(splits.train_idx) & set(splits.test_idx):
        masalah.append("indeks latih dan uji tumpang tindih")

    kelas_uji = set(labels[splits.test_idx])
    kurang_uji = set(classes) - kelas_uji
    if kurang_uji:
        masalah.append(f"data uji tidak memuat kelas {sorted(kurang_uji)}")

    for i, (tr, va) in enumerate(splits.folds, start=1):
        pasien_tr = set(groups[tr])
        pasien_va = set(groups[va])
        if pasien_tr & pasien_va:
            masalah.append(f"lipatan {i}: pasien bocor antar latih dan validasi")
        if set(tr) & set(va):
            masalah.append(f"lipatan {i}: indeks latih dan validasi tumpang tindih")

        kurang_tr = set(classes) - set(labels[tr])
        if kurang_tr:
            masalah.append(f"lipatan {i}: sub-latih tidak memuat kelas {sorted(kurang_tr)}")
        kurang_va = set(classes) - set(labels[va])
        if kurang_va:
            masalah.append(f"lipatan {i}: validasi tidak memuat kelas {sorted(kurang_va)}")

        if set(va) & set(splits.test_idx):
            masalah.append(f"lipatan {i}: validasi menyentuh data uji")

    return masalah


def split_report(
    manifest: pd.DataFrame,
    splits: SplitResult,
    classes: Sequence[str],
    label_column: str = "label",
    group_column: str = "patient_id",
) -> pd.DataFrame:
    """Tabel sebaran kelas dan jumlah pasien untuk setiap partisi."""
    baris: list[dict] = []

    def _tambah(nama: str, idx: np.ndarray) -> None:
        sub = manifest.iloc[idx]
        entri: dict = {
            "partisi": nama,
            "jendela": len(sub),
            "pasien": sub[group_column].nunique(),
        }
        hitung = sub[label_column].value_counts()
        for kelas in classes:
            entri[kelas] = int(hitung.get(kelas, 0))
        baris.append(entri)

    _tambah("latih (80%)", splits.train_idx)
    _tambah("uji (20%)", splits.test_idx)
    for i, (tr, va) in enumerate(splits.folds, start=1):
        _tambah(f"lipatan {i} latih", tr)
        _tambah(f"lipatan {i} validasi", va)

    return pd.DataFrame(baris)


def patient_assignment(
    manifest: pd.DataFrame,
    splits: SplitResult,
    group_column: str = "patient_id",
) -> pd.DataFrame:
    """Tabel penempatan setiap pasien, untuk lampiran laporan dan audit."""
    groups = manifest[group_column].to_numpy()
    penempatan: dict[str, dict] = {}

    for pasien in sorted(set(groups)):
        penempatan[pasien] = {"patient_id": pasien, "partisi": "?", "lipatan_validasi": ""}

    for pasien in set(groups[splits.train_idx]):
        penempatan[pasien]["partisi"] = "latih"
    for pasien in set(groups[splits.test_idx]):
        penempatan[pasien]["partisi"] = "uji"

    for i, (_, va) in enumerate(splits.folds, start=1):
        for pasien in set(groups[va]):
            penempatan[pasien]["lipatan_validasi"] = str(i)

    return pd.DataFrame(list(penempatan.values()))
