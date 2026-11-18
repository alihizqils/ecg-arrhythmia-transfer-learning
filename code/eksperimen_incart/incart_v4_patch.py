"""Penyesuaian runtime pipeline V4 agar rekaman INCART (12 lead) terbaca; rekaman V4 tidak berubah perilakunya.

Dipakai uji lintas database (incart_external_test.py). Kode V4 dan hash metodenya tidak diubah.
1. Lead kedua (Revisi 11): loader V4 mensyaratkan tepat 2 lead. Untuk INCART dibaca lead II (utama) dan V1 (kedua).
2. Ritme lokal (Revisi 14/15): loader fitur_ns diganti untuk INCART saja dengan load_record_incart (incart_features.py),
   lewat fungsi worker modul ini agar berlaku juga di proses worker.
3. label_valid: incartdb beranotasi jenis beat, jadi window N INCART sah (aturan sama dengan mitdb/svdb).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import wfdb

HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "eksperimen_v4_svdb"
for path in [str(V4 / "fase_e_ns"), str(V4), str(HERE.parent / "src"), str(HERE)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import fitur_ns as NS  # noqa: E402

_ORIGINAL_LOAD = NS.load_record
import incart_features  # noqa: E402  (mengganti NS.load_record; dikembalikan lewat dispatcher di bawah)


def _load_record(database, record, channel, fs, annotations=True):
    if database == "incartdb":
        return incart_features.load_record_incart(database, record, channel, fs, annotations)
    return _ORIGINAL_LOAD(database, record, channel, fs, annotations)


NS.load_record = _load_record


def record_features(args):
    """Fungsi asli v4_local_rhythm; dipanggil lewat modul ini agar worker ikut memasang _load_record."""
    return _ORIGINAL_RF(args)


def _rdrecord(path, *args, **kwargs):
    if "incartdb" in str(path).replace("\\", "/").split("/"):
        return wfdb.rdrecord(path, *args, channel_names=["II", "V1"], **kwargs)
    return wfdb.rdrecord(path, *args, **kwargs)


import v4_local_rhythm  # noqa: E402
_ORIGINAL_RF = v4_local_rhythm.record_features


def install():
    import v4_pipeline as v4
    import v4_second_lead
    v4_second_lead.wfdb = types.SimpleNamespace(rdrecord=_rdrecord)
    v4_local_rhythm.record_features = record_features
    rule = v4.METHOD["label_validity"]
    if "incartdb" not in rule["beat_annotated_databases"]:
        rule["beat_annotated_databases"] = [*rule["beat_annotated_databases"], "incartdb"]
