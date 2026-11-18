"""Pemuat konfigurasi YAML berlapis.

Satu berkas dasar (`configs/base.yaml`) memuat seluruh parameter default.
Berkas lain hanya menimpa sebagian kunci, sehingga tidak ada duplikasi nilai
antar berkas konfigurasi dan hanya ada satu sumber kebenaran.

Contoh:
    cfg = load_config("configs/base.yaml", overrides=["configs/windowing/w10_224.yaml"])
    cfg.get("preprocessing.bandpass.high_hz")   # 40.0
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

_MISSING = object()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Menggabungkan dua mapping secara rekursif.

    Nilai pada `override` menang. Sub-mapping digabung, bukan ditimpa utuh,
    sehingga berkas override hanya perlu menyebut kunci yang berubah. Nilai
    non-mapping (termasuk list) ditimpa utuh, karena penggabungan list secara
    implisit lebih sering menimbulkan kejutan daripada manfaat.
    """
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        existing = result.get(key, _MISSING)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_cli_override(item: str) -> tuple[str, Any]:
    """Mengurai override gaya baris perintah `a.b.c=nilai`.

    Nilai diurai sebagai YAML agar tipe data terjaga: `true` menjadi bool,
    `40` menjadi int, `40.0` menjadi float, `[1, 2]` menjadi list, dan teks
    biasa tetap menjadi str.
    """
    if "=" not in item:
        raise ValueError(f"override harus berbentuk kunci=nilai, diterima: {item!r}")
    key, raw_value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"kunci override kosong pada: {item!r}")
    return key, yaml.safe_load(raw_value)


def _set_by_path(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


class Config:
    """Pembungkus tipis di atas dict konfigurasi, dengan akses jalur bertitik."""

    def __init__(self, data: Mapping[str, Any]):
        self._data: dict[str, Any] = copy.deepcopy(dict(data))

    @property
    def data(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        """Mengambil nilai lewat jalur bertitik, mis. `training.lr.frozen`."""
        cursor: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                if default is _MISSING:
                    raise KeyError(f"kunci konfigurasi tidak ditemukan: {dotted_key!r}")
                return default
            cursor = cursor[part]
        return copy.deepcopy(cursor) if isinstance(cursor, (dict, list)) else cursor

    def require(self, dotted_key: str) -> Any:
        """Sama dengan `get` tanpa default: memunculkan KeyError bila absen."""
        return self.get(dotted_key)

    def __repr__(self) -> str:  # pragma: no cover - hanya untuk debugging
        return f"Config({json.dumps(self._data, indent=2, default=str)})"

    def to_yaml(self) -> str:
        return yaml.safe_dump(self._data, sort_keys=False, allow_unicode=True)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"berkas konfigurasi tidak ditemukan: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"konfigurasi harus berupa mapping di akarnya: {path}")
    return dict(loaded)


def load_config(
    base_path: str | Path,
    overrides: Iterable[str | Path] | None = None,
    cli_overrides: Iterable[str] | None = None,
) -> Config:
    """Memuat konfigurasi dasar lalu menimpanya berlapis.

    Urutan presedensi, dari terendah ke tertinggi: `base_path`, tiap berkas di
    `overrides` sesuai urutan, lalu `cli_overrides`.
    """
    merged = load_yaml(base_path)
    for override_path in overrides or []:
        merged = deep_merge(merged, load_yaml(override_path))
    for item in cli_overrides or []:
        key, value = parse_cli_override(item)
        _set_by_path(merged, key, value)
    return Config(merged)


def project_root() -> Path:
    """Akar proyek kode, yaitu direktori `code/`."""
    return Path(__file__).resolve().parents[2]


def resolve_path(cfg: Config, dotted_key: str) -> Path:
    """Mengubah jalur relatif di konfigurasi menjadi jalur absolut."""
    raw = Path(str(cfg.require(dotted_key)))
    return raw if raw.is_absolute() else project_root() / raw
