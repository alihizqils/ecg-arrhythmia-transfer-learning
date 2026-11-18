"""Paket Beat-Level Specialist untuk model lengkap MA1 (dijalankan sekali secara lokal; hasilnya ikut bundel Colab).

    python eksperimen_v4_svdb/fase_e_ns/paket_spesialis_r15.py

Specialist (GBM N/S/V/AF pada 59 fitur beat + 9 fitur klaster pasien KL) hanya memakai sinyal EKG dan label
development; tidak bergantung pada CNN maupun backbone. Karena itu peluangnya dihitung sekali dan dikunci dengan
sha256, sehingga ketiga backbone di Colab memakai specialist yang identik dengan MA1.

Isi paket (per varian W = berbobot (1/frekuensi)^0,5, U = tanpa bobot):
  cv_<v>   [5, n_window, 13]  agregat window untuk CV fold luar k (fold dev j memakai model pasangan (k, j))
  fin_<v>  [n_window, 13]     agregat window untuk model akhir (dev fold j: model (j, j); holdout: model seluruh dev)
  window_id [n_window]        urutan window = global_manifest tanpa window N dari afdb/vfdb (sama dengan load_base)
Sumber peluang beat:
  W: hasil/spesialis_s_pasien/peluang_beat.npz (P_KL, PH_KL) = specialist MA1 tersimpan;
  U: hasil/e6/_cache_bobot/spec_U.npy (pasangan) + model seluruh dev tanpa bobot (dilatih di sini).
Pemeriksaan: GBM seluruh dev berbobot yang dilatih ulang di sini harus identik dengan PH_KL tersimpan.
Holdout tidak dipakai untuk melatih apa pun (hanya diberi peluang dari model seluruh development).
Keluaran: eksperimen_v4_svdb/model_lengkap_data/spesialis_r15.npz + spesialis_r15.json (tidak menimpa).
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from e5_common import HERE, agg13, beat_features, kl_probs, load_base, log
from e6_bobot_ujung_ke_ujung import CACHE, gbm

OUT = HERE.parent / "model_lengkap_data"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    target = OUT / "spesialis_r15.npz"
    if target.exists():
        sys.exit(f"{target} sudah ada; tidak ditimpa.")
    OUT.mkdir(parents=True, exist_ok=True)
    B = load_base(); g = B["g"]; bi, y4, pdev, pfold = B["bi"], B["y4"], B["pdev"], B["pfold"]
    XA, KL, _ = beat_features(B); X = np.hstack([XA[bi], KL[bi]]); del XA, KL
    Z = np.load(HERE / "hasil" / "spesialis_s_pasien" / "peluang_beat.npz")
    Q = {"W": kl_probs(B), "U": np.load(CACHE / "spec_U.npy")}
    tr = pdev & (y4 >= 0); th = np.flatnonzero(~pdev); PH = {}
    for v in ["W", "U"]:
        m = gbm(X[tr], y4[tr], v == "W"); pr = m.predict_proba(X[th]); cl = list(m.classes_)
        PH[v] = np.full((len(bi), 2), np.nan); PH[v][th, 0] = pr[:, cl.index(1)]; PH[v][th, 1] = pr[:, cl.index(3)]
        log(f"specialist seluruh dev {v} selesai")
    repro = float(np.nanmax(np.abs(PH["W"][th] - Z["PH_KL"][th])))
    assert repro < 1e-6, f"specialist seluruh dev W tidak mereproduksi PH_KL ({repro})"
    out = {"window_id": g.window_id.to_numpy().astype(str)}
    for v in ["W", "U"]:
        cv = np.zeros((5, len(g), 13))
        for k in range(5):
            p = np.zeros((len(bi), 2))
            for j in range(5):
                mm = pdev & (pfold == j); p[mm] = np.nan_to_num(Q[v][k, j, mm])
            cv[k] = agg13(B, p)
        p = np.zeros((len(bi), 2))
        for j in range(5):
            mm = pdev & (pfold == j); p[mm] = np.nan_to_num(Q[v][j, j, mm])
        p[~pdev] = np.nan_to_num(PH[v][~pdev])
        out[f"cv_{v}"] = cv; out[f"fin_{v}"] = agg13(B, p)
    np.savez(target, **out)
    info = {"berkas": target.name, "sha256": sha(target), "n_window": int(len(g)),
            "reproduksi_PH_KL_selisih_maks": repro,
            "sumber": {"peluang_beat.npz": sha(HERE / "hasil" / "spesialis_s_pasien" / "peluang_beat.npz"),
                       "spec_U.npy": sha(CACHE / "spec_U.npy")},
            "catatan": "Specialist tidak bergantung pada CNN/backbone; holdout tidak dipakai untuk pelatihan."}
    (OUT / "spesialis_r15.json").write_text(json.dumps(info, indent=2))
    log(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
