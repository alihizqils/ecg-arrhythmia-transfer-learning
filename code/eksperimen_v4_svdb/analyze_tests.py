"""Analisis hasil uji tambahan Colab (G0L4/U2/U3/U4/U5/U6) dari prediksi evaluate_tests.py.

Semua hold-out memakai template bebas label. Model lengkap = resep v4_model_lengkap: specialist (paket resmi) +
fusi LR dilatih pada log-prob out-of-fold development + kunci VT. Bila CNN final diganti (U4) dan CV-nya resmi, OOF
dan fusi resmi dipakai (hanya masukan hold-out CNN yang diganti, sama dengan B4c). Bila CV-nya baru (U3, U5), OOF
dibangun dari CV uji itu pada E* uji (E* diperiksa ulang dengan aturan resmi). Bootstrap per pasien 2.000 ulangan
(seed 0) + Holm atas 8 metrik (v4_model_lengkap._boot), sama dengan B4b/B4c. Hold-out tidak dipakai memilih apa pun.

Kontrol: model resmi lewat jalur ini harus mereproduksi model_lengkap_labelfree (CNN path dan model lengkap 100 %).
Keluaran baru: outputs_colab_r15/additional_tests/analisis_colab/
Pakai: python analyze_tests.py <folder hasil uji tambahan>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "colab"))
import v4_model_lengkap as M  # noqa: E402
import colab_tests as U  # noqa: E402

OUT = HERE / "outputs_colab_r15"
PRED = OUT / "additional_tests" / "uji_colab"
B4 = OUT / "additional_tests" / "b4_tabular_only"
DEST = OUT / "additional_tests" / "analisis_colab"
ARCHS = ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"]
SEEDS = (142, 1042, 2042)
UJI = M.UJI


class Data:
    def __init__(self, base: Path):
        self.base = base
        g = M.windows(); self.g = g; self.Z, _ = M.paket()
        self.y = g.label_idx.to_numpy(); self.dev = (g.split == "development").to_numpy()
        self.hold = (g.split == "holdout").to_numpy()
        self.ids = g.window_id.astype(str).to_numpy(); self.hid = self.ids[self.hold]
        self.yh = self.y[self.hold]; self.pid = (g.database + "/" + g.record.astype(str)).to_numpy()[self.hold].astype(str)
        self.db_h = g.database.to_numpy()[self.hold]
        self.pos = pd.Series(np.arange(len(g)), index=self.ids)

    def root(self, uid):
        arch = next(a for a in ARCHS if a in uid)
        return self.base / arch / "TA_R15_uji_tambahan_hasil" / uid

    def prob(self, uid):
        """Probabilitas hold-out uji (urutan self.hid) dan, untuk U2, probabilitas window tambahan."""
        z = np.load(PRED / uid / "prediksi.npz"); n = int(z["n_holdout"])
        s = pd.DataFrame(z["prob"][:n], index=z["window_id"][:n].astype(str))
        if set(s.index) != set(self.hid) or len(s) != len(self.hid):
            raise ValueError(f"{uid}: window hold-out berbeda.")
        yy = pd.Series(z["y"][:n], index=s.index).loc[self.hid].to_numpy()
        assert np.array_equal(yy, self.yh), uid
        ekstra = None
        if len(z["window_id"]) > n:
            ekstra = {"window_id": z["window_id"][n:].astype(str), "prob": z["prob"][n:], "y": z["y"][n:],
                      "pasien": z["pasien"][n:].astype(str)}
        return s.loc[self.hid].to_numpy(), ekstra

    def oof(self, out_root, arch, sk):
        """Log-prob OOF development pada E* (aturan resmi diperiksa ulang), seperti M.cnn_logprob bagian dev."""
        estar = M.estar_resmi(out_root, arch, sk)
        L = np.full((len(self.g), 5), np.nan)
        for k in range(5):
            d = out_root / "cv" / arch / sk / f"fold_{k}"
            m = pd.read_csv(d / "manifest.csv", dtype={"window_id": str})
            L[self.pos.loc[m.window_id].to_numpy()] = np.load(d / "epoch_probabilities" / f"epoch_{estar:03d}.npy")
        return np.log(np.clip(np.nan_to_num(L, nan=1.0), 1e-6, 1)), estar

    def lengkap(self, Ldev, Phold, berbobot):
        L = Ldev.copy(); L[self.hold] = np.log(np.clip(Phold, 1e-6, 1)); v = "W" if berbobot else "U"
        F = np.hstack([L, self.Z[f"fin_{v}"]]); fu = M.fuser(F[self.dev], self.y[self.dev], berbobot)
        return M.decide(fu, F[self.hold], L[self.hold])


def boot(D, a, b, label):
    T = M._boot(D.yh, a, b, D.pid); T.insert(0, "banding", label)
    return T


def boot_rerata(D, A, B, label, n=2000, seed=0):
    """Selisih rerata metrik antar beberapa prediksi (mis. 3 seed) a - b, bootstrap per pasien + Holm."""
    rng = np.random.default_rng(seed); ids = np.unique(D.pid); idx = {i: np.flatnonzero(D.pid == i) for i in ids}

    def mean_metrik(s):
        ra = pd.DataFrame([M.ringkas(D.yh[s], a[s]) for a in A])[UJI].mean()
        rb = pd.DataFrame([M.ringkas(D.yh[s], b[s]) for b in B])[UJI].mean()
        return ra - rb
    base = mean_metrik(np.arange(len(D.yh))); draws = []
    for _ in range(n):
        s = np.concatenate([idx[i] for i in rng.choice(ids, len(ids))]); draws.append(mean_metrik(s))
    Dd = pd.DataFrame(draws); rows = []
    for k in UJI:
        p2 = 2 * min((Dd[k] <= 0).mean(), (Dd[k] >= 0).mean())
        rows.append({"metrik": k, "selisih": base[k], "ci95_bawah": Dd[k].quantile(.025),
                     "ci95_atas": Dd[k].quantile(.975), "p_dua_sisi": min(1.0, p2)})
    T = pd.DataFrame(rows); p = T.p_dua_sisi.to_numpy(); order = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    T["p_holm"] = adj; T.insert(0, "banding", label)
    return T


def main(base: Path):
    DEST.mkdir(parents=True, exist_ok=True)
    D = Data(base); rows, banding, kontrol = [], [], {}

    def catat(arch, uji, varian, seed, pred, **kw):
        rows.append({"arch": arch, "uji": uji, "varian": varian, "seed_final": seed, **kw, **M.ringkas(D.yh, pred)})

    b4 = np.load(B4 / "prediksi_holdout.npz"); assert np.array_equal(b4["window_id"], D.hid)
    tf = {s: np.load(B4 / "seed_final" / f"seed{s}" / "prediksi_holdout.npz") for s in SEEDS}
    for s in SEEDS:
        assert np.array_equal(tf[s]["window_id"], D.hid)
        catat("-", "B4c", "tabular-only", s, tf[s]["tabular"])
        catat("-", "B4c", "tabular+specialist+fusi (kunci VT)", s, tf[s]["tabular_fusi_kunci_vt"])

    for arch in ARCHS:
        res = {}       # (skenario, seed) -> (cnn, lengkap)
        off_L = {}
        for sc in ("S1", "S2", "S3"):
            Ldev, _ = D.oof(OUT, arch, sc); off_L[sc] = Ldev
            P, _ = D.prob(f"RESMI_{arch}_{sc}")
            cnn, full = P.argmax(1), D.lengkap(Ldev, P, M.BOBOT[sc])
            ref = np.load(OUT / "main_results" / "model_lengkap_labelfree" / arch / sc / "prediksi.npz")
            assert np.array_equal(ref["window_id_holdout"], D.hid)
            kontrol[f"{arch}_{sc}"] = {"cnn_sama": float((cnn == ref["cnn_holdout"]).mean()),
                                       "lengkap_sama": float((full == ref["lengkap_holdout"]).mean())}
            if min(kontrol[f"{arch}_{sc}"].values()) < 1:
                raise ValueError(f"Kontrol {arch} {sc} tidak mereproduksi model_lengkap_labelfree: {kontrol}")
            res[(sc, 142)] = (cnn, full)
            for s in SEEDS[1:]:
                Ps, _ = D.prob(U.uji_id("U4", arch, sc, s))
                res[(sc, s)] = (Ps.argmax(1), D.lengkap(Ldev, Ps, M.BOBOT[sc]))
            for s in SEEDS:
                catat(arch, "U4", f"{sc} CNN path", s, res[(sc, s)][0])
                catat(arch, "U4", f"{sc} model lengkap", s, res[(sc, s)][1])

        # G0L4
        Pg, _ = D.prob(f"G0L4_{arch}"); Pr, _ = D.prob(f"RESMI_{arch}_S2")
        kontrol[f"G0L4_{arch}"] = {"argmax_sama": float((Pg.argmax(1) == Pr.argmax(1)).mean()),
                                   "selisih_prob_maks": float(np.abs(Pg - Pr).max())}

        # U4: efek skenario per seed dan rerata 3 seed; model lengkap vs baseline tanpa citra (B4c) per seed
        for s in SEEDS:
            for m_i, nama in ((0, "CNN path"), (1, "model lengkap")):
                banding.append(boot(D, res[("S2", s)][m_i], res[("S1", s)][m_i], f"{arch} U4 seed{s} S2-S1 {nama}"))
                banding.append(boot(D, res[("S3", s)][m_i], res[("S2", s)][m_i], f"{arch} U4 seed{s} S3-S2 {nama}"))
            banding.append(boot(D, res[("S2", s)][1], tf[s]["tabular_fusi_kunci_vt"],
                                f"{arch} U4 seed{s} model lengkap S2 - TF kunci VT seed{s}"))
        for m_i, nama in ((0, "CNN path"), (1, "model lengkap")):
            banding.append(boot_rerata(D, [res[("S2", s)][m_i] for s in SEEDS], [res[("S1", s)][m_i] for s in SEEDS],
                                       f"{arch} U4 rerata3seed S2-S1 {nama}"))
            banding.append(boot_rerata(D, [res[("S3", s)][m_i] for s in SEEDS], [res[("S2", s)][m_i] for s in SEEDS],
                                       f"{arch} U4 rerata3seed S3-S2 {nama}"))
        banding.append(boot_rerata(D, [res[("S2", s)][1] for s in SEEDS], [tf[s]["tabular_fusi_kunci_vt"] for s in SEEDS],
                                   f"{arch} U4 rerata3seed model lengkap S2 - TF kunci VT"))
        banding.append(boot_rerata(D, [res[("S2", s)][0] for s in SEEDS], [tf[s]["tabular"] for s in SEEDS],
                                   f"{arch} U4 rerata3seed CNN path S2 - tabular-only"))

        # U3: image-only (CNN) dan image-only + specialist + fusi (kunci VT dari image-only)
        r3 = D.root(f"U3_{arch}"); L3, e3 = D.oof(r3, arch, U.U3_NAMA); P3, _ = D.prob(f"U3_{arch}")
        img, img_f = P3.argmax(1), D.lengkap(L3, P3, True)
        catat(arch, "U3", "image-only", 142, img, E_star=e3)
        catat(arch, "U3", "image-only+specialist+fusi (kunci VT)", 142, img_f, E_star=e3)
        banding.append(boot(D, res[("S2", 142)][0], img, f"{arch} U3 CNN path S2 - image-only"))
        banding.append(boot(D, img, b4["tabular"], f"{arch} U3 image-only - tabular-only"))
        banding.append(boot(D, res[("S2", 142)][1], img_f, f"{arch} U3 model lengkap S2 - image-only+spesialis+fusi"))

        # U5: backbone acak beku (BatchNorm dikalibrasi) vs ImageNet beku, protokol S2
        r5 = D.root(f"U5_{arch}"); L5, e5 = D.oof(r5, arch, "S2"); P5, _ = D.prob(f"U5_{arch}")
        rnd, rnd_f = P5.argmax(1), D.lengkap(L5, P5, True)
        catat(arch, "U5", "acak beku CNN path", 142, rnd, E_star=e5)
        catat(arch, "U5", "acak beku model lengkap", 142, rnd_f, E_star=e5)
        banding.append(boot(D, res[("S2", 142)][0], rnd, f"{arch} U5 ImageNet - acak, CNN path"))
        banding.append(boot(D, res[("S2", 142)][1], rnd_f, f"{arch} U5 ImageNet - acak, model lengkap"))
        banding.append(boot_rerata(D, [res[("S2", s)][0] for s in SEEDS], [rnd],
                                   f"{arch} U5 ImageNet (rerata 3 seed) - acak, CNN path"))
        banding.append(boot(D, rnd, b4["tabular"], f"{arch} U5 acak CNN path - tabular-only"))

        # U2: kontrol hold-out
        P2, ek = D.prob(f"U2_{arch}")
        catat(arch, "U2", "tanpa 12 pasien AF/VT luar sumber, CNN path (hold-out)", 142, P2.argmax(1))
        banding.append(boot(D, P2.argmax(1), res[("S2", 142)][0], f"{arch} U2 - resmi, CNN path hold-out"))
        np.savez(DEST / f"u2_prediksi_{arch}.npz", **ek)

        # U6
        if arch == "efficientnet_b0":
            for fr in U.U6_FRAKSI:
                for ul in U.U6_ULANG:
                    uid = U.uji_id("U6", arch, "S2", None, False, fr, ul); P6, _ = D.prob(uid)
                    n_p = json.loads((D.root(uid) / "uji.json").read_text())["n_pasien_latih"]
                    catat(arch, "U6", f"{int(fr * 100)}% pasien", 142, P6.argmax(1), ulang=ul, n_pasien=n_p)

    R = pd.DataFrame(rows); R.to_csv(DEST / "ringkas_holdout.csv", index=False)
    Bd = pd.concat(banding); Bd.to_csv(DEST / "banding_holdout.csv", index=False)
    u2 = analisis_u2(D)
    info = {"kontrol": kontrol, "u2": u2, "kode_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "holdout_dipakai_untuk_seleksi": False}
    (DEST / "complete.json").write_text(json.dumps(json.loads(json.dumps(info, default=float)), indent=2))
    print(json.dumps(kontrol, indent=1))


def analisis_u2(D):
    """U2 pada 12 pasien yang dikeluarkan: recall AF/VT per sumber, alarm AF palsu; pembanding OOF CV resmi (U1)."""
    u1 = np.load(HERE / "outputs_additional_tests" / "u1_recall_per_sumber" / "prediksi_oof.npz", allow_pickle=True)
    u1i = pd.Series(np.arange(len(u1["window_id"])), index=u1["window_id"].astype(str))
    out, per_pasien = [], []
    for arch in ARCHS:
        ek = dict(np.load(DEST / f"u2_prediksi_{arch}.npz"))
        w = ek["window_id"]; pr = ek["prob"].argmax(1); yy = ek["y"]; pas = ek["pasien"]; db = np.array([p.split("/")[0] for p in pas])
        j = u1i.loc[w].to_numpy(); assert np.array_equal(u1["y"][j], yy)
        oof = u1[f"{arch}_S2_cnn"][j]
        for kelas, ci in (("AF", 3), ("VT", 4)):
            for d in sorted(set(db[yy == ci])):
                m = (yy == ci) & (db == d)
                out.append({"arch": arch, "kelas": kelas, "database": d, "n_window": int(m.sum()),
                            "n_pasien": int(len(set(pas[m]))), "recall_U2": float((pr[m] == ci).mean()),
                            "recall_OOF_resmi": float((oof[m] == ci).mean())})
        non_af = yy != 3
        out.append({"arch": arch, "kelas": "alarm AF palsu (non-AF -> AF)", "database": "12 pasien",
                    "n_window": int(non_af.sum()), "n_pasien": int(len(set(pas[non_af]))),
                    "recall_U2": float((pr[non_af] == 3).mean()), "recall_OOF_resmi": float((oof[non_af] == 3).mean())})
        for p in sorted(set(pas)):
            m = pas == p
            for kelas, ci in (("AF", 3), ("VT", 4)):
                mm = m & (yy == ci)
                if mm.any():
                    per_pasien.append({"arch": arch, "pasien": p, "kelas": kelas, "n_window": int(mm.sum()),
                                       "recall_U2": float((pr[mm] == ci).mean()), "recall_OOF_resmi": float((oof[mm] == ci).mean())})
    T = pd.DataFrame(out); T.to_csv(DEST / "u2_recall_per_sumber.csv", index=False)
    pd.DataFrame(per_pasien).to_csv(DEST / "u2_recall_per_pasien.csv", index=False)
    return T.to_dict(orient="records")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
