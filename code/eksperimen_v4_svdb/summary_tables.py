"""Rekap hasil utama (hold-out bebas label + CV terisolasi).

Masukan:
  main_results/holdout_labelfree/<arch>/<sc>/probabilities_labelfree.npy  (holdout_labelfree.py)
  main_results/cv_terisolasi_per_fold.csv                                  (cv_isolated.py)
Model lengkap hold-out: resep v4_model_lengkap (fusi dilatih pada seluruh development dengan masukan yang sama
seperti sebelumnya), hanya masukan CNN hold-out diganti probabilitas bebas label. Holdout tidak dipakai untuk
memilih apa pun. Keluaran: main_results/*.csv dan model_lengkap_labelfree/<arch>/<sc>/prediksi.npz.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import v4_model_lengkap as M
from cv_isolated import ARCHS, DEST, OUT

SC = ["S1", "S2", "S3"]


def main():
    g = M.windows(); Z, _ = M.paket(); y = g.label_idx.to_numpy()
    dev = (g.split == "development").to_numpy(); hold = (g.split == "holdout").to_numpy()
    pos = pd.Series(np.arange(len(g)), index=g.window_id); gh = g[hold].reset_index(drop=True)
    main_rows, cls_rows, db_rows, abl_rows, pred = [], [], [], [], {}
    for arch in ARCHS:
        for sc in SC:
            estar = M.estar_resmi(OUT, arch, sc); L = M.cnn_logprob(OUT, arch, sc, g, estar)
            L_old = L.copy()
            m = pd.read_csv(OUT / "holdout" / arch / sc / "manifest.csv", dtype={"window_id": str})
            P = np.load(DEST / "holdout_labelfree" / arch / sc / "probabilities_labelfree.npy")
            L[pos.loc[m.window_id].to_numpy()] = np.log(np.clip(P, 1e-6, 1))
            w = M.BOBOT[sc]; v = "W" if w else "U"
            F = np.hstack([L, Z[f"fin_{v}"]]); fu = M.fuser(F[dev], y[dev], w)
            full = M.decide(fu, F[hold], L[hold]); fus = fu.classes_[fu.predict_proba(F[hold]).argmax(1)]
            cnn = L[hold].argmax(1); yh = y[hold]
            Fo = np.hstack([L_old, Z[f"fin_{v}"]]); full_old = M.decide(M.fuser(Fo[dev], y[dev], w), Fo[hold], L_old[hold])
            ref = np.load(OUT / "model_lengkap" / arch / sc / "prediksi.npz")
            assert (full_old == ref["lengkap_holdout"]).mean() > 0.999
            pred[(arch, sc)] = full
            dst = DEST / "model_lengkap_labelfree" / arch / sc; dst.mkdir(parents=True, exist_ok=True)
            np.savez(dst / "prediksi.npz", window_id_holdout=g.window_id.to_numpy()[hold].astype(str), y_holdout=yh,
                     lengkap_holdout=full, cnn_holdout=cnn, fusi_holdout=fus,
                     pasien_holdout=(g.database + "/" + g.record).to_numpy()[hold].astype(str))
            r_cnn, r_full = M.ringkas(yh, cnn), M.ringkas(yh, full)
            main_rows.append({"arch": arch, "sc": sc, "E*": estar, "holdout_cnn": r_cnn["macro_F1"],
                              "holdout_lengkap": r_full["macro_F1"], **{f"F1_{c}": r_full[f"F1_{c}"] for c in M.C},
                              "recall_S": r_full["recall_S"], "presisi_S": r_full["presisi_S"],
                              "holdout_lengkap_lama": M.ringkas(yh, full_old)["macro_F1"],
                              "window_lengkap_berubah": int((full != full_old).sum()),
                              "window_cnn_berubah": int((cnn != L_old[hold].argmax(1)).sum())})
            cm, t = M.per_kelas(yh, full)
            np.save(dst / "confusion_holdout.npy", cm)
            cls_rows += [{"arch": arch, "sc": sc, "kelas": c, **t.loc[c].to_dict()} for c in M.C]
            for db in sorted(gh.database.unique()):
                mm = (gh.database == db).to_numpy()
                for ci, cn in enumerate(M.C):
                    sel = mm & (yh == ci)
                    if sel.sum():
                        db_rows.append({"arch": arch, "sc": sc, "database": db, "kelas": cn, "n": int(sel.sum()),
                                        "recall": float((full[sel] == ci).mean()), "pasien": int(gh.patient_id[sel].nunique())})
            if sc == "S2":
                ko = json.load(open(DEST / "holdout_labelfree" / arch / sc / "complete.json"))["knockout_labelfree"]
                sp = M.fuser(Z[f"fin_{v}"][dev], y[dev], w)
                spec = sp.classes_[sp.predict_proba(Z[f"fin_{v}"][hold]).argmax(1)]
                for name, key in [("Image (tabular KO)", "tanpa_tabular"), ("Tabular (image KO)", "tanpa_citra")]:
                    abl_rows.append({"arch": arch, "konfigurasi": name, "macro": ko[key]["macro"],
                                     **{c: ko[key][c] for c in M.C}})
                for name, p in [("Specialist + LR", spec), ("CNN path", cnn), ("+ fusion", fus), ("+ VT lock", full)]:
                    r = M.ringkas(yh, p)
                    abl_rows.append({"arch": arch, "konfigurasi": name, "macro": r["macro_F1"],
                                     **{c: r[f"F1_{c}"] for c in M.C}})
            print(arch, sc, round(r_cnn["macro_F1"], 4), round(r_full["macro_F1"], 4), flush=True)
    eff = []
    for arch in ARCHS:
        for a, b in [("S2", "S1"), ("S3", "S2")]:
            T = M._boot(y[hold], pred[(arch, a)], pred[(arch, b)], (g.database + "/" + g.record).to_numpy()[hold])
            eff.append(T.assign(arch=arch, kontras=f"{a}-{b}"))
    pd.DataFrame(main_rows).to_csv(DEST / "tabel_utama_holdout.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(DEST / "per_kelas_holdout.csv", index=False)
    pd.DataFrame(db_rows).to_csv(DEST / "per_database_holdout.csv", index=False)
    pd.DataFrame(abl_rows).to_csv(DEST / "ablation_holdout_S2.csv", index=False)
    pd.concat(eff).to_csv(DEST / "efek_skenario_holdout.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(main_rows).round(3).to_string())
    print(pd.DataFrame(abl_rows).round(3).to_string())
    print(pd.concat(eff).round(3).to_string())


if __name__ == "__main__":
    main()
