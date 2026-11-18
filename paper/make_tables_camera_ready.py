"""Tabel paper camera-ready SOFTT dari berkas hasil (tidak ada angka yang diketik manual).

Sumber:
  code/eksperimen_v4_svdb/outputs_colab_r15/main_results/   (Tabel II kolom CV dan hold-out seed acuan, Tabel III)
  code/eksperimen_v4_svdb/outputs_colab_r15/additional_tests/  (rentang 3 seed, baseline terlatih, latensi pipeline)

Pakai: python make_tables_camera_ready.py          -> cetak blok
       python make_tables_camera_ready.py --tulis  -> ganti blok di main.tex di antara
                                                       '% BEGIN auto:<nama>' dan '% END auto:<nama>'
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "code/eksperimen_v4_svdb/outputs_colab_r15"
R3, RV = ROOT / "main_results", ROOT / "additional_tests"
NAMA = {"efficientnet_b0": "EfficientNet-B0", "mobilenet_v2": "MobileNetV2", "shufflenet_v2_x1_0": "ShuffleNetV2"}
ARCH = list(NAMA)


def f(x):
    return f"{x:.3f}"


def s(x, dag=False):
    v = round(x, 3)
    t = "0.000" if v == 0 else (f"+{v:.3f}" if v > 0 else f"$-${-v:.3f}")
    return t + ("$^{\\dagger}$" if dag else "")


def ringkas():
    return pd.read_csv(RV / "analisis_colab/ringkas_holdout.csv")


def tab_main():
    H = pd.read_csv(R3 / "tabel_utama_holdout.csv").set_index(["arch", "sc"])
    T = pd.read_csv(R3 / "cv_terisolasi_per_fold.csv")
    cnn = T[T.prosedur.str.startswith("CNN path terisolasi")].groupby(["arch", "skenario"]).macro_F1.agg(["mean", "std"])
    full = T[T.prosedur.str.startswith("terisolasi")].groupby(["arch", "skenario"]).macro_F1.agg(["mean", "std"])
    K = ringkas(); out = []
    for i, a in enumerate(ARCH):
        if i:
            out.append("\\midrule")
        for sc in ["S1", "S2", "S3"]:
            h = H.loc[(a, sc)]; c, u = cnn.loc[(a, sc)], full.loc[(a, sc)]
            seeds = K[(K.arch == a) & (K.uji == "U4") & (K.varian == f"{sc} model lengkap")].macro_F1
            assert len(seeds) == 3 and abs(seeds[K.seed_final == 142].iloc[0] - h.holdout_lengkap) < 1e-9
            out.append(f"{NAMA[a] if sc == 'S1' else ''} & {sc} & {int(h['E*'])} & {f(c['mean'])} $\\pm$ {f(c['std'])} & "
                       f"{f(u['mean'])} $\\pm$ {f(u['std'])} && {f(h.holdout_cnn)} & {f(h.holdout_lengkap)} & "
                       f"{seeds.min():.3f}--{seeds.max():.3f} && "
                       + " & ".join(f(h[f"F1_{k}"]) for k in ["N", "S", "V", "AF", "VT"]) + " \\\\")
    return "\n".join(out)


def tab_effects():
    E = pd.read_csv(R3 / "efek_skenario_holdout.csv"); out = []
    cols = ["F1_N", "F1_S", "F1_V", "F1_AF", "recall_S", "presisi_S", "recall_VT"]
    for i, a in enumerate(ARCH):
        if i:
            out.append("\\midrule")
        out.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{NAMA[a]}}}}} \\\\")
        for k in ["S2-S1", "S3-S2"]:
            e = E[(E.arch == a) & (E.kontras == k)].set_index("metrik")
            m = e.loc["macro_F1"]
            cells = [f"{s(m.selisih, m.p_holm < .05)} [{s(m.ci95_bawah)}, {s(m.ci95_atas)}]"]
            cells += [s(e.loc[c].selisih, e.loc[c].p_holm < .05) for c in cols]
            a_, b_ = k.split("-")
            out.append(f"{a_} $-$ {b_} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def tab_baselines():
    """Hold-out, protokol S2, seed acuan 142. Dagger: ImageNet - acak signifikan (p_Holm < .05) pada model lengkap."""
    K = ringkas(); K = K[K.seed_final == 142]
    B = pd.read_csv(RV / "analisis_colab/banding_holdout.csv")

    def row(arch, uji, varian):
        r = K[(K.arch == arch) & (K.uji == uji) & (K.varian == varian)]
        assert len(r) == 1, (arch, uji, varian)
        return r.iloc[0]

    def sig(a):
        b = B[(B.banding == f"{a} U5 ImageNet - acak, model lengkap") & (B.metrik == "macro_F1")]
        assert len(b) == 1
        return b.p_holm.iloc[0] < .05

    # Kelompok simetris: setiap model dasar tanpa salah satu masukan diberi baris "+ specialist + fusion".
    # Judul kelompok menamai jaringan dasar; baris "+ specialist" menambah fitur beat, jadi bukan "tanpa tabular".
    spec = [("Tabular only", None),
            ("\\quad Tabular expert", ("-", "B4c", "tabular-only")),
            ("\\quad\\quad + specialist + fusion", ("-", "B4c", "tabular+specialist+fusi (kunci VT)")),
            ("Image only (ImageNet, frozen)", None),
            ("\\quad Image-only CNN", ("A", "U3", "image-only")),
            ("\\quad\\quad + specialist + fusion", ("A", "U3", "image-only+specialist+fusi (kunci VT)")),
            ("CNN path (image path + tabular expert)", None),
            ("\\quad Random, frozen", ("A", "U5", "acak beku CNN path")),
            ("\\quad ImageNet, frozen", ("A", "U4", "S2 CNN path")),
            ("Full model (CNN path + specialist + fusion)", None),
            ("\\quad Random, frozen", ("A", "U5", "acak beku model lengkap")),
            ("\\quad ImageNet, frozen", ("A", "U4", "S2 model lengkap"))]
    out = []
    for label, key in spec:
        if key is None:
            out.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{label}}}}} \\\\"); continue
        a0, uji, var = key
        if a0 == "-":
            r = row("-", uji, var)
            # Tanpa backbone: satu model, satu nilai; dibentangkan dengan keterangan agar tidak terbaca sebagai tiga model.
            cells = [f"\\multicolumn{{3}}{{c}}{{{f(r.macro_F1)}\\rlap{{$^{{\\ast}}$}}}}"]; e = r
        else:
            cells = []
            for a in ARCH:
                r = row(a, uji, var); v = f(r.macro_F1)
                if var == "S2 model lengkap" and sig(a):
                    v += "\\rlap{$^{\\dagger}$}"  # rlap: dagger tidak menggeser perataan angka
                cells.append(v)
            e = row("efficientnet_b0", uji, var)
        kelas = " & ".join(f(e[c]) for c in ("F1_N", "F1_S", "F1_V", "F1_AF", "F1_VT"))
        out.append(f"{label} & " + " & ".join(cells) + f" & {kelas} \\\\")
    return "\n".join(out)


def pipeline_row():
    L = pd.read_csv(RV / "b1_latensi/cnn_vs_crop/latensi_end_to_end_skala_resmi.csv")
    g = L.groupby("arch").ms_total.agg(["min", "max"])
    return "Pipeline CPU latency (ms) & " + " & ".join(f"{g.loc[a, 'min']:.0f}--{g.loc[a, 'max']:.0f}" for a in ARCH) + " \\\\"


BLOK = {"tab_main": tab_main, "tab_effects": tab_effects, "tab_baselines": tab_baselines, "pipeline_row": pipeline_row}

if __name__ == "__main__":
    blok = {k: fn() for k, fn in BLOK.items()}
    if "--tulis" not in sys.argv:
        for k, v in blok.items():
            print(f"% --- {k}\n{v}\n")
        sys.exit()
    tex = Path(__file__).with_name("main.tex"); t = tex.read_text(encoding="utf-8")
    for k, v in blok.items():
        pat = re.compile(rf"(% BEGIN auto:{k}\n).*?(% END auto:{k})", re.S)
        if not pat.search(t):
            raise SystemExit(f"penanda {k} tidak ditemukan")
        t = pat.sub(lambda m: m.group(1) + v + "\n" + m.group(2), t)
    tex.write_text(t, encoding="utf-8"); print("main.tex diperbarui:", ", ".join(blok))
