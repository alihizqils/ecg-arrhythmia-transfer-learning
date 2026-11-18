"""Tabel dan gambar rekap V4 untuk notebook 04 (presentasi dan paper).

Hanya membaca berkas hasil di `outputs/` (tanpa training/inferensi). Model atau skenario
yang belum selesai dilewati, sehingga notebook dapat dijalankan kapan saja.
Gambar disimpan ke outputs/paper/<nama>.png (300 dpi) dan .pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

import v4_pipeline as v4
from ecgtl.hierarchical.data import read_manifest
from ecgtl.hierarchical.protocol import CLASSES
from v4_data import ProtocolV4

ARCHS = ("efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0")
ARCH_NAMES = {"efficientnet_b0": "EfficientNet-B0", "mobilenet_v2": "MobileNetV2", "shufflenet_v2_x1_0": "ShuffleNetV2 x1.0"}
SCENARIOS = ("S1", "S2", "S3")
SCENARIO_NAMES = {"S1": "S1 beku, tanpa bobot", "S2": "S2 beku + class weighting", "S3": "S3 fine-tuning penuh"}
COLORS = {"S1": "#8c8c8c", "S2": "#1f77b4", "S3": "#ff7f0e"}
CLASS_NAMES = {"N": "N (normal)", "S": "S (supraventrikular)", "V": "V (ventrikular)", "AF": "AF", "VT": "VT"}
TARGET = 0.80
_P = None
_CACHE: dict = {}

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})


def protocol():
    global _P
    if _P is None:
        import os
        _P = ProtocolV4(output_root=os.environ.get("V4_OUTPUT_ROOT") or None)  # mis. outputs_colab_r15
    return _P


# --------------------------------------------------------------------------- model lengkap (R15)
LENGKAP_METRIK = ["macro_F1", "F1_N", "F1_S", "recall_S", "presisi_S", "F1_V", "F1_AF", "F1_VT", "recall_VT"]


def lengkap_table(source="holdout") -> pd.DataFrame:
    """Model lengkap (CNN + Beat-Level Specialist + fusi + kunci VT) per backbone × skenario."""
    rows = []
    for arch in ARCHS:
        path = out() / "model_lengkap" / arch / "ringkasan_S1_S2_S3.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0)
        for scenario in SCENARIOS:
            key = f"{scenario} {'holdout' if source == 'holdout' else 'dev (CV)'}"
            if key in frame.index:
                r = frame.loc[key]
                rows.append({"Backbone": ARCH_NAMES[arch], "Skenario": scenario, "E*": int(r["E*"]),
                             **{m: round(float(r[m]), 4) for m in LENGKAP_METRIK}})
    return save_table(pd.DataFrame(rows), f"B00_model_lengkap_{source}") if rows else pd.DataFrame()


def cnn_vs_lengkap() -> pd.DataFrame:
    """Holdout: CNN saja vs model lengkap (Macro-F1, F1 S, recall S, recall VT)."""
    rows = []
    for arch in ARCHS:
        for scenario in SCENARIOS:
            path = out() / "model_lengkap" / arch / scenario / "complete.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            cnn, full = data["holdout_cnn_saja"], data["holdout_lengkap"]
            rows.append({"Backbone": ARCH_NAMES[arch], "Skenario": scenario,
                         **{f"{m} CNN": round(float(cnn[m]), 4) for m in ["macro_F1", "F1_S", "recall_S", "recall_VT"]},
                         **{f"{m} lengkap": round(float(full[m]), 4) for m in ["macro_F1", "F1_S", "recall_S", "recall_VT"]}})
    return save_table(pd.DataFrame(rows), "B00_cnn_vs_lengkap_holdout") if rows else pd.DataFrame()


def lengkap_effects(level="lengkap") -> pd.DataFrame:
    """Uji efek holdout (bootstrap pasien berpasangan 2.000 ulangan, Holm atas 8 metrik): S2−S1 dan S3−S2."""
    rows = []
    for arch in ARCHS:
        for pair in ["S2_minus_S1", "S3_minus_S2"]:
            path = out() / "model_lengkap" / arch / f"banding_{pair}_{level}.csv"
            if not path.exists():
                continue
            for _, r in pd.read_csv(path).iterrows():
                rows.append({"Backbone": ARCH_NAMES[arch], "Perbandingan": pair.replace("_minus_", " − "), "Metrik": r["metrik"],
                             "Selisih": round(r["selisih"], 4), "CI 95%": f"[{r['ci95_bawah']:+.3f}; {r['ci95_atas']:+.3f}]",
                             "p Holm": round(r["p_holm"], 3), "Bermakna (p Holm < 0,05)": bool(r["p_holm"] < 0.05)})
    return save_table(pd.DataFrame(rows), f"C00_uji_efek_{level}") if rows else pd.DataFrame()


def fig_lengkap_bars():
    """Holdout Macro-F1 dan F1 S: CNN saja (batang pudar) vs model lengkap, per backbone × skenario."""
    table = cnn_vs_lengkap()
    if table.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    w = 0.36
    for ax, metric, title in zip(axes, ["macro_F1", "F1_S"], ["Macro-F1 holdout", "F1 kelas S holdout"]):
        ticks, labels = [], []
        for i, arch in enumerate(ARCHS):
            for j, scenario in enumerate(SCENARIOS):
                row = table[(table.Backbone == ARCH_NAMES[arch]) & (table.Skenario == scenario)]
                if row.empty:
                    continue
                x = i * 3.6 + j * 1.0; color = COLORS[scenario]
                cnn, full = row[f"{metric} CNN"].iloc[0], row[f"{metric} lengkap"].iloc[0]
                ax.bar(x - w / 2, cnn, w, color=color, alpha=0.35, hatch="//", edgecolor=color, linewidth=0)
                ax.bar(x + w / 2, full, w, color=color)
                ax.text(x + w / 2, full + 0.003, f"{full:.3f}", ha="center", va="bottom", fontsize=6.5)
                ticks.append(x); labels.append(scenario)
        ax.set_xticks(ticks, labels, fontsize=8)
        for i, arch in enumerate(ARCHS):
            ax.text(i * 3.6 + 1.0, -0.12, ARCH_NAMES[arch], transform=ax.get_xaxis_transform(), ha="center", fontsize=9, fontweight="bold")
        low = min(table[f"{metric} CNN"].min(), table[f"{metric} lengkap"].min())
        high = max(table[f"{metric} CNN"].max(), table[f"{metric} lengkap"].max())
        ax.set_ylim(max(0, low - 0.03), min(1.0, high + 0.03)); ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in SCENARIOS]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb", alpha=0.5, hatch="//", edgecolor="#888888"),
                plt.Rectangle((0, 0), 1, 1, color="#555555")]
    fig.legend(handles, [SCENARIO_NAMES[s] for s in SCENARIOS] + ["CNN saja (komponen)", "Model lengkap (CNN + specialist + fusi)"],
               loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); save(fig, "B00_cnn_vs_lengkap"); return fig


def out():
    return protocol().out


def paper_dir() -> Path:
    folder = out() / "paper"; folder.mkdir(parents=True, exist_ok=True)
    return folder


def save(fig, name):
    fig.savefig(paper_dir() / f"{name}.png", bbox_inches="tight")
    fig.savefig(paper_dir() / f"{name}.pdf", bbox_inches="tight")
    return paper_dir() / f"{name}.png"


def save_table(frame: pd.DataFrame, name: str):
    frame.to_csv(paper_dir() / f"{name}.csv", index=False)
    try:
        (paper_dir() / f"{name}.tex").write_text(frame.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
    except Exception:  # jinja2 tidak wajib
        pass
    return frame


# --------------------------------------------------------------------------- pemuatan

def manifest(split: str) -> pd.DataFrame:
    key = ("manifest", split)
    if key not in _CACHE:
        rows = read_manifest(protocol().prepared / "global_manifest.csv")
        _CACHE[key] = v4.label_valid(rows[rows.split == split].reset_index(drop=True))
    return _CACHE[key]


def done(arch, scenario, stage="holdout") -> bool:
    paths = {"cv": out() / "tables" / arch / f"{scenario}_cv_summary.json",
             "holdout": out() / "holdout" / arch / scenario / "complete.json",
             "xai": out() / "xai" / arch / scenario / "complete.json"}
    return paths[stage].exists()


def models(stage="holdout"):
    return [(a, s) for a in ARCHS for s in SCENARIOS if done(a, s, stage)]


def holdout(arch, scenario) -> dict:
    folder = out() / "holdout" / arch / scenario
    return {"summary": json.loads((folder / "summary.json").read_text()), "y": np.load(folder / "labels.npy"),
            "prob": np.load(folder / "probabilities.npy").astype(np.float64),
            "per_class": pd.read_csv(folder / "per_class.csv"), "s_patients": pd.read_csv(folder / "s_patients.csv")}


def cv(arch, scenario) -> dict:
    folder = out() / "tables" / arch
    return {"summary": json.loads((folder / f"{scenario}_cv_summary.json").read_text()),
            "folds": pd.read_csv(folder / f"{scenario}_cv_folds.csv"),
            "curve": pd.read_csv(out() / "selection" / arch / f"{scenario}_final_epoch_curve.csv")}


def cv_pooled(arch, scenario):
    """Label dan probabilitas validasi seluruh fold pada E* (dasar confusion matrix CV)."""
    epoch = cv(arch, scenario)["summary"]["selected_epoch"]; ys, probs = [], []
    for fold in range(5):
        folder = out() / "cv" / arch / scenario / f"fold_{fold}" / "epoch_probabilities"
        ys.append(np.load(folder / "labels.npy")); probs.append(np.load(folder / f"epoch_{epoch:03d}.npy").astype(np.float64))
    return np.concatenate(ys), np.concatenate(probs)


# --------------------------------------------------------------------------- A. data dan desain

def fig_flow():
    steps = ["Data PhysioNet\n4 database, 161 pasien", "Split pasien\n129 dev / 32 holdout", "Citra global 10 dtk\n+ crop lokal per beat\n+ fitur RR/fisiologi",
             "Skenario\nS1 · S2 · S3", "CV 5 fold\n(pasien terpisah)", "E* = argmax\nrerata Macro-F1", "Model final\nseluruh development",
             "Holdout\n(sekali)", "XAI (SHAP)\n+ benchmark"]
    fig, ax = plt.subplots(figsize=(13, 2.4)); ax.axis("off")
    for i, text in enumerate(steps):
        x = i * 1.45
        ax.add_patch(plt.Rectangle((x, 0), 1.25, 1, fc="#eef3fb" if i not in (3, 7) else "#fde9d9", ec="#1f4e79", lw=1.2))
        ax.text(x + 0.625, 0.5, text, ha="center", va="center", fontsize=8.2)
        if i < len(steps) - 1:
            ax.annotate("", (x + 1.43, 0.5), (x + 1.27, 0.5), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlim(-0.1, len(steps) * 1.45); ax.set_ylim(-0.1, 1.1)
    ax.set_title("Alur penelitian V4"); save(fig, "A01_alur_penelitian")
    return fig


def dataset_tables() -> dict:
    tables = {}
    for split in ("development", "holdout"):
        rows = manifest(split)
        table = pd.crosstab(rows.label, rows.database).reindex(CLASSES).fillna(0).astype(int)
        table["Total"] = table.sum(1); table["Pasien"] = rows.groupby("label").patient_id.nunique().reindex(CLASSES)
        table.loc["Total"] = [*table.drop(columns="Pasien").sum().tolist(), rows.patient_id.nunique()]
        tables[split] = table
        save_table(table.reset_index().rename(columns={"label": "Kelas"}), f"A02_dataset_{split}")
    dev = manifest("development")
    folds = pd.crosstab(dev.fold, dev.label)[list(CLASSES)]
    folds["Total"] = folds.sum(1); folds["Pasien"] = dev.groupby("fold").patient_id.nunique()
    folds["Pasien S"] = dev[dev.label == "S"].groupby("fold").patient_id.nunique()
    tables["folds"] = folds; save_table(folds.reset_index(), "A02_dataset_fold")
    return tables


def fig_dataset():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    palette = {"mitdb": "#4c72b0", "svdb": "#55a868", "afdb": "#c44e52", "vfdb": "#8172b2"}
    for ax, split in zip(axes, ("development", "holdout")):
        rows = manifest(split); table = pd.crosstab(rows.label, rows.database).reindex(CLASSES).fillna(0)
        bottom = np.zeros(len(CLASSES))
        for database in table.columns:
            ax.bar(CLASSES, table[database], bottom=bottom, label=database, color=palette.get(database)); bottom += table[database].to_numpy()
        patients = rows.groupby("label").patient_id.nunique().reindex(CLASSES)
        for i, c in enumerate(CLASSES):
            ax.text(i, bottom[i], f"{int(bottom[i])}\n({patients[c]} pasien)", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{split.capitalize()}: {len(rows):,} window, {rows.patient_id.nunique()} pasien".replace(",", "."))
        ax.set_ylabel("Jumlah window"); ax.set_ylim(0, bottom.max() * 1.25)
    axes[0].legend(title="Database"); fig.tight_layout(); save(fig, "A03_komposisi_dataset")
    return fig


def fig_examples(seed=20261105):
    dev = manifest("development"); rng = np.random.default_rng(seed)
    images = np.load(protocol().source / "images.npy", mmap_mode="r")
    local = read_manifest(protocol().prepared / "local_manifest.csv")
    fig, axes = plt.subplots(len(CLASSES), 4, figsize=(13, 2.1 * len(CLASSES)), gridspec_kw={"width_ratios": [2, 1, 1, 1]})
    for i, c in enumerate(CLASSES):
        row = dev[dev.label == c].iloc[int(rng.integers((dev.label == c).sum()))]
        axes[i, 0].imshow(images[int(row["index"])], cmap="gray", aspect="auto")
        axes[i, 0].set_ylabel(CLASS_NAMES[c], rotation=0, ha="right", va="center", fontsize=9)
        crops = local[local.parent_window_id == row.window_id]
        shard = np.load(protocol().prepared / crops.iloc[0].shard, mmap_mode="r")
        for j in range(3):
            if j < len(crops):
                axes[i, j + 1].imshow(shard[int(crops.shard_index.iloc[j])], cmap="gray")
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title("Citra global (10 detik)"); [axes[0, j].set_title(f"Crop lokal beat {j}") for j in (1, 2, 3)]
    fig.tight_layout(); save(fig, "A04_contoh_citra")
    return fig


# --------------------------------------------------------------------------- B. hasil utama

def main_table() -> pd.DataFrame:
    rows = []
    for arch in ARCHS:
        for scenario in SCENARIOS:
            row = {"Backbone": ARCH_NAMES[arch], "Skenario": scenario}
            if done(arch, scenario, "cv"):
                s = cv(arch, scenario)["summary"]
                row.update({"E*": s["selected_epoch"], "CV Macro-F1": f"{s['macro_f1_at_E_mean']:.3f} ± {s['macro_f1_at_E_std']:.3f}",
                            "CV F1 S": f"{s['f1_S_at_E_mean']:.3f} ± {s['f1_S_at_E_std']:.3f}"})
            if done(arch, scenario):
                h = holdout(arch, scenario)["summary"]; ci = h["ci95_patient_bootstrap"]["macro_f1"]
                row.update({"Holdout Macro-F1": f"{h['macro_f1']:.3f} [{ci['low']:.3f}–{ci['high']:.3f}]",
                            **{f"Holdout F1 {c}": round(h["f1"][c], 3) for c in CLASSES}})
            rows.append(row)
    return save_table(pd.DataFrame(rows), "B01_tabel_utama")


def fig_macro_bars():
    fig, ax = plt.subplots(figsize=(11, 4.2)); width = 0.13
    for i, arch in enumerate(ARCHS):
        for j, scenario in enumerate(SCENARIOS):
            x = i + (j - 1) * 2.2 * width
            if done(arch, scenario, "cv"):
                s = cv(arch, scenario)["summary"]
                ax.bar(x - width / 2, s["macro_f1_at_E_mean"], width, yerr=s["macro_f1_at_E_std"], color=COLORS[scenario], alpha=0.45, capsize=3)
            if done(arch, scenario):
                h = holdout(arch, scenario)["summary"]; ci = h["ci95_patient_bootstrap"]["macro_f1"]
                ax.bar(x + width / 2, h["macro_f1"], width, color=COLORS[scenario], capsize=3,
                       yerr=[[h["macro_f1"] - ci["low"]], [ci["high"] - h["macro_f1"]]])
                ax.text(x + width / 2, ci["high"] + 0.01, f"{h['macro_f1']:.3f}", ha="center", fontsize=7.5)
    ax.axhline(TARGET, ls="--", color="crimson", lw=1); ax.text(2.45, TARGET + 0.005, "target 0,80", color="crimson", fontsize=8)
    ax.set_xticks(range(len(ARCHS)), [ARCH_NAMES[a] for a in ARCHS]); ax.set_ylim(0.5, 1.0); ax.set_ylabel("Macro-F1")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in SCENARIOS]
    handles += [plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.45), plt.Rectangle((0, 0), 1, 1, color="gray")]
    ax.legend(handles, [SCENARIO_NAMES[s] for s in SCENARIOS] + ["CV (rerata ± SD)", "Holdout (CI 95% pasien)"], ncol=3, fontsize=8, loc="lower center")
    ax.set_title("Macro-F1 per backbone dan skenario"); save(fig, "B02_macro_f1"); return fig


_CMAP = LinearSegmentedColormap.from_list("biru", ["#ffffff", "#c6dbef", "#2171b5", "#08306b"])


def draw_confusion(ax, y, pred, title, diff_base=None):
    """Sel: jumlah window (persen terhadap kelas sebenarnya). Warna = persen baris."""
    cm = confusion_matrix(y, pred, labels=range(5)); support = cm.sum(1, keepdims=True)
    pct = 100 * cm / np.maximum(support, 1)
    if diff_base is not None:
        delta = pct - diff_base
        ax.imshow(delta, cmap="RdBu", vmin=-30, vmax=30)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{delta[i, j]:+.1f}", ha="center", va="center", fontsize=8.5,
                        fontweight="bold" if i == j else None)
    else:
        ax.imshow(pct, cmap=_CMAP, vmin=0, vmax=100)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)", ha="center", va="center", fontsize=8,
                        color="white" if pct[i, j] > 55 else "black", fontweight="bold" if i == j else None)
        precision = np.diag(cm) / np.maximum(cm.sum(0), 1)
        for j in range(5):
            ax.text(j, 5.0, f"P {100 * precision[j]:.1f}%", ha="center", va="center", fontsize=7.5, color="#444")
        for i in range(5):
            ax.text(5.0, i, f"n={support[i, 0]}", ha="left", va="center", fontsize=7.5, color="#444")
    ax.set_xticks(range(5), CLASSES); ax.set_yticks(range(5), CLASSES)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Sebenarnya"); ax.set_title(title, fontsize=9.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, 5.6 if diff_base is None else 4.5); ax.set_ylim(5.5 if diff_base is None else 4.5, -0.5)
    return pct


def fig_confusion(arch, source="holdout"):
    """1×3: S1 | S2 | S3 untuk satu backbone (holdout atau CV gabungan pada E*)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4)); available = False
    for ax, scenario in zip(axes, SCENARIOS):
        if not done(arch, scenario, "holdout" if source == "holdout" else "cv"):
            ax.axis("off"); ax.set_title(f"{scenario}: belum selesai"); continue
        available = True
        if source == "holdout":
            h = holdout(arch, scenario); y, prob = h["y"], h["prob"]; score = h["summary"]["macro_f1"]
        else:
            y, prob = cv_pooled(arch, scenario); score = f1_score(y, prob.argmax(1), average="macro")
        draw_confusion(ax, y, prob.argmax(1), f"{SCENARIO_NAMES[scenario]}\nMacro-F1 {score:.3f}")
    label = "Holdout" if source == "holdout" else "CV (gabungan fold validasi pada E*)"
    fig.suptitle(f"Confusion matrix {label} — {ARCH_NAMES[arch]}", fontweight="bold"); fig.tight_layout()
    if available:
        save(fig, f"B03_confusion_{source}_{arch}")
    return fig


def fig_confusion_single(arch, scenario, source="holdout"):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    if source == "holdout":
        h = holdout(arch, scenario); y, prob = h["y"], h["prob"]
    else:
        y, prob = cv_pooled(arch, scenario)
    draw_confusion(ax, y, prob.argmax(1), f"{ARCH_NAMES[arch]} {SCENARIO_NAMES[scenario]} ({source})")
    fig.tight_layout(); save(fig, f"B03_confusion_{source}_{arch}_{scenario}"); return fig


def fig_confusion_diff(arch, source="holdout"):
    """Selisih persen baris: S2−S1 dan S3−S2 (poin persen); diagonal positif = recall naik."""
    pairs = [("S2", "S1"), ("S3", "S2")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8)); any_pair = False
    for ax, (a, b) in zip(axes, pairs):
        stage = "holdout" if source == "holdout" else "cv"
        if not (done(arch, a, stage) and done(arch, b, stage)):
            ax.axis("off"); ax.set_title(f"{a}−{b}: belum tersedia"); continue
        any_pair = True
        if source == "holdout":
            ya, pa = holdout(arch, a)["y"], holdout(arch, a)["prob"]; yb, pb = holdout(arch, b)["y"], holdout(arch, b)["prob"]
        else:
            ya, pa = cv_pooled(arch, a); yb, pb = cv_pooled(arch, b)
        base_cm = confusion_matrix(yb, pb.argmax(1), labels=range(5)); base = 100 * base_cm / np.maximum(base_cm.sum(1, keepdims=True), 1)
        draw_confusion(ax, ya, pa.argmax(1), f"{a} − {b} (poin persen baris)", diff_base=base)
    fig.suptitle(f"Perubahan confusion matrix {source} — {ARCH_NAMES[arch]} (biru = naik, merah = turun)", fontweight="bold")
    fig.tight_layout()
    if any_pair:
        save(fig, f"B04_confusion_selisih_{source}_{arch}")
    return fig


def fig_training_curves(arch):
    fig, ax = plt.subplots(figsize=(8, 4))
    for scenario in SCENARIOS:
        if not done(arch, scenario, "cv"):
            continue
        data = cv(arch, scenario); curve = data["curve"]; e = data["summary"]["selected_epoch"]
        ax.plot(curve.epoch, curve.mean_macro_f1, color=COLORS[scenario], label=f"{SCENARIO_NAMES[scenario]} (E*={e})")
        ax.fill_between(curve.epoch, curve.mean_macro_f1 - curve.std_macro_f1, curve.mean_macro_f1 + curve.std_macro_f1, color=COLORS[scenario], alpha=0.15)
        ax.axvline(e, color=COLORS[scenario], ls=":", lw=1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro-F1 validasi (rerata 5 fold ± SD)"); ax.legend(fontsize=8)
    ax.set_title(f"Kurva validasi dan epoch final bersama — {ARCH_NAMES[arch]}"); save(fig, f"B05_kurva_{arch}"); return fig


# --------------------------------------------------------------------------- C. efek skenario

def paired_bootstrap(arch, a, b, replicates=2000, seed=20261105) -> dict:
    """Selisih holdout a−b dengan resampling pasien yang sama untuk kedua model."""
    test = manifest("holdout"); ha, hb = holdout(arch, a), holdout(arch, b)
    if not np.array_equal(ha["y"], hb["y"]):
        raise ValueError("Urutan label holdout berbeda.")
    y, pa, pb = ha["y"], ha["prob"].argmax(1), hb["prob"].argmax(1)
    patients = test.patient_id.astype(str).to_numpy(); index = {p: np.flatnonzero(patients == p) for p in np.unique(patients)}
    keys = list(index); rng = np.random.default_rng(seed); draws = []
    def scores(sel):
        fa = f1_score(y[sel], pa[sel], labels=range(5), average=None, zero_division=0)
        fb = f1_score(y[sel], pb[sel], labels=range(5), average=None, zero_division=0)
        present = np.isin(range(5), y[sel])
        return [fa[present].mean() - fb[present].mean(), *np.where(present, fa - fb, np.nan)]
    point = scores(np.arange(len(y)))
    for _ in range(replicates):
        draws.append(scores(np.concatenate([index[keys[i]] for i in rng.integers(0, len(keys), len(keys))])))
    draws = np.asarray(draws); names = ["macro", *CLASSES]
    return {n: {"delta": float(point[i]), "low": float(np.nanpercentile(draws[:, i], 2.5)), "high": float(np.nanpercentile(draws[:, i], 97.5)),
                "p_le_0": float(np.nanmean(draws[:, i] <= 0))} for i, n in enumerate(names)}


def scenario_effects() -> pd.DataFrame:
    rows = []
    for arch in ARCHS:
        for a, b, effect in [("S2", "S1", "class weighting"), ("S3", "S2", "fine-tuning")]:
            row = {"Backbone": ARCH_NAMES[arch], "Efek": f"{a}−{b} ({effect})"}
            if done(arch, a, "cv") and done(arch, b, "cv"):
                fa, fb = cv(arch, a)["folds"], cv(arch, b)["folds"]
                for metric in ["macro_f1_at_E", *[f"f1_{c}_at_E" for c in CLASSES]]:
                    delta = (fa[metric] - fb[metric]).to_numpy()
                    row[f"CV Δ{metric.replace('_at_E', '')}"] = f"{delta.mean():+.3f} ({(delta > 0).sum()}/5 fold naik)"
            if done(arch, a) and done(arch, b):
                boot = paired_bootstrap(arch, a, b)
                row["Holdout ΔMacro-F1 [CI 95%]"] = f"{boot['macro']['delta']:+.3f} [{boot['macro']['low']:+.3f}, {boot['macro']['high']:+.3f}]"
                for c in CLASSES:
                    row[f"Holdout ΔF1 {c}"] = f"{boot[c]['delta']:+.3f}"
            rows.append(row)
    return save_table(pd.DataFrame(rows), "C01_efek_skenario")


def fig_scenario_effects():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    metrics = ["macro", *CLASSES]
    for ax, (a, b, effect) in zip(axes, [("S2", "S1", "class weighting"), ("S3", "S2", "fine-tuning")]):
        for k, arch in enumerate(ARCHS):
            offset = (k - 1) * 0.25
            if done(arch, a, "cv") and done(arch, b, "cv"):
                fa, fb = cv(arch, a)["folds"], cv(arch, b)["folds"]
                for i, m in enumerate(metrics):
                    column = "macro_f1_at_E" if m == "macro" else f"f1_{m}_at_E"
                    delta = (fa[column] - fb[column]).to_numpy()
                    ax.scatter(np.full(5, i + offset), delta, s=10, color=f"C{k}", alpha=0.5)
            if done(arch, a) and done(arch, b):
                boot = paired_bootstrap(arch, a, b)
                for i, m in enumerate(metrics):
                    ax.errorbar(i + offset, boot[m]["delta"], yerr=[[boot[m]["delta"] - boot[m]["low"]], [boot[m]["high"] - boot[m]["delta"]]],
                                fmt="D", color=f"C{k}", ms=5, capsize=2, label=ARCH_NAMES[arch] if i == 0 else None)
        ax.axhline(0, color="black", lw=0.8); ax.set_xticks(range(len(metrics)), ["Macro"] + list(CLASSES))
        ax.set_title(f"{a} − {b}: efek {effect}"); ax.set_ylabel("Δ F1 (titik = fold CV, ◆ = holdout ± CI 95%)")
    axes[0].legend(fontsize=8); fig.tight_layout(); save(fig, "C02_efek_skenario"); return fig


def fig_f1_heatmap(source="holdout"):
    labels, values = [], []
    for arch, scenario in models("holdout" if source == "holdout" else "cv"):
        labels.append(f"{ARCH_NAMES[arch]} {scenario}")
        if source == "holdout":
            values.append([holdout(arch, scenario)["summary"]["f1"][c] for c in CLASSES])
        else:
            folds = cv(arch, scenario)["folds"]; values.append([folds[f"f1_{c}_at_E"].mean() for c in CLASSES])
    if not values:
        return None
    values = np.asarray(values); fig, ax = plt.subplots(figsize=(7, 0.45 * len(labels) + 1.5))
    ax.imshow(values, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(len(labels)):
        for j in range(5):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8.5)
    ax.set_xticks(range(5), CLASSES); ax.set_yticks(range(len(labels)), labels)
    ax.set_title(f"F1 per kelas ({source})"); fig.tight_layout(); save(fig, f"C03_heatmap_f1_{source}"); return fig


# --------------------------------------------------------------------------- D. per kelas

def per_class_table() -> pd.DataFrame:
    rows = []
    for arch, scenario in models():
        table = holdout(arch, scenario)["per_class"]
        for record in table.to_dict("records"):
            rows.append({"Backbone": ARCH_NAMES[arch], "Skenario": scenario, **record})
    return save_table(pd.DataFrame(rows), "D01_per_kelas_holdout") if rows else pd.DataFrame()


def fig_precision_recall():
    table = per_class_table()
    if table.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    markers = dict(zip(ARCHS, ["o", "s", "^"]))
    for record in table.to_dict("records"):
        arch = [a for a in ARCHS if ARCH_NAMES[a] == record["Backbone"]][0]; color = COLORS[record["Skenario"]]
        axes[0].scatter(record["sensitivity"], record["precision"], marker=markers[arch], color=color, s=40)
        axes[0].annotate(record["class"], (record["sensitivity"], record["precision"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axes[1].scatter(record["specificity"], record["sensitivity"], marker=markers[arch], color=color, s=40)
        axes[1].annotate(record["class"], (record["specificity"], record["sensitivity"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel("Recall / sensitivity"); axes[0].set_ylabel("Precision"); axes[0].set_title("Precision vs recall per kelas (holdout)")
    axes[1].set_xlabel("Specificity"); axes[1].set_ylabel("Sensitivity"); axes[1].set_title("Sensitivity vs specificity per kelas (holdout)")
    handles = [plt.Line2D([], [], marker=markers[a], ls="", color="gray", label=ARCH_NAMES[a]) for a in ARCHS]
    handles += [plt.Line2D([], [], marker="o", ls="", color=COLORS[s], label=s) for s in SCENARIOS]
    axes[1].legend(handles=handles, fontsize=8); fig.tight_layout(); save(fig, "D02_precision_recall"); return fig


def fig_roc(arch, scenario):
    h = holdout(arch, scenario); fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(h["y"] == i, h["prob"][:, i]); ax.plot(fpr, tpr, label=f"{c} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC holdout — {ARCH_NAMES[arch]} {scenario}"); ax.legend(fontsize=8)
    save(fig, f"D03_roc_{arch}_{scenario}"); return fig


def best_scenario(arch):
    candidates = [(holdout(arch, s)["summary"]["macro_f1"], s) for s in SCENARIOS if done(arch, s)]
    return max(candidates)[1] if candidates else None


def fig_s_patients():
    fig, ax = plt.subplots(figsize=(12, 4.4)); labels = []
    for k, (arch, scenario) in enumerate(models()):
        s = holdout(arch, scenario)["s_patients"]; labels.append(f"{ARCH_NAMES[arch]}\n{scenario}")
        jitter = np.random.default_rng(k).uniform(-0.15, 0.15, len(s))
        for database, color in [("mitdb", "#4c72b0"), ("svdb", "#55a868")]:
            part = s.database == database
            ax.scatter(k + jitter[part], s["mean"][part], color=color, s=12 + 2 * s["size"][part], alpha=0.7, label=database if k == 0 else None)
        ax.hlines(s["mean"].median(), k - 0.3, k + 0.3, color="black")
        for _, row in s[s["mean"] < 0.2].iterrows():
            ax.annotate(row.patient_id.split("/")[-1], (k, row["mean"]), fontsize=6.5, xytext=(6, 0), textcoords="offset points")
    if not labels:
        plt.close(fig); return None
    ax.axhline(0.2, color="crimson", ls=":", lw=0.8)
    ax.set_xticks(range(len(labels)), labels, fontsize=8); ax.set_ylabel("Recall S per pasien")
    ax.set_title("Recall kelas S per pasien holdout (garis = median; ukuran titik = jumlah window)"); ax.legend()
    fig.tight_layout(); save(fig, "D04_recall_s_pasien"); return fig


def fig_per_database():
    rows = []
    for arch, scenario in models():
        table = pd.read_csv(out() / "holdout" / arch / scenario / "per_database_class.csv")
        for c in ("S", "V"):
            for database in ("mitdb", "svdb"):
                value = table[(table.database == database) & (table["class"] == c)]
                if len(value):
                    rows.append({"model": f"{ARCH_NAMES[arch]} {scenario}", "kelas": c, "database": database, "f1": float(value.f1.iloc[0])})
    if not rows:
        return None
    frame = pd.DataFrame(rows); fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, c in zip(axes, ("S", "V")):
        pivot = frame[frame.kelas == c].pivot(index="model", columns="database", values="f1")
        pivot.plot.bar(ax=ax, color=["#4c72b0", "#55a868"], rot=45); ax.set_title(f"F1 kelas {c} per database (holdout)"); ax.set_xlabel("")
    fig.tight_layout(); save(fig, "D05_f1_per_database"); return fig


# --------------------------------------------------------------------------- E. kontribusi transfer learning

def modality_table() -> pd.DataFrame:
    rows = []
    for arch, scenario in models("xai"):
        data = json.loads((out() / "xai" / arch / scenario / "modalitas_holdout.json").read_text())
        for condition, scores in data["skor"].items():
            rows.append({"Backbone": ARCH_NAMES[arch], "Skenario": scenario, "Kondisi": condition, **{k: round(v, 3) for k, v in scores.items()}})
        rows.append({"Backbone": ARCH_NAMES[arch], "Skenario": scenario, "Kondisi": "% Shapley citra",
                     **{k: (round(v["persen_citra"], 1) if v["persen_citra"] is not None else None) for k, v in data["shapley"].items()}})
    return save_table(pd.DataFrame(rows), "E01_kontribusi_modalitas") if rows else pd.DataFrame()


def fig_modality():
    available = models("xai")
    if not available:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    conditions = ["gabungan", "tanpa_citra", "tanpa_tabular", "tanpa_keduanya"]; width = 0.8 / len(conditions)
    shares = []
    for k, (arch, scenario) in enumerate(available):
        data = json.loads((out() / "xai" / arch / scenario / "modalitas_holdout.json").read_text())
        for j, condition in enumerate(conditions):
            axes[0].bar(k + (j - 1.5) * width, data["skor"][condition]["macro"], width, color=["#2b8cbe", "#fdae61", "#a6d96a", "#d9d9d9"][j],
                        label=condition.replace("_", " ") if k == 0 else None)
        shares.append([data["shapley"][m]["persen_citra"] or 0 for m in ["macro", *CLASSES]])
    labels = [f"{ARCH_NAMES[a]}\n{s}" for a, s in available]
    axes[0].set_xticks(range(len(labels)), labels, fontsize=7.5); axes[0].set_ylabel("Macro-F1 holdout"); axes[0].legend(fontsize=8)
    axes[0].set_title("Knock-out modalitas (holdout)")
    shares = np.clip(np.asarray(shares), 0, 100); metrics = ["Macro", *CLASSES]
    for i in range(len(metrics)):
        axes[1].bar(np.arange(len(labels)) + (i - 2.5) * 0.13, shares[:, i], 0.13, label=metrics[i])
    axes[1].set_xticks(range(len(labels)), labels, fontsize=7.5); axes[1].set_ylabel("% kontribusi jalur citra (Shapley)")
    axes[1].set_ylim(0, 100); axes[1].legend(fontsize=7.5, ncol=3); axes[1].set_title("Porsi jalur citra (transfer learning) vs tabular")
    fig.tight_layout(); save(fig, "E02_kontribusi_modalitas"); return fig


# --------------------------------------------------------------------------- F. XAI

def _overlay(ax, image, attr, segments=None, width=None, offset=0.0, scale=1.0):
    ax.imshow(image, cmap="gray", aspect="auto")
    a = attr.sum(0) if attr.ndim == 3 else attr; limit = np.percentile(np.abs(a), 99.5) or 1
    ax.imshow(a, cmap="bwr", vmin=-limit, vmax=limit, alpha=0.55, aspect="auto")
    if segments:
        for beat in segments:
            for name, color in [("P", "#2ca02c"), ("QRS", "#9467bd")]:
                if beat.get(name):
                    x0, x1 = [(v - offset) * scale for v in beat[name]]
                    if 0 <= x1 and x0 <= width:
                        ax.axvspan(max(x0, 0), min(x1, width), ymin=0, ymax=0.06, color=color)
    ax.set_xticks([]); ax.set_yticks([])


def fig_xai_gallery(arch, scenarios=("S2", "S3")):
    samples = json.loads((out() / "xai" / "sampel_xai.json").read_text())
    available = [s for s in scenarios if done(arch, s, "xai")]
    if not available:
        return None
    fig, axes = plt.subplots(len(CLASSES), 2 * len(available), figsize=(7 * len(available), 2.0 * len(CLASSES)),
                             gridspec_kw={"width_ratios": [2.2, 1] * len(available)})
    axes = np.atleast_2d(axes)
    for i, c in enumerate(CLASSES):
        window_id = samples["gallery"][c][0]
        for k, scenario in enumerate(available):
            z = np.load(out() / "xai" / arch / scenario / "citra" / f"{window_id}.npz"); segments = json.loads(str(z["segments"]))
            prob = z["probabilities"]
            _overlay(axes[i, 2 * k], z["global_image"][0], z["attr_global"], segments, 448, 0.0, 448 / 2500)
            beat = len(segments) // 2
            _overlay(axes[i, 2 * k + 1], z["local_images"][beat, 0], z["attr_local"][beat], segments, 224,
                     segments[beat]["r"] - 156, 224 / 312)
            axes[i, 2 * k].set_title(f"{scenario} | {c} → {CLASSES[int(prob.argmax())]} (p={prob[CLASSES.index(c)]:.2f})", fontsize=8.5)
        axes[i, 0].set_ylabel(c, rotation=0, ha="right", fontsize=10)
    fig.suptitle(f"SHAP citra (merah = mendukung kelas sebenarnya; hijau = P, ungu = QRS) — {ARCH_NAMES[arch]}", fontweight="bold")
    fig.tight_layout(); save(fig, f"F01_galeri_shap_{arch}"); return fig


def xai_segment_table() -> pd.DataFrame:
    rows = []
    for arch, scenario in models("xai"):
        frame = pd.read_csv(out() / "xai" / arch / scenario / "shap_citra_ringkasan.csv")
        for label, part in frame.groupby("label"):
            row = {"Backbone": ARCH_NAMES[arch], "Skenario": scenario, "Kelas": label, "n": len(part)}
            for view in ("global", "lokal"):
                for segment in ("P", "QRS", "ST-T"):
                    row[f"{view} {segment} (massa/luas)"] = round(float(part[f"{view}_{segment}_rasio"].median()), 2)
            row["|residual additivity| median"] = round(float(part.additivity_residual.abs().median()), 2)
            rows.append(row)
    return save_table(pd.DataFrame(rows), "F02_shap_segmen") if rows else pd.DataFrame()


def fig_xai_segments(view="lokal"):
    table = xai_segment_table()
    if table.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, segment in zip(axes, ("P", "QRS", "ST-T")):
        column = f"{view} {segment} (massa/luas)"
        pivot = table.assign(model=table.Backbone + " " + table.Skenario).pivot(index="Kelas", columns="model", values=column).reindex(CLASSES)
        pivot.plot.bar(ax=ax, rot=0, legend=segment == "ST-T"); ax.axhline(1, color="black", lw=0.8, ls="--")
        ax.set_title(f"Segmen {segment} ({view})"); ax.set_ylabel("Rasio massa |SHAP| / luas (>1 = terkonsentrasi)")
    if axes[-1].get_legend():
        axes[-1].legend(fontsize=7)
    fig.tight_layout(); save(fig, f"F03_shap_segmen_{view}"); return fig


def fig_xai_tabular():
    available = models("xai")
    if not available:
        return None
    import v4_xai as X
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(18, 4.2), sharey=True)
    for ax, c in zip(axes, CLASSES):
        values = []
        for arch, scenario in available:
            frame = pd.read_csv(out() / "xai" / arch / scenario / "shap_tabular_kelompok.csv")
            values.append(frame[frame.label == c][X.GROUPS].abs().mean().to_numpy())
        values = np.asarray(values); y = np.arange(len(X.GROUPS)); height = 0.8 / len(available)
        for k, (arch, scenario) in enumerate(available):
            ax.barh(y + k * height, values[k], height, color=COLORS[scenario], alpha=0.5 + 0.25 * ARCHS.index(arch) / 2,
                    label=f"{ARCH_NAMES[arch]} {scenario}")
        ax.set_yticks(y + 0.4, X.GROUPS, fontsize=8); ax.set_title(f"Kelas {c}"); ax.set_xlabel("rerata |SHAP| kelompok (logit)")
    axes[-1].legend(fontsize=6.5); fig.suptitle("SHAP fitur tabular berkelompok", fontweight="bold"); fig.tight_layout()
    save(fig, "F04_shap_tabular"); return fig


# --------------------------------------------------------------------------- G. efisiensi

def efficiency_table() -> pd.DataFrame:
    folder = out() / "benchmark"
    rows = [json.loads(p.read_text()) for p in sorted(folder.glob("*_S*.json"))] if folder.exists() else []
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    keep = {"arch": "Backbone", "skenario": "Skenario", "parameter_total": "Parameter total", "parameter_dilatih": "Parameter dilatih",
            "ukuran_model_mb": "Ukuran (MB)", "gflops_per_window": "GFLOPs/window", "gpu_median_ms": "Latensi GPU (ms)",
            "cpu_median_ms": "Latensi CPU (ms)", "gpu_peak_memory_mb": "Memori GPU inferensi (MB)",
            "training_vram_puncak_mb": "VRAM training (MB)", "detik_per_epoch_cv": "Detik/epoch", "cv_total_jam": "Total CV (jam)"}
    frame = frame[[k for k in keep if k in frame]].rename(columns=keep)
    frame["Backbone"] = frame.Backbone.map(ARCH_NAMES)
    return save_table(frame.round(2), "G01_efisiensi")


def fig_tradeoff():
    table = efficiency_table()
    if table.empty:
        return None
    # R15 (13 Nov 2026): sumbu Y memakai Macro-F1 holdout model lengkap (model utama paper) bila tersedia.
    # Latensi hanya jalur CNN: paket specialist menyimpan probabilitas terkunci, bukan model GBM, sehingga waktu
    # ekstraksi fitur beat + specialist + fusi tidak diukur; komponen itu sama untuk semua backbone (offset tetap).
    lengkap = lengkap_table("holdout")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    points = {}
    for record in table.to_dict("records"):
        arch = [a for a in ARCHS if ARCH_NAMES[a] == record["Backbone"]][0]
        if not done(arch, record["Skenario"]):
            continue
        if not lengkap.empty:
            score = float(lengkap[(lengkap.Backbone == record["Backbone"]) & (lengkap.Skenario == record["Skenario"])].macro_F1.iloc[0])
        else:
            score = holdout(arch, record["Skenario"])["summary"]["macro_f1"]
        ax.scatter(record["Latensi CPU (ms)"], score, s=20 * record["Ukuran (MB)"], color=COLORS[record["Skenario"]], alpha=0.7, edgecolor="black")
        points.setdefault(record["Backbone"], []).append((record["Latensi CPU (ms)"], score))
    scores = [s for v in points.values() for _, s in v]
    for name, values in points.items():  # satu label per backbone; skenario dibedakan warna
        x = np.mean([p[0] for p in values]); y = min(p[1] for p in values)
        ax.annotate(f"{name}\n{np.mean([p[0] for p in values]):.0f} ms", (x, y), xytext=(0, -26), textcoords="offset points",
                    ha="center", va="top", fontsize=8, fontweight="bold")
    if scores and min(scores) - 0.02 > TARGET:
        ax.set_ylim(min(scores) - 0.02, max(scores) + 0.008)  # target 0,80 di luar rentang; disebut di judul
        ax.set_title("Trade-off kinerja vs efisiensi (ukuran titik = ukuran model; semua di atas target 0,80)")
    else:
        ax.axhline(TARGET, color="crimson", ls="--", lw=0.8)
    ax.set_xlabel("Latensi CPU per window, jalur CNN (ms, median)")
    ax.set_ylabel("Macro-F1 holdout" + (" (model lengkap)" if not lengkap.empty else ""))
    if not ax.get_title():
        ax.set_title("Trade-off kinerja vs efisiensi (ukuran titik = ukuran model)")
    if not lengkap.empty:
        fig.text(0.01, 0.01, "Latensi tidak mencakup ekstraksi fitur beat, Beat-Level Specialist, dan fusi; komponen ini sama untuk semua backbone.",
                 fontsize=7, color="#555555")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[s], markeredgecolor="black", alpha=0.7) for s in SCENARIOS]
    ax.legend(handles, [SCENARIO_NAMES[s] for s in SCENARIOS], fontsize=8, loc="lower right")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, "G02_tradeoff"); return fig


# --------------------------------------------------------------------------- poin bahasan otomatis

def talking_points_main() -> str:
    lines = []
    for arch, scenario in models():
        h = holdout(arch, scenario)["summary"]; ci = h["ci95_patient_bootstrap"]["macro_f1"]
        status = "mencapai" if h["macro_f1"] >= TARGET else "di bawah"
        weakest = min(h["f1"], key=h["f1"].get)
        lines.append(f"- **{ARCH_NAMES[arch]} {scenario}**: Macro-F1 holdout {h['macro_f1']:.3f} (CI 95% {ci['low']:.3f}–{ci['high']:.3f}), "
                     f"{status} target {TARGET:.2f}; kelas terlemah {weakest} (F1 {h['f1'][weakest]:.3f}).")
    return "\n".join(lines) or "_Belum ada model yang dievaluasi pada holdout._"


def talking_points_confusion(arch) -> str:
    lines = []
    for scenario in SCENARIOS:
        if not done(arch, scenario):
            continue
        h = holdout(arch, scenario); cm = confusion_matrix(h["y"], h["prob"].argmax(1), labels=range(5))
        pct = 100 * cm / cm.sum(1, keepdims=True); off = pct.copy(); np.fill_diagonal(off, 0)
        i, j = np.unravel_index(off.argmax(), off.shape)
        recall = ", ".join(f"{c} {pct[k, k]:.1f}%" for k, c in enumerate(CLASSES))
        lines.append(f"- **{scenario}**: recall per kelas {recall}. Kesalahan terbesar: {CLASSES[i]} ditebak {CLASSES[j]} "
                     f"({cm[i, j]} window, {pct[i, j]:.1f}% dari kelas {CLASSES[i]}).")
    return "\n".join(lines) or "_Belum tersedia._"


def talking_points_effects() -> str:
    lines = []
    for arch in ARCHS:
        for a, b, effect in [("S2", "S1", "class weighting"), ("S3", "S2", "fine-tuning")]:
            if done(arch, a) and done(arch, b):
                boot = paired_bootstrap(arch, a, b); m = boot["macro"]
                verdict = "signifikan (CI tidak memuat 0)" if m["low"] > 0 or m["high"] < 0 else "tidak signifikan (CI memuat 0)"
                lines.append(f"- **{ARCH_NAMES[arch]} {a}−{b}** ({effect}): ΔMacro-F1 holdout {m['delta']:+.3f} "
                             f"[{m['low']:+.3f}, {m['high']:+.3f}], {verdict}; ΔF1 S {boot['S']['delta']:+.3f}.")
    return "\n".join(lines) or "_Belum tersedia._"


def _efek_lengkap(arch, pair):
    frame = pd.read_csv(out() / "model_lengkap" / arch / f"banding_{pair}_lengkap.csv").set_index("metrik")
    m, s = frame.loc["macro_F1"], frame.loc["recall_S"]
    return f"{ARCH_NAMES[arch]} ΔMacro {m.selisih:+.3f} (p Holm {m.p_holm:.3f}), Δrecall S {s.selisih:+.3f} (p Holm {s.p_holm:.3f})"


def answers_table() -> pd.DataFrame:
    if all((out() / "model_lengkap" / a / "ringkasan_S1_S2_S3.csv").exists() for a in ARCHS):
        # R15: jawaban memakai model lengkap (model utama paper); CNN saja ada di B01/C01 sebagai komponen.
        holdout_s2 = {a: pd.read_csv(out() / "model_lengkap" / a / "ringkasan_S1_S2_S3.csv", index_col=0).loc["S2 holdout"] for a in ARCHS}
        rows = [
            {"Tujuan / rumusan": "Kinerja model (S2, model lengkap) pada data terbatas & tidak seimbang", "Bukti": "B00",
             "Angka kunci": "; ".join(f"{ARCH_NAMES[a]} Macro {r.macro_F1:.3f}, F1 S {r.F1_S:.3f}" for a, r in holdout_s2.items())},
            {"Tujuan / rumusan": "Efek class weighting (S2 vs S1), model lengkap", "Bukti": "C00",
             "Angka kunci": "; ".join(_efek_lengkap(a, "S2_minus_S1") for a in ARCHS)},
            {"Tujuan / rumusan": "Efek fine-tuning penuh (S3 vs S2), model lengkap", "Bukti": "C00",
             "Angka kunci": "; ".join(_efek_lengkap(a, "S3_minus_S2") for a in ARCHS)},
            {"Tujuan / rumusan": "Kontribusi transfer learning (jalur citra)", "Bukti": "B00.1, E01, E02", "Angka kunci": "CNN saja vs model lengkap; % Shapley citra"},
            {"Tujuan / rumusan": "XAI: fokus model pada fitur klinis (S2 vs S3)", "Bukti": "F01–F04", "Angka kunci": "kualitatif"},
            {"Tujuan / rumusan": "Kelayakan resource-constrained", "Bukti": "G01, G02", "Angka kunci": "latensi CPU/GPU, memori, ukuran"},
        ]
        return save_table(pd.DataFrame(rows), "H01_jawaban_rumusan")
    rows = [
        {"Tujuan / rumusan": "Kinerja model dasar transfer learning dengan class weighting (S2) pada data terbatas & tidak seimbang",
         "Bukti": "B01, B02, B03, C03", "Angka kunci": "; ".join(f"{ARCH_NAMES[a]} {holdout(a, 'S2')['summary']['macro_f1']:.3f}" for a in ARCHS if done(a, "S2"))},
        {"Tujuan / rumusan": "Efek class weighting (S2 vs S1)", "Bukti": "C01, C02, B04",
         "Angka kunci": "; ".join(f"{ARCH_NAMES[a]} {paired_bootstrap(a, 'S2', 'S1')['macro']['delta']:+.3f}" for a in ARCHS if done(a, "S2") and done(a, "S1"))},
        {"Tujuan / rumusan": "Efek fine-tuning (S3 vs S2)", "Bukti": "C01, C02, B04",
         "Angka kunci": "; ".join(f"{ARCH_NAMES[a]} {paired_bootstrap(a, 'S3', 'S2')['macro']['delta']:+.3f}" for a in ARCHS if done(a, "S3") and done(a, "S2"))},
        {"Tujuan / rumusan": "Kontribusi transfer learning (jalur citra)", "Bukti": "E01, E02", "Angka kunci": "lihat % Shapley citra"},
        {"Tujuan / rumusan": "XAI: fokus model pada fitur klinis (S2 vs S3)", "Bukti": "F01–F04", "Angka kunci": "kualitatif"},
        {"Tujuan / rumusan": "Kelayakan resource-constrained", "Bukti": "G01, G02", "Angka kunci": "latensi CPU/GPU, memori, ukuran"},
    ]
    return save_table(pd.DataFrame(rows), "H01_jawaban_rumusan")
