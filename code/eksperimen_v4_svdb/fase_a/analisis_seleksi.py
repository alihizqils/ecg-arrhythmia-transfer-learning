"""Fase A lanjutan: risiko AF tak beranotasi di SVDB dan simulasi komposisi.

SVDB praktis tidak mempunyai anotasi ritme, sehingga jendela N dilabeli hanya
dari anotasi beat. Ketidakteraturan RR (dari anotasi beat) dibandingkan dengan
jendela N dan AF mitdb, yang mempunyai anotasi ritme dan beat sekaligus.
Simulasi komposisi memakai pemilihan merata `select_capped_windows` milik proyek.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from ecgtl.data.build_dataset import select_capped_windows  # noqa: E402
from ecgtl.data.labeling import LabeledWindow, Sumber  # noqa: E402

OUT = HERE / "hasil"
CLASSES = ["N", "S", "V", "AF", "VT"]
BEATS = set("NLRejAaJSVEF/fQ")
WINDOW = {"svdb": 1280, "mitdb": 3600}  # 10 detik pada laju asli


def rr_irregularity(database: str, windows: pd.DataFrame) -> pd.DataFrame:
    """Median |dRR| / median RR per jendela, dari anotasi beat (label-free terhadap kelas)."""
    rows = []
    for record, group in windows.groupby("record"):
        ann = wfdb.rdann(str(ROOT / "data/raw" / database / str(record)), "atr")
        beats = np.array([s for s, c in zip(ann.sample, ann.symbol) if c in BEATS])
        for w in group.itertuples():
            b = beats[(beats >= w.start) & (beats < w.start + WINDOW[database])]
            rr = np.diff(b)
            value = np.median(np.abs(np.diff(rr))) / np.median(rr) if len(rr) >= 3 else np.nan
            rows.append({"database": database, "record": str(record), "start": w.start,
                         "label": w.label, "irregularity": value})
    return pd.DataFrame(rows)


def capped(windows: pd.DataFrame, cap: int) -> pd.DataFrame:
    items = [LabeledWindow(r.database, str(r.record), r.start, r.start + 1, r.label, Sumber.DETAK, 0.0, 0)
             for r in windows.itertuples()]
    chosen = select_capped_windows(items, cap)
    return pd.DataFrame([{"record": w.record, "label": w.label} for w in chosen])


def main():
    svdb = pd.read_csv(OUT / "svdb_window_berlabel.csv", dtype={"record": str})
    from audit_svdb import window_audit  # noqa: E402
    mit = []
    for r in sorted(p.stem for p in (ROOT / "data/raw/mitdb").glob("*.hea")):
        mit += window_audit("mitdb", ROOT / "data/raw/mitdb", r, True)[0]
    mit = pd.DataFrame(mit)

    irr = pd.concat([
        rr_irregularity("svdb", svdb[svdb.label == "N"]),
        rr_irregularity("mitdb", mit[mit.label.isin(["N", "AF"])]),
    ])
    irr.to_csv(OUT / "rr_irregularity.csv", index=False)
    ref_n = irr[(irr.database == "mitdb") & (irr.label == "N")].irregularity.dropna()
    ref_af = irr[(irr.database == "mitdb") & (irr.label == "AF")].irregularity.dropna()
    threshold = float(ref_n.quantile(0.99))
    sv_n = irr[irr.database == "svdb"]
    per_record = sv_n.groupby("record").irregularity.apply(lambda x: float((x > threshold).mean()))
    risk = {
        "ambang_p99_mitdb_N": threshold,
        "mitdb_AF_di_atas_ambang": float((ref_af > threshold).mean()),
        "mitdb_N_median": float(ref_n.median()),
        "mitdb_AF_median": float(ref_af.median()),
        "svdb_N_median": float(sv_n.irregularity.median()),
        "svdb_N_di_atas_ambang": float((sv_n.irregularity > threshold).mean()),
        "svdb_rekaman_N_tidak_teratur_gt30pct": per_record[per_record > 0.3].round(3).to_dict(),
    }

    # Komposisi gabungan untuk beberapa batas per rekaman per kelas.
    manifest = pd.read_csv(ROOT / "data/processed/hierarchical_10s_local_v1/global_manifest.csv")
    base = manifest.label.value_counts().reindex(CLASSES, fill_value=0)
    s_per_record = svdb[svdb.label == "S"].record.value_counts()
    scenarios = []
    for min_s in [1, 10, 20, 40]:
        eligible = s_per_record[s_per_record >= min_s].index
        for cap in [20, 40, 60]:
            chosen = capped(svdb[svdb.record.isin(eligible)], cap)
            add = chosen.label.value_counts().reindex(CLASSES, fill_value=0)
            total = base + add
            scenarios.append({
                "min_S_windows_per_record": min_s, "cap_per_record_per_class": cap,
                "svdb_records": len(eligible),
                **{f"add_{c}": int(add[c]) for c in ["N", "S", "V"]},
                "total_windows": int(total.sum()),
                **{f"pct_{c}": round(100 * total[c] / total.sum(), 2) for c in CLASSES},
                "ratio_N_to_S": round(total["N"] / total["S"], 2),
            })
    scenarios = pd.DataFrame(scenarios)
    scenarios.to_csv(OUT / "simulasi_komposisi.csv", index=False)
    (OUT / "risiko_af_tak_beranotasi.json").write_text(json.dumps(risk, indent=2))
    print(json.dumps(risk, indent=2))
    print("komposisi data lama:", base.to_dict(), int(base.sum()))
    print(scenarios.to_string(index=False))


if __name__ == "__main__":
    main()
