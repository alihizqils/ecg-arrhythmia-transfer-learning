"""XAI diulang dengan template rekaman bebas label pada window hold-out.

Sama persis dengan v4_xai.explain (sampel beku, nsamples, background), kecuali fitur template rekaman window
hold-out diganti versi rekaman lengkap (full_record_template.py). Background (development) tetap.
Keluaran: outputs_colab_r15/main_results/xai/<arch>/<sc>/ (berkas baru).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("V4_OUTPUT_ROOT", str(HERE / "outputs_colab_r15"))
sys.path.insert(0, str(HERE))
import v4_xai as X  # noqa: E402
from holdout_labelfree import load_new, patch  # noqa: E402

DEST = HERE / "outputs_colab_r15" / "main_results" / "xai"
NEW = load_new()
_orig = X.load_final


def load_final_labelfree(arch, scenario):
    exp, model, scaler, ck = _orig(arch, scenario)
    hold = exp.all_rows[exp.all_rows.split == "holdout"].window_id.astype(str).tolist()
    exp.record_template, exp.beat_record_template = patch(exp, hold, NEW)
    scaler.record_template, scaler.beat_record_template = exp.record_template, exp.beat_record_template
    return exp, model, scaler, ck


X.load_final = load_final_labelfree

if __name__ == "__main__":
    cfg = [("efficientnet_b0", "S2")] + [(a, s) for a in ["efficientnet_b0", "mobilenet_v2", "shufflenet_v2_x1_0"]
                                         for s in ["S1", "S2", "S3"] if (a, s) != ("efficientnet_b0", "S2")]
    for arch, sc in cfg:
        print(X.explain(arch, sc, output=DEST / arch / sc), flush=True)
