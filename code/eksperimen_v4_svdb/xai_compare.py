"""Bandingkan XAI EfficientNet-B0 S2 lama (template dari window terpilih) dan baru (template bebas label)."""
import numpy as np
import pandas as pd

from cv_isolated import OUT

old = OUT / "xai" / "efficientnet_b0" / "S2"; new = OUT / "main_results" / "xai" / "efficientnet_b0" / "S2"
a = pd.read_csv(old / "shap_citra_ringkasan.csv").set_index("window_id")
b = pd.read_csv(new / "shap_citra_ringkasan.csv").set_index("window_id").loc[a.index]
cols = [c for c in a.columns if c.endswith("_rasio")]
d = (b[cols] - a[cols]).abs()
print("selisih |rasio massa/luas| per window: median", round(float(np.nanmedian(d.values)), 4), "maks", round(float(np.nanmax(d.values)), 4))
print("prediksi sama:", float((a.pred == b.pred).mean()))
med_a = a.groupby("label")[cols].median(); med_b = b.groupby("label")[cols].median()
print("selisih median per kelas maks:", round(float((med_b - med_a).abs().max().max()), 4))
ta = pd.read_csv(old / "shap_tabular_kelompok.csv"); tb = pd.read_csv(new / "shap_tabular_kelompok.csv")
g = [c for c in ta.columns if c not in ("window_id", "label", "pred") and ta[c].dtype != object]
sa = ta[g].abs().mean(); sb = tb[g].abs().mean()
print(pd.DataFrame({"lama_%": 100 * sa / sa.sum(), "baru_%": 100 * sb / sb.sum()}).round(1))
out = pd.DataFrame({"metrik": ["median_selisih_rasio", "maks_selisih_rasio", "maks_selisih_median_kelas"],
                    "nilai": [float(np.nanmedian(d.values)), float(np.nanmax(d.values)), float((med_b - med_a).abs().max().max())]})
out.to_csv(OUT / "main_results" / "banding_xai_effb0_s2.csv", index=False)
