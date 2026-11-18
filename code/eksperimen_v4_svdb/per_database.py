"""Uji B3: tabel recall hold-out per kelas dan database sumber.

Tanpa inferensi baru. Sumber: main_results/per_database_holdout.csv (model lengkap, template bebas label).
Keluaran: additional_tests/b3_per_database.csv (rentang semua konfigurasi) dan b3_per_database_S2.csv.
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs_colab_r15"
DEST = OUT / "additional_tests"; DEST.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(OUT / "main_results" / "per_database_holdout.csv")
semua = d.groupby(["kelas", "database"]).agg(n_window=("n", "first"), pasien=("pasien", "first"),
                                              recall_min=("recall", "min"), recall_max=("recall", "max")).reset_index()
semua.to_csv(DEST / "b3_per_database.csv", index=False)
s2 = d[d.sc == "S2"].pivot_table(index=["kelas", "database", "n", "pasien"], columns="arch", values="recall").reset_index()
s2.to_csv(DEST / "b3_per_database_S2.csv", index=False)
pd.set_option("display.width", 200)
print(semua.round(3).to_string(index=False)); print(s2.round(3).to_string(index=False))
