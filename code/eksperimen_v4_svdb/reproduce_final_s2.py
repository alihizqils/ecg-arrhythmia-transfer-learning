"""Uji tambahan G0: apakah mesin lokal (RTX 4050) mereproduksi model final resmi (Colab L4)?

Model final EfficientNet-B0 S2 dilatih ulang secara lokal dengan kode, data, cache embedding, seed (142), dan E* resmi
(6, dari outputs_colab_r15/selection) ke folder keluaran baru outputs_additional_tests/g0_laptop. Tidak ada seleksi:
E* diambil dari receipt resmi, hold-out hanya dipakai untuk membandingkan prediksi dua model.

Kriteria lolos (dikunci sebelum run):
  argmax CNN path sama >= 99 % pada window development dan hold-out, dan |selisih macro-F1 hold-out| <= 0,005.
Pembanding tambahan: model resmi diinferensi ulang secara lokal (memisahkan efek inferensi dari efek training).
Keluaran: outputs_additional_tests/g0_laptop/ (run final baru) dan outputs_additional_tests/g0_laptop/banding_g0.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFF = HERE / "outputs_colab_r15"
NEW = HERE / "outputs_additional_tests" / "g0_laptop"
os.environ["V4_OUTPUT_ROOT"] = str(NEW)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import v4_holdout  # noqa: E402
import v4_pipeline as v4  # noqa: E402
import v4_runner as R  # noqa: E402
from ecgtl.hierarchical.data import read_manifest  # noqa: E402
from ecgtl.hierarchical.training import load_checkpoint  # noqa: E402

ARCH, SC = "efficientnet_b0", "S2"


def macro(y, p):
    return float(f1_score(y, p, labels=range(5), average="macro", zero_division=0))


def scaler_diff(a, b):
    """Selisih absolut maksimum antar state scaler (struktur harus identik)."""
    if isinstance(a, dict):
        assert set(a) == set(b), set(a) ^ set(b)
        return max([scaler_diff(a[k], b[k]) for k in a] + [0.0])
    if isinstance(a, (list, tuple)) and a and isinstance(a[0], (dict, list, tuple, str)):
        assert len(a) == len(b)
        return max(scaler_diff(x, y) for x, y in zip(a, b))
    if isinstance(a, str) or a is None:
        assert a == b, (a, b); return 0.0
    A, B = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert A.shape == B.shape and (np.isnan(A) == np.isnan(B)).all()
    return float(np.nanmax(np.abs(A - B))) if A.size else 0.0


def main():
    dest = NEW / "banding_g0.json"
    if dest.exists():
        print(dest.read_text()); return
    sel = json.loads((OFF / "selection" / ARCH / f"{SC}_final_epoch.json").read_text())
    off_res = json.loads((OFF / "final_models" / ARCH / SC / "final" / "result.json").read_text())
    estar = int(sel["selected_epoch"]); assert estar == off_res["epochs_run"] == 6, (estar, off_res["epochs_run"])
    p = R.protocol(); assert p.out == NEW, p.out
    assert p.hash == off_res["protocol_hash"], "protokol data berbeda dari run resmi"
    assert v4.method_hash() == off_res["method_hash"], "hash metode berbeda dari run resmi"
    exp = R.experiment(ARCH, SC)
    print(time.strftime("%H:%M"), "verifikasi cache embedding", flush=True)
    exp.base.cache_features()
    t0 = time.time()
    res = exp.fit(-1, True, estar)
    train_s = time.time() - t0
    print(time.strftime("%H:%M"), "training selesai", res["training_s"], flush=True)

    ck_new = load_checkpoint(NEW / "final_models" / ARCH / SC / "final" / "best.pt")
    ck_off = load_checkpoint(OFF / "final_models" / ARCH / SC / "final" / "best.pt")
    sdiff_train = scaler_diff(ck_new["rr_scaler"], ck_off["rr_scaler"])
    assert sdiff_train < 1e-6, f"scaler berbeda ({sdiff_train})"   # urutan penjumlahan float ~1e-8
    wdiff = max(float((ck_new["model"][k].float() - ck_off["model"][k].float()).abs().max())
                for k in ck_off["model"] if ck_off["model"][k].is_floating_point())
    a_new = np.load(NEW / "final_models" / ARCH / SC / "final" / "initialization_anchor.npz")
    a_off = np.load(OFF / "final_models" / ARCH / SC / "final" / "initialization_anchor.npz")
    assert (a_new["window_ids"] == a_off["window_ids"]).all()

    scaler = v4.V4Scaler(exp.store, exp.dev.window_id, exp.record_template, exp.beat_record_template)
    sdiff_eval = scaler_diff(scaler.state_dict(), ck_off["rr_scaler"]); assert sdiff_eval < 1e-6, sdiff_eval
    out = {"E*": estar, "detik_training_laptop": res["training_s"], "detik_training_L4": off_res["training_s"],
           "detik_total_laptop": train_s, "selisih_bobot_maks": wdiff,
           "scaler_selisih_training": sdiff_train, "scaler_selisih_evaluasi": sdiff_eval,
           "anchor_prob_selisih_maks": float(np.abs(a_new["probabilities"] - a_off["probabilities"]).max()),
           "anchor_argmax_sama": float((a_new["probabilities"].argmax(1) == a_off["probabilities"].argmax(1)).mean())}
    # development (jalur cache embedding, head saja)
    heads = {}
    for name, ck in [("laptop", ck_new), ("resmi", ck_off)]:
        h = exp._head(); h.load_state_dict(ck["model"], strict=True); heads[name] = h.eval()
    dev = exp.dev.reset_index(drop=True)
    pd_ = {k: exp.predict(h, dev, scaler, True) for k, h in heads.items()}
    yd = pd_["resmi"]["y"]; P = {k: v["prob"].argmax(1) for k, v in pd_.items()}
    out["dev_window"] = int(len(dev)); out["dev_argmax_sama"] = float((P["laptop"] == P["resmi"]).mean())
    out["dev_prob_selisih_maks"] = float(np.abs(pd_["laptop"]["prob"] - pd_["resmi"]["prob"]).max())
    # hold-out (jalur citra mentah, template lama = sama dengan probabilities.npy resmi)
    man = read_manifest(p.prepared / "global_manifest.csv")
    test = v4.label_valid(man[man.split == "holdout"].reset_index(drop=True))
    ref = np.load(OFF / "holdout" / ARCH / SC / "probabilities.npy"); yref = np.load(OFF / "holdout" / ARCH / SC / "labels.npy")
    hp = {}
    for name, ck in [("laptop", ck_new), ("resmi", ck_off)]:
        model = v4_holdout.build_final_model(exp, SC, ck)
        hp[name] = exp.predict(model, test, scaler, False)
        del model; torch.cuda.empty_cache()
    yh = hp["resmi"]["y"]; assert (yh == yref).all()
    pl, pr = hp["laptop"]["prob"].argmax(1), hp["resmi"]["prob"].argmax(1)
    out.update({"holdout_window": int(len(test)),
                "holdout_inferensi_ulang_resmi_vs_colab_argmax_sama": float((pr == ref.argmax(1)).mean()),
                "holdout_inferensi_ulang_resmi_vs_colab_prob_selisih_maks": float(np.abs(hp["resmi"]["prob"] - ref).max()),
                "holdout_argmax_sama_laptop_vs_resmi": float((pl == ref.argmax(1)).mean()),
                "holdout_macro_resmi_colab": macro(yh, ref.argmax(1)), "holdout_macro_laptop": macro(yh, pl),
                "holdout_macro_resmi_inferensi_laptop": macro(yh, pr)})
    out["holdout_selisih_macro"] = out["holdout_macro_laptop"] - out["holdout_macro_resmi_colab"]
    out["lolos"] = bool(out["dev_argmax_sama"] >= 0.99 and out["holdout_argmax_sama_laptop_vs_resmi"] >= 0.99
                        and abs(out["holdout_selisih_macro"]) <= 0.005)
    out["holdout_dipakai_untuk_seleksi"] = False
    dest.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
