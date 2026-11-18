"""Ekstraksi fitur tingkat beat untuk studi N vs S (7 Nov 2026).

    python eksperimen_v4_svdb/fase_e_ns/fitur_ns.py [--fs 250] [--workers 10]

Satu baris = satu R-peak terdeteksi (GQRS, sama dengan V4) pada rekaman MITDB/SVDB yang
dipakai V4. Label dari simbol anotasi asli yang cocok dengan R-peak itu (toleransi V4):
    N = simbol "N";  S = simbol "A", "a", "J", "S";  lainnya = konteks saja (tidak dilatih/dievaluasi).
Label TIDAK dipakai untuk menghitung fitur apa pun. Template "beat stabil" dipilih dari
ritme dan bentuk QRS saja, hanya dari beat sebelumnya (kausal).

Kelompok fitur (fungsi terpisah):
    E0  fitur_e0_baseline         fitur tabular model V4 (RR, fisiologi, gelombang P/QRS, template rekaman,
                                  RR dominan, lead kedua), dihitung pada blok 10 detik seperti window V4
    E1  fitur_e1_rr_lokal         Local RR Dynamics (prefix e1c_ = kausal, e1o_ = butuh beat berikutnya)
    E2  fitur_e2_deviasi_p        Patient-specific P-wave deviation terhadap template P lokal kausal
    E3  fitur_e3_konteks          Multi-beat temporal context, L = 5/7/11 (e3c{L}_ kausal, e3o{L}_ offline)
    E4  fitur_e4_gelombang_t      T-wave deformation / hidden atrial activity
    E5  fitur_e5_wavelet          Multi-scale wavelet (DWT db4)

Keluaran: fase_e_ns/fitur/<fs>hz/ (meta.csv, fitur.npz, nama_fitur.csv, info.json). Tidak menimpa:
berkas yang sudah ada membuat skrip berhenti.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import detrend, savgol_filter
from wfdb.processing import gqrs_detect

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
for path in [CODE, CODE / "src", CODE / "eksperimen_v4_svdb"]:
    sys.path.insert(0, str(path))

from ecgtl.data.build_dataset import load_channel_signal  # noqa: E402
from ecgtl.data.segmentation import preprocess_record  # noqa: E402
from ecgtl.hierarchical.data import match_peaks  # noqa: E402
from tools.a5_physiology import (GLOBAL_PHYSIOLOGY_NAMES, LOCAL_PHYSIOLOGY_NAMES,  # noqa: E402
                                 _append_global_morphology, _beat_features, _global_rr_features)
from tools.a6_wave_delineation import GLOBAL_WAVE_NAMES, LOCAL_WAVE_NAMES, extract_wave_features  # noqa: E402
from tools.rr_ablation import GLOBAL_RR_NAMES, LOCAL_RR_NAMES  # noqa: E402
import v4_record_template as RT  # noqa: E402

RAW = CODE / "data" / "raw"
PREP = CODE / "data" / "processed" / "v4_svdb" / "prepared"
BANDPASS = {"enabled": True, "low_hz": 0.5, "high_hz": 40.0, "order": 4, "zero_phase": True}
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
S_SYMBOLS = {"A", "a", "J", "S"}
CONTEXTS = (5, 7, 11)
BASELINES = (5, 10, 20)
# db4 (Daubechies 4), filter dekomposisi.
DB4_LO = np.array([-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114,
                   -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523])
DB4_HI = np.array([-0.23037781330885523, 0.7148465705525415, -0.6308807679295904, -0.02798376941698385,
                   0.18703481171888114, 0.030841381835986965, -0.032883011666982945, -0.010597401784997278])


# ----------------------------------------------------------------------------- utilitas
def _corr(a, b):
    if a is None or b is None or len(a) != len(b) or len(a) < 3:
        return np.nan
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else np.nan


def _seg(signal, center, left, right):
    start, stop = center - left, center + right + 1
    return signal[start:stop].astype(np.float64) if start >= 0 and stop <= len(signal) else None


def _robust_scale(signal):
    med = np.median(signal)
    scale = 1.4826 * np.median(np.abs(signal - med))
    return float(scale) if scale > 1e-9 else float(np.std(signal) + 1e-9)


def _robust_median(values):
    """Median dua tahap: buang interval yang menyimpang > 20% dari median awal."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    m0 = np.median(values)
    kept = values[np.abs(values / m0 - 1) < 0.2]
    return float(np.median(kept)) if len(kept) >= 3 else float(m0)


def _mad(values):
    values = values[np.isfinite(values)]
    return float(np.median(np.abs(values - np.median(values)))) if len(values) else np.nan


# ----------------------------------------------------------------------------- rekaman
def records(databases=("mitdb", "svdb")):
    g = pd.read_csv(PREP / "global_manifest.csv", dtype={"record": str})
    g = g[g.database.isin(list(databases))]
    return (g.groupby(["database", "record"]).agg(patient_id=("patient_id", "first"), channel=("channel", "first"),
                                                 split=("split", "first"), fold=("fold", "first"))
            .reset_index())


def load_record(database, record, channel, fs, annotations=True):
    """annotations=False: tanpa membaca anotasi (dipakai V4 Revisi 14; fitur tidak memerlukan label)."""
    raw, fs_raw, used = load_channel_signal(RAW / database, record, [channel])
    assert used == channel
    signal = preprocess_record(raw, fs_raw, fs, BANDPASS).signal.astype(np.float64)
    full = wfdb.rdrecord(str(RAW / database / record))
    other = 1 - list(full.sig_name).index(channel)
    second = preprocess_record(np.nan_to_num(full.p_signal[:, other].astype(np.float64)), fs_raw, fs, BANDPASS).signal
    offset = int(round(8 * fs / 250))  # offset detektor V4 (8 sampel pada 250 Hz)
    peaks = gqrs_detect(sig=signal, fs=fs) + offset
    peaks = np.unique(peaks[(peaks >= 0) & (peaks < len(signal))]).astype(np.int64)
    if not annotations:
        return {"signal": signal, "second": np.asarray(second, np.float64), "peaks": peaks, "fs": float(fs)}
    ann = wfdb.rdann(str(RAW / database / record), "atr")
    keep =[i for i, s in enumerate(ann.symbol) if s in BEAT_SYMBOLS]
    ref = np.rint(np.asarray(ann.sample)[keep] * fs / fs_raw).astype(np.int64)
    symbols = np.asarray(ann.symbol)[keep]
    matches = match_peaks(ref, peaks, int(round(38 * fs / 250)))
    symbol = np.full(len(peaks), "unmatched", dtype=object)
    distance_ms = np.full(len(peaks), np.nan)
    for j, (i, dist) in matches.items():
        symbol[j] = symbols[i]; distance_ms[j] = dist * 1000 / fs
    return {"signal": signal, "second": np.asarray(second, np.float64), "peaks": peaks, "fs": float(fs),
            "symbol": symbol, "distance_ms": distance_ms, "n_reference": len(ref), "n_matched": len(matches)}


# ----------------------------------------------------------------------------- E0
def _record_template_beats(signal, peaks, fs):
    ql, qr = int(round(RT._QRS[0] * fs)), int(round(RT._QRS[1] * fs))
    pl, pr = int(round(RT._PRE[0] * fs)), int(round(RT._PRE[1] * fs))
    qrs = [None if (s := _seg(signal, p, ql, qr)) is None else RT._znorm(s) for p in peaks]
    pre = [None if (s := _seg(signal, p, pl, -pr)) is None else RT._znorm(s) for p in peaks]
    qt = RT._znorm(np.median(np.stack([q for q in qrs if q is not None]), 0))
    pt = RT._znorm(np.median(np.stack([q for q in pre if q is not None]), 0))
    return (np.array([RT._correlation(q, qt) for q in qrs]), np.array([RT._correlation(q, pt) for q in pre]))


def fitur_e0_baseline(rec):
    signal, peaks, fs = rec["signal"], rec["peaks"], rec["fs"]
    n, width = len(peaks), int(round(10 * fs))
    rr_prev = np.r_[np.nan, np.diff(peaks) / fs]
    rr_next = np.r_[np.diff(peaks) / fs, np.nan]
    record_median = float(np.median(np.diff(peaks) / fs))
    qrs_rec, p_rec = _record_template_beats(signal, peaks, fs)
    qrs_rec2, p_rec2 = _record_template_beats(rec["second"], peaks, fs)
    ratio_rec = rr_prev / record_median
    cols = {}
    beat_local = np.full((n, 6), np.nan); phys = np.full((n, 12), np.nan); wave = np.full((n, 13), np.nan)
    win = {k: np.full((n, d), np.nan) for k, d in [("rr", 9), ("phys", 12), ("wave", 10), ("rt", 9), ("dom", 8), ("rt2", 9)]}
    block = peaks // width
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        start = int(b) * width
        seg = signal[start:start + width]
        local = peaks[idx] - start
        rr = np.diff(local) / fs
        median = float(np.median(rr)) if len(rr) else np.nan
        with np.errstate(all="ignore"):
            beat_local[idx] = np.column_stack([rr_prev[idx], rr_next[idx], rr_prev[idx] / median, rr_next[idx] / median,
                                               rr_next[idx] / rr_prev[idx], (rr_prev[idx] + rr_next[idx]) / (2 * median)])
        lp = _beat_features(seg, local, fs)
        phys[idx] = lp
        win["phys"][idx] = _append_global_morphology(_global_rr_features(local, fs), lp)
        gw, lw = extract_wave_features(seg, local, fs)
        wave[idx] = lw; win["wave"][idx] = gw
        g = np.full(9, np.nan)
        g[0] = len(local)
        if len(rr):
            mean = rr.mean(); q25, q75 = np.percentile(rr, [25, 75])
            g[1:] = [median, 60 / mean, rr.std(ddof=1) if len(rr) > 1 else np.nan,
                     np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else np.nan, rr.std() / mean, q75 - q25,
                     rr.min() / median, rr.max() / median]
        win["rr"][idx] = g
        win["rt"][idx] = RT._window_features(np.column_stack([ratio_rec[idx], qrs_rec[idx], p_rec[idx]]))
        win["rt2"][idx] = RT._window_features(np.column_stack([ratio_rec[idx], qrs_rec2[idx], p_rec2[idx]]))
        win["dom"][idx] = RT.dominant_rr_features(lp[:, 5], qrs_rec[idx], rr_prev[idx])
    for i, name in enumerate(LOCAL_RR_NAMES):
        cols[f"e0_{name}"] = beat_local[:, i]
    cols["e0_rr_previous_over_record_median"] = ratio_rec
    cols["e0_rr_next_over_record_median"] = rr_next / record_median
    cols["e0_drr_next_minus_previous_over_record_median"] = (rr_next - rr_prev) / record_median
    for i, name in enumerate(LOCAL_PHYSIOLOGY_NAMES):
        cols[f"e0_{name}"] = phys[:, i]
    for i, name in enumerate(LOCAL_WAVE_NAMES):
        cols[f"e0_{name}"] = wave[:, i]
    cols["e0_beat_rec_qrs_correlation"] = qrs_rec
    cols["e0_beat_rec_p_correlation"] = p_rec
    cols["e0_beat_rec_p_correlation_previous_beat"] = np.r_[np.nan, p_rec[:-1]]
    for key, names in [("rr", GLOBAL_RR_NAMES), ("phys", GLOBAL_PHYSIOLOGY_NAMES), ("wave", GLOBAL_WAVE_NAMES),
                       ("rt", RT.RECORD_TEMPLATE_NAMES), ("dom", RT.DOMINANT_RR_NAMES),
                       ("rt2", [f"lead2_{x}" for x in RT.RECORD_TEMPLATE_NAMES])]:
        for i, name in enumerate(names):
            cols[f"e0w_{name}"] = win[key][:, i]
    cols["e0w_record_median_rr_s"] = np.full(n, record_median)
    return pd.DataFrame(cols), {"qrs_rec": qrs_rec, "p_present": wave[:, 1], "p_conf": wave[:, 0],
                                "pr": wave[:, 4], "p_pol": wave[:, 5]}


# ----------------------------------------------------------------------------- E1
def fitur_e1_rr_lokal(rec):
    peaks, fs = rec["peaks"], rec["fs"]
    n = len(peaks)
    rr = np.r_[np.nan, np.diff(peaks) / fs]          # rr[i] = RR_i (beat i-1 -> i)
    rr_next = np.r_[rr[1:], np.nan]                  # RR_{i+1}
    rr_before = np.r_[np.nan, rr[:-1]]               # RR_{i-1}
    c = {}
    base = {}
    for k in BASELINES:
        base[k] = np.array([_robust_median(rr[max(1, i - k):i]) if i - 1 >= 3 else np.nan for i in range(n)])
        c[f"e1c_rr_relative_b{k}"] = rr / base[k]
    with np.errstate(all="ignore"):
        c["e1c_rr_change_ratio"] = (rr - rr_before) / rr_before
        c["e1c_rr_ratio_previous"] = rr / rr_before
    stats = np.full((n, 4), np.nan)
    for i in range(n):
        w = rr[max(1, i - 10):i]
        w = w[np.isfinite(w)]
        if len(w) >= 3:
            med = np.median(w)
            stats[i] = [med, w.std(ddof=1), _mad(w), w.std(ddof=1) / w.mean()]
    for j, name in enumerate(["median", "std", "mad", "cv"]):
        c[f"e1c_local_rr_{name}_10"] = stats[:, j]
    c["e1c_rr_minus_local_median_over_mad"] = (rr - stats[:, 0]) / (stats[:, 2] + 1e-3)
    # Offline (memerlukan RR_{i+1})
    with np.errstate(all="ignore"):
        c["e1o_rr_asymmetry"] = (rr_next - rr) / (rr_next + rr)
        c["e1o_rr_next_relative_b10"] = rr_next / base[10]
    centered = np.full(n, np.nan); cstats = np.full((n, 3), np.nan)
    for i in range(n):
        w = np.r_[rr[max(1, i - 5):i], rr[i + 2:i + 7]]
        w = w[np.isfinite(w)]
        if len(w) >= 4:
            centered[i] = _robust_median(w)
            cstats[i] = [w.std(ddof=1), _mad(w), w.std(ddof=1) / w.mean()]
    c["e1o_rr_relative_centered"] = rr / centered
    for j, name in enumerate(["std", "mad", "cv"]):
        c[f"e1o_local_rr_{name}_centered"] = cstats[:, j]
    return pd.DataFrame(c), {"rr_rel10": c["e1c_rr_relative_b10"], "base10": base[10], "rr": rr, "rr_next": rr_next}


# ----------------------------------------------------------------------------- beat stabil (kausal, tanpa label)
def beat_stabil(qrs_local_corr, rr_rel10, rr_next_rel10):
    """Beat j stabil bila tidak prematur, tidak diikuti jeda, dan QRS mirip template lokal."""
    with np.errstate(invalid="ignore"):
        return ((np.abs(rr_rel10 - 1) < 0.10) & (np.abs(rr_next_rel10 - 1) < 0.15) & (qrs_local_corr >= 0.90))


def _qrs_local(signal, peaks, fs, depth=20):
    """Korelasi QRS beat i vs median QRS 20 beat sebelumnya (kausal)."""
    ql, qr = int(round(0.10 * fs)), int(round(0.14 * fs))
    segs = [_seg(signal, p, ql, qr) for p in peaks]
    out = np.full(len(peaks), np.nan)
    for i in range(len(peaks)):
        prev = [s for s in segs[max(0, i - depth):i] if s is not None]
        if len(prev) >= 5 and segs[i] is not None:
            out[i] = _corr(segs[i], np.median(np.stack(prev), 0))
    return out


def _template_indices(stable, i, need=5, depth=40, use=20):
    """Indeks beat stabil sebelum i (maks. `use` terbaru dari `depth` beat terakhir). j+1 <= i agar RR_{j+1} diketahui."""
    lo = max(0, i - depth)
    cand = np.flatnonzero(stable[lo:i]) + lo
    cand = cand[cand + 1 <= i]
    cand = cand[-use:]
    return cand if len(cand) >= need else None


# ----------------------------------------------------------------------------- E2
def _p_width(seg, scale):
    k = int(np.argmax(np.abs(seg))); amp = seg[k]
    thr = max(0.1 * abs(amp), 1e-9)
    left = k
    while left > 0 and abs(seg[left]) > thr and np.sign(seg[left]) == np.sign(amp):
        left -= 1
    right = k
    while right < len(seg) - 1 and abs(seg[right]) > thr and np.sign(seg[right]) == np.sign(amp):
        right += 1
    return right - left, amp


def fitur_e2_deviasi_p(rec, e0, e1, stable):
    signal, peaks, fs = rec["signal"], rec["peaks"], rec["fs"]
    smooth = savgol_filter(signal, window_length=max(5, int(round(0.04 * fs)) // 2 * 2 + 1), polyorder=2)
    pl, pr = int(round(0.30 * fs)), int(round(0.06 * fs))
    lag_max = int(round(0.04 * fs))
    regions = []
    for p in peaks:
        s = _seg(smooth, p, pl, -pr)
        regions.append(None if s is None else detrend(s, type="linear"))
    scale = _robust_scale(signal)
    n = len(peaks)
    names = ["p_template_corr", "p_deviation_dp", "p_alignment_lag_s", "p_amplitude_deviation", "p_duration_deviation_s",
             "p_area_deviation", "pr_relative", "p_polarity", "p_polarity_change", "p_confidence_relative",
             "p_template_available", "p_template_beats"]
    out = np.full((n, len(names)), np.nan)
    pr_all, conf, pol = e0["pr"], e0["p_conf"], e0["p_pol"]
    usable = stable & np.array([r is not None for r in regions])
    for i in range(n):
        out[i, 7] = pol[i]
        cand = regions[i]
        idx = _template_indices(usable, i)
        if cand is None or idx is None:
            out[i, 10] = 0; out[i, 11] = 0 if idx is None else len(idx)
            continue
        tmpl = np.median(np.stack([regions[j] for j in idx]), 0)
        best, lag_best = -np.inf, 0
        core = slice(lag_max, len(tmpl) - lag_max)
        for lag in range(-lag_max, lag_max + 1):
            c = _corr(cand[lag_max + lag:len(cand) - lag_max + lag], tmpl[core])
            if np.isfinite(c) and c > best:
                best, lag_best = c, lag
        aligned = cand[lag_max + lag_best:len(cand) - lag_max + lag_best]
        t = tmpl[core]
        wc, ac = _p_width(aligned, scale); wt, at = _p_width(t, scale)
        out[i, 0] = best if np.isfinite(best) else np.nan
        out[i, 1] = 1 - out[i, 0]
        out[i, 2] = lag_best / fs
        out[i, 3] = (abs(ac) - abs(at)) / (abs(at) + 1e-9)
        out[i, 4] = (wc - wt) / fs
        out[i, 5] = (np.abs(aligned).sum() - np.abs(t).sum()) / (np.abs(t).sum() + 1e-9)
        prs = pr_all[idx][np.isfinite(pr_all[idx])]
        out[i, 6] = pr_all[i] / np.median(prs) if len(prs) and np.isfinite(pr_all[i]) else np.nan
        pols = pol[idx][np.isfinite(pol[idx])]
        if len(pols) and np.isfinite(pol[i]):
            out[i, 8] = float(pol[i] != np.sign(np.sum(pols)))
        confs = conf[idx][np.isfinite(conf[idx])]
        out[i, 9] = conf[i] / np.median(confs) if len(confs) and np.isfinite(conf[i]) else np.nan
        out[i, 10] = 1; out[i, 11] = len(idx)
    return pd.DataFrame(out, columns=[f"e2_{x}" for x in names])


# ----------------------------------------------------------------------------- E3
def fitur_e3_konteks(e1, e2, qrs_local):
    rr_rel = np.asarray(e1["rr_rel10"], np.float64)
    dp = e2["e2_p_deviation_dp"].to_numpy()
    rr = e1["rr"]
    n = len(rr_rel)
    cols = {}
    for L in CONTEXTS:
        for mode, offsets in [("c", np.arange(-(L - 1), 1)), ("o", np.arange(-(L // 2), L // 2 + 1))]:
            pre = f"e3{mode}{L}_"
            idx = np.arange(n)[:, None] + offsets[None, :]
            valid = (idx >= 0) & (idx < n)
            idx = np.clip(idx, 0, n - 1)
            seq = {"rr_rel": np.where(valid, rr_rel[idx], np.nan), "p_dev": np.where(valid, dp[idx], np.nan),
                   "qrs_sim": np.where(valid, qrs_local[idx], np.nan)}
            for key, values in seq.items():
                for k, off in enumerate(offsets):
                    if off != 0 or key != "rr_rel":  # posisi 0 rr_rel sudah ada di E1
                        cols[f"{pre}{key}_{off:+d}"] = values[:, k]
            r = np.where(valid, rr[idx], np.nan)
            with np.errstate(all="ignore"):
                x = offsets - offsets.mean()
                finite = np.isfinite(r)
                rm = np.where(finite, r, 0)
                cnt = finite.sum(1)
                mean = rm.sum(1) / np.maximum(cnt, 1)
                slope = ((np.where(finite, (r - mean[:, None]) * x, 0)).sum(1)
                         / np.maximum((np.where(finite, x * x, 0)).sum(1), 1e-9))
                cols[f"{pre}rr_trend"] = np.where(cnt >= 3, slope / mean, np.nan)
                sd = np.sqrt(np.where(finite, (r - mean[:, None]) ** 2, 0).sum(1) / np.maximum(cnt - 1, 1))
                cols[f"{pre}rr_cv"] = np.where(cnt >= 3, sd / mean, np.nan)
                d = np.diff(r, axis=1)
                sign = np.sign(d)
                for k in range(sign.shape[1]):
                    cols[f"{pre}drr_sign_{k}"] = sign[:, k]
                alt = (sign[:, 1:] * sign[:, :-1] < 0).astype(float)
                alt[~np.isfinite(sign[:, 1:] * sign[:, :-1])] = np.nan
                cols[f"{pre}short_long_alternation"] = np.nanmean(alt, 1) if alt.shape[1] else np.nan
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------------- E4
def _region_vs_template(signal, peaks, fs, i, idx, start_off, stop_off):
    """Bandingkan signal[R_{i-1}+start_off : R_{i-1}+stop_off] dengan median segmen yang sama setelah beat template."""
    p = peaks[i - 1]
    a, b = p + start_off, p + stop_off
    if b - a < int(round(0.08 * fs)) or a < 0 or b > len(signal):
        return None
    x = signal[a:b]
    segs = [signal[peaks[j] + start_off:peaks[j] + stop_off] for j in idx if peaks[j] + stop_off <= len(signal)]
    if len(segs) < 5:
        return None
    t = np.median(np.stack(segs), 0)
    x = x - x[:max(2, len(x) // 10)].mean(); t = t - t[:max(2, len(t) // 10)].mean()
    return x, t


def fitur_e4_gelombang_t(rec, stable):
    signal, peaks, fs = rec["signal"], rec["peaks"], rec["fs"]
    n = len(peaks)
    scale = _robust_scale(signal)
    names = ["t_morphology_corr", "t_residual_energy", "t_slope_deviation", "t_area_deviation",
             "pre_candidate_corr", "pre_candidate_residual_energy", "transition_residual_fraction", "t_template_available"]
    out = np.full((n, len(names)), np.nan)
    out[:, 7] = 0
    rr_next = np.r_[np.diff(peaks), 10 ** 9]  # sampel; RR_{j+1}
    p_off = int(round(0.30 * fs)); qrs_guard = int(round(0.06 * fs))
    for i in range(1, n):
        rr_i = peaks[i] - peaks[i - 1]
        base = _template_indices(stable, i, need=5, depth=60, use=40)
        if base is None:
            continue
        # (a) gelombang T beat sebelumnya: R_{i-1}+120 ms s.d. min(R_{i-1}+400 ms, R_i-60 ms)
        t0 = int(round(0.12 * fs)); t1 = min(int(round(0.40 * fs)), rr_i - qrs_guard)
        idx = base[rr_next[base] - p_off >= t1]
        idx = idx[-20:]
        got = _region_vs_template(signal, peaks, fs, i, idx, t0, t1) if len(idx) >= 5 else None
        if got is not None:
            x, t = got
            r = x - t
            out[i, 0] = _corr(x, t)
            out[i, 1] = (r ** 2).sum() / ((t - t.mean()) ** 2).sum().clip(1e-9)
            out[i, 2] = (np.abs(np.diff(x)).max() - np.abs(np.diff(t)).max()) / scale
            out[i, 3] = (np.abs(x - x.mean()).sum() - np.abs(t - t.mean()).sum()) / (np.abs(t - t.mean()).sum() + 1e-9)
            out[i, 7] = 1
        # (b) wilayah P kandidat dinyatakan dalam waktu beat sebelumnya: R_i-300 ms s.d. R_i-60 ms.
        # Template = kelanjutan sinyal setelah beat stabil yang beat berikutnya belum mulai (tanpa P baru).
        a0, a1 = max(int(round(0.10 * fs)), rr_i - p_off), rr_i - qrs_guard
        idx = base[rr_next[base] - p_off >= a1]
        idx = idx[-20:]
        got = _region_vs_template(signal, peaks, fs, i, idx, a0, a1) if len(idx) >= 5 else None
        if got is not None:
            x, t = got
            r = x - t
            energy = (r ** 2).sum()
            out[i, 4] = _corr(x, t)
            out[i, 5] = energy / ((t - t.mean()) ** 2).sum().clip(1e-9)
            tail = r[int(0.6 * len(r)):]
            out[i, 6] = (tail ** 2).sum() / max(energy, 1e-12)
    return pd.DataFrame(out, columns=[f"e4_{x}" for x in names])


# ----------------------------------------------------------------------------- E5
def _dwt(x, levels=4):
    coeffs, a = [], x
    for _ in range(levels):
        pad = np.pad(a, (len(DB4_LO) - 1, len(DB4_LO) - 1), mode="symmetric")
        lo = np.convolve(pad, DB4_LO[::-1], "valid")[1::2]
        hi = np.convolve(pad, DB4_HI[::-1], "valid")[1::2]
        coeffs.append(hi); a = lo
    return coeffs, a  # coeffs[0] = D1 (paling halus)


def fitur_e5_wavelet(rec, stable):
    signal, peaks, fs = rec["signal"], rec["peaks"], rec["fs"]
    n = len(peaks)
    left, right = int(round(0.25 * fs)), int(round(0.45 * fs))
    scale = _robust_scale(signal)
    # Pada 250 Hz: D1 62-125 Hz (di atas bandpass 40 Hz, tidak dipakai), D2 31-62, D3 16-31, D4 8-16, A4 0-8 Hz.
    bands = ["d2", "d3", "d4", "a4"]
    decomposed = []
    for p in peaks:
        s = _seg(signal, p, left, right)
        if s is None:
            decomposed.append(None); continue
        d, a = _dwt(s / scale)
        decomposed.append([d[1], d[2], d[3], a])
    names = ([f"wav_rel_energy_{b}" for b in bands] + ["wav_log_energy"] + [f"wav_std_{b}" for b in bands]
             + [f"wav_maxabs_{b}" for b in bands] + ["wav_entropy"] + [f"wav_residual_{b}" for b in bands]
             + ["wav_qrs_similarity", "wav_template_available"])
    out = np.full((n, len(names)), np.nan)
    usable = stable & np.array([d is not None for d in decomposed])
    for i in range(n):
        c = decomposed[i]
        if c is None:
            continue
        e = np.array([(x ** 2).sum() for x in c]); total = e.sum() + 1e-12
        rel = e / total
        out[i, 0:4] = rel; out[i, 4] = np.log(total)
        out[i, 5:9] = [x.std() for x in c]; out[i, 9:13] = [np.abs(x).max() for x in c]
        out[i, 13] = -(rel * np.log(rel + 1e-12)).sum()
        idx = _template_indices(usable, i)
        out[i, 19] = 0
        if idx is None:
            continue
        tmpl = [np.median(np.stack([decomposed[j][k] for j in idx]), 0) for k in range(4)]
        out[i, 14:18] = [((c[k] - tmpl[k]) ** 2).sum() / ((tmpl[k] ** 2).sum() + 1e-9) for k in range(4)]
        # Kemiripan QRS: koefisien D3+D4 di sekitar R (+-100 ms)
        parts_c, parts_t = [], []
        for k, level in [(1, 3), (2, 4)]:
            step = 2 ** level
            mid = left // step; half = max(1, int(round(0.10 * fs)) // step)
            sl = slice(max(0, mid - half), mid + half + 1)
            parts_c.append(c[k][sl]); parts_t.append(tmpl[k][sl])
        out[i, 18] = _corr(np.concatenate(parts_c), np.concatenate(parts_t))
        out[i, 19] = 1
    return pd.DataFrame(out, columns=[f"e5_{x}" for x in names])


# ----------------------------------------------------------------------------- per rekaman
def process(args):
    row, fs = args
    t = time.time()
    rec = load_record(row["database"], row["record"], row["channel"], fs)
    e0, aux = fitur_e0_baseline(rec)
    e1, aux1 = fitur_e1_rr_lokal(rec)
    qrs_local = _qrs_local(rec["signal"], rec["peaks"], fs)
    rr_next_rel = aux1["rr_next"] / aux1["base10"]
    stable = beat_stabil(qrs_local, aux1["rr_rel10"], rr_next_rel)
    e2 = fitur_e2_deviasi_p(rec, aux, aux1, stable)
    e3 = fitur_e3_konteks(aux1, e2, qrs_local)
    e4 = fitur_e4_gelombang_t(rec, stable)
    e5 = fitur_e5_wavelet(rec, stable)
    sym = rec["symbol"]
    label = np.where(sym == "N", "N", np.where(np.isin(sym, list(S_SYMBOLS)), "S", "lain"))
    meta = pd.DataFrame({"database": row["database"], "record": row["record"], "patient_id": row["patient_id"],
                         "split": row["split"], "fold": row["fold"], "beat_index": np.arange(len(rec["peaks"])),
                         "rpeak_sample": rec["peaks"], "symbol": sym, "label": label,
                         "match_distance_ms": rec["distance_ms"], "beat_stable_template": stable.astype(int)})
    feats = pd.concat([e0, e1, e2, e3, e4, e5], axis=1).astype(np.float32)
    qa = {"database": row["database"], "record": row["record"], "detected": len(rec["peaks"]),
          "reference": rec["n_reference"], "matched": rec["n_matched"], "seconds": round(time.time() - t, 1)}
    return meta, feats, qa


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="uji cepat: hanya N rekaman pertama")
    ap.add_argument("--semua", action="store_true", help="semua database V4 (termasuk afdb/vfdb) untuk proksi window")
    args = ap.parse_args(argv)
    out = HERE / "fitur" / f"{args.fs}hz"
    if args.limit:
        out = HERE / "fitur" / f"uji_{args.fs}hz_{args.limit}"
    if args.semua:
        out = HERE / "fitur" / f"{args.fs}hz_semua"
    if (out / "info.json").exists():
        sys.exit(f"{out} sudah ada; tidak ditimpa.")
    out.mkdir(parents=True, exist_ok=True)
    recs = records(("mitdb", "svdb", "afdb", "vfdb") if args.semua else ("mitdb", "svdb"))
    # rekaman panjang (afdb 10 jam) dijadwalkan lebih dulu
    recs = recs.assign(_urut=recs.database.map({"afdb": 0, "vfdb": 1}).fillna(2)).sort_values("_urut").drop(columns="_urut")
    if args.limit:
        recs = recs.groupby("database").head(args.limit)
    t = time.time()
    jobs = [(r, args.fs) for r in recs.to_dict("records")]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(process, jobs))
    meta = pd.concat([r[0] for r in results], ignore_index=True)
    feats = pd.concat([r[1] for r in results], ignore_index=True)
    qa = pd.DataFrame([r[2] for r in results])
    meta.to_csv(out / "meta.csv", index=False)
    np.savez_compressed(out / "fitur.npz", X=feats.to_numpy(np.float32), names=np.asarray(feats.columns))
    groups = feats.columns.str.extract(r"^(e\d)")[0]
    pd.DataFrame({"fitur": feats.columns, "kelompok": groups,
                  "mode": np.where(feats.columns.str.contains(r"^e1o_|^e3o"), "offline", "kausal"),
                  "ketersediaan": feats.notna().mean().round(4).to_numpy()}).to_csv(out / "nama_fitur.csv", index=False)
    qa.to_csv(out / "qa_detektor.csv", index=False)
    info = {"fs": args.fs, "rekaman": len(recs), "beat": len(meta), "fitur": feats.shape[1],
            "label": meta.label.value_counts().to_dict(), "detik": round(time.time() - t, 1),
            "label_dipakai_untuk_fitur": False, "S_simbol": sorted(S_SYMBOLS), "N_simbol": ["N"]}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
