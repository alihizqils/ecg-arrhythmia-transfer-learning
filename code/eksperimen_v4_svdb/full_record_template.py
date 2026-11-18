"""Template rekaman dari rekaman lengkap, tidak bergantung pada seleksi window.

Masalah prosedur lama (v4_record_template.py, v4_second_lead.py): template QRS/pra-QRS dan median RR rekaman
dihitung dari beat pada window TERPILIH (global_manifest), padahal seleksi window memakai label (batas jumlah
window per kelas per rekaman, window N hanya dari database beranotasi beat, aturan pelabelan). Komposisi beat
pembentuk template karena itu bergantung pada label, yang tidak diketahui saat inferensi.

Versi baru: acuan rekaman (template QRS, template pra-QRS, median RR) dihitung dari SELURUH beat hasil detektor
GQRS pada rekaman lengkap (preprocessing dan detektor sama dengan V4, fase_e_ns/fitur_ns.load_record dengan
annotations=False). Manifest, label, split, dan fold tidak dibaca untuk membentuk acuan. Fitur per beat/per
window lalu dihitung dengan definisi yang sama persis (segmen beat dari window, RR di dalam window), sehingga
satu-satunya perubahan adalah acuan rekaman. Keluaran: outputs_colab_r15/main_results/template_rekaman_penuh.npz.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "fase_e_ns"))
import fitur_ns as NS  # noqa: E402
import v4_record_template as RT  # noqa: E402
from ecgtl.data.segmentation import preprocess_record  # noqa: E402

CODE = HERE.parent
PREP = CODE / "data" / "processed" / "v4_svdb" / "prepared"
DEST = HERE / "outputs_colab_r15" / "main_results"
FS = 250.0; WIN = 2500
QL, QR = int(round(RT._QRS[0] * FS)), int(round(RT._QRS[1] * FS))
PL, PR = int(round(RT._PRE[0] * FS)), int(round(RT._PRE[1] * FS))


def _templates(signal, peaks):
    """Template QRS dan pra-QRS (median segmen ter-z) dari SEMUA beat rekaman."""
    q = [RT._znorm(signal[p - QL:p + QR + 1]) for p in peaks if p - QL >= 0 and p + QR < len(signal)]
    r = [RT._znorm(signal[p - PL:p - PR + 1]) for p in peaks if p - PL >= 0 and p - PR < len(signal)]
    q = [x for x in q if x is not None]; r = [x for x in r if x is not None]
    return (RT._znorm(np.median(np.stack(q), 0)) if q else None, RT._znorm(np.median(np.stack(r), 0)) if r else None)


def _window_beats(segment, peaks, qt, pt, rr_median):
    out = []
    for order, peak in enumerate(peaks):
        qrs = segment[peak - QL:peak + QR + 1] if peak - QL >= 0 and peak + QR < len(segment) else None
        pre = segment[peak - PL:peak - PR + 1] if peak - PL >= 0 and peak - PR < len(segment) else None
        rr = (peak - peaks[order - 1]) / FS if order > 0 else np.nan
        out.append((rr / rr_median if np.isfinite(rr_median) and rr_median > 0 else np.nan,
                    RT._correlation(None if qrs is None else RT._znorm(qrs), qt),
                    RT._correlation(None if pre is None else RT._znorm(pre), pt)))
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def per_record(args):
    database, record, channel, rows = args
    rec = NS.load_record(database, record, channel, FS, annotations=False)
    signal, peaks = rec["signal"], rec["peaks"]
    rr_median = float(np.median(np.diff(peaks) / FS))
    qt, pt = _templates(signal, peaks)
    # lead kedua: preprocessing sama dengan v4_second_lead (segmen window di-nan_to_num)
    full = wfdb.rdrecord(str(NS.RAW / database / record))
    other = 1 - list(full.sig_name).index(channel)
    second = preprocess_record(full.p_signal[:, other].astype(np.float64), full.fs, FS, NS.BANDPASS).signal
    second_clean = np.nan_to_num(np.asarray(second, np.float64))
    qt2, pt2 = _templates(second_clean, peaks)
    res = {}
    for wid, start, wpeaks in rows:
        seg = signal[start:start + WIN]
        b1 = _window_beats(seg, wpeaks - start, qt, pt, rr_median)
        b2 = _window_beats(np.nan_to_num(np.asarray(second[start:start + WIN], np.float64)), wpeaks - start, qt2, pt2,
                           rr_median)
        res[wid] = (RT._window_features(b1), RT._beat_features(b1), RT._window_features(b2), seg)
    return database, record, res, {"beats_record": int(len(peaks)), "rr_median_record_s": rr_median}


def main():
    DEST.mkdir(exist_ok=True)
    g = pd.read_csv(PREP / "global_manifest.csv", dtype={"window_id": str, "record": str})
    local = pd.read_csv(PREP / "local_manifest.csv", dtype={"parent_window_id": str})
    local = local[local.parent_window_id.isin(set(g.window_id))].sort_values(["parent_window_id", "rpeak_sample"])
    pk = {k: f.rpeak_sample.to_numpy(np.int64) for k, f in local.groupby("parent_window_id")}
    jobs = []
    for (db, rec, ch), rows in g.groupby(["database", "record", "channel"], sort=False):
        jobs.append((db, rec, ch, [(r.window_id, int(r.start_target_fs), pk.get(r.window_id, np.empty(0, np.int64)))
                                   for r in rows.itertuples(index=False)]))
    out, info, seg_err = {}, [], 0.0
    with ProcessPoolExecutor(max_workers=10) as pool:
        for db, rec, res, meta in pool.map(per_record, jobs):
            wf = np.load(PREP / "records" / db / rec / "waveforms.npy", mmap_mode="r")
            pos = {int(v): i for i, v in enumerate(np.load(PREP / "records" / db / rec / "window_indices.npy"))}
            idx = g.set_index("window_id").loc[list(res), "index"]
            for wid, (a, b, c, seg) in res.items():
                seg_err = max(seg_err, float(np.abs(np.asarray(wf[pos[int(idx[wid])]], np.float64) - seg).max()))
                out[wid] = (a, b, c)
            info.append({"database": db, "record": rec, **meta})
            print(db, rec, len(res), flush=True)
    ids = g.window_id.tolist()
    np.savez_compressed(DEST / "template_rekaman_penuh.npz", window_ids=np.asarray(ids),
                        rt=np.stack([out[i][0] for i in ids]).astype(np.float32),
                        lead2=np.stack([out[i][2] for i in ids]).astype(np.float32),
                        beat=np.concatenate([out[i][1] for i in ids]).astype(np.float32),
                        beat_offsets=np.cumsum([0] + [len(out[i][1]) for i in ids]).astype(np.int64))
    pd.DataFrame(info).to_csv(DEST / "template_rekaman_penuh_info.csv", index=False)
    print("selisih maks segmen window vs waveforms.npy:", seg_err)


if __name__ == "__main__":
    main()
