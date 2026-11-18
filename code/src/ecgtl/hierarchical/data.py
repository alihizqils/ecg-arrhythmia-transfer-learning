"""Manifest pasien, QA detektor, dan citra turunan dengan split parent."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from wfdb.processing import gqrs_detect

from ecgtl.data.build_dataset import load_channel_signal
from ecgtl.data.rendering import render_waveform
from ecgtl.data.segmentation import preprocess_record, minmax_normalize
from .protocol import CLASSES, atomic_json, digest

BEATS = {s: c for c, symbols in {'N': ['N', 'L', 'R', 'e', 'j'],
         'S': ['A', 'a', 'J', 'S'], 'V': ['V', 'E'], 'F': ['F'],
         'Q': ['/', 'f', 'Q']}.items() for s in symbols}


def read_manifest(path):
    return pd.read_csv(path, dtype={'record': str, 'record_id': str, 'patient_id': str})


def assign_folds(p):
    cfg = p.cfg['data']
    df = read_manifest(p.source / 'manifest.csv').sort_values('index').reset_index(drop=True)
    assignment = pd.read_csv(p.source / 'patient_assignment.csv').set_index('patient_id')
    if not assignment.index.is_unique or not df['index'].is_unique:
        raise ValueError('Identitas pasien/window tidak unik.')
    expected_id = [cfg['aliases'].get(d, {}).get(r, f'{d}/{r}')
                   for d, r in zip(df.database, df.record)]
    if list(df.patient_id) != expected_id:
        raise ValueError('Alias pasien tidak sesuai.')
    df['split'] = df.patient_id.map(assignment.partisi).map({'latih': 'development', 'uji': 'holdout'})
    if df.split.isna().any() or not np.array_equal(df['index'], np.arange(len(df))):
        raise ValueError('Manifest dan assignment tidak lengkap.')
    for part in ['development', 'holdout']:
        sub = df[df.split == part]
        if len(sub) != cfg[f'expected_{part}_windows'] or sub.patient_id.nunique() != cfg[f'expected_{part}_patients']:
            raise ValueError(f'Jumlah {part} tidak sesuai protokol.')
    if len(df) != cfg['expected_windows'] or df.patient_id.nunique() != cfg['expected_patients']:
        raise ValueError('Jumlah dataset global tidak sesuai protokol.')
    if not np.array_equal(df.label_idx, df.label.map({c: i for i, c in enumerate(CLASSES)})):
        raise ValueError('Indeks kelas tidak sesuai.')
    dev = df[df.split == 'development']
    patient_counts = pd.crosstab(dev.patient_id, dev.label).reindex(columns=CLASSES, fill_value=0)
    wc = patient_counts.to_numpy(dtype=float)
    pc = (wc > 0).astype(float)
    normalized_w = wc / wc.sum(0)
    normalized_p = pc / pc.sum(0)
    best = None
    for candidate in range(cfg['split_candidates']):
        rng = np.random.default_rng(p.cfg['seed'] + candidate)
        priority = np.maximum(normalized_w.max(1), normalized_p.max(1)) * rng.uniform(.7, 1.3, len(wc))
        allocations = np.empty(len(wc), dtype=int)
        fw, fp = np.zeros((5, 5)), np.zeros((5, 5))
        size = np.zeros(5)
        for patient in np.argsort(-priority):
            costs = []
            for fold in range(5):
                tw, tp, ts = fw.copy(), fp.copy(), size.copy()
                tw[fold] += normalized_w[patient]
                tp[fold] += normalized_p[patient]
                ts[fold] += 1 / len(wc)
                costs.append(np.square(tw).sum() + 2 * np.square(tp).sum() + np.square(ts).sum())
            fold = int(np.argmin(np.asarray(costs) + rng.uniform(0, 1e-10, 5)))
            allocations[patient] = fold
            fw[fold] += normalized_w[patient]
            fp[fold] += normalized_p[patient]
            size[fold] += 1 / len(wc)
        folds = dev.patient_id.map(dict(zip(patient_counts.index, allocations))).to_numpy()
        counts = pd.crosstab(folds, dev.label.to_numpy()).reindex(columns=CLASSES, fill_value=0)
        unique = dev.assign(fold=folds).drop_duplicates(['fold', 'label', 'patient_id'])
        patients = pd.crosstab(unique.fold, unique.label).reindex(index=range(5), columns=CLASSES, fill_value=0)
        if (patients['S'] < cfg['min_s_patients']).any() or (counts == 0).any().any():
            continue
        score = (counts / counts.sum()).std().mean() + (patients / patients.sum()).std().mean()
        if best is None or score < best[0]:
            best = (float(score), folds.copy(), candidate)
    if best is None:
        raise ValueError('Tidak ditemukan fold dengan dukungan pasien kelas yang memadai.')
    df['fold'] = -1
    df.loc[dev.index, 'fold'] = best[1]
    df['window_id'] = df['index'].map(lambda i: f'w{i:06d}')
    p.prepared.mkdir(parents=True, exist_ok=True)
    path = p.prepared / 'global_manifest.csv'
    if path.exists() and path.read_text() != df.to_csv(index=False):
        # CSV CRLF/LF dinormalisasi oleh read_text.
        if not read_manifest(path).equals(df):
            raise ValueError('Manifest tersimpan berbeda dari pembagian deterministik.')
    df.to_csv(path, index=False)
    distribution = df.groupby(['split', 'fold', 'label']).agg(windows=('window_id', 'size'), patients=('patient_id', 'nunique')).reset_index()
    distribution.to_csv(p.prepared / 'distribution.csv', index=False)
    atomic_json(p.prepared / 'split_audit.json', {'candidate': best[2], 'score': best[0],
                'patient_overlap': 0, 'holdout_assignment_sha256': digest(p.source / 'patient_assignment.csv')})
    return distribution


def match_peaks(reference, detected, tolerance):
    candidates = []
    for i, r in enumerate(reference):
        lo, hi = np.searchsorted(detected, [r - tolerance, r + tolerance], side='left')
        hi = np.searchsorted(detected, r + tolerance, side='right')
        for j in range(lo, hi):
            candidates.append((abs(int(detected[j]) - int(r)), i, j))
    used_r, used_d, result = set(), set(), {}
    for distance, i, j in sorted(candidates):
        if i not in used_r and j not in used_d:
            used_r.add(i)
            used_d.add(j)
            result[j] = (i, distance)
    return result


def detected_peaks(signal, cfg):
    peaks = gqrs_detect(sig=np.asarray(signal, dtype=np.float64), fs=cfg['fs']) + cfg['detector_offset']
    return np.unique(peaks[(peaks >= 0) & (peaks < len(signal))]).astype(int)


def window_views(signal, peaks, cfg):
    norm = minmax_normalize(signal)
    fallback = len(peaks) == 0
    if fallback:
        energy = np.convolve(np.diff(signal, prepend=signal[0]) ** 2, np.ones(20), 'same')
        peaks = np.asarray([int(energy.argmax())])
    half = cfg['crop_samples'] // 2
    padded = np.pad(norm, (half, half), mode='reflect')
    crops = np.stack([render_waveform(padded[r:r + cfg['crop_samples']],
                                      width=cfg['local_shape'][1], height=cfg['local_shape'][0]) for r in peaks])
    return render_waveform(norm, width=448, height=224), crops, peaks, fallback


def prepare_record(p, rows):
    cfg = p.cfg['data']
    first = rows.iloc[0]
    db, record, part = first.database, first.record, first.split
    folder = p.prepared / 'records' / db / record
    folder.mkdir(parents=True, exist_ok=True)
    source_files = sorted((p.raw / db).glob(f'{record}.*'))
    source_hashes = {f.name: digest(f) for f in source_files}
    source_key = {'sources': source_hashes, 'manifest': digest(p.prepared / 'global_manifest.csv'),
                  'config': cfg}
    if (folder / 'done.json').exists():
        done = json.loads((folder / 'done.json').read_text())
        if done['source_key'] != source_key:
            raise ValueError(f'Cache sumber berubah: {db}/{record}')
        for name, expected in done['hashes'].items():
            if digest(folder / name) != expected:
                raise ValueError(f'Cache rusak: {folder / name}')
        return read_manifest(folder / 'crops.csv'), done['qa']
    raw, fs, channel = load_channel_signal(p.raw / db, record, [first.channel])
    if channel != first.channel:
        raise ValueError('Kanal sumber berbeda dari manifest.')
    signal = preprocess_record(raw, fs, cfg['fs'], cfg['bandpass']).signal
    peaks = detected_peaks(signal, cfg)
    ref, labels, matches = np.array([], dtype=int), [], {}
    qa = None
    # Label detak hanya dimuat untuk development MITDB.
    if db == 'mitdb' and part == 'development':
        ann = wfdb.rdann(str(p.raw / db / record), 'atr')
        selected = [(s, BEATS[c]) for s, c in zip(ann.sample, ann.symbol) if c in BEATS]
        ref = np.rint(np.array([s for s, _ in selected]) * cfg['fs'] / fs).astype(int)
        labels = [c for _, c in selected]
        matches = match_peaks(ref, peaks, cfg['tolerance_samples'])
        hit = {i for i, _ in matches.values()}
        qa = {'record': record, 'patient_id': first.patient_id, 'fold': int(first.fold),
              'reference': len(ref), 'detected': len(peaks), 'matched': len(matches),
              'errors_ms': [float((peaks[j] - ref[i]) * 1000 / cfg['fs']) for j, (i, _) in matches.items()]}
        for c in CLASSES[:3]:
            qa[f'reference_{c}'] = labels.count(c)
            qa[f'matched_{c}'] = sum(labels[i] == c for i in hit)
    local, waveform, entries = [], [], []
    original = np.load(p.source / 'images.npy', mmap_mode='r')
    offset = 0
    for row in rows.itertuples():
        start = int(row.start_target_fs)
        segment = signal[start:start + cfg['window_samples']]
        if len(segment) != cfg['window_samples']:
            raise ValueError('Panjang parent window tidak lengkap.')
        indices = np.flatnonzero((peaks >= start) & (peaks < start + len(segment)))
        global_image, crops, centers, fallback = window_views(segment, peaks[indices] - start, cfg)
        if not np.array_equal(global_image, original[row.index]):
            raise ValueError(f'Renderer global tidak sama dengan sumber: {row.window_id}')
        waveform.append(segment)
        for k, center in enumerate(centers):
            pair = None if fallback else matches.get(int(indices[k]))
            ann_index = pair[0] if pair else None
            label = labels[ann_index] if pair else 'unmatched'
            entries.append({'crop_id': f'{row.window_id}_r{k:03d}', 'parent_window_id': row.window_id,
                'patient_id': row.patient_id, 'record_id': record, 'database': db, 'channel': channel,
                'rpeak_sample': int(start + center), 'annotation_sample': int(ref[ann_index]) if pair else -1,
                'match_distance_ms': pair[1] * 1000 / cfg['fs'] if pair else -1,
                'beat_label': label, 'split': part, 'fold': int(row.fold),
                'fallback_peak': fallback, 'edge_padding': bool(center < 156 or center + 156 > len(segment)),
                'shard': f'records/{db}/{record}/local.npy', 'shard_index': offset + k})
        offset += len(crops)
        local.append(crops)
    np.save(folder / 'local.npy', np.concatenate(local))
    np.save(folder / 'waveforms.npy', np.stack(waveform))
    np.save(folder / 'window_indices.npy', rows['index'].to_numpy())
    result = pd.DataFrame(entries)
    result.to_csv(folder / 'crops.csv', index=False)
    names = ['local.npy', 'waveforms.npy', 'window_indices.npy', 'crops.csv']
    atomic_json(folder / 'done.json', {'source_key': source_key, 'qa': qa,
                                      'hashes': {n: digest(folder / n) for n in names}})
    return result, qa


def detector_report(p, audits):
    reference = sum(a['reference'] for a in audits)
    detected = sum(a['detected'] for a in audits)
    matched = sum(a['matched'] for a in audits)
    result = {'precision': matched / detected, 'recall': matched / reference,
              'f1': 2 * matched / (reference + detected)}
    errors = np.concatenate([a['errors_ms'] for a in audits])
    result.update(error_median_ms=float(np.median(abs(errors))), error_p95_ms=float(np.percentile(abs(errors), 95)))
    rows = []
    for fold in [-1, *range(5)]:
        subset = audits if fold == -1 else [a for a in audits if a['fold'] == fold]
        for c in CLASSES[:3]:
            n = sum(a[f'reference_{c}'] for a in subset)
            hit = sum(a[f'matched_{c}'] for a in subset)
            rows.append({'fold': fold, 'class': c, 'reference': n, 'matched': hit, 'recall': hit / n if n else 0})
    frame = pd.DataFrame(rows)
    result['passed'] = bool(result['f1'] >= p.cfg['data']['min_detector_f1'] and
                            frame.recall.min() >= p.cfg['data']['min_class_recall'])
    out = p.out / 'detector_qc'
    out.mkdir(exist_ok=True)
    atomic_json(out / 'summary.json', result)
    frame.to_csv(out / 'recall.csv', index=False)
    pd.DataFrame([{k: v for k, v in a.items() if k != 'errors_ms'} for a in audits]).to_csv(out / 'records.csv', index=False)
    np.save(out / 'errors_ms.npy', errors)
    if not result['passed']:
        raise ValueError('Gerbang mutu detektor development belum lulus.')
    return result


def prepare_data(p):
    assign_folds(p)
    df = read_manifest(p.prepared / 'global_manifest.csv')
    all_crops, audits = [], []
    for part in ['development', 'holdout']:
        groups = list(df[df.split == part].groupby(['database', 'record'], sort=True))
        for _, rows in tqdm(groups, desc=f'Persiapan {part}'):
            crops, qa = prepare_record(p, rows)
            all_crops.append(crops)
            if qa is not None:
                audits.append(qa)
        if part == 'development':
            detector_report(p, audits)
    local = pd.concat(all_crops, ignore_index=True)
    validate_inheritance(df, local)
    local.to_csv(p.prepared / 'local_manifest.csv', index=False)
    files = list((p.prepared / 'records').rglob('*.npy'))
    atomic_json(p.prepared / 'data_checksums.json',
                {'source_images': digest(p.source / 'images.npy'),
                 'prepared': {f.relative_to(p.prepared).as_posix(): digest(f) for f in files}})
    p.lock()
    return pd.DataFrame([{'windows': len(df), 'crops': len(local), 'patients': df.patient_id.nunique(), 'status': 'lengkap'}])


def validate_inheritance(global_df, local):
    if not local.crop_id.is_unique:
        raise ValueError('Crop ID tidak unik.')
    parent = global_df.set_index('window_id')
    if set(local.parent_window_id) != set(parent.index):
        raise ValueError('Parent window tidak lengkap.')
    for col in ['patient_id', 'split', 'fold']:
        if not np.array_equal(local[col].to_numpy(), local.parent_window_id.map(parent[col]).to_numpy()):
            raise ValueError(f'Pewarisan {col} tidak konsisten.')
    if global_df.groupby('patient_id').split.nunique().max() != 1 or global_df.groupby('patient_id').fold.nunique().max() != 1:
        raise ValueError('Overlap pasien ditemukan.')


class WindowDataset(Dataset):
    def __init__(self, p, rows, features=None):
        self.p, self.rows, self.features = p, rows.reset_index(drop=True), features
        local = read_manifest(p.prepared / 'local_manifest.csv')
        local = local[local.parent_window_id.isin(self.rows.window_id)]
        self.groups = {wid: g for wid, g in local.groupby('parent_window_id', sort=False)}
        self.maps = {}
        self.global_images = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        crops = self.groups[row.window_id]
        if self.features:
            with np.load(Path(self.features) / f'{row.window_id}.npz') as f:
                global_x, local_x = f['global_features'].copy(), f['local_features'].copy()
        else:
            if self.global_images is None:
                self.global_images = np.load(self.p.source / 'images.npy', mmap_mode='r')
            global_x = self.global_images[int(row['index'])][None].astype('float32') / 255
            shard = str(crops.iloc[0].shard)
            if shard not in self.maps:
                self.maps[shard] = np.load(self.p.prepared / shard, mmap_mode='r')
            local_x = self.maps[shard][crops.shard_index.to_numpy()][:, None].astype('float32') / 255
        beat = np.array([CLASSES.index(c) if c in CLASSES[:3] and row.label in CLASSES[:3] else -1
                         for c in crops.beat_label], dtype=np.int64)
        return {'global': torch.from_numpy(global_x), 'local': torch.from_numpy(local_x),
                'beat': torch.from_numpy(beat), 'y': int(row.label_idx), 'id': row.window_id}


def collate(items):
    lengths = [len(x['local']) for x in items]
    maximum = max(lengths)
    local = items[0]['local'].new_zeros((len(items), maximum, *items[0]['local'].shape[1:]))
    mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    beat = torch.full((len(items), maximum), -1, dtype=torch.long)
    for i, item in enumerate(items):
        n = lengths[i]
        local[i, :n], mask[i, :n], beat[i, :n] = item['local'], True, item['beat']
    return {'global': torch.stack([x['global'] for x in items]), 'local': local,
            'mask': mask, 'beat': beat, 'y': torch.tensor([x['y'] for x in items]), 'id': [x['id'] for x in items]}
