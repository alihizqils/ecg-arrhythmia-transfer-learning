"""Rekap terverifikasi, evaluasi idempotent, dan pengukuran efisiensi."""
from __future__ import annotations

import json
import os
import platform
import threading
import time

import numpy as np
import pandas as pd
import psutil
import torch

from ecgtl.data.build_dataset import load_channel_signal
from ecgtl.data.segmentation import preprocess_record
from .data import read_manifest, detected_peaks, window_views, collate, WindowDataset
from .metrics import export_metrics
from .protocol import atomic_json, digest, environment
from .training import Experiment, load_checkpoint, to_device


def final_models(p):
    result = []
    for arch in p.cfg['backbones']:
        selection = json.loads((p.out / 'selection' / arch / 'selected.json').read_text())
        if selection['protocol_hash'] != p.hash:
            raise ValueError('Pemilihan model tidak konsisten.')
        for scenario in p.cfg['scenarios']:
            tau = selection['tau'] if scenario in ['S3', 'S4'] else 1
            tag = f'{scenario}_tau{tau:g}' if scenario == 'S3' else scenario
            folder = p.out / 'full_models' / arch / tag / 'final'
            meta = json.loads((folder / 'result.json').read_text())
            if meta['protocol_hash'] != p.hash or digest(folder / 'best.pt') != meta['checkpoint_sha256']:
                raise ValueError('Checkpoint final tidak valid.')
            result.append((arch, scenario, tau, folder, meta))
    return result


def recap(p):
    p.verify()
    finals = final_models(p)
    rows = []
    for arch, scenario, tau, _, _ in finals:
        tag = f'{scenario}_tau{tau:g}' if scenario == 'S3' else scenario
        for fold in range(5):
            folder = p.out / 'cv' / arch / tag / f'fold_{fold}'
            row = json.loads((folder / 'result.json').read_text())
            if row['protocol_hash'] != p.hash or digest(folder / 'best.pt') != row['checkpoint_sha256']:
                raise ValueError('Run CV tidak valid.')
            rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 60 or frame.duplicated(['arch', 'scenario', 'fold']).any():
        raise ValueError('Matriks 60 run belum lengkap.')
    out = p.out / 'tables'; out.mkdir(exist_ok=True)
    frame.to_csv(out / 'cv_runs.csv', index=False)
    metrics = ['macro_f1', 'f1_S', 'macro_sensitivity', 'macro_specificity', 'macro_auc', 'accuracy', 'training_s']
    summary = frame.groupby(['arch', 'scenario'])[metrics].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(c).rstrip('_') if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(out / 'cv_summary.csv', index=False)
    effects = []
    for arch, sub in frame.groupby('arch'):
        for after, before in [('S2', 'S1'), ('S3', 'S2'), ('S4', 'S3')]:
            a, b = sub[sub.scenario == after].set_index('fold'), sub[sub.scenario == before].set_index('fold')
            for fold in range(5):
                effects.append({'arch': arch, 'comparison': f'{after}-{before}', 'fold': fold,
                                **{m: float(a.loc[fold, m] - b.loc[fold, m]) for m in metrics}})
    pd.DataFrame(effects).to_csv(out / 'paired_effects.csv', index=False)
    return summary


def evaluate_holdout(p):
    recap(p)
    manifest = read_manifest(p.prepared / 'global_manifest.csv')
    test = manifest[manifest.split == 'holdout'].reset_index(drop=True)
    roster = {f'{a}/{s}': m['checkpoint_sha256'] for a, s, _, _, m in final_models(p)}
    roster_path = p.out / 'holdout' / 'model_roster.json'
    if roster_path.exists() and json.loads(roster_path.read_text()) != roster:
        raise ValueError('Daftar model berubah setelah evaluasi holdout.')
    atomic_json(roster_path, roster)
    scores = []
    for arch, scenario, tau, folder, meta in final_models(p):
        out = p.out / 'holdout' / arch / scenario
        marker = out / 'complete.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved['checkpoint_sha256'] != meta['checkpoint_sha256'] or saved['protocol_hash'] != p.hash:
                raise ValueError('Prediksi tersimpan tidak sesuai model.')
            for name, sha in saved['files'].items():
                if digest(out / name) != sha:
                    raise ValueError('Artefak holdout berubah.')
            score = json.loads((out / 'metrics.json').read_text())
        else:
            exp = Experiment(p, arch)
            model = exp.model(scenario, pretrained=False)
            model.load_state_dict(load_checkpoint(folder / 'best.pt')['model'])
            prediction = exp.predict(model, test)
            score = export_metrics(out, prediction['y'], prediction['prob'], test, prediction['global_prob'])
            atomic_json(marker, {'protocol_hash': p.hash, 'checkpoint_sha256': meta['checkpoint_sha256'],
                                 'files': {f.name: digest(f) for f in out.iterdir() if f.is_file()}})
        scores.append({'arch': arch, 'scenario': scenario, **score})
    frame = pd.DataFrame(scores)
    frame.to_csv(p.out / 'tables/holdout_summary.csv', index=False)
    return frame


class PeakRSS:
    def __enter__(self):
        self.stop = threading.Event()
        self.peak = psutil.Process().memory_info().rss
        def sample():
            process = psutil.Process()
            while not self.stop.wait(0.005):
                self.peak = max(self.peak, process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self.stop.set(); self.thread.join()


def benchmark(p):
    p.verify()
    cfg = p.cfg['benchmark']
    torch.set_num_threads(cfg['threads'])
    identity = {'hostname': platform.node(), 'platform': platform.platform(),
                'processor': platform.processor(), 'threads': torch.get_num_threads(),
                'device': cfg['device'], 'environment': environment(), 'protocol_hash': p.hash}
    if cfg['device'] != 'cpu':
        raise ValueError('Benchmark protokol utama menggunakan CPU pada mesin tetap.')
    out = p.out / 'benchmark'; out.mkdir(exist_ok=True)
    machine = out / 'machine.json'
    if machine.exists() and json.loads(machine.read_text()) != identity:
        raise ValueError('Seluruh model harus diukur pada mesin dan lingkungan yang sama.')
    atomic_json(machine, identity)
    manifest = read_manifest(p.prepared / 'global_manifest.csv')
    dev = manifest[manifest.split == 'development']
    selected = dev.sample(n=min(cfg['windows'], len(dev)), random_state=p.cfg['seed']).sort_values('index')
    selected.to_csv(out / 'windows.csv', index=False)
    # Model offline memakai filter dan deteksi per rekaman; biaya tersebut
    # diamortisasi atas seluruh window terpilih dari rekaman yang sama.
    preprocess_times = {}
    for (db, rec), rows in selected.groupby(['database', 'record']):
        raw, fs, channel = load_channel_signal(p.raw / db, rec, [rows.iloc[0].channel])
        begin = time.perf_counter()
        signal = preprocess_record(raw, fs, p.cfg['data']['fs'], p.cfg['data']['bandpass']).signal
        peaks = detected_peaks(signal, p.cfg['data'])
        duration = time.perf_counter() - begin
        count = len(dev[(dev.database == db) & (dev.record == rec)])
        for row in rows.itertuples():
            start = int(row.start_target_fs)
            begin = time.perf_counter()
            gx, lx, _, _ = window_views(signal[start:start + 2500], peaks[(peaks >= start) & (peaks < start + 2500)] - start, p.cfg['data'])
            torch.from_numpy(gx.astype('float32') / 255)[None, None]
            torch.from_numpy(lx.astype('float32') / 255)[None, :, None]
            torch.ones(1, len(lx), dtype=torch.bool)
            preprocess_times[row.window_id] = duration / count + time.perf_counter() - begin
    atomic_json(out / 'preprocessing_seconds.json', preprocess_times)
    results, measurements = [], []
    for arch, scenario, _, folder, meta in final_models(p):
        exp = Experiment(p, arch, device='cpu')
        model = exp.model(scenario, pretrained=False).eval()
        model.load_state_dict(load_checkpoint(folder / 'best.pt')['model'])
        dataset = WindowDataset(p, selected)
        with torch.inference_mode(), PeakRSS() as memory:
            warmup = collate([dataset[0]])
            for _ in range(cfg['warmup']):
                model(warmup['global'], warmup['local'], warmup['mask'])
            times, end_to_end, crop_counts = [], [], []
            for i in range(len(dataset)):
                batch = collate([dataset[i]])
                begin = time.perf_counter()
                model(batch['global'], batch['local'], batch['mask'])
                elapsed = time.perf_counter() - begin
                times.append(elapsed)
                end_to_end.append(elapsed + preprocess_times[batch['id'][0]])
                crop_counts.append(int(batch['mask'].sum()))
                measurements.append({'arch': arch, 'scenario': scenario, 'window_id': batch['id'][0],
                                     'model_s': elapsed, 'end_to_end_s': end_to_end[-1], 'crops': crop_counts[-1]})
        export_path = out / f'{arch}_{scenario}_inference.pt'
        torch.save({'arch': arch, 'scenario': scenario, 'protocol_hash': p.hash, 'model': model.state_dict()}, export_path)
        results.append({'arch': arch, 'scenario': scenario, 'parameters': meta['total_parameters'],
                        'trainable_parameters': meta['trainable_parameters'], 'model_bytes': export_path.stat().st_size,
                        'checkpoint_bytes': (folder / 'best.pt').stat().st_size, 'training_s': meta['training_s'],
                        'training_peak_vram_bytes': meta['peak_vram_bytes'], 'peak_process_rss_bytes': memory.peak,
                        'latency_median_ms': float(np.median(times) * 1000), 'latency_p95_ms': float(np.percentile(times, 95) * 1000),
                        'e2e_median_ms': float(np.median(end_to_end) * 1000), 'e2e_p95_ms': float(np.percentile(end_to_end, 95) * 1000),
                        'throughput_windows_s': len(times) / sum(times), 'e2e_throughput_windows_s': len(times) / sum(end_to_end),
                        'crops_mean': float(np.mean(crop_counts)), 'crops_p95': float(np.percentile(crop_counts, 95)),
                        'timing_scope': 'offline_record_preprocessing_amortized_excluding_disk_io'})
    frame = pd.DataFrame(results)
    frame.to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(measurements).to_csv(out / 'measurements.csv', index=False)
    return frame


def seal_results(p):
    required = [p.out / 'tables' / name for name in
                ['cv_runs.csv', 'cv_summary.csv', 'paired_effects.csv', 'holdout_summary.csv']]
    required += [p.out / 'benchmark/summary.csv', p.out / 'shap/complete.json']
    for file in required:
        if not file.exists():
            raise FileNotFoundError(file)
    files = []
    for sub in ['tables', 'holdout', 'benchmark', 'shap', 'detector_qc']:
        files += [f for f in (p.out / sub).rglob('*') if f.suffix in ['.csv', '.npy', '.npz', '.json']]
    files += list((p.out / 'cv').rglob('history.csv'))
    files += list((p.out / 'cv').rglob('per_class.csv'))
    atomic_json(p.out / 'artifact_checksums.json', {f.relative_to(p.out).as_posix(): digest(f) for f in files})
    return pd.DataFrame([{'artifact': str(f.relative_to(p.out)), 'status': 'terverifikasi'} for f in required])
