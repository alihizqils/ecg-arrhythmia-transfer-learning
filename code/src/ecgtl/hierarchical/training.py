"""CV, caching fitur, resume deterministik, dan pelatihan development penuh."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import WindowDataset, collate, read_manifest
from .metrics import calculate, export_metrics
from .model import HierarchicalNet, inverse_weights, loss_terms
from .protocol import atomic_json, atomic_torch, digest, seed_all, environment


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=False)


class Experiment:
    def __init__(self, protocol, arch, device=None, resume=True):
        self.p, self.arch = protocol, arch
        self.resume = resume
        self.p.verify()
        if arch not in self.p.cfg['backbones']:
            raise ValueError('Backbone tidak terdaftar.')
        manifest = read_manifest(self.p.prepared / 'global_manifest.csv')
        self.dev = manifest[manifest.split == 'development'].reset_index(drop=True)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.cfg = self.p.cfg['training']
        self.cache = self.p.prepared / 'features' / self.arch / self.p.hash[:16]

    def model(self, scenario, pretrained=True):
        return HierarchicalNet(self.arch, scenario, self.p.cfg, pretrained=pretrained).to(self.device)

    def loader(self, rows, cached=False, shuffle=False, seed=42, batch_size=None):
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(WindowDataset(self.p, rows, self.cache if cached else None),
                          batch_size=batch_size or self.cfg['batch_size'], shuffle=shuffle,
                          collate_fn=collate, num_workers=self.cfg['num_workers'],
                          generator=generator, pin_memory=self.device.type == 'cuda')

    def cache_features(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        marker = self.cache / 'complete.json'
        if marker.exists():
            meta = json.loads(marker.read_text())
            if meta['protocol_hash'] != self.p.hash:
                raise ValueError('Hash cache tidak cocok.')
            for name, sha in tqdm(meta['files'].items(), desc='Verifikasi fitur'):
                if digest(self.cache / name) != sha:
                    raise ValueError(f'Cache fitur rusak: {name}')
            return pd.DataFrame([{'windows': len(meta['files']), 'status': 'lengkap'}])
        seed_all(self.p.cfg['seed'])
        model = self.model('S2').eval()
        hashes = {}
        with torch.inference_mode():
            for batch in tqdm(self.loader(self.dev, batch_size=self.cfg['feature_batch_size']), desc=f'Fitur {self.arch}'):
                batch = to_device(batch, self.device)
                paths = [self.cache / f'{wid}.npz' for wid in batch['id']]
                complete = []
                for path in paths:
                    receipt = path.with_suffix('.json')
                    saved = json.loads(receipt.read_text()) if receipt.exists() else {}
                    valid = path.exists() and saved.get('protocol_hash') == self.p.hash and saved.get('sha256') == digest(path)
                    complete.append(valid)
                    if valid:
                        hashes[path.name] = saved['sha256']
                if all(complete):
                    continue
                g = model.encode(batch['global']).cpu().numpy()
                local = model.encode(batch['local'][batch['mask']]).cpu().numpy()
                offset = 0
                for i, path in enumerate(paths):
                    length = int(batch['mask'][i].sum())
                    if not complete[i]:
                        tmp = path.with_suffix('.tmp')
                        with tmp.open('wb') as f:
                            np.savez(f, global_features=g[i], local_features=local[offset:offset + length])
                        os.replace(tmp, path)
                        hashes[path.name] = digest(path)
                        atomic_json(path.with_suffix('.json'), {'protocol_hash': self.p.hash, 'sha256': hashes[path.name]})
                    offset += length
        atomic_json(marker, {'protocol_hash': self.p.hash, 'files': hashes})
        return pd.DataFrame([{'windows': len(hashes), 'status': 'lengkap'}])

    def fit_scaler(self, model, rows):
        sums = {key: [np.zeros(model.dim), np.zeros(model.dim), 0] for key in ['global', 'local']}
        for wid in rows.window_id:
            with np.load(self.cache / f'{wid}.npz') as f:
                for key in sums:
                    values = f[f'{key}_features'].reshape(-1, model.dim).astype('float64')
                    sums[key][0] += values.sum(0)
                    sums[key][1] += (values * values).sum(0)
                    sums[key][2] += len(values)
        for key, (total, square, n) in sums.items():
            mean = total / n
            std = np.sqrt(np.maximum(square / n - mean ** 2, 0))
            std[std < 1e-6] = 1
            getattr(model, f'{key}_mean').copy_(torch.tensor(mean, device=self.device))
            getattr(model, f'{key}_std').copy_(torch.tensor(std, device=self.device))

    def weights(self, rows, tau):
        y = rows.label_idx.to_numpy()
        local = read_manifest(self.p.prepared / 'local_manifest.csv')
        local = local[local.parent_window_id.isin(rows[rows.label_idx < 3].window_id)]
        counts = {'final': np.bincount(y, minlength=5),
                  'global': np.bincount(np.where(y < 3, 0, y - 2), minlength=3),
                  'local': local.beat_label.value_counts().reindex(['N', 'S', 'V'], fill_value=0).to_numpy()}
        return {k: inverse_weights(v, tau).to(self.device) for k, v in counts.items()}, counts

    def run_dir(self, scenario, fold, tau=1, full=False):
        base = self.p.out / ('full_models' if full else 'cv') / self.arch
        tag = f'{scenario}_tau{tau:g}' if scenario == 'S3' else scenario
        return base / tag / ('final' if full else f'fold_{fold}')

    def fit(self, scenario, fold, tau=1, full=False, epochs=None):
        folder = self.run_dir(scenario, fold, tau, full)
        if not getattr(self, 'resume', True) and folder.exists() and any(folder.iterdir()):
            raise FileExistsError('RESUME=False membutuhkan direktori run baru.')
        folder.mkdir(parents=True, exist_ok=True)
        done = folder / 'result.json'
        if done.exists():
            result = json.loads(done.read_text())
            if result['protocol_hash'] != self.p.hash or digest(folder / 'best.pt') != result['checkpoint_sha256']:
                raise ValueError('Integritas run tidak sesuai.')
            return result
        train = self.dev if full else self.dev[self.dev.fold != fold]
        valid = None if full else self.dev[self.dev.fold == fold]
        seed = self.p.cfg['seed'] + (100 if full else int(fold))
        seed_all(seed, self.cfg['deterministic'])
        last_path = folder / 'last.pt'
        model = self.model(scenario, pretrained=not last_path.exists())
        if scenario in ['S2', 'S3']:
            self.fit_scaler(model, train)
        elif scenario == 'S4' and not last_path.exists():
            source = self.run_dir('S3', fold, tau, full) / 'best.pt'
            model.load_state_dict(load_checkpoint(source)['model'])
        weights, counts = self.weights(train, tau if scenario in ['S3', 'S4'] else 1)
        if scenario == 'S4':
            params = [{'params': model.encoder.parameters(), 'lr': self.cfg['lr_finetune_backbone']},
                      {'params': [v for k, v in model.named_parameters() if not k.startswith('encoder.')], 'lr': self.cfg['lr_finetune_head']}]
        else:
            params = [{'params': [v for v in model.parameters() if v.requires_grad],
                       'lr': self.cfg['lr_frozen'] if model.frozen else self.cfg['lr_scratch']}]
        optimizer = torch.optim.AdamW(params, weight_decay=self.cfg['weight_decay'])
        max_epochs = int(epochs or self.cfg['max_epochs'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        scaler = torch.amp.GradScaler('cuda', enabled=self.cfg['amp'] and self.device.type == 'cuda')
        history, best, best_epoch, start, elapsed = [], -1., 0, 0, 0.
        if last_path.exists():
            ckpt = load_checkpoint(last_path)
            if ckpt['protocol_hash'] != self.p.hash or ckpt['max_epochs'] != max_epochs:
                raise ValueError('Checkpoint tidak sesuai protokol/jumlah epoch.')
            model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler']); scaler.load_state_dict(ckpt['amp'])
            history, best, best_epoch, start, elapsed = ckpt['history'], ckpt['best'], ckpt['best_epoch'], ckpt['epoch'], ckpt['elapsed']
            torch.set_rng_state(ckpt['rng_torch']); np.random.set_state(ckpt['rng_numpy']); random.setstate(ckpt['rng_python'])
            if self.device.type == 'cuda':
                torch.cuda.set_rng_state_all(ckpt['rng_cuda'])
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(start, max_epochs):
            if not full and epoch - best_epoch >= self.cfg['patience']:
                break
            begin = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
            loader = self.loader(train, cached=model.frozen, shuffle=True, seed=seed + epoch)
            losses, n_windows = 0., 0
            accumulation = self.cfg['accumulation_steps']
            for step, batch in enumerate(tqdm(loader, desc=f'{self.arch} {scenario} {fold} epoch {epoch + 1}', leave=False)):
                batch = to_device(batch, self.device)
                group_start = (step // accumulation) * accumulation
                group_windows = min(accumulation * loader.batch_size, len(train) - group_start * loader.batch_size)
                with torch.autocast(self.device.type, enabled=scaler.is_enabled()):
                    output = model(batch['global'], batch['local'], batch['mask'])
                    loss, _ = loss_terms(output, batch, weights, self.cfg)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Loss tidak finite.')
                scaler.scale(loss * len(batch['y']) / group_windows).backward()
                losses += float(loss.detach()) * len(batch['y']); n_windows += len(batch['y'])
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            score = None
            if valid is not None:
                predictions = self.predict(model, valid, cached=model.frozen)
                score = calculate(predictions['y'], predictions['prob'])[0]
            scheduler.step()
            elapsed += time.perf_counter() - begin
            value = score['macro_f1'] if score else float(epoch + 1)
            improved = value > best
            if improved:
                best, best_epoch = value, epoch + 1
            history.append({'epoch': epoch + 1, 'train_loss': losses / n_windows,
                            'val_macro_f1': score['macro_f1'] if score else None, 'elapsed_s': elapsed})
            payload = {'protocol_hash': self.p.hash, 'arch': self.arch, 'scenario': scenario, 'tau': tau,
                       'fold': fold, 'full': full, 'epoch': epoch + 1, 'max_epochs': max_epochs,
                       'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(), 'amp': scaler.state_dict(),
                       'history': history, 'best': best, 'best_epoch': best_epoch, 'elapsed': elapsed,
                       'counts': {k: v.tolist() for k, v in counts.items()},
                       'weights': {k: v.cpu().tolist() for k, v in weights.items()},
                       'train_ids': train.window_id.tolist(), 'rng_torch': torch.get_rng_state(),
                       'rng_numpy': np.random.get_state(), 'rng_python': random.getstate(),
                       'rng_cuda': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else []}
            if improved:
                atomic_torch(folder / 'best.pt', payload)
            atomic_torch(last_path, payload)
            pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
            print(f'{scenario} | fold {fold} | epoch {epoch + 1} | loss {losses / n_windows:.4f} | macro-F1 {value:.4f}' if score else f'{scenario} | development | epoch {epoch + 1}/{max_epochs}')
        checkpoint = load_checkpoint(folder / 'best.pt')
        model.load_state_dict(checkpoint['model'])
        score = {}
        if valid is not None:
            predictions = self.predict(model, valid, cached=model.frozen)
            score = export_metrics(folder, predictions['y'], predictions['prob'], valid.reset_index(drop=True),
                                   predictions['global_prob'], predictions['beat_y'], predictions['beat_prob'])
        result = {'arch': self.arch, 'scenario': scenario, 'fold': int(fold), 'tau': float(tau),
                  'full': full, 'best_epoch': best_epoch, 'protocol_hash': self.p.hash,
                  'checkpoint_sha256': digest(folder / 'best.pt'), 'training_s': elapsed,
                  'peak_vram_bytes': torch.cuda.max_memory_allocated() if self.device.type == 'cuda' else 0,
                  'total_parameters': sum(x.numel() for x in model.parameters()),
                  'trainable_parameters': sum(x.numel() for x in model.parameters() if x.requires_grad), **score}
        result['environment'] = environment()
        result['device'] = torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'cpu'
        atomic_json(done, result)
        return result

    @torch.inference_mode()
    def predict(self, model, rows, cached=False):
        model.eval()
        y, prob, gp, by, bp = [], [], [], [], []
        for batch in self.loader(rows, cached=cached):
            batch = to_device(batch, self.device)
            output = model(batch['global'], batch['local'], batch['mask'])
            y.extend(batch['y'].cpu().tolist()); prob.append(output['logp'].exp().cpu().numpy())
            gp.append(output['global_logits'].softmax(-1).cpu().numpy())
            valid = batch['mask'] & (batch['beat'] >= 0) & (batch['y'][:, None] < 3)
            by.extend(batch['beat'][valid].cpu().tolist())
            bp.append(output['beat_logits'].softmax(-1)[valid].cpu().numpy())
        return {'y': np.asarray(y), 'prob': np.concatenate(prob), 'global_prob': np.concatenate(gp),
                'beat_y': np.asarray(by), 'beat_prob': np.concatenate(bp)}

    def cv(self, scenario, tau=1):
        return pd.DataFrame([self.fit(scenario, fold, tau) for fold in range(5)])

    def ablation(self):
        results = []
        for tau in self.cfg['tau_candidates']:
            frame = self.cv('S3', tau)
            results.append({'tau': tau, 'macro_f1': frame.macro_f1.mean(),
                            'f1_S': frame.f1_S.mean(), 'sd': frame.macro_f1.std(ddof=1)})
        # Kesetaraan diuji pada hasil aktual, bukan dengan menyalin run S2.
        delta = []
        for fold in range(5):
            a = np.load(self.run_dir('S2', fold) / 'probabilities.npy')
            b = np.load(self.run_dir('S3', fold, 1) / 'probabilities.npy')
            delta.append(float(np.max(np.abs(a - b))))
        if max(delta) > 1e-5:
            raise AssertionError(f'S2 dan S3 tau=1 tidak ekuivalen: {delta}')
        table = pd.DataFrame(results)
        candidates = table[table.macro_f1 >= table.macro_f1.max() - self.cfg['tie_tolerance']]
        chosen = candidates.sort_values(['f1_S', 'sd', 'tau'], ascending=[False, True, True]).iloc[0]
        folder = self.p.out / 'selection' / self.arch
        folder.mkdir(parents=True, exist_ok=True)
        table.to_csv(folder / 'ablation.csv', index=False)
        atomic_json(folder / 'selected.json', {'protocol_hash': self.p.hash, 'tau': float(chosen.tau),
                    'equivalence_max_error': max(delta), 'selection_partition': 'development'})
        return table

    def selected_tau(self):
        result = json.loads((self.p.out / 'selection' / self.arch / 'selected.json').read_text())
        if result['protocol_hash'] != self.p.hash:
            raise ValueError('Pemilihan tau tidak sesuai protokol.')
        return result['tau']

    def full_training(self, scenario, tau=1):
        cv = self.cv(scenario, tau)
        epochs = max(1, int(np.floor(np.median(cv.best_epoch) + 0.5)))
        return pd.DataFrame([self.fit(scenario, -1, tau, full=True, epochs=epochs)])

    def status(self):
        rows = []
        tau = self.selected_tau()
        for scenario in self.p.cfg['scenarios']:
            for fold in range(5):
                path = self.run_dir(scenario, fold, tau if scenario in ['S3', 'S4'] else 1) / 'result.json'
                rows.append({'scenario': scenario, 'fold': fold, 'artifact': str(path.relative_to(self.p.out)), 'complete': path.exists()})
            path = self.run_dir(scenario, -1, tau if scenario in ['S3', 'S4'] else 1, True) / 'result.json'
            rows.append({'scenario': scenario, 'fold': 'final', 'artifact': str(path.relative_to(self.p.out)), 'complete': path.exists()})
        return pd.DataFrame(rows)
