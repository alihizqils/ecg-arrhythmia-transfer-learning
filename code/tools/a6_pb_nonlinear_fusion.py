"""Controlled standalone pilot: nonlinear local fusion on frozen ImageNet features."""
import hashlib
import json
from pathlib import Path

import torch
import pandas as pd
from torch import nn

from tools.rr_ablation_a6_standalone import A6StandaloneHead
from tools.rr_ablation_a6_patient_balanced import A6PatientBalancedExperiment


class ResidualFusion(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.linear = nn.Linear(inputs, 3)
        self.interaction = nn.Sequential(nn.Linear(inputs, 32), nn.ReLU(),
                                         nn.Dropout(0.2), nn.Linear(32, 3))
        nn.init.zeros_(self.interaction[-1].weight)
        nn.init.zeros_(self.interaction[-1].bias)

    def forward(self, x):
        return self.linear(x) + self.interaction(x)


class NonlinearHead(A6StandaloneHead):
    def __init__(self, dim=1280, dropout=0.2):
        super().__init__(dim, dropout)
        self.beat_head = ResidualFusion(self.beat_head.in_features)


class NonlinearExperiment(A6PatientBalancedExperiment):
    variant = 'A6_PB_LOCAL_NONLINEAR_STANDALONE'
    implementation_file = Path(__file__)

    def compare_fold(self, fold):
        base = self.p.out / 'ablation_rr' / self.arch / 'S3_tau2'
        rows = []
        for name, folder in [('A6_PB', base/'A6_PATIENT_BALANCED'/f'fold_{fold}'),
                             ('Nonlinear', self.run_dir(fold))]:
            result = json.loads((folder/'result.json').read_text())
            per = pd.read_csv(folder/'per_class.csv').set_index('class')
            rows.append({'variant': name, 'macro_f1': result['macro_f1'],
                         **{f'f1_{c}': per.loc[c, 'f1'] for c in ['N','S','V','AF','VT']}})
        table = pd.DataFrame(rows).set_index('variant').T
        table['delta'] = table.Nonlinear - table.A6_PB
        return table

    def build_model(self, fold):
        return NonlinearHead(dropout=self.p.cfg['model']['dropout']).to(self.device)

    def fit_fold(self, fold, **kwargs):
        # Track all shared pilot/core code, including inherited training logic.
        root = Path(__file__).resolve().parents[1]
        files = sorted((root / 'tools').glob('*.py')) + sorted(
            (root / 'src/ecgtl/hierarchical').glob('*.py'))
        fingerprint = hashlib.sha256(b''.join(
            str(p.relative_to(root)).encode() + p.read_bytes() for p in files)).hexdigest()
        folder = self.run_dir(fold)
        folder.mkdir(parents=True, exist_ok=True)
        lock = folder / 'implementation_lock.json'
        if lock.exists():
            if json.loads(lock.read_text())['sha256'] != fingerprint:
                raise ValueError('Kode berubah sejak run dimulai; gunakan run baru.')
        else:
            if (folder / 'last.pt').exists() or (folder / 'result.json').exists():
                raise ValueError('Artefak tanpa implementation lock tidak boleh digunakan.')
            lock.write_text(json.dumps({'sha256': fingerprint}), encoding='utf-8')
        return super().fit_fold(fold, **kwargs)
