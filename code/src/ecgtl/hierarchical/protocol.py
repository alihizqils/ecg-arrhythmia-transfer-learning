"""Konfigurasi, integritas artefak, dan pencatatan lingkungan."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

CLASSES = ['N', 'S', 'V', 'AF', 'VT']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def atomic_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def environment():
    names = ['torch', 'torchvision', 'wfdb', 'numpy', 'scipy', 'pandas',
             'scikit-learn', 'shap', 'matplotlib', 'PyYAML']
    versions = {n: importlib.metadata.version(n) for n in names}
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'cuda': torch.version.cuda, 'packages': versions}


def seed_all(seed, deterministic=True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic)


class Protocol:
    def __init__(self, code_root, output_root=None, data_root=None):
        self.code = Path(code_root).resolve()
        self.folder = self.code / 'eksperimen_hierarkis_10s_lokal'
        self.cfg = yaml.safe_load((self.folder / 'hierarchical_v1.yaml').read_text())
        self.root = Path(data_root).resolve() if data_root else self.code
        self.out = Path(output_root).resolve() if output_root else self.folder / 'outputs'
        self.prepared = self.root / self.cfg['data']['prepared']
        self.source = self.root / self.cfg['data']['source']
        self.raw = self.root / self.cfg['data']['raw']
        self.selection = self.code / 'pilot_durasi_lokal/selected_local_window.json'
        locked = json.loads(self.selection.read_text())
        if (locked['implemented_crop_samples'], locked['detector_offset_samples'],
                locked['holdout_used_for_selection']) != (312, 8, False):
            raise ValueError('Konfigurasi pilot tidak sesuai protokol.')
        if self.cfg['classes'] != CLASSES:
            raise ValueError('Urutan kelas tidak konsisten.')
        self.out.mkdir(parents=True, exist_ok=True)

    def lock(self):
        files = [self.folder / 'hierarchical_v1.yaml', self.folder / 'requirements-colab.txt', self.selection]
        files += sorted((self.code / 'src/ecgtl/hierarchical').glob('*.py'))
        files += [self.code / 'src/ecgtl/data' / n for n in
                  ['segmentation.py', 'rendering.py', 'build_dataset.py', 'inventory.py']]
        hashes = {p.relative_to(self.code).as_posix(): digest(p) for p in files}
        for root_name, root, names in [('source', self.source, ['manifest.csv', 'patient_assignment.csv']),
                ('prepared', self.prepared, ['global_manifest.csv', 'local_manifest.csv', 'data_checksums.json', 'distribution.csv'])]:
            hashes.update({f'{root_name}/{n}': digest(root / n) for n in names})
        hash_value = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        path = self.out / 'protocol_lock.json'
        if path.exists():
            if json.loads(path.read_text())['protocol_hash'] != hash_value:
                raise ValueError('Protokol atau artefak berubah; gunakan run ID/direktori hasil baru.')
            self.hash = hash_value
            return hash_value
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=self.code,
                             capture_output=True, text=True, check=False)
        atomic_json(path, {'protocol_hash': hash_value, 'files': hashes,
                          'run_id': self.cfg['run_id'], 'environment': environment(),
                          'commit': git.stdout.strip() or None})
        self.hash = hash_value
        return hash_value

    def verify(self):
        path = self.out / 'protocol_lock.json'
        if not path.exists():
            raise FileNotFoundError('Artefak Notebook 00 belum lengkap.')
        return self.lock()

    def verify_data(self):
        expected = json.loads((self.prepared / 'data_checksums.json').read_text())
        if digest(self.source / 'images.npy') != expected['source_images']:
            raise ValueError('Checksum citra global tidak sesuai.')
        for name, sha in expected['prepared'].items():
            if digest(self.prepared / name) != sha:
                raise ValueError(f'Checksum data turunan tidak sesuai: {name}')
        return {'data_integrity': 'passed', 'files': len(expected['prepared']) + 1}
