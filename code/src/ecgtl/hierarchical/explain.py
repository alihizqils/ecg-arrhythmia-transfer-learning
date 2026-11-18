"""Atribusi SHAP head global dan lokal dengan anchor berpasangan."""
import json

import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

from .data import WindowDataset, collate, read_manifest
from .protocol import atomic_json, seed_all
from .training import Experiment, load_checkpoint, to_device
from .evaluation import final_models


class HeadView(nn.Module):
    def __init__(self, model, view):
        super().__init__()
        self.model, self.view = model, view

    def forward(self, images):
        feature = self.model.encode(images)
        if self.view == 'global':
            return self.model.global_head((feature - self.model.global_mean) / self.model.global_std).softmax(-1)
        return self.model.beat_head((feature - self.model.local_mean) / self.model.local_std).softmax(-1)


def explain(p):
    p.verify()
    cfg = p.cfg['shap']
    out = p.out / 'shap'; out.mkdir(exist_ok=True)
    manifest = read_manifest(p.prepared / 'global_manifest.csv')
    test = manifest[manifest.split == 'holdout'].reset_index(drop=True)
    dev = manifest[manifest.split == 'development'].reset_index(drop=True)
    # Background berasal dari development; seluruh pasangan memakai ID yang sama.
    background_rows = dev.sample(n=min(cfg['background'], len(dev)), random_state=p.cfg['seed'])
    background_rows[['window_id']].to_csv(out / 'background.csv', index=False)
    finals = {(a, s): f for a, s, _, f, _ in final_models(p)}
    # Pemilihan anchor deterministik dari gabungan TP/FP/FN keenam model.
    anchors = {}
    for arch in p.cfg['backbones']:
        for scenario in ['S3', 'S4']:
            pred = np.load(p.out / 'holdout' / arch / scenario / 'probabilities.npy').argmax(1)
            for cls in range(5):
                for kind, mask in [('TP', (test.label_idx == cls) & (pred == cls)),
                                   ('FP', (test.label_idx != cls) & (pred == cls)),
                                   ('FN', (test.label_idx == cls) & (pred != cls))]:
                    for i in np.flatnonzero(mask)[:cfg['anchors_per_class']]:
                        anchors.setdefault(test.iloc[i].window_id, set()).add(f'{arch}:{scenario}:{cls}:{kind}')
    anchor_rows = test[test.window_id.isin(anchors)].reset_index(drop=True)
    metadata = []
    for arch in p.cfg['backbones']:
        exp = Experiment(p, arch)
        reference = exp.model('S3', pretrained=False).eval()
        reference.load_state_dict(load_checkpoint(finals[(arch, 'S3')] / 'best.pt')['model'])
        dataset = WindowDataset(p, anchor_rows)
        backgrounds = WindowDataset(p, background_rows)
        bg_global = torch.stack([backgrounds[i]['global'] for i in range(len(backgrounds))]).to(exp.device)
        bg_local = torch.stack([backgrounds[i]['local'][0] for i in range(len(backgrounds))]).to(exp.device)
        selected_crops = []
        for i in range(len(dataset)):
            batch = to_device(collate([dataset[i]]), exp.device)
            with torch.no_grad():
                prediction = reference(batch['global'], batch['local'], batch['mask'])
            cls = min(int(prediction['local_logits'].argmax(1)), 2)
            selected_crops.append(int(prediction['attention'][0, :, cls].argmax()))
        # Crop anchor dipilih oleh S3 dan dipertahankan pada S4 untuk perbandingan.
        for scenario in ['S3', 'S4']:
            seed_all(p.cfg['seed'])
            model = exp.model(scenario, pretrained=False).eval()
            model.load_state_dict(load_checkpoint(finals[(arch, scenario)] / 'best.pt')['model'])
            model.requires_grad_(True)
            global_explainer = shap.GradientExplainer(HeadView(model, 'global'), bg_global)
            local_explainer = shap.GradientExplainer(HeadView(model, 'local'), bg_local)
            for i, row in anchor_rows.iterrows():
                item = dataset[i]
                gx = item['global'][None].to(exp.device)
                lx = item['local'][selected_crops[i]][None].to(exp.device)
                gshap = np.asarray(global_explainer.shap_values(gx, nsamples=cfg['nsamples'], rseed=p.cfg['seed']))
                lshap = np.asarray(local_explainer.shap_values(lx, nsamples=cfg['nsamples'], rseed=p.cfg['seed']))
                name = f'{arch}_{scenario}_{row.window_id}.npz'
                np.savez_compressed(out / name, global_image=gx.cpu().numpy(), local_image=lx.cpu().numpy(),
                                    global_shap=gshap, local_shap=lshap)
                metadata.append({'arch': arch, 'scenario': scenario, 'window_id': row.window_id,
                                 'true_class': row.label, 'crop_index': selected_crops[i],
                                 'anchor_categories': ';'.join(sorted(anchors[row.window_id])), 'artifact': name})
    frame = pd.DataFrame(metadata)
    frame.to_csv(out / 'anchors.csv', index=False)
    atomic_json(out / 'complete.json', {'protocol_hash': p.hash, 'models': 6, 'artifacts': len(frame),
                'scope': 'head_probabilities_global_and_local_not_final_fused_decision',
                'clinical_validation': False})
    return frame
