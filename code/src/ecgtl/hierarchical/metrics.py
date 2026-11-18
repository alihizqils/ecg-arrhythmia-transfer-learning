"""Metrik dengan urutan kelas tetap dan kurva numerik tersimpan."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

from .protocol import CLASSES, atomic_json


def calculate(y, probabilities, classes=CLASSES):
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError('Probabilitas tidak valid.')
    pred = probabilities.argmax(1)
    cm = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    rows = []
    for i, c in enumerate(classes):
        tp = int(cm[i, i]); fn = int(cm[i].sum() - tp); fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        rows.append({'class': c, 'support': tp + fn, 'precision': tp / (tp + fp) if tp + fp else 0.,
                     'sensitivity': tp / (tp + fn) if tp + fn else None,
                     'specificity': tn / (tn + fp) if tn + fp else None,
                     'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                     'auc': float(roc_auc_score(y == i, probabilities[:, i])) if 0 < (y == i).sum() < len(y) else None})
    def average(key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None
    result = {'accuracy': float(np.mean(pred == y)), 'macro_f1': average('f1'),
              'macro_sensitivity': average('sensitivity'), 'macro_specificity': average('specificity'),
              'macro_auc': average('auc'), 'auc_defined_classes': sum(r['auc'] is not None for r in rows)}
    if 'S' in classes:
        result['f1_S'] = rows[classes.index('S')]['f1']
    return result, pd.DataFrame(rows), cm


def export_metrics(folder, y, probabilities, manifest, global_prob=None, beat_y=None, beat_prob=None):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    summary, per_class, cm = calculate(y, probabilities)
    atomic_json(folder / 'metrics.json', summary)
    per_class.to_csv(folder / 'per_class.csv', index=False)
    np.save(folder / 'confusion.npy', cm)
    np.save(folder / 'probabilities.npy', probabilities)
    np.save(folder / 'labels.npy', y)
    manifest.to_csv(folder / 'manifest.csv', index=False)
    for group in ['patient_id', 'database', 'channel']:
        results = []
        for value, rows in manifest.reset_index(drop=True).groupby(group):
            ix = rows.index.to_numpy()
            score, per, _ = calculate(np.asarray(y)[ix], probabilities[ix])
            results.append({group: value, 'windows': len(ix), **score})
            per.insert(0, group, value)
            if len(results) == 1:
                parts = []
            parts.append(per)
        pd.DataFrame(results).to_csv(folder / f'per_{group}.csv', index=False)
        pd.concat(parts).to_csv(folder / f'per_{group}_class.csv', index=False)
    curves = []
    for i, c in enumerate(CLASSES):
        binary = np.asarray(y) == i
        if 0 < binary.sum() < len(binary):
            fpr, tpr, _ = roc_curve(binary, probabilities[:, i])
            precision, recall, _ = precision_recall_curve(binary, probabilities[:, i])
            curves.extend({'class': c, 'curve': 'ROC', 'x': float(x), 'y': float(v)} for x, v in zip(fpr, tpr))
            curves.extend({'class': c, 'curve': 'PR', 'x': float(x), 'y': float(v)} for x, v in zip(recall, precision))
    pd.DataFrame(curves).to_csv(folder / 'curves.csv', index=False)
    if global_prob is not None:
        sy, pc, mat = calculate(np.where(np.asarray(y) < 3, 0, np.asarray(y) - 2), global_prob, ['beat-based', 'AF', 'VT'])
        atomic_json(folder / 'global_metrics.json', sy)
        pc.to_csv(folder / 'global_per_class.csv', index=False)
        np.save(folder / 'global_confusion.npy', mat)
    if beat_y is not None and len(beat_y):
        sy, pc, mat = calculate(beat_y, beat_prob, ['N', 'S', 'V'])
        atomic_json(folder / 'local_metrics.json', sy)
        pc.to_csv(folder / 'local_per_class.csv', index=False)
        np.save(folder / 'local_confusion.npy', mat)
    return summary
