"""Visual presentasi dari artefak numerik yang telah diverifikasi."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .protocol import CLASSES, digest

COLORS = dict(zip(CLASSES, ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377']))


def render_all(p):
    checksums = json.loads((p.out / 'artifact_checksums.json').read_text())
    for name, expected in checksums.items():
        if digest(p.out / name) != expected:
            raise ValueError(f'Integritas artefak tidak sesuai: {name}')
    out = p.out / 'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'figure.figsize': (10, 5.625), 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300})
    saved = []
    def save(fig, name):
        fig.tight_layout()
        for ext in ['png', 'pdf', 'svg']:
            path = out / f'{name}.{ext}'
            fig.savefig(path, bbox_inches='tight')
        plt.close(fig); saved.append({'figure': name, 'formats': 'PNG, PDF, SVG', 'status': 'tersedia'})
    fig, ax = plt.subplots(); ax.axis('off')
    nodes = [(0.05, .5, 'Rekaman EKG\n250 Hz'), (.32, .75, 'Citra global\n10 detik'),
             (.32, .25, 'GQRS +8 sampel\nCitra lokal 1,25 detik'), (.61, .75, 'Backbone bersama\nHead B / AF / VT'),
             (.61, .25, 'Backbone bersama\nPerhatian N / S / V'), (.89, .5, 'Fusi\nN, S, V, AF, VT')]
    for x, y, label in nodes:
        ax.text(x, y, label, ha='center', va='center', transform=ax.transAxes,
                bbox={'boxstyle': 'round,pad=0.6', 'fc': '#EAF0F6', 'ec': '#4477AA'})
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)]:
        ax.annotate('', xy=nodes[b][:2], xytext=nodes[a][:2], xycoords='axes fraction', arrowprops={'arrowstyle': '->'})
    save(fig, 'alur_eksperimen')
    distribution = pd.read_csv(p.prepared / 'distribution.csv')
    for metric in ['windows', 'patients']:
        fig, ax = plt.subplots()
        distribution.groupby('label')[metric].sum().reindex(CLASSES).plot.bar(ax=ax, color=[COLORS[c] for c in CLASSES])
        ax.set(xlabel='Kelas', ylabel='Jumlah window' if metric == 'windows' else 'Jumlah pasien', title='Distribusi dataset')
        save(fig, f'distribusi_{metric}')
    qc = pd.read_csv(p.out / 'detector_qc/recall.csv')
    fig, ax = plt.subplots()
    for c in CLASSES[:3]:
        sub = qc[(qc['class'] == c) & (qc.fold >= 0)]
        ax.plot(sub.fold + 1, sub.recall, marker='o', label=c, color=COLORS[c])
    ax.set(xlabel='Fold', ylabel='Recall detektor', ylim=(0, 1.02), title='Mutu deteksi R-peak pada development'); ax.legend()
    save(fig, 'recall_detektor')
    fig, ax = plt.subplots()
    ax.hist(np.load(p.out / 'detector_qc/errors_ms.npy'), bins=41, color='#4477AA')
    ax.set(xlabel='Galat peak terhadap anotasi (ms)', ylabel='Jumlah detak', title='Distribusi galat R-peak')
    save(fig, 'galat_detektor')
    cv = pd.read_csv(p.out / 'tables/cv_summary.csv')
    fig, ax = plt.subplots()
    for j, arch in enumerate(p.cfg['backbones']):
        sub = cv[cv.arch == arch].set_index('scenario').reindex(p.cfg['scenarios'])
        ax.bar(np.arange(4) + (j - 1) * .24, sub.macro_f1_mean, width=.24, yerr=sub.macro_f1_std, capsize=3, label=arch)
    ax.set(xticks=np.arange(4), xticklabels=p.cfg['scenarios'], ylabel='Macro-F1 (mean ± SD)', ylim=(0, 1), title='Kinerja validasi silang'); ax.legend()
    save(fig, 'macro_f1_cv')
    effects = pd.read_csv(p.out / 'tables/paired_effects.csv')
    fig, ax = plt.subplots()
    groups = list(effects.groupby(['arch', 'comparison']))
    ax.boxplot([g.macro_f1 for _, g in groups], tick_labels=['\n'.join(k) for k, _ in groups])
    ax.axhline(0, color='black', linewidth=.8); ax.tick_params(axis='x', labelsize=8)
    ax.set(ylabel='Perubahan macro-F1 per fold', title='Efek skenario berpasangan')
    save(fig, 'efek_berpasangan')
    heat = []
    for arch in p.cfg['backbones']:
        for scenario in p.cfg['scenarios']:
            folder = p.out / 'holdout' / arch / scenario
            cm = np.load(folder / 'confusion.npy')
            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap='Blues'); fig.colorbar(im, ax=ax)
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
            ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
                   xlabel='Prediksi', ylabel='Referensi', title=f'{arch} — {scenario}')
            save(fig, f'confusion_{arch}_{scenario}')
            pc = pd.read_csv(folder / 'per_class.csv').set_index('class').reindex(CLASSES)
            heat.append({'model': f'{arch} {scenario}', **pc.f1.to_dict()})
            curves = pd.read_csv(folder / 'curves.csv')
            for kind in ['ROC', 'PR']:
                fig, ax = plt.subplots()
                for c in CLASSES:
                    sub = curves[(curves['class'] == c) & (curves.curve == kind)]
                    ax.plot(sub.x, sub.y, label=c, color=COLORS[c])
                ax.set(xlabel='False positive rate' if kind == 'ROC' else 'Recall',
                       ylabel='True positive rate' if kind == 'ROC' else 'Precision',
                       xlim=(0, 1), ylim=(0, 1), title=f'{kind}: {arch} — {scenario}'); ax.legend()
                save(fig, f'{kind}_{arch}_{scenario}')
    table = pd.DataFrame(heat).set_index('model')
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(table, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto'); fig.colorbar(im, ax=ax)
    for i in range(len(table)):
        for j in range(5):
            ax.text(j, i, f'{table.iloc[i, j]:.3f}', ha='center', va='center', fontsize=9)
    ax.set(xticks=range(5), xticklabels=CLASSES, yticks=range(len(table)), yticklabels=table.index, title='F1 per kelas pada holdout')
    save(fig, 'heatmap_f1')
    fig, ax = plt.subplots()
    table['S'].plot.barh(ax=ax, color=COLORS['S']); ax.set(xlim=(0, 1), xlabel='F1 kelas S', title='Kinerja kelas S pada holdout')
    save(fig, 'kelas_s')
    for path in (p.out / 'cv').rglob('history.csv'):
        frame = pd.read_csv(path)
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(frame.epoch, frame.train_loss); axes[0].set(xlabel='Epoch', ylabel='Training loss')
        axes[1].plot(frame.epoch, frame.val_macro_f1); axes[1].set(xlabel='Epoch', ylabel='Validation macro-F1', ylim=(0, 1))
        save(fig, 'kurva_' + '_'.join(path.relative_to(p.out / 'cv').parts[:-1]))
    efficiency = pd.read_csv(p.out / 'benchmark/summary.csv').merge(cv, on=['arch', 'scenario'])
    for metric, label in [('parameters', 'Jumlah parameter'), ('model_bytes', 'Ukuran model (byte)'), ('e2e_median_ms', 'Latensi end-to-end offline (ms)')]:
        fig, ax = plt.subplots()
        for row in efficiency.itertuples():
            ax.scatter(getattr(row, metric), row.macro_f1_mean)
            ax.annotate(f'{row.arch} {row.scenario}', (getattr(row, metric), row.macro_f1_mean), fontsize=7)
        ax.set(xlabel=label, ylabel='Mean CV macro-F1', title='Trade-off keandalan dan efisiensi')
        save(fig, f'tradeoff_{metric}')
    anchors = pd.read_csv(p.out / 'shap/anchors.csv')
    for row in anchors.itertuples():
        with np.load(p.out / 'shap' / row.artifact) as values:
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            for v, view in enumerate(['global', 'local']):
                image = values[f'{view}_image'][0, 0]
                shap_values = values[f'{view}_shap']
                if shap_values.shape != (1, 1, *image.shape, 3):
                    raise ValueError(f'Bentuk SHAP tidak sesuai: {shap_values.shape}')
                maps = shap_values[0, 0]
                limit = max(float(np.abs(maps).max()), 1e-10)
                for c in range(3):
                    axes[v, c].imshow(image, cmap='gray', vmin=0, vmax=1)
                    axes[v, c].imshow(maps[..., c], cmap='coolwarm', vmin=-limit, vmax=limit, alpha=.65)
                    axes[v, c].set_title((['B', 'AF', 'VT'] if v == 0 else CLASSES[:3])[c]); axes[v, c].axis('off')
            fig.suptitle(f'{row.arch} {row.scenario} — {row.window_id} — referensi {row.true_class}')
            save(fig, f'shap_{row.arch}_{row.scenario}_{row.window_id}')
    frame = pd.DataFrame(saved)
    frame.to_csv(out / 'visual_audit.csv', index=False)
    return frame
