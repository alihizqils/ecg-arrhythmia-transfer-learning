# Transfer Learning for Arrhythmia Detection with Limited and Imbalanced ECG Data

Code, data protocol, and result tables for the paper

> A. H. Syauqani and S. Mandala, "Performance Study of Transfer Learning for Arrhythmia Detection with Limited and
> Imbalanced Datasets in Resource-Constrained Environments," accepted at SOFTT 2026.

The study compares three ImageNet-pretrained lightweight CNNs (EfficientNet-B0, MobileNetV2, ShuffleNetV2) for
five-class classification (N, S, V, AF/AFL, VT/VFL/VF) of 10-s ECG windows from four PhysioNet databases, under a
patient-wise protocol without augmentation or oversampling. Three transfer strategies are compared: frozen backbone
(S1), frozen backbone with class weighting (S2), and full fine-tuning (S3).

## Repository layout

```
code/
  eksperimen_v4_svdb/          main experiment (pipeline, full model, baselines, analyses)
    metode_v4.yaml             locked method configuration (S1/S2); metode_v4_s3.yaml for S3
    protokol_v4.yaml           data protocol (databases, classes, caps, split rules)
    v4_pipeline.py, v4_s3.py   CNN path training: nested CV, epoch selection, final models
    v4_model_lengkap.py        full model: CNN path + beat-level specialist + logistic fusion + VT lock
    v4_tl_murni.py             image-only baseline
    cv_isolated.py, summary_tables.py, baseline_*.py, ...  evaluation and analysis scripts for the paper tables
    fase_a/, fase_b/           SVDB audit, record selection, label manifest, patient split
    fase_e_ns/                 tabular features and beat-level specialist
    model_lengkap_data/        fitted beat-level specialist package (spesialis_r15.npz)
    outputs_colab_r15/         result tables and method/protocol locks of the official runs
  src/ecgtl/                   shared library (data preparation, rendering, hierarchical model)
  tools/                       feature modules used by the locked pipeline
  configs/                     base configuration
  data/processed/v4_svdb/      label manifest, patient split, checksums (no signals or images)
notebooks/
  colab_training/              official training runs (one notebook per backbone, Colab L4)
  colab_additional_tests/      baselines, seeds, and robustness tests (Colab L4)
  analysis/                    preparation/verification, result summary, efficiency benchmark
paper/make_tables_camera_ready.py  generates the LaTeX tables of the paper from the result files
scripts/verify_hashes.py       checks the code and data protocol against the recorded locks
```

File and variable names are partly in Indonesian (e.g. *hasil* = results, *pasien* = patient, *rekaman* =
record, *acak* = random, *beku* = frozen, *kunci VT* = VT lock, *model lengkap* = full model).

## Data

All recordings are public on PhysioNet and are not redistributed here:

| Database | Use |
|---|---|
| [MIT-BIH Arrhythmia Database (MITDB)](https://physionet.org/content/mitdb/1.0.0/) | development and hold-out |
| [MIT-BIH Atrial Fibrillation Database (AFDB)](https://physionet.org/content/afdb/1.0.0/) | development and hold-out |
| [MIT-BIH Malignant Ventricular Ectopy Database (VFDB)](https://physionet.org/content/vfdb/1.0.0/) | development and hold-out |
| [MIT-BIH Supraventricular Arrhythmia Database (SVDB)](https://physionet.org/content/svdb/1.0.0/) | development and hold-out |
| [St Petersburg INCART 12-lead Arrhythmia Database](https://physionet.org/content/incartdb/1.0.0/) | external test only |

The patient-wise split is in `code/data/processed/v4_svdb/patient_assignment.csv`
(columns `patient_id, database, split, fold`): 129 development patients in five CV folds and 32 hold-out
patients. The hold-out set was never used for training, model selection, or tuning.
`label_manifest.csv` lists every selected 10-s window and its label; `prepared/*_manifest.csv` and
`prepared/data_checksums.json` describe the rendered global and local images.

## Verifying the locked method

The official runs recorded SHA-256 locks of the method code and the data protocol
(`code/eksperimen_v4_svdb/outputs_colab_r15/*_lock.json`). To check that this repository contains the same code and
protocol:

```bash
python scripts/verify_hashes.py
```

The script checks that:

- all 27 files of the data protocol are byte-identical to the run (protocol hash `9db60710fd60`);
- of the 22 files in the method hashes (official `d7f73001bb35`, S3 `44315f01cd18`), 19 are byte-identical and 3
  (`metode_v4.yaml`, `v4_local_rhythm.py`, `fase_e_ns/fitur_ns.py`) differ only in full-line comments or the module
  docstring. For these, the code fingerprint (`scripts/_fingerprint.py`) matches the version that was run.
  `scripts/lock_equivalence.json` holds the SHA-256 of every method file as run; these files reproduce the official
  method hashes.

Expected output ends with `ALL CHECKS PASSED`. Because of the comment edits, new runs from this repository record
method hashes `5a131004958b` (S3 `8bb5f1d88a8a`); the notebooks check these values. Files are stored
byte-for-byte (`.gitattributes` disables line-ending conversion), so the checks also hold after cloning on Windows.

## Reproducing the results

Environment (official runs): Python 3.13, PyTorch 2.6.0 + CUDA 12.4, NVIDIA L4 on Google Colab.
Efficiency was measured on a laptop CPU and an RTX 4050 GPU.

```bash
pip install -r requirements.txt
```

1. **Data preparation.** Download the PhysioNet databases, then run `fase_a/audit_svdb.py`,
   `fase_b/buat_protokol.py` (label manifest, caps, patient split, protocol lock), and the rendering in
   `v4_data.py` (global 224×448 and local 224×224 images). `scripts/verify_hashes.py` confirms that the regenerated
   manifests match the published ones.
2. **Training (CNN path).** `notebooks/colab_training/0{1,2,3}_*_colab.ipynb`, one per backbone, run S1, S2,
   and S3: five-fold CV, epoch selection on CV only, final model, and hold-out evaluation.
3. **Full model.** `v4_model_lengkap.py` adds the beat-level specialist, logistic fusion, and VT lock.
   The fitted specialist is provided in `model_lengkap_data/`.
4. **Patient-isolated CV, label-free hold-out templates, effects, and leakage control.**
   `cv_isolated.py`, `size_control.py`, `holdout_labelfree.py`, `full_record_template.py`,
   `cv_sensitivity_*.py`, `selection_consistency.py`, `xai_labelfree.py`, `xai_compare.py`, and `summary_tables.py`.
5. **Baselines, seeds, and robustness tests.** `notebooks/colab_additional_tests/` (training on Colab), then
   `evaluate_tests.py` and `analyze_tests.py` (evaluation and statistics),
   `baseline_tabular*.py` (tabular-only baseline), `pipeline_latency.py` and `latency_per_crop.py`,
   `per_database.py`, `recall_per_source_cv.py`, `template_selection.py`, and `reproduce_final_s2.py`
   (bit-level reproduction check of the S2 final model). Test IDs in the result files: U3 image-only,
   U4 seeds, U5 random backbone, U6 learning curve, U2 leave-source-out, B4c tabular-only.
6. **Efficiency.** `notebooks/analysis/05_benchmark_efisiensi_r15.ipynb` (`v4_benchmark.py`).

Trained checkpoints and image caches are not included because of their size.

## Where each result comes from

All paths are relative to `code/eksperimen_v4_svdb/outputs_colab_r15/`.
`paper/make_tables_camera_ready.py` builds the LaTeX tables directly from these files.

| Paper item | Result files |
|---|---|
| Table II (CV and hold-out per backbone and strategy, three seeds) | `main_results/tabel_utama_holdout.csv`, `main_results/cv_terisolasi_per_fold.csv`, `additional_tests/analisis_colab/ringkas_holdout.csv` |
| Table III (S2 − S1 and S3 − S2 effects, patient bootstrap, Holm) | `main_results/efek_skenario_holdout.csv` |
| Table IV (baselines and random-backbone comparison) | `additional_tests/analisis_colab/ringkas_holdout.csv`, `additional_tests/analisis_colab/banding_holdout.csv`, `additional_tests/b4_tabular_only/` |
| Table V (size, latency, memory, training time) | `benchmark/*.json`, `additional_tests/b1_latensi/` |
| Epoch selection | `selection/<backbone>/S*_final_epoch*.{json,csv}` |
| Leakage control and template sensitivity | `main_results/kontrol_kebocoran_*.csv`, `main_results/cv_sensitivitas_*.csv`, `additional_tests/b5_seleksi_template.csv` |
| Per-database results and leave-source-out test | `main_results/per_database_holdout.csv`, `additional_tests/b3_per_database*.csv`, `additional_tests/analisis_colab/u2_recall_per_*.csv` |
| Learning curve | `additional_tests/analisis_colab/ringkas_holdout.csv` (test `U6`) |
| SHAP comparison | `main_results/banding_xai_effb0_s2.csv` |

## Citation

```bibtex
@inproceedings{syauqani2026transfer,
  author    = {Syauqani, Ali Hizqil and Mandala, Satria},
  title     = {Performance Study of Transfer Learning for Arrhythmia Detection with Limited and Imbalanced
               Datasets in Resource-Constrained Environments},
  booktitle = {Proceedings of SOFTT 2026},
  year      = {2026}
}
```
