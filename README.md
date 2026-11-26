# Evaluating Arabic Readability under Cross-Level Ambiguity: An Ordinal Analysis of the SAMER Corpus

Code and test-set predictions for the article:

> Alnefaie, S. Evaluating Arabic Readability under Cross-Level Ambiguity: An Ordinal Analysis of the SAMER Corpus. *Electronics* **2026**. DOI: to be added after publication.

Author: Sarah Alnefaie, Department of Computer Science, Faculty of Computing and Information Technology, King Abdulaziz University, Jeddah, Saudi Arabia.

## Contents

| Path | Purpose |
|---|---|
| `original_experiments/SAMER_main_experiments.ipynb` | Full experimental pipeline: preprocessing, all seven models, metrics, ambiguity analysis, figures. Saved outputs of the reported run are included in the notebook. |
| `original_experiments/environment_versions.json` | Software versions of the reported run (Python 3.12, scikit-learn 1.6.1, PyTorch 2.11.0, Transformers 5.13.1). |
| `predictions/` | Test-set predictions of the seven main models from the reported run and of the six additional baselines (one row per test instance). |
| `revision/revision_analyses.py` | Reproduces every analysis added in the revision (Sections 3.1, 3.3, 5.1, 5.6, 6.3, 6.5; Tables 3, 6, 8, A1-A3), including the sensitivity analysis with merged resampling units (Section 5.6.3). CPU only. |
| `revision/additional_baselines_colab.py` | Trains the GPU-based additional baselines (single-stage AraBERT, frozen AraBERT + LR, SOAR with fine-tuned encoder). Requires a GPU. |
| `revision/char_svd_lr_baseline.py` | Trains the Char. SVD-256 + LR baseline (surface branch of SOAR alone) and saves its predictions. CPU only. |
| `revision/additional_baselines_stats.py` | Reproduces Table 9 and the group-level comparisons of Section 6.5.3 from the saved predictions. CPU only. |
| `revision/additional_baselines_environment.json` | Software versions of the additional-baseline run (Python 3.13, PyTorch 2.11.0, Transformers 5.20.2). |

## Data

The SAMER corpus is not redistributed here. Obtain the official `train.tsv`, `dev.tsv` and `test.tsv` partitions from the SAMER release (Alhafni et al., 2024) and place them in a `data/` folder.

## Reproducing the revision analyses

```
pip install -r requirements.txt   # or, for the revision analyses only: pip install numpy pandas scipy scikit-learn
python revision/revision_analyses.py --data_dir data --pred_dir predictions --out_dir results
python revision/additional_baselines_stats.py --pred_dir predictions --out_dir results
```

Runtime is about 6 minutes in total on a standard CPU. All random procedures use fixed seeds (bootstrap: 42; permutation test: 2026).

## Reproducing the main experiments

Open `original_experiments/SAMER_main_experiments.ipynb` in Google Colab with a GPU runtime, upload the three SAMER files to `/content/sample_data/`, and run all cells. Classical models reproduce exactly; AraBERT-based results may differ in a few predictions because GPU inference is not bit-wise deterministic.

## Note on file paths

The main notebook was run in Google Colab and saves its outputs to Google Drive (`/content/drive/MyDrive/...`). If you run it elsewhere, change these output paths or skip the cells that mount Google Drive.

## Citation

If you use this code, please cite the article above.

```bibtex
@article{alnefaie2026samer,
  author  = {Alnefaie, Sarah},
  title   = {Evaluating Arabic Readability under Cross-Level Ambiguity: An Ordinal Analysis of the {SAMER} Corpus},
  journal = {Electronics},
  year    = {2026},
  note    = {Volume, article number and DOI to be added after publication}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The SAMER corpus is subject to its own license and is not included.
