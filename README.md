# Airline passenger satisfaction: choosing models through controlled experiments

An Asia Pacific University Optimisation and Deep Learning coursework project that predicts whether a passenger is satisfied using 22 passenger, trip and service-rating features. It compares optimizers and network architectures, then investigates whether hyperparameter search improves the stronger model.

**Status:** a notebook-based research project with raw/processed CSVs and saved result figures. The preprocessing check executes real notebook code. The reported neural-network metrics are **historical figure results**: the optimizer and tuning notebooks have no saved cell outputs, and trained weights and the final selected tuned configuration are not retained.

![Recorded baseline and tuned MLP comparison](Airline%20Satisfaction/ODL/Task2_DataPrep/tune_vs_baseline.png)

*Existing result figure from the repository, not a fresh evaluation. Accuracy increases, while recall decreases slightly.*

## Experiment design

- Six optimizers: SGD, momentum SGD, RMSprop, Adam, Nadam and AdamW, compared over five seeds for an MLP and 1D-CNN.
- An MLP with 64/32 hidden units and a CNN over the ordered tabular feature sequence.
- Random versus Bayesian search with custom k-fold trials, followed by a focused search and checks of fixed settings.
- Accuracy, precision, recall, specificity, F1, MCC and ROC-AUC; confusion matrices and training/search curves.

```mermaid
flowchart LR
    A[Raw train and test CSVs] --> B[Encode and stratify development data]
    B --> C[Training median + log delays + standard scaling]
    C --> D[6 optimizers x 5 seeds x 2 architectures]
    D --> E[MLP selected for tuning]
    E --> F[Random / Bayesian CV search]
    F --> G[Test evaluation and recorded figures]
```

The supplied 103,904-row training file is split into 83,123 training and 20,781 validation rows. The separate test file contains 25,976 passengers. Imputation and scaling learn from the training partition; the same transformations are applied to validation and test. Search reuses preprocessed development data, so preprocessing is **not refitted independently within each CV fold**.

## Recorded results

| Metric | Baseline MLP | Tuned MLP |
|---|---:|---:|
| Accuracy | 95.74% | **96.32%** |
| Precision | 96.23% | **98.09%** |
| Recall | **93.98%** | 93.45% |
| ROC-AUC | 0.993 | **0.995** |

The retained tuned confusion matrix has TN 14,365, FP 208, FN 747 and TP 10,656. The comparison supports a tradeoff: fewer false positives, with more missed satisfied passengers. These values were checked against the saved figures, not recomputed from model predictions.

MLP/Nadam is the recorded baseline winner over CNN/Adam. The historical figures show 28 broad-search trials and 16 focused trials; the current notebook uses reduced budgets of 15 and 12. A fresh run should not be expected to reproduce the figures exactly.

## Run and verify

Use a fresh Python environment with versions supported by TensorFlow on your platform. No environment lockfile is supplied. The lightweight preprocessing check needs:

```powershell
python -m pip install numpy pandas scikit-learn matplotlib seaborn
python "Airline Satisfaction/ODL/docs/verify_preprocessing.py"
```

It executes the notebook's actual preparation cells in a temporary directory and compares all three regenerated CSVs to the committed data. It does not train a model or overwrite repository data. See the [verification record](Airline%20Satisfaction/ODL/docs/VERIFICATION.md) for results and scope.

For the complete experiment, also install TensorFlow, Keras Tuner and Jupyter:

```powershell
python -m pip install tensorflow keras-tuner jupyter
python -m jupyter notebook
```

Run these notebooks in order under `Airline Satisfaction/ODL/`:

1. `Task2_DataPrep/ODL_Task2_DataPreparation.ipynb`, with `Task2_DataPrep/` as the working directory.
2. `ODL_A2_Optimizer_Comparison.ipynb`, from `ODL/`.
3. `ODL_A2_Task2_ModelTuning.ipynb`, from `ODL/`.

Training and cross-validation perform many model fits. Saved plots can be inspected without running this expensive stage. The modelling notebooks write new figures beside themselves; the historical figures shown here remain in `Task2_DataPrep/`.

## Scope and documentation

This is one survey dataset, with no external generalization test or deployed application. Column order gives the tabular CNN an arbitrary neighborhood structure. CV uses each fold's highest validation accuracy, and its preprocessing is shared; validation estimates need that qualification. The exact tuned architecture is not retained, so the original final model cannot be reconstructed from a saved artifact alone.

The [full project guide](Airline%20Satisfaction/ODL/docs/PROJECT_GUIDE.md) preserves the original methodology, source walkthrough, full metric tables and figure catalog. This repository documents the coursework implementation; no collaborator-specific contribution is asserted. Dataset origin and redistribution terms are not established by the retained files, and no repository license file was found. Only the airline project is published; unrelated university materials remain excluded by ignore rules.
