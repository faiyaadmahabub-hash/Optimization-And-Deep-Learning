# Verification record

Date: 17 October 2026. Windows PowerShell, Python 3.10.9; existing numpy, pandas, scikit-learn, matplotlib and seaborn. No dependency install or neural-network training performed.

## Executed

```powershell
$env:MPLBACKEND='Agg'
python "Airline Satisfaction/ODL/docs/verify_preprocessing.py"
```

Exit 0. Executed original data-preparation notebook cells 1, 15, 17, 19, 21, 23, 25 and 27 in a temporary folder using copies of the raw CSVs. This includes source loading, ID removal, encoding, stratified split, training-median imputation, log-delay transforms, training-fitted scaling and CSV export.

Regenerated training (83,123 x 23), validation (20,781 x 23) and test (25,976 x 23) files match the committed CSVs within `rtol=1e-10, atol=1e-10`. All outputs have finite values and binary targets. Temporary files are removed automatically; checked-in CSVs were not rewritten.

## Historical evidence

`Task2_DataPrep/tune_vs_baseline.png` and `tune_confusion.png` were visually inspected. README results agree with those existing figures. They show a historical tuned test accuracy of 96.32%, AUC 0.995, and confusion counts TN 14,365 / FP 208 / FN 747 / TP 10,656. These are not fresh inference or evaluation results. The optimizer/tuning notebooks contain no saved code-cell outputs; model weights and final selected parameters are absent.

## Limits

No trained classifier was rerun. Keras Tuner is unavailable in this system environment, and full search would train many models. The retained search uses 15/12 trials versus 28/16 shown in historical figures. CV consumes globally preprocessed development data rather than refitting preprocessing within each fold, and scores each fit at its highest validation-accuracy epoch. Dataset provenance/license is unresolved in retained files. No browser UI applies to this notebook project. The old README remains in `PROJECT_GUIDE.md`; unrelated university documents remain ignored.
