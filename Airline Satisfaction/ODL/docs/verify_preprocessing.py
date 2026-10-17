"""Execute the notebook's preparation cells in a temporary working folder."""
from pathlib import Path
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

ODL = Path(__file__).resolve().parents[1]
notebook = json.loads((ODL / "Task2_DataPrep/ODL_Task2_DataPreparation.ipynb").read_text(encoding="utf-8"))
before = Path.cwd()
with tempfile.TemporaryDirectory(prefix="airline-prep-") as temporary:
    root = Path(temporary)
    (root / "Dataset").mkdir()
    (root / "Task2_DataPrep").mkdir()
    for name in ["train.csv", "test.csv"]:
        shutil.copyfile(ODL / "Dataset" / name, root / "Dataset" / name)
    try:
        os.chdir(root / "Task2_DataPrep")
        scope = {}
        for index in [1, 15, 17, 19, 21, 23, 25, 27]:
            exec(compile("".join(notebook["cells"][index]["source"]), f"preparation-cell-{index}", "exec"), scope)
        for name, expected_rows in [("train_processed.csv", 83123), ("val_processed.csv", 20781), ("test_processed.csv", 25976)]:
            actual = pd.read_csv(name)
            saved = pd.read_csv(ODL / "Task2_DataPrep" / name)
            assert actual.shape == (expected_rows, 23)
            assert list(actual.columns) == list(saved.columns)
            assert np.isfinite(actual.to_numpy()).all()
            assert set(actual.satisfaction.unique()) == {0, 1}
            np.testing.assert_allclose(actual.to_numpy(), saved.to_numpy(), rtol=1e-10, atol=1e-10)
            print(f"PASS: {name}: {actual.shape}; matches committed values within 1e-10 tolerance")
    finally:
        os.chdir(before)
print("PASS: exact notebook preparation cells rerun; no model training or repository CSV writes")
