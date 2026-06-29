#!/usr/bin/env python3
"""
train.py

Train LASER_POWERS -> PROTON from all batch folders in outputs/batches.
Each run saves a new incremented model folder.

Written by ChatGPT with help from Scott Feister on 2026-06-16.
Modified by Scott directly on June 29 2026.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import h5py
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# -----------------------
# Settings
# -----------------------

outputs_dir = Path("outputs")
batches_dir = outputs_dir / "batches"
models_dir = outputs_dir / "models"

test_size = 0.10
split_seed = 0
ridge_alpha = 1.0


# -----------------------
# Load all batches
# -----------------------

batch_files = sorted(batches_dir.glob("batch_*"))

#with h5py.File(batch_files[0], "r") as f:
#    f.visititems(lambda name, obj: print(name, obj.shape if hasattr(obj, "shape") else ""))
    

Xs = []
Ys = []

for batch_file in batch_files:
    with h5py.File(batch_file, "r") as f:
        Xs.append(f["primary/laser_powers"][:])
        Ys.append(f["primary/proton_trace"][:])
        print(f"Loaded {batch_file}")

X_all = np.concatenate(Xs, axis=0)
Y_all = np.concatenate(Ys, axis=0)

print(f"Combined LASER_POWERS shape: {X_all.shape}")
print(f"Combined PROTON shape:   {Y_all.shape}")

# -----------------------
# Random train/test split
# -----------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X_all,
    Y_all,
    test_size=test_size,
    random_state=split_seed,
    shuffle=True,
)


# -----------------------
# Train model
# -----------------------

ridge_model = Ridge(alpha=ridge_alpha)

ridge_model.fit(X_train, Y_train)

Y_pred = ridge_model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)

print(f"Train examples: {X_train.shape[0]}")
print(f"Test examples:  {X_test.shape[0]}")
print(f"Test MSE:       {mse}")


# -----------------------
# Save incremented model
# -----------------------

models_dir.mkdir(parents=True, exist_ok=True)

existing_model_dirs = sorted(models_dir.glob("model_*"))
model_number = len(existing_model_dirs) + 1

model_dir = models_dir / f"model_{model_number:03d}"
model_dir.mkdir()

model_path = model_dir / "ridge_model.joblib"
joblib.dump(ridge_model, model_path)

print(f"Saved model to {model_path}")