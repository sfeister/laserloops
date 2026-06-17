#!/usr/bin/env python3
"""
train.py

Train ELECTRON -> PROTON from all batch folders in outputs/batches.
Each run saves a new incremented model folder.

Written by ChatGPT with help from Scott Feister on 2026-06-16.
"""

from pathlib import Path

import joblib
import numpy as np
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

batch_dirs = sorted(batches_dir.glob("batch_*"))

Xs = []
Ys = []

for batch_dir in batch_dirs:
    Xs.append(np.load(batch_dir / "electron.npy"))
    Ys.append(np.load(batch_dir / "proton.npy"))
    print(f"Loaded {batch_dir}")

X_all = np.concatenate(Xs)
Y_all = np.concatenate(Ys)

print(f"Combined ELECTRON shape: {X_all.shape}")
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