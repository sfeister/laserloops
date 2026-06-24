#!/usr/bin/env python3
"""
train_better.py

Train ELECTRON -> PROTON from all batch folders in outputs/batches.
Each run saves a new incremented model folder.

This version keeps the same calling pattern and output files as train.py,
but uses a better nonlinear model internally.

Created by ChatGPT with help from Scott Feister on 2026-06-24.
"""

from pathlib import Path
import re

import joblib
import numpy as np
from tqdm import tqdm

from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------
# Settings
# -----------------------

outputs_dir = Path("outputs")
batches_dir = outputs_dir / "batches"
models_dir = outputs_dir / "models"

test_size = 0.10
split_seed = 0

# Better model settings
hidden_layer_sizes = (256, 256, 128)
activation = "relu"
alpha = 1e-4
learning_rate_init = 1e-3
max_iter = 2000

early_stopping = True
validation_fraction = 0.10
n_iter_no_change = 50


# -----------------------
# Load all batches
# -----------------------

batch_dirs = sorted(batches_dir.glob("batch_*"))

if not batch_dirs:
    raise RuntimeError(f"No batch folders found in {batches_dir}")

Xs = []
Ys = []

for batch_dir in tqdm(batch_dirs, desc="Loading batches"):
    electron_path = batch_dir / "electron.npy"
    proton_path = batch_dir / "proton.npy"

    if not electron_path.exists():
        raise FileNotFoundError(f"Missing {electron_path}")

    if not proton_path.exists():
        raise FileNotFoundError(f"Missing {proton_path}")

    X = np.load(electron_path)
    Y = np.load(proton_path)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Batch {batch_dir} has mismatched example counts: "
            f"electron {X.shape}, proton {Y.shape}"
        )

    Xs.append(X)
    Ys.append(Y)

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

base_model = Pipeline(
    steps=[
        ("x_scaler", StandardScaler()),
        (
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change,
                random_state=split_seed,
                verbose=True,
            ),
        ),
    ]
)

model = TransformedTargetRegressor(
    regressor=base_model,
    transformer=StandardScaler(),
)

print("Training model...")
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)

print(f"Train examples: {X_train.shape[0]}")
print(f"Test examples:  {X_test.shape[0]}")
print(f"Test MSE:       {mse}")


# -----------------------
# Save incremented model
# -----------------------

models_dir.mkdir(parents=True, exist_ok=True)

existing_model_numbers = []

for path in models_dir.glob("model_*"):
    if not path.is_dir():
        continue

    match = re.fullmatch(r"model_(\d+)", path.name)

    if match:
        existing_model_numbers.append(int(match.group(1)))

model_number = max(existing_model_numbers, default=0) + 1

model_dir = models_dir / f"model_{model_number:03d}"
model_dir.mkdir()

# Keep the same output filename as the old script.
model_path = model_dir / "ridge_model.joblib"

joblib.dump(model, model_path)

print(f"Saved model to {model_path}")