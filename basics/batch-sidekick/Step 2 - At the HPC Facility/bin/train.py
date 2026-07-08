#!/usr/bin/env python3
"""
train.py

Train a Ridge regression model:

    LASER_POWERS -> PROTON_TRACE

This script trains from one HDF5 experiment data file and writes one model file.

Expected input HDF5 structure:

    primary/laser_powers
    primary/proton_trace

Example usage:

    python train.py -i experiment_data.h5 -o mymodel.joblib

Written by ChatGPT with help from Scott Feister on 2026-06-16.
Modified by Scott directly on June 29 2026.
Modified by ChatGPT with help from Scott Feister on 2026-07-06.
"""

from pathlib import Path
import argparse
import sys

import h5py
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# -----------------------
# Settings
# -----------------------

test_size = 0.10
split_seed = 0
ridge_alpha = 1.0


# -----------------------
# Command-line arguments
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Ridge model from one HDF5 experiment file."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input HDF5 file, for example experiment_data.h5.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output joblib model file, for example mymodel.joblib.",
    )

    return parser.parse_args()


# -----------------------
# Main script
# -----------------------

def main():
    args = parse_args()

    input_file = args.input
    output_file = args.output

    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    # -----------------------
    # Load data
    # -----------------------

    with h5py.File(input_file, "r") as f:
        X_all = f["primary/laser_powers"][:]
        Y_all = f["primary/proton_trace"][:]

    print(f"Loaded input file: {input_file}")
    print(f"LASER_POWERS shape: {X_all.shape}")
    print(f"PROTON_TRACE shape: {Y_all.shape}")

    if X_all.shape[0] != Y_all.shape[0]:
        raise ValueError(
            "laser_powers and proton_trace have different numbers of examples: "
            f"{X_all.shape[0]} and {Y_all.shape[0]}"
        )

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
    # Save model
    # -----------------------

    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_model, output_file)

    print(f"Saved model to: {output_file}")


if __name__ == "__main__":
    sys.exit(main())