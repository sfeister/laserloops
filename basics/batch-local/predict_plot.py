#!/usr/bin/env python3
"""
predict_plot.py

Generate new random ELECTRON traces, compute true PROTON traces,
use the latest trained model to predict PROTON, and save incremented plots.

Written by ChatGPT with help from Scott Feister on 2026-06-16.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from synthetic_data import generate_random_input_trace, complicated_trace_map


# -----------------------
# Settings
# -----------------------

outputs_dir = Path("outputs")
models_dir = outputs_dir / "models"
plots_dir = outputs_dir / "plots"

n_new_plots = 3
noise_std = 0.03
seed_offset = 2000


# -----------------------
# Find latest model
# -----------------------

model_dirs = sorted(models_dir.glob("model_*"))

if not model_dirs:
    print(f"No model folders found in {models_dir}")
    print("Run python train.py first.")
    raise SystemExit(1)

latest_model_dir = model_dirs[-1]
model_path = latest_model_dir / "ridge_model.joblib"

if not model_path.exists():
    print("Model folder exists, but model file is missing:")
    print(f"  {model_path}")
    print("Run python train.py again.")
    raise SystemExit(1)


# -----------------------
# Load model
# -----------------------

model = joblib.load(model_path)


# -----------------------
# Find next plot number
# -----------------------

plots_dir.mkdir(parents=True, exist_ok=True)

existing_plots = sorted(plots_dir.glob("plot_*.png"))
next_plot_number = len(existing_plots) + 1


# -----------------------
# Make plots
# -----------------------

for i in range(n_new_plots):
    plot_number = next_plot_number + i
    seed = seed_offset + plot_number

    rng = np.random.default_rng(seed)

    x = generate_random_input_trace(rng)
    y = complicated_trace_map(x, noise_std=noise_std, rng=rng)

    y_pred = model.predict(x[None, :])[0]

    output_path = plots_dir / f"plot_{plot_number:03d}.png"

    trace_index = np.arange(100)

    plt.figure(figsize=(8, 5))
    plt.plot(trace_index, y, label="true PROTON")
    plt.plot(trace_index, y_pred, "--", label="predicted PROTON")
    plt.xlabel("Trace point")
    plt.ylabel("Signal")
    plt.title("Fresh random ELECTRON -> PROTON prediction")
    plt.legend()

    plt.text(
        0.02,
        0.02,
        f"Model: {latest_model_dir.name}\nSeed: {seed}",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="bottom",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Wrote {output_path} using {latest_model_dir.name}, seed {seed}")