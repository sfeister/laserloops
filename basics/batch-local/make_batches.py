#!/usr/bin/env python3
"""
make_batches.py

Create more ELECTRON/PROTON batch folders.
This is a simple wrapper around generate_dataset().

Written by ChatGPT with help from Scott Feister on 2026-06-16.
"""

from pathlib import Path
import re
import numpy as np

from synthetic_data import generate_dataset


# -----------------------
# Settings
# -----------------------

outputs_dir = Path("outputs")
batches_dir = outputs_dir / "batches"

n_new_batches = 3
n_examples_per_batch = 2000
noise_std = 0.03

seed_offset = 1000


# -----------------------
# Helper: find next batch number
# -----------------------

def get_existing_batch_numbers():
    batch_numbers = []

    for batch_dir in batches_dir.glob("batch_*"):
        match = re.fullmatch(r"batch_(\d+)", batch_dir.name)

        if match:
            batch_numbers.append(int(match.group(1)))

    return sorted(batch_numbers)


def get_next_batch_number():
    existing_numbers = get_existing_batch_numbers()

    if not existing_numbers:
        return 1

    return max(existing_numbers) + 1


# -----------------------
# Make new batches
# -----------------------

batches_dir.mkdir(parents=True, exist_ok=True)

next_batch_number = get_next_batch_number()

for i in range(n_new_batches):
    batch_number = next_batch_number + i
    batch_name = f"batch_{batch_number:03d}"
    batch_dir = batches_dir / batch_name

    # Simple deterministic seed so each batch is different.
    seed = seed_offset + batch_number

    X, Y = generate_dataset(
        n_examples=n_examples_per_batch,
        noise_std=noise_std,
        seed=seed,
    )

    batch_dir.mkdir(parents=True, exist_ok=False)

    np.save(batch_dir / "electron.npy", X.astype(np.float32))
    np.save(batch_dir / "proton.npy", Y.astype(np.float32))

    print(f"Wrote {batch_dir}")
    print(f"  seed: {seed}")
    print(f"  electron shape: {X.shape}")
    print(f"  proton shape:   {Y.shape}")