#!/usr/bin/env python3
"""
export_batch_hdf5.py

Export one Tiled run to the next numbered HDF5 batch file.

Each exported run becomes one file:

    outputs/batches/batch_001.h5
    outputs/batches/batch_002.h5
    ...

Written by ChatGPT with help from Scott Feister on 2026-06-26.
"""

import h5py
import numpy as np
from pathlib import Path
import re


# -----------------------
# Settings
# -----------------------

outputs_dir = Path("outputs")
batches_dir = outputs_dir / "batches"


def export_run_to_hdf5_custom2(tiled_client, uid, filename=None):
    """
    Export one specific kind of run to from Tiled to HDF5.

    Created by ChatGPT with help from Scott Feister on 2026-06-25.
    """
    if filename is None:
        filename = f"run_{uid}.h5"

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    primary = tiled_client[uid, "primary"]

    with h5py.File(filename, "w") as f:
        f.attrs["uid"] = uid

        g = f.create_group("primary")

        g.create_dataset("scan_step", data=np.asarray(primary["scan_step"]))
        g.create_dataset("scan_rep", data=np.asarray(primary["scan_rep"]))
        g.create_dataset("laser_powers", data=np.asarray(primary["laser-powers-readback"]))
        g.create_dataset("electron_trace", data=np.asarray(primary["electron_pva-trace-array"]))
        g.create_dataset("electron_shot_num", data=np.asarray(primary["electron_pva-trace-uniqueId"]))
        g.create_dataset("proton_trace", data=np.asarray(primary["proton_pva-trace-array"]))
        g.create_dataset("proton_shot_num", data=np.asarray(primary["proton_pva-trace-uniqueId"]))

    return filename

# -----------------------
# Helper: find next batch number
# -----------------------

def get_existing_batch_numbers():
    batch_numbers = []

    for path in batches_dir.glob("batch_*"):
        # Accept both old-style folders:
        #     batch_001/
        #
        # and new-style HDF5 files:
        #     batch_001.h5
        match = re.fullmatch(r"batch_(\d+)(?:\.h5)?", path.name)

        if match:
            batch_numbers.append(int(match.group(1)))

    return sorted(batch_numbers)


def get_next_batch_number():
    existing_numbers = get_existing_batch_numbers()

    if not existing_numbers:
        return 1

    return max(existing_numbers) + 1


def get_next_batch_hdf5_path():
    batches_dir.mkdir(parents=True, exist_ok=True)

    batch_number = get_next_batch_number()
    batch_name = f"batch_{batch_number:03d}.h5"

    return batches_dir / batch_name


# -----------------------
# Export one run
# -----------------------

def export_run_as_next_batch(tiled_client, uid):
    batch_path = get_next_batch_hdf5_path()

    export_run_to_hdf5_custom2(
        tiled_client,
        uid,
        batch_path,
    )

    print(f"Wrote {batch_path}")

    return batch_path

