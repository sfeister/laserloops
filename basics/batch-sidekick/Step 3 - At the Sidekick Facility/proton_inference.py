#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proton_inference.py

Simple importable inference code for:

    laser_powers -> proton_trace

The main function is:

    proton_trace = infer(laser_powers)

This file does not:
    - start a PVA server
    - make plots
    - generate synthetic data
    - write output files

It only loads the latest trained model and uses it to predict one proton trace
from one laser-powers array.

Written by ChatGPT with help from Scott Feister on 2026-06-30.
"""

from pathlib import Path

import joblib
import numpy as np


# -----------------------
# Settings
# -----------------------

MODELS_DIR = Path("outputs") / "models"
MODEL_FILE = "ridge_model.joblib"


# -----------------------
# Cached model
# -----------------------
#
# The model is loaded once, on the first call to infer().
# Later calls reuse the same loaded model.

model = None
model_path = None


def latest_model_path():
    """
    Return the path to the newest trained model.

    Expected folder structure:

        outputs/models/model_001/ridge_model.joblib
        outputs/models/model_002/ridge_model.joblib
        outputs/models/model_003/ridge_model.joblib
        ...
    """
    model_dirs = sorted(MODELS_DIR.glob("model_*"))

    if not model_dirs:
        raise FileNotFoundError(
            f"No model folders found in {MODELS_DIR}. "
            "Run train.py first."
        )

    path = model_dirs[-1] / MODEL_FILE

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return path


def get_model():
    """
    Load the newest model once, then reuse it.
    """
    global model
    global model_path

    if model is None:
        model_path = latest_model_path()
        model = joblib.load(model_path)
        print(f"Loaded model: {model_path}")

    return model


def infer(laser_powers):
    """
    Predict one proton trace from one laser-powers array.

    Parameters
    ----------
    laser_powers : array-like
        One laser-powers array, usually shape:

            (100,)

    Returns
    -------
    proton_trace : numpy.ndarray
        One predicted proton-trace array, usually shape:

            (100,)
    """
    laser_powers = np.asarray(laser_powers, dtype=np.float64)

    if laser_powers.ndim != 1:
        raise ValueError(
            f"infer() expects one 1D laser_powers array, got shape {laser_powers.shape}"
        )

    model = get_model()

    # scikit-learn expects a batch:
    #
    #     one trace:     (100,)
    #     batch of one:  (1, 100)
    #
    # So we add one temporary batch dimension, predict, then take the first
    # and only output trace.
    proton_trace = model.predict(laser_powers[None, :])[0]

    return np.asarray(proton_trace, dtype=np.float64)


def current_model_path():
    """
    Return the currently loaded model path.

    Returns None if infer() has not loaded a model yet.
    """
    return model_path