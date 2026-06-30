#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict-pva-server.py

Minimal PVAccess server for a simple LASER_POWERS -> PROTON_TRACE prediction.

This server exposes three PVs:

    <PREFIX>:info
        Human-readable description of the server.

    <PREFIX>:inputs:laser_powers
        Writable input array. A client writes laser powers here.

    <PREFIX>:outputs:proton_trace
        Readable output array. This server updates it whenever laser_powers
        is written.

For now, the "prediction" is deliberately simple:

    proton_trace = 2 * laser_powers

Later, this is where we can replace the simple math with a real ML model.

Example command line usage:

    python predict-pva-server.py --prefix "PREDICT-PVA"

Original intention by Scott Feister, July 30, 2026.
Rewritten as a simple p4p example by ChatGPT with help from Scott Feister on 2026-06-30.
"""

import time
import argparse
import numpy as np

from p4p.nt import NTScalar
from p4p.server import Server
from p4p.server.thread import SharedPV

from ScottNTNDArray import ScottNTNDArray


def predict_proton_trace(laser_powers):
    """
    Convert laser powers into a predicted proton trace.

    This is the main science hook.

    Right now this is intentionally trivial:

        proton_trace = 2 * laser_powers

    Later, replace this function with something like:

        proton_trace = model.predict(laser_powers)

    Parameters
    ----------
    laser_powers : array-like
        Input laser powers.

    Returns
    -------
    proton_trace : numpy.ndarray
        Predicted proton trace.
    """
    laser_powers = np.asarray(laser_powers, dtype=np.float64)
    proton_trace = 2.0 * laser_powers

    return proton_trace


def post_proton_trace(proton_trace, *, unique_id=None):
    """
    Post a new proton trace to the output PV.

    In p4p language, "post" means:
        update the server-side value and notify any monitors/subscribers.
    """
    now = time.time()

    dev_proton_trace.post(
        np.asarray(proton_trace, dtype=np.float64),
        uniqueId=unique_id,
        dataTimeStamp=now,
        timestamp=now,
        attrib={"source": "predict_proton_trace(laser_powers)"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal PVA server for LASER_POWERS -> PROTON_TRACE prediction"
    )
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        help='EPICS prefix for this device. Example: --prefix "PREDICT-PVA"',
    )

    args = parser.parse_args()
    DEVICE_NAME = str(args.prefix)

    # -----------------------
    # Create the PVs
    # -----------------------

    dev_info = SharedPV(
        nt=NTScalar("s"),
        initial="PREDICT-PVA Inference Server, 20260630",
    )

    dev_laser_powers = SharedPV(
        nt=ScottNTNDArray(),
        initial=np.zeros(8, dtype=np.uint16),
    )

    dev_proton_trace = SharedPV(
        nt=ScottNTNDArray(),
        initial=np.zeros(8, dtype=np.float64),
    )

    # -----------------------
    # Make laser_powers writable
    # -----------------------
    #
    # This function runs whenever a PVA client writes to:
    #
    #     <PREFIX>:inputs:laser_powers
    #
    # In p4p language:
    #   - "put" is a client-requested write
    #   - "post" is the server updating a PV value
    #
    # So this handler accepts the client put, updates laser_powers, computes
    # proton_trace, and posts the result.

    @dev_laser_powers.put
    def handle_laser_powers_put(pv, op):
        try:
            now = time.time()

            # Get the value written by the client.
            laser_powers = np.asarray(op.value(), dtype=np.uint16)

            # Store the new input value so that reads/monitors of laser_powers
            # see what the client most recently wrote.
            pv.post(
                laser_powers,
                dataTimeStamp=now,
                timestamp=now,
                attrib={"source": "client put"},
            )

            # Run the prediction.
            proton_trace = predict_proton_trace(laser_powers)

            # Use a simple timestamp-derived unique ID for now.
            # Later this could become a shot number, scan index, etc.
            unique_id = int(now * 1_000_000)

            # Publish the prediction.
            post_proton_trace(
                proton_trace,
                unique_id=unique_id,
            )

            # Tell the PVA client that the put succeeded.
            op.done()

        except Exception as err:
            # Tell the PVA client that the put failed.
            op.done(error=str(err))

    # -----------------------
    # Register PV names
    # -----------------------

    providers = {
        DEVICE_NAME + ":info": dev_info,
        DEVICE_NAME + ":inputs:laser_powers": dev_laser_powers,
        DEVICE_NAME + ":outputs:proton_trace": dev_proton_trace,
    }

    print("Starting PREDICT-PVA server with PVs:")
    for name in providers:
        print(f"  {name}")

    # -----------------------
    # Run forever
    # -----------------------

    with Server(providers=[providers]):
        while True:
            time.sleep(1)