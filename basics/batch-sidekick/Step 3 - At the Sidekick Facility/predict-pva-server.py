#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict-pva-server.py

Minimal PVAccess server for LASER_POWERS -> PROTON_TRACE prediction.

This server exposes three PVs:

    <PREFIX>:info
        Human-readable description of the server.

    <PREFIX>:inputs:laser_powers
        Writable input array. A client writes laser powers here.

    <PREFIX>:outputs:proton_trace
        Readable output array. This server updates it after inference.

For now, the "prediction" is deliberately simple:

    proton_trace = 2 * laser_powers

Later, replace predict_proton_trace() with a real ML model.

Important behavior
------------------

This server uses a "newest request wins" policy.

That means:

    - If inference is idle, a new laser_powers write starts inference.
    - If inference is already running, a new laser_powers write is remembered.
    - If several writes happen while inference is running, only the newest one
      is kept.
    - Old pending requests are discarded before inference starts.

This is useful for live prediction/control, where stale predictions are not
usually worth computing.

Example command line usage:

    python predict-pva-server.py --prefix "PREDICT-PVA"

Original intention by Scott Feister, July 30, 2026.
Rewritten as a simple p4p example by ChatGPT with help from Scott Feister on 2026-06-30.
"""

import time
import argparse
import threading
import numpy as np

from p4p.nt import NTScalar
from p4p.server import Server
from p4p.server.thread import SharedPV

from ScottNTNDArray import ScottNTNDArray

# Local model inference
from proton_inference import infer


# -----------------------
# Shared worker state
# -----------------------
#
# These variables coordinate the PVA put handler and the background inference
# worker.
#
# latest_request:
#     Holds the newest laser_powers request waiting to be processed.
#     It is either None or a tuple:
#
#         (laser_powers, unique_id)
#
# worker_running:
#     True if the background inference worker is currently active.
#
# request_lock:
#     Protects latest_request and worker_running so that the PVA server thread
#     and the worker thread do not modify them at the same time.

latest_request = None
worker_running = False
request_lock = threading.Lock()


# -----------------------
# Science hook
# -----------------------

def predict_proton_trace(laser_powers):
    """
    Convert laser powers into a predicted proton trace.

    This is the main science hook.

    Commented out is something intentionally trivial:

        proton_trace = 2 * laser_powers

    But have replaced this function with something like:

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
    #laser_powers = np.asarray(laser_powers, dtype=np.float64)
    #proton_trace = 2.0 * laser_powers
    proton_trace = infer(laser_powers)

    return proton_trace


def post_proton_trace(proton_trace, *, unique_id=None):
    """
    Post a new proton trace to the output PV.

    In p4p language, "post" means:
        update the server-side value and notify any monitors/subscribers.

    This function is called by the background inference worker, not directly
    by the PVA put handler.
    """
    now = time.time()

    dev_proton_trace.post(
        np.asarray(proton_trace, dtype=np.float64),
        uniqueId=unique_id,
        dataTimeStamp=now,
        timestamp=now,
        attrib={"source": "predict_proton_trace(laser_powers)"},
    )


def inference_worker():
    """
    Background worker that runs inference without blocking PVA puts.

    This worker uses a "newest request wins" policy.

    The loop is:

        1. Take the newest pending request.
        2. Clear the pending slot.
        3. Run inference.
        4. Post the result.
        5. Check whether a newer request arrived while inference was running.
        6. If yes, process that newest request next.
        7. If no, mark the worker idle and exit.

    This means the server does not build up a long queue of stale predictions.
    """
    global latest_request
    global worker_running

    while True:
        # Take the newest pending request.
        with request_lock:
            if latest_request is None:
                worker_running = False
                return

            laser_powers, unique_id = latest_request
            latest_request = None

        # Run the slow work outside the lock.
        #
        # This is important: while inference is running, the put handler can
        # still accept a newer laser_powers value and store it as latest_request.
        try:
            proton_trace = predict_proton_trace(laser_powers)
            post_proton_trace(proton_trace, unique_id=unique_id)

        except Exception as err:
            print("Inference failed:", err)


# -----------------------
# Main server
# -----------------------

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
    #
    #   put:
    #       A client-requested write.
    #
    #   post:
    #       A server-side update of a PV value.
    #
    # This handler should stay fast. It should not run slow ML inference
    # directly. Instead, it:
    #
    #   1. Accepts the new laser_powers value.
    #   2. Posts laser_powers so reads/monitors see the new input.
    #   3. Stores this as the newest pending inference request.
    #   4. Starts the background worker if it is not already running.
    #   5. Calls op.done() to tell the client the put was accepted.
    #
    # Important:
    #
    #   op.done() means "the input was accepted."
    #
    # It does not mean "the prediction is finished."

    @dev_laser_powers.put
    def handle_laser_powers_put(pv, op):
        global latest_request
        global worker_running

        try:
            now = time.time()

            # Get the value written by the client.
            laser_powers = np.asarray(op.value(), dtype=np.uint16)

            # Use a simple timestamp-derived unique ID for now.
            #
            # This same ID is used for both:
            #
            #   - the input laser_powers
            #   - the output proton_trace made from that input
            #
            # That lets downstream code tell which input produced which output.
            unique_id = int(now * 1_000_000)

            # Store the new input value so that reads/monitors of laser_powers
            # see what the client most recently wrote.
            pv.post(
                laser_powers,
                uniqueId=unique_id,
                dataTimeStamp=now,
                timestamp=now,
                attrib={"source": "client put"},
            )

            # Save this as the newest inference request.
            #
            # The copy is intentional. It protects the worker from any later
            # mutation/reuse of the input array object.
            with request_lock:
                latest_request = (laser_powers.copy(), unique_id)

                # If no worker is running, start one.
                #
                # If a worker is already running, do not start another one.
                # The running worker will pick up latest_request when it
                # finishes its current inference.
                if not worker_running:
                    worker_running = True

                    thread = threading.Thread(
                        target=inference_worker,
                        daemon=True,
                    )
                    thread.start()

            # Tell the PVA client the put was accepted.
            #
            # The prediction may still be running in the background.
            op.done()

        except Exception as err:
            # Tell the PVA client the put failed.
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

    print()
    print("Newest-only inference policy is active.")
    print("A put to inputs:laser_powers is accepted immediately.")
    print("The output proton_trace is posted when inference finishes.")
    print()

    # -----------------------
    # Run forever
    # -----------------------

    with Server(providers=[providers]):
        while True:
            time.sleep(1)