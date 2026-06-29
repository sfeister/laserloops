
# Global Imports

import numpy as np
import os
from pathlib import Path
import h5py

from bluesky import RunEngine
from bluesky.callbacks import LiveTable
from bluesky.plans import count, list_scan
import bluesky.plan_stubs as bps

from bluesky.callbacks.tiled_writer import TiledWriter
from tiled.server import SimpleTiledServer
from tiled.client import from_uri

from ophyd_async.core import init_devices

import matplotlib.pyplot as plt

# Local File Imports

from sidekick_model3_CA_devices_v4 import PulseGenerator, LilLaser, Diode
from sidekick_model3_PVA_devices_v4 import DiodePVA
from array_steps import array_one_nd_step_with_reps
from sidekick_model3_helpers import condition_pulse
from tiled_helpers import get_tiled_client
from export_batch_hdf5 import export_run_as_next_batch

# Functions


def main():
    # Start Bluesky RunEngine
    print("Starting Run Engine.")
    RE = RunEngine()
    
    # Initialize Ophyd-Async Devices
    print("Initializing Devices.")
    with init_devices():
        pulsegen = PulseGenerator("PULSEGEN:", name="pulsegen")
        laser = LilLaser("LASER:", name="laser")
        electron = Diode("ELECTRON:", name="electron")
        proton = Diode("PROTON:", name="proton")
        electron_pva = DiodePVA(prefix="pva://ELECTRON-DAQ:", name="electron_pva")
        proton_pva = DiodePVA(prefix="pva://PROTON-DAQ:", name="proton_pva")
    
    # Connect to our Lab-Wide Tiled Server
    print("Connecting to Tiled Server.")
    tiled_client = get_tiled_client()
    tw = TiledWriter(tiled_client)
    RE.subscribe(tw)
    
    # Execute initial settings of Sidekick
    print("Executing run plan for initial settings.")

    def prepare_for_run():
        """Put the Sidekick Model 3 into the standard initial state before a run.
        
        Written by ChatGPT with help from Scott Feister on 2026-06-22.
        Simple Bluesky pre-run setup helper for putting Sidekick devices
        into a known initial state before opening a run.

        """

        # Set all delta-t values.
        yield from bps.mv(
            proton.dt, 5.0e-6,        # seconds
            electron.dt, 5.0e-6,      # seconds
            laser.powers_dt, 5.0,     # microseconds
        )

        # Set trigger delays to known values.
        yield from bps.mv(
            pulsegen.ch2_delay, 100.0,    # microseconds; proton delay
            pulsegen.ch3_delay, 100.0,    # microseconds; electron delay
            pulsegen.ch4_delay, 100.0,    # microseconds; laser delay
        )

        # Set system repetition rate.
        yield from bps.mv(
            pulsegen.reprate, 10.0,       # Hz
        )

    RE(prepare_for_run())
    
    # Run and Save in a Loop
    print("Executing run plan for batch data collection.")
    npulses = 50
    reps = 3
    detectors = [electron_pva.trace, proton_pva.trace]
    motor = laser.powers

    for i in range(3):
        print(f"Beginning batch iteration {i}.")
        pulse_list = [condition_pulse(np.round(np.random.rand(100)*255)) for i in range(npulses)]
        motor_points = pulse_list
        md = {"user_note" : "Acquire and save in a loop, repeated list scans on Sidekick Model 3", "outer_loop_i": i}
        uid, = RE(list_scan(detectors, motor, motor_points, md=md, per_step=array_one_nd_step_with_reps(reps=reps)))
        export_run_as_next_batch(tiled_client, uid)

if __name__ == "__main__":
    main()