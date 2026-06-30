# Written by ChatGPT with help from Scott Feister on 2026-06-30.
#
# Loops forever over three possible pulses.
#
# Each pulse is written to:
#   - PREDICT-PVA:inputs:laser_powers   using PVAccess / PVA
#   - LASER:powers:set                  using Channel Access / CA
#
# Note:
#   This p4p installation does not include a "ca" provider, so we use:
#       p4p    for PVA
#       pyepics for CA

from time import sleep

import epics
import numpy as np
from p4p.client.thread import Context


pulse_descent = np.uint8(np.round(np.linspace(1, 0, 100) * 255))
pulse_ascent = np.uint8(np.round(np.linspace(0, 1, 100) * 255))
pulse_spiky = np.uint8(np.tile(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    5,
))

pulses = [pulse_descent, pulse_ascent, pulse_spiky]

pva = Context("pva")

while True:
    for pulse in pulses:
        print("Putting pulse")

        # Write to the PVA prediction server.
        pva.put("PREDICT-PVA:inputs:laser_powers", pulse)

        # Write to the real/simulated laser CA PV.
        epics.caput("LASER:powers:set", pulse)

        sleep(3)