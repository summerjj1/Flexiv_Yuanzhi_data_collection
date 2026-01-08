"""Multi-Tac sensor helper (new module)

Provides:
- Tac3DSensor: thin wrapper around PyTac3D.Sensor with a unified .read() API
- MultiTacSensor: manage and read multiple tactile sensors at once

This is a standalone, clean replacement that won't modify the existing
`touch_reader/tac_sensor.py` file (which may be malformed). Use this module
by importing `touch_reader.multitac_sensor`.
"""

from typing import List, Dict, Any, Optional
import threading
import time
import logging


import PyTac3D


logger = logging.getLogger("multitac")



class MultiTacSensor:
    """Manage multiple tactile sensors and read them together.

    Usage examples:
      # create from ports
      mts = MultiTacSensor(ports=[9989, 9988])
      readings = mts.read_all()

      # or provide existing sensor objects
      s1 = PyTac3D.Sensor(port=9989)
      s2 = PyTac3D.Sensor(port=9988)
      mts = MultiTacSensor(sensors=[s1, s2])

    The class also supports a background thread that continuously reads the
    sensors and stores the latest snapshot accessible via `get_latest()`.
    """

    def __init__(self, ports: Optional[List[int]] = None,  names: Optional[List[str]] = None):
        self.sensors = {}
        for i, port in enumerate(ports):
            self.sensors[names[i]] = PyTac3D.Sensor(port=port, maxQSize=1)
        # warm up
        for s in self.sensors.values():
            time.sleep(0.1)
            s.getFrame()

    def get_observation(self, sensor) -> Dict[str, Any]:
        """Get a combined observation from all sensors."""
        while True:
            obs = sensor.getFrame()
            if obs is not None:
                return obs

    def states(self) -> List[Dict[str, Any]]:
        """Get list of all sensor readings."""
        results = {}
        for s in self.sensors.keys():
            # print("Reading sensor:", s)
            r = self.get_observation(self.sensors[s])
            # print("  Frame :", r)
            results[s] = r
        return results


# Example usage when running directly
if __name__ == "__main__":
    mts = MultiTacSensor(ports=[9989, 9988], names=["Tac3D-75L", "Tac3D-06R"])
    print("Reading once from sensors:")
    r = mts.states()
    print(r)

