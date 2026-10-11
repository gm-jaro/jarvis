"""Jarvis pan-tilt client: talks to the firmware on the Wemos over USB serial."""

import re
import sys
import time

import serial

POS_RE = re.compile(r"P=(-?\d+) T=(-?\d+)")
REPLY_PREFIXES = ("OK", "POS", "LIM", "ERR")
DEMO = ["C", "M 50 90", "M 130 90", "M 90 60", "M 90 120", "C"]


class PanTilt:
    def __init__(self, port="/dev/ttyUSB0", baud=115200, boot_timeout=3.0):
        self.ser = serial.Serial(port, baud, timeout=0.2)
        # Opening the port usually resets the D1 mini, so we wait for READY; with no reset the timeout just passes.
        deadline = time.monotonic() + boot_timeout
        while time.monotonic() < deadline:
            if self._readline().startswith("READY"):
                break
        self.ser.reset_input_buffer()

    def _readline(self):
        return self.ser.readline().decode(errors="replace").strip()

    def command(self, cmd, timeout=10.0):
        """Sends a command and returns the reply line; move commands reply only after the move ends."""
        self.ser.write((cmd + "\n").encode())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._readline()
            if line.startswith(REPLY_PREFIXES):
                if line.startswith("ERR"):
                    raise RuntimeError(f"{cmd!r}: {line}")
                return line
        raise TimeoutError(f"brak odpowiedzi na {cmd!r}")

    def move(self, pan, tilt):
        """Turns the head and returns the actual position (the firmware clamps angles outside the limits)."""
        return self._position(self.command(f"M {int(pan)} {int(tilt)}"))

    def center(self):
        return self._position(self.command("C"))

    def position(self):
        return self._position(self.command("?"))

    @staticmethod
    def _position(line):
        pan, tilt = POS_RE.search(line).groups()
        return int(pan), int(tilt)

    def close(self):
        self.ser.close()


if __name__ == "__main__":
    cmds = sys.argv[1:] or ["?"]
    if cmds == ["demo"]:
        cmds = DEMO
    pt = PanTilt()
    try:
        for cmd in cmds:
            print(f"> {cmd}\n{pt.command(cmd)}")
    finally:
        pt.close()
