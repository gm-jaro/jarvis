"""Finds the servo directions with the camera: how far and which way the image shifts between two positions."""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from pantilt import PanTilt
from vision import Camera

SIZE = (640, 360)
STEP = 20  # degrees between the compared positions; fewer = more shared frame
MIN_SHIFT_PX = 8  # a smaller shift means the camera does not move together with the head


def gray(img):
    return np.float32(cv2.cvtColor(cv2.resize(img, SIZE), cv2.COLOR_BGR2GRAY))


def measure(cam, pt, a, b):
    window = cv2.createHanningWindow(SIZE, cv2.CV_32F)
    pt.move(*a)
    time.sleep(0.4)
    first = gray(cam.frame())
    pt.move(*b)
    time.sleep(0.4)
    second = gray(cam.frame())
    (dx, dy), response = cv2.phaseCorrelate(first, second, window)
    return dx, dy, response


if __name__ == "__main__":
    pt, cam = PanTilt(), Camera()
    try:
        c = 90
        pdx, pdy, pr = measure(cam, pt, (c - STEP // 2, c), (c + STEP // 2, c))
        tdx, tdy, tr = measure(cam, pt, (c, c - STEP // 2), (c, c + STEP // 2))
        pt.center()
    finally:
        cam.close()
        pt.close()

    print(f"PAN  +{STEP}°: dx={pdx:+.1f} px  dy={pdy:+.1f} px  (confidence {pr:.2f})")
    print(f"TILT +{STEP}°: dx={tdx:+.1f} px  dy={tdy:+.1f} px  (confidence {tr:.2f})")
    if abs(pdx) < MIN_SHIFT_PX or abs(tdy) < MIN_SHIFT_PX:
        raise SystemExit("The image barely shifts: the camera is probably not mounted on the head.")

    cal = {
        "pan_right_increases": bool(pdx < 0),
        "tilt_up_increases": bool(tdy > 0),
        "px_per_deg_at_640": round((abs(pdx) + abs(tdy)) / (2 * STEP), 2),
    }
    Path(__file__).with_name("calibration.json").write_text(json.dumps(cal, indent=2))
    print(json.dumps(cal, indent=2))
    print(f"estimated horizontal field of view ≈ {SIZE[0] / cal['px_per_deg_at_640']:.0f}°")
