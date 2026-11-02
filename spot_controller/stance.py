"""Contact (stance) detection from foot height."""
import numpy as np


def detect_stance(feet_body, R_body_to_world, height_threshold=0.02):
    """Return a list of booleans (True = foot on the ground), one per foot.

    The foot heights are taken in a gravity-aligned frame (rotate body-frame foot vectors by
    the estimated attitude, keep z). On flat ground the stance feet are at the lowest height;
    a foot more than `height_threshold` metres above the lowest one is in swing.
    """
    heights = [float((R_body_to_world @ np.asarray(r))[2]) for r in feet_body]
    lowest = min(heights)
    return [h - lowest < height_threshold for h in heights]
