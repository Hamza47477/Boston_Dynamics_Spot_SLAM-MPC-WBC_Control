"""Stereo block matching on the CPU: rectified pair -> disparity -> metric depth.

    left, right (rectified)  --SGBM-->  disparity d [px]  --z = fx * B / d-->  depth [m]

OpenCV's semi-global matching (SGBM, 3-way) with a left-right consistency check; invalid pixels
(no texture, occluded, out of range) are 0. Runs at 512x288 in about 20 ms on one core.
"""
import cv2
import numpy as np

MAX_DEPTH = 15.0


class StereoMatcher:

    def __init__(self, intr, num_disp=96, block=7, max_depth=MAX_DEPTH):
        self.intr = intr
        self.num_disp = num_disp
        self.max_depth = max_depth
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disp, blockSize=block,
            P1=8 * block * block, P2=32 * block * block, disp12MaxDiff=1,
            uniquenessRatio=8, speckleWindowSize=80, speckleRange=2, preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def disparity(self, left_gray, right_gray):
        # SGBM leaves the first `num_disp` columns invalid (no match window to the left of the image);
        # padding both images lets the real left edge be matched, at the cost of ~15 % more pixels.
        pad = self.num_disp
        left_p = cv2.copyMakeBorder(left_gray, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
        right_p = cv2.copyMakeBorder(right_gray, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
        disp = self.sgbm.compute(left_p, right_p)[:, pad:].astype(np.float32) / 16.0
        disp[disp < 0.8] = 0.0
        return disp

    def depth_from_disparity(self, disp):
        depth = np.zeros_like(disp)
        valid = disp > 0
        depth[valid] = self.intr.fx * self.intr.baseline / disp[valid]
        depth[depth > self.max_depth] = 0.0
        return depth

    def compute(self, left_gray, right_gray):
        disp = self.disparity(left_gray, right_gray)
        return disp, self.depth_from_disparity(disp)


def depth_error(depth, gt, max_range=10.0):
    """Score a depth image against ground truth: (valid fraction, MAE m, median rel. error, bad %)."""
    use = (gt > 0.3) & (gt < max_range)
    if not use.any():
        return None
    have = use & (depth > 0)
    frac = float(have.sum()) / float(use.sum())
    if not have.any():
        return frac, float('nan'), float('nan'), float('nan')
    err = np.abs(depth[have] - gt[have])
    rel = err / gt[have]
    bad = float(np.mean((err > 0.25) & (rel > 0.05))) * 100.0
    return frac, float(err.mean()), float(np.median(rel)), bad
