import cv2
import numpy as np

from spot_controller.perception.geometry import Intrinsics
from spot_controller.perception.stereo_depth import depth_error, StereoMatcher


def make_pair(width=512, height=288, fx=305.0, baseline=0.12, seed=0):
    """Random-texture scene: a slanted floor plane plus a frontal wall at 4 m. Returns left, right, true depth."""
    rng = np.random.default_rng(seed)
    cx, cy = width / 2.0, height / 2.0
    v = np.arange(height)[:, None].repeat(width, 1)
    u = np.arange(width)[None, :].repeat(height, 0)
    depth = np.full((height, width), 4.0)
    floor = v > cy + 20
    depth[floor] = fx * 0.7 / (v[floor] - cy)            # camera 0.7 m above a flat floor
    depth = np.clip(depth, 0.5, 8.0)
    disp = fx * baseline / depth
    texture = cv2.GaussianBlur(rng.random((height, width + 120)).astype(np.float32), (0, 0), 1.0)
    texture = (255 * (texture - texture.min()) / (np.ptp(texture))).astype(np.float32)
    uu = (u + 60).astype(np.float32)
    left = cv2.remap(texture, uu, v.astype(np.float32), cv2.INTER_LINEAR)
    right = cv2.remap(texture, (uu + disp).astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
    return left.astype(np.uint8), right.astype(np.uint8), depth


def test_depth_matches_truth_and_covers_the_left_edge():
    left, right, truth = make_pair()
    intr = Intrinsics(305.0, 256.0, 144.0, 512, 288)
    matcher = StereoMatcher(intr, num_disp=96)
    disp, depth = matcher.compute(left, right)
    frac, mae, rel, bad = depth_error(depth, truth, max_range=7.0)
    assert frac > 0.85 and rel < 0.05 and bad < 6.0
    edge = depth[:, :40]                                  # the strip plain SGBM leaves invalid
    assert (edge > 0).mean() > 0.5


def test_depth_error_handles_empty_depth():
    truth = np.full((10, 10), 3.0)
    frac, mae, rel, bad = depth_error(np.zeros((10, 10)), truth)
    assert frac == 0.0 and np.isnan(mae)
