"""Replay a recorded dataset (tools/record_dataset.py) through SLAM (and optionally the mapper) and score it.

    python3 tools/replay_slam.py --data ~/spot_slam_data/run2 [--every 1] [--max 800] [--no-loop] [--map]

Prints progress, per-stage timing, the absolute trajectory error (ATE) of the SLAM keyframes and of the
leg-odometry EKF against ground truth, and writes replay.png (top-down paths) and, with --map, map.png.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from spot_controller.perception.detector import OracleDetector
from spot_controller.perception.geometry import Intrinsics, T_body_camcv
from spot_controller.perception.mapper import Mapper
from spot_controller.perception.pose_graph import ate_rmse, umeyama
from spot_controller.perception.stereo_slam import StereoSlam


def load(path):
    d = np.load(path, allow_pickle=True)
    return (float(d['t']), cv2.imdecode(d['left'], cv2.IMREAD_COLOR), d['depth'].astype(np.float32),
            d['odom'], d['gt'], json.loads(str(d['objects'])))


def oracle_boxes(oracle, gtp, objects, depth, grow=0.15):
    """Image boxes of people and cars from ground truth (simulation cheat), slightly enlarged."""
    boxes = []
    for det in oracle.detect(gtp, objects, depth):
        u0, v0, u1, v1 = det.box
        w, h = u1 - u0, v1 - v0
        boxes.append((u0 - grow * w, v0 - grow * h, u1 + grow * w, v1 + grow * h))
    return boxes


def draw_paths(paths, size=800, margin=40):
    allp = np.vstack([p[:, :2] for p in paths.values() if len(p)])
    lo, hi = allp.min(0), allp.max(0)
    scale = (size - 2 * margin) / max(hi - lo)
    img = np.full((size, size, 3), 255, np.uint8)
    colors = dict(gt=(0, 160, 0), odom=(200, 120, 0), slam=(0, 0, 220))
    for i, (name, p) in enumerate(paths.items()):
        pts = ((p[:, :2] - lo) * scale + margin).astype(np.int32)
        pts[:, 1] = size - pts[:, 1]
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, colors[name], 2, cv2.LINE_AA)
        cv2.putText(img, name, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[name], 2)
    return img


def render_map(mapper, kf_xy, path, pixels_per_cell=4):
    grid = mapper.grid_2d()
    known = np.argwhere(grid >= 0)
    r0, c0 = known.min(0) - 5
    r1, c1 = known.max(0) + 6
    r0, c0 = max(r0, 0), max(c0, 0)
    sub = grid[r0:r1, c0:c1]
    img = np.full(sub.shape + (3,), 150, np.uint8)
    img[sub == 0] = (255, 255, 255)
    img[sub == 100] = (0, 0, 0)
    img = cv2.resize(img[::-1], None, fx=pixels_per_cell, fy=pixels_per_cell, interpolation=cv2.INTER_NEAREST)
    vm = mapper.vm
    for x, y in kf_xy:
        col = (x - vm.x0) / vm.res - c0
        row = (sub.shape[0] - ((y - vm.y0) / vm.res - r0)) 
        cv2.circle(img, (int(col * pixels_per_cell), int(row * pixels_per_cell)), 3, (0, 0, 255), -1)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--max', type=int, default=100000)
    ap.add_argument('--no-loop', action='store_true')
    ap.add_argument('--no-odom', action='store_true')
    ap.add_argument('--map', action='store_true')
    ap.add_argument('--param', action='append', default=[], help='SLAM parameter override key=value')
    ap.add_argument('--mask', choices=['none', 'oracle'], default='none',
                    help='dynamic-object masking: ground-truth boxes (a cheat, to measure the effect)')
    args = ap.parse_args()
    cam = json.load(open(os.path.join(args.data, 'camera.json')))
    intr = Intrinsics(cam['k'][0], cam['k'][2], cam['k'][5], cam['width'], cam['height'])
    overrides = {k: float(v) for k, v in (a.split('=') for a in args.param)}
    if args.no_loop:
        overrides['loop_min_path'] = 10 ** 9
    slam = StereoSlam(intr, **overrides)
    mapper = Mapper(intr) if args.map else None
    oracle = OracleDetector(intr, T_body_camcv()) if args.mask == 'oracle' else None
    files = sorted(glob.glob(os.path.join(args.data, 'f*.npz')))[::args.every][:args.max]
    T_base_cam = T_body_camcv()
    kf_gt, kf_odo, states, times = {}, {}, [], {}
    for n, f in enumerate(files):
        t, left, depth, odom, gtp, objects = load(f)
        boxes = oracle_boxes(oracle, gtp, objects, depth) if oracle else None
        res = slam.process(t, cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), depth, None if args.no_odom else odom,
                           boxes)
        for k, v in res.ms.items():
            times.setdefault(k, []).append(v)
        if res.state == 'init':
            continue
        states.append(res.state)
        if res.keyframe:
            kf_gt[res.kf_id] = (gtp @ T_base_cam)[:3, 3]
            kf_odo[res.kf_id] = (odom @ T_base_cam)[:3, 3]
        if mapper:
            if res.keyframe or res.graph_changed:
                mapper.update_keyframes([k.T_w_cam for k in slam.kfs])
            if res.state == 'tracking':
                mapper.add_frame(t, res.kf_id, res.T_kf_cam, res.T_w_cam, depth, boxes)
        if res.loop:
            print('  loop closure: kf {} <-> kf {} ({} inliers), {} keyframes'.format(
                res.kf_id, res.loop[0], res.loop[1], len(slam.kfs)))
        if (n + 1) % 200 == 0:
            ids = sorted(kf_gt)
            print('frame {:4d} t={:6.1f}s kfs={:3d} loops={} lost={:3d} ate(kf)={:.3f} m'.format(
                n + 1, t, len(slam.kfs), slam.loop_count, states.count('lost'),
                ate_rmse(np.array([slam.kfs[i].T_w_cam[:3, 3] for i in ids]), np.array([kf_gt[i] for i in ids]))[0]))
    ids = sorted(kf_gt)
    est = np.array([slam.kfs[i].T_w_cam[:3, 3] for i in ids])
    gt = np.array([kf_gt[i] for i in ids])
    odo = np.array([kf_odo[i] for i in ids])
    print('\n==== RESULT ({} frames, {} keyframes, {} loops, {} lost frames) ===='.format(
        len(states), len(slam.kfs), slam.loop_count, states.count('lost')))
    print('ATE slam keyframes : rmse %.3f m  max %.3f m' % ate_rmse(est, gt))
    print('ATE leg odometry   : rmse %.3f m  max %.3f m' % ate_rmse(odo, gt))
    print('path length %.1f m' % np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    print('mean ms per frame: ' + ', '.join('%s %.1f' % (k, np.mean(v)) for k, v in times.items()))
    Ra, ta = umeyama(est, gt)
    Rb, tb = umeyama(odo, gt)
    out = os.path.join(args.data, 'replay.png')
    cv2.imwrite(out, draw_paths(dict(gt=gt, odom=odo @ Rb.T + tb, slam=est @ Ra.T + ta)))
    print('plot:', out)
    if mapper:
        mp = os.path.join(args.data, 'map.png')
        render_map(mapper, est[:, :2], mp)
        occ = int((mapper.vm.L > 1.0).sum())
        print('map: {} occupied voxels, {} integrated frames, {} rebuilds, {:.1f} ms/frame; {}'.format(
            occ, len(mapper.frames), mapper.rebuilds, mapper.ms_integrate, mp))


if __name__ == '__main__':
    main()
