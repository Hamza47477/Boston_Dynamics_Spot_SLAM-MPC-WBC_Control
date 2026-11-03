"""Render a dashboard video (left image + SLAM features, depth, growing map, stats) from a recorded run.

    python3 tools/make_demo_video.py --data ~/spot_slam_data/run2 --out media/spot_slam_demo_raw.mp4

It runs the real SLAM, mapper and HOG detector on every recorded frame (offline, no Webots) and writes every
`--every`-th frame to the video. Encode afterwards with ffmpeg (H.264) for sharing.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from spot_controller.perception.detector import HogDetector
from spot_controller.perception.geometry import Intrinsics
from spot_controller.perception.mapper import Mapper
from spot_controller.perception.stereo_slam import StereoSlam

from make_pipeline_figures import color_depth, draw_boxes, grid_image, load, text

W, H = 640, 360


def fit(img, w=W, h=H):
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
    out = np.full((h, w, 3), 245, np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out, s, x0, y0


def title(img, s):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (35, 35, 35), -1)
    text(img, s, (8, 18), 0.55, (255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--max', type=int, default=100000)
    args = ap.parse_args()
    cam = json.load(open(os.path.join(args.data, 'camera.json')))
    intr = Intrinsics(cam['k'][0], cam['k'][2], cam['k'][5], cam['width'], cam['height'])
    slam, mapper, hog = StereoSlam(intr), Mapper(intr), HogDetector()
    orb = cv2.ORB_create(nfeatures=400, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    files = sorted(glob.glob(os.path.join(args.data, 'f*.npz')))[:args.max]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (2 * W, 2 * H + 32))
    boxes, t0, shown = [], None, 0
    for n, f in enumerate(files):
        t, left, depth, odom, gtp, objects = load(f)
        t0 = t if t0 is None else t0
        res = slam.process(t, cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), depth, odom, None)
        if res.state == 'init':
            continue
        if res.keyframe or res.graph_changed:
            mapper.update_keyframes([k.T_w_cam for k in slam.kfs])
        if res.state == 'tracking':
            mapper.add_frame(t, res.kf_id, res.T_kf_cam, res.T_w_cam, depth, None)
        if n % args.every:
            continue
        if (n // args.every) % 3 == 0:
            boxes = [(d.box, 'person %.2f' % d.score) for d in hog.detect(left)]
        kps = orb.detect(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None)
        img = cv2.drawKeypoints(left, kps, None, color=(0, 255, 0), flags=0)
        img = draw_boxes(img, [b for b, _ in boxes], [l for _, l in boxes], (0, 0, 230))
        a, _, _, _ = fit(title(img, 'ORB features (green), HOG people (red)'))
        b, _, _, _ = fit(color_depth(depth))
        title(b, 'stereo depth (CPU): blue near, red far')
        grid = mapper.grid_2d()
        if (grid >= 0).any():
            gimg, px = grid_image(grid, scale=1)
            vm = mapper.vm
            pts = [px((k.T_w_cam[1, 3] - vm.y0) / vm.res, (k.T_w_cam[0, 3] - vm.x0) / vm.res) for k in slam.kfs]
            cur = px((res.T_w_cam[1, 3] - vm.y0) / vm.res, (res.T_w_cam[0, 3] - vm.x0) / vm.res)
            big, s, x0, y0 = fit(gimg, W, H)
            big = np.ascontiguousarray(big)
            pp = np.array([[int(x * s + x0), int(y * s + y0)] for x, y in pts], np.int32)
            if len(pp) > 1:
                cv2.polylines(big, [pp.reshape(-1, 1, 2)], False, (0, 0, 220), 2, cv2.LINE_AA)
            cv2.circle(big, (int(cur[0] * s + x0), int(cur[1] * s + y0)), 6, (0, 160, 0), -1)
        else:
            big = np.full((H, W, 3), 245, np.uint8)
        c = title(big, 'live 2D map, red = SLAM path')
        info = np.full((H, W, 3), 30, np.uint8)
        lines = ['SLAM state: %s' % res.state, 'keyframes: %d' % len(slam.kfs), 'loop closures: %d' % slam.loop_count,
                 'sim time: %.0f s' % (t - t0), 'SLAM compute: %.0f ms / frame' % res.ms.get('total', 0.0),
                 'map rebuilds after loop closure: %d' % mapper.rebuilds]
        for i, ln in enumerate(lines):
            text(info, ln, (24, 70 + 44 * i), 0.8, (255, 255, 255), 1)
        title(info, 'pipeline status')
        frame = np.vstack([np.hstack([a, b]), np.hstack([c, info])])
        bar = np.full((32, 2 * W, 3), 20, np.uint8)
        text(bar, 'Replay of a recorded Webots simulation. Spot + stereo camera, CPU only, 2x speed. '
                  'Not real-world data.', (10, 22), 0.55, (200, 200, 200))
        vw.write(np.vstack([frame, bar]))
        shown += 1
        if shown % 100 == 0:
            print('rendered', shown, 'frames (source frame', n, 'of', len(files), ')', flush=True)
    vw.release()
    print('wrote', args.out, shown, 'frames')


if __name__ == '__main__':
    main()
