"""Render one "input -> algorithm -> output" figure per perception stage from real data.

    python3 tools/make_pipeline_figures.py --data ~/spot_slam_data/run2 --samples docs/images/samples \
        --out docs/images/pipeline

The stereo figure runs SGBM on the sample pair in --samples. Every other figure runs the real algorithm classes
(ORB + PnP, SLAM, mapper, HOG detector, tracker, frontier search) on the recorded run (tools/record_dataset.py).
The replay takes a few minutes and is cached in --cache so that re-rendering is quick.
"""
import argparse
import glob
import json
import os
import pickle

import cv2
import numpy as np

from spot_controller.perception.detector import HogDetector, OracleDetector
from spot_controller.perception.explorer_node import find_frontiers
from spot_controller.perception.geometry import Intrinsics, T_body_camcv
from spot_controller.perception.mapper import Mapper, points_to_scan
from spot_controller.perception.pose_graph import umeyama
from spot_controller.perception.stereo_depth import StereoMatcher
from spot_controller.perception.stereo_slam import StereoSlam
from spot_controller.perception.tracker import Tracker

FONT = cv2.FONT_HERSHEY_SIMPLEX
INK, MUTED, BG = (40, 40, 40), (120, 120, 120), (255, 255, 255)
PANEL_W = 384


# ---------------------------------------------------------------------------------------- drawing helpers

def text(img, s, org, scale=0.5, color=INK, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def wrap(s, width):
    lines, cur = [], ''
    for word in s.split():
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + ' ' + word).strip()
    return lines + [cur]


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def color_depth(depth, vmax=10.0):
    v = np.clip(depth / vmax, 0, 1)
    out = cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[depth <= 0] = 0
    return out


def panel(img, w=PANEL_W):
    img = to_bgr(img)
    h = int(round(img.shape[0] * w / img.shape[1]))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(150, 150, 150))


def figure(title, subtitle, rows, gap=104):
    """rows: list of rows; a row is a list of (image, caption) and optional arrow labels between items:
    [(img, cap), 'SGBM', (img, cap), ...]."""
    built = []
    for row in rows:
        items, x = [], 0
        for it in row:
            if isinstance(it, str):
                items.append(('arrow', it, x))
                x += gap
            else:
                p = panel(it[0], it[2] if len(it) > 2 else PANEL_W)
                cap = wrap(it[1], max(p.shape[1] // 9, 20))
                items.append(('panel', p, cap, x))
                x += p.shape[1]
        built.append((items, x))
    width = max(max(w for _, w in built) + 60, 900)
    y = 70
    sub = wrap(subtitle, int(width / 9.5))
    y += 20 * len(sub)
    heights = []
    for items, _ in built:
        ph = max(it[1].shape[0] for it in items if it[0] == 'panel')
        cl = max(len(it[2]) for it in items if it[0] == 'panel')
        heights.append((ph, cl))
        y += ph + 22 * cl + 40
    canvas = np.full((y + 10, width, 3), 255, np.uint8)
    tscale = 0.9
    while cv2.getTextSize(title, FONT, tscale, 2)[0][0] > width - 60:
        tscale -= 0.05
    text(canvas, title, (30, 36), tscale, (20, 20, 20), 2)
    for i, line in enumerate(sub):
        text(canvas, line, (30, 66 + 20 * i), 0.5, MUTED)
    y = 70 + 20 * len(sub) + 14
    for (items, rw), (ph, cl) in zip(built, heights):
        x0 = (width - rw) // 2
        for it in items:
            if it[0] == 'panel':
                _, p, cap, x = it
                canvas[y:y + p.shape[0], x0 + x:x0 + x + p.shape[1]] = p
                for k, line in enumerate(cap):
                    text(canvas, line, (x0 + x, y + p.shape[0] + 18 + 18 * k), 0.45, INK)
            else:
                _, label, x = it
                cy = y + ph // 2
                cv2.arrowedLine(canvas, (x0 + x + 8, cy), (x0 + x + gap - 8, cy), (60, 60, 200), 2, tipLength=0.3)
                lines = wrap(label, 13)
                for k, line in enumerate(lines):
                    text(canvas, line, (x0 + x + 4, cy - 12 - 16 * (len(lines) - 1 - k)), 0.4, (60, 60, 200), 1)
        y += ph + 22 * cl + 40
    return canvas


def draw_boxes(img, boxes, labels, color):
    img = img.copy()
    for (u0, v0, u1, v1), lab in zip(boxes, labels):
        cv2.rectangle(img, (int(u0), int(v0)), (int(u1), int(v1)), color, 2)
        if lab:
            (tw, th), _ = cv2.getTextSize(lab, FONT, 0.5, 1)
            cv2.rectangle(img, (int(u0), int(v0) - th - 6), (int(u0) + tw + 4, int(v0)), color, -1)
            text(img, lab, (int(u0) + 2, int(v0) - 4), 0.5, (255, 255, 255))
    return img


def grid_image(grid, scale=3, pad=4):
    """Occupancy grid (y rows, x cols) -> BGR image with y up, cropped to the known area.
    Returns image and the function mapping (row, col) of the grid to image pixels."""
    known = np.argwhere(grid >= 0)
    r0, c0 = np.maximum(known.min(0) - pad, 0)
    r1, c1 = known.max(0) + pad + 1
    sub = grid[r0:r1, c0:c1]
    img = np.full(sub.shape + (3,), 160, np.uint8)
    img[sub == 0] = (255, 255, 255)
    img[sub == 100] = (30, 30, 30)
    img = cv2.resize(img[::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h = sub.shape[0]

    def px(row, col):
        return int((col - c0 + 0.5) * scale), int((h - (row - r0) - 0.5) * scale)
    return img, px


def paths_image(paths, loops, size=432, margin=30):
    allp = np.vstack([p[:, :2] for p in paths.values()])
    lo, hi = allp.min(0), allp.max(0)
    scale = (size - 2 * margin) / max(hi - lo)
    img = np.full((size + 90, size, 3), 255, np.uint8)
    colors = dict(gt=(0, 160, 0), odom=(200, 120, 0), slam=(0, 0, 220))

    def conv(p):
        q = ((p[:, :2] - lo) * scale + margin).astype(np.int32)
        q[:, 1] = size - q[:, 1]
        return q
    for k, (name, p) in enumerate(paths.items()):
        cv2.polylines(img, [conv(p).reshape(-1, 1, 2)], False, colors[name], 2, cv2.LINE_AA)
        text(img, {'gt': 'ground truth', 'odom': 'leg odometry only', 'slam': 'SLAM (after loop closures)'}[name],
             (10, size + 22 + 20 * k), 0.5, colors[name], 1)
    if loops:
        s = conv(paths['slam'])
        for a, b in loops:
            cv2.line(img, tuple(s[a]), tuple(s[b]), (255, 0, 255), 1, cv2.LINE_AA)
        text(img, 'magenta = loop closure', (10, size + 22 + 20 * 3), 0.5, (255, 0, 255), 1)
    return img


# ------------------------------------------------------------------------------------------ data loading

def load(path):
    d = np.load(path, allow_pickle=True)
    return (float(d['t']), cv2.imdecode(d['left'], cv2.IMREAD_COLOR), d['depth'].astype(np.float32),
            d['odom'], d['gt'], json.loads(str(d['objects'])))


def lift(intr, depth, box, T_w_cam):
    u0, v0, u1, v1 = [int(round(v)) for v in box]
    u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, intr.width - 1), min(v1, intr.height - 1)
    patch = depth[v0:v1 + 1, u0:u1 + 1]
    patch = patch[patch > 0.3]
    if len(patch) < 20:
        return None
    z = float(np.median(patch))
    p_cam = intr.backproject(np.array([(u0 + u1) / 2.0]), np.array([(v0 + v1) / 2.0]), np.array([z]))[0]
    return T_w_cam[:3, :3] @ p_cam + T_w_cam[:3, 3], z


# ----------------------------------------------------------------------------------------------- replay

def replay(data, cache):
    if os.path.exists(cache):
        return pickle.load(open(cache, 'rb'))
    cam = json.load(open(os.path.join(data, 'camera.json')))
    intr = Intrinsics(cam['k'][0], cam['k'][2], cam['k'][5], cam['width'], cam['height'])
    slam, mapper, tracker = StereoSlam(intr), Mapper(intr), Tracker()
    oracle = OracleDetector(intr, T_body_camcv())
    T_base_cam = T_body_camcv()
    files = sorted(glob.glob(os.path.join(data, 'f*.npz')))
    kf_gt, kf_odo, loops, trails, best = {}, {}, [], {}, None
    mid = None
    hist = {}                                       # GT object id -> list of (t, xy), for the true speed
    agree = dict(ok=0, bad=0)
    for n, f in enumerate(files):
        t, left, depth, odom, gtp, objects = load(f)
        res = slam.process(t, cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), depth, odom, None)
        if res.state == 'init':
            continue
        if res.keyframe:
            kf_gt[res.kf_id] = (gtp @ T_base_cam)[:3, 3]
            kf_odo[res.kf_id] = (odom @ T_base_cam)[:3, 3]
        if res.keyframe or res.graph_changed:
            mapper.update_keyframes([k.T_w_cam for k in slam.kfs])
        if res.state == 'tracking':
            mapper.add_frame(t, res.kf_id, res.T_kf_cam, res.T_w_cam, depth, None)
            if n >= 900 and mid is None:
                mid = dict(n=n, left=left, depth=depth, T_w_cam=res.T_w_cam.copy(), floor_z=mapper.floor_z)
            for ob in objects:
                hist.setdefault(ob['id'], []).append((t, np.array(ob['pos'][:2])))
                hist[ob['id']] = hist[ob['id']][-40:]
            if n % 3 == 0:
                T_wm = gtp @ T_base_cam @ np.linalg.inv(res.T_w_cam)        # map frame -> simulator world
                dets = []
                for d in oracle.detect(gtp, objects, depth):
                    got = lift(intr, depth, d.box, res.T_w_cam)
                    if got:
                        dets.append(dict(cls=d.cls, pos=got[0], score=1.0, box=d.box, range=got[1]))
                tracks = tracker.update(t, dets)

                def truth(tr):
                    p = T_wm[:3, :3] @ np.array([tr.x[0], tr.x[1], tr.z]) + T_wm[:3, 3]
                    obs = [(np.hypot(*(o[-1][1] - p[:2])), k) for k, o in hist.items() if o[-1][0] == t]
                    if not obs or min(obs)[0] > 2.0:
                        return None, None
                    k = min(obs)[1]
                    old = hist[k][0]
                    return k, float(np.linalg.norm(hist[k][-1][1] - old[1]) / max(t - old[0], 1e-3))
                for tr in tracks:
                    if tr.confirmed and tr.misses == 0:
                        _, gv = truth(tr)
                        if gv is not None:
                            agree['ok' if (gv > 0.3) == bool(tr.moving) else 'bad'] += 1
                    trails.setdefault(tr.id, dict(cls=tr.cls, pts=[], moving=False, confirmed=False))
                    trails[tr.id]['pts'].append((n, tr.x[0], tr.x[1]))
                    trails[tr.id]['moving'] |= bool(tr.moving)
                    trails[tr.id]['confirmed'] |= bool(tr.confirmed)
                truths = {tr.id: truth(tr) for tr in tracks}
                score = sum((2 if tr.moving else 1) for tr in tracks if tr.confirmed and tr.misses == 0
                            and truths[tr.id][1] is not None and (truths[tr.id][1] > 0.3) == bool(tr.moving))
                if score > 0 and (best is None or score > best['score']):
                    best = dict(score=score, n=n, left=left, dets=dets,
                                tracks=[dict(id=tr.id, cls=tr.cls, box=tr.box, speed=float(tr.speed),
                                             moving=bool(tr.moving), confirmed=bool(tr.confirmed),
                                             pos=tr.x[:2].copy(), gt_id=truths[tr.id][0], gt_speed=truths[tr.id][1])
                                        for tr in tracks if tr.misses == 0])
        if res.loop:
            loops.append((res.kf_id, res.loop[0]))
        if (n + 1) % 200 == 0:
            print('replay', n + 1, '/', len(files), 'kfs', len(slam.kfs), 'loops', slam.loop_count, flush=True)
    ids = sorted(kf_gt)
    vm = mapper.vm
    k0 = int(np.ceil((mapper.floor_z + 0.15 - vm.z0) / vm.res))
    occ = vm.L[:, :, k0:] > 1.0
    top = np.where(occ.any(axis=2), k0 + occ.shape[2] - 1 - np.argmax(occ[:, :, ::-1], axis=2), -1)
    out = dict(est=np.array([slam.kfs[i].T_w_cam[:3, 3] for i in ids]), gt=np.array([kf_gt[i] for i in ids]),
               odo=np.array([kf_odo[i] for i in ids]), loops=[(ids.index(a), ids.index(b)) for a, b in loops
                                                              if a in ids and b in ids],
               grid=mapper.grid_2d(), height=top.T.copy(), res=vm.res, x0=vm.x0, y0=vm.y0, z0=vm.z0,
               floor_z=mapper.floor_z, robot=slam.T_w_cam[:3, 3].copy(), trails=trails, best=best, mid=mid,
               n_frames=len(files), rebuilds=mapper.rebuilds, n_kfs=len(slam.kfs), agree=agree)
    pickle.dump(out, open(cache, 'wb'))
    return out


# ------------------------------------------------------------------------------------------------ figures

def fig_stereo(samples, out):
    left = cv2.imread(os.path.join(samples, 'stereo_left.png'))
    right = cv2.imread(os.path.join(samples, 'stereo_right.png'), cv2.IMREAD_GRAYSCALE)
    intr = Intrinsics(305.17, 256.0, 144.0, 512, 288)
    matcher = StereoMatcher(intr)
    disp, depth = matcher.compute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), right)
    dvis = cv2.applyColorMap(np.clip(disp / 96.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dvis[disp <= 0] = 0
    dep = color_depth(depth, 12.0)
    ys, xs = np.nonzero(disp > 6)
    k = len(ys) // 2
    y, x = ys[k], xs[k]
    for img in (dvis, dep):
        cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 2)
    note = 'disparity d = %.1f px at the white circle  ->  z = fx*B/d = %.1f*%.2f/%.1f = %.2f m' % (
        disp[y, x], intr.fx, intr.baseline, disp[y, x], depth[y, x])
    img = figure('1. Stereo depth (SGBM)',
                 'Problem: one camera cannot measure distance. Two cameras 12 cm apart see the same point at slightly '
                 'different columns; that shift (disparity) gives depth. Input: a rectified left/right pair. '
                 'Output: a depth image in metres.',
                 [[(left, 'INPUT left image 512x288'), '', (right, 'INPUT right image (same row = same point)'),
                   'SGBM 7x7 patches, 96 shifts', (dvis, 'disparity (colour = pixel shift, black = no match)'),
                   'z = fx*B/d', (dep, 'OUTPUT depth (colour = metres, black = invalid). ' + note)]])
    cv2.imwrite(os.path.join(out, '01_stereo_depth.png'), img)


def orb_features(gray, n=1000):
    orb = cv2.ORB_create(nfeatures=n, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    return orb.detectAndCompute(gray, None)


def fig_slam(data, rep, out):
    files = sorted(glob.glob(os.path.join(data, 'f*.npz')))
    cam = json.load(open(os.path.join(data, 'camera.json')))
    intr = Intrinsics(cam['k'][0], cam['k'][2], cam['k'][5], cam['width'], cam['height'])
    A, B = load(files[700]), load(files[708])
    ga, gb = (cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY) for x in (A, B))
    ka, da = orb_features(ga)
    kb, db = orb_features(gb)
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 70 and m.distance < 0.85 * n.distance]
    obj, img_pts, keep = [], [], []
    for m in good:
        u, v = ka[m.queryIdx].pt
        z = A[2][int(v), int(u)]
        if 0.5 < z < 10:
            obj.append(intr.backproject(np.array([u]), np.array([v]), np.array([z]))[0])
            img_pts.append(kb[m.trainIdx].pt)
            keep.append(m)
    obj, img_pts = np.array(obj, np.float64), np.array(img_pts, np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img_pts, intr.K, None, iterationsCount=80,
                                             reprojectionError=3.0)
    inl = set(int(i) for i in (inl.ravel() if inl is not None else []))
    side = np.hstack([A[1], B[1]])
    for i, m in enumerate(keep):
        p, q = ka[m.queryIdx].pt, kb[m.trainIdx].pt
        col = (0, 200, 0) if i in inl else (0, 0, 230)
        cv2.line(side, (int(p[0]), int(p[1])), (int(q[0]) + 512, int(q[1])), col, 1, cv2.LINE_AA)
    kp_img = cv2.drawKeypoints(A[1], ka, None, color=(0, 255, 0), flags=0)
    move = float(np.linalg.norm(tvec))
    paths = None
    if rep:
        est, gt, odo = rep['est'], rep['gt'], rep['odo']
        Ra, ta = umeyama(est, gt)
        Rb, tb = umeyama(odo, gt)
        paths = dict(gt=gt, odom=odo @ Rb.T + tb, slam=est @ Ra.T + ta)
    rows = [[(A[1], 'INPUT frame t (left image)'), '', (color_depth(A[2]), 'INPUT depth of frame t (stage 1)'),
             'ORB corners', (kp_img, 'OUTPUT 1: %d ORB features, each with a 256-bit descriptor and, from the depth '
                                     'image, a 3D position' % len(ka))],
            [(side, 'Match descriptors (Hamming, ratio test 0.85) between frame t and t+8, then PnP + RANSAC on the '
                    '3D-2D pairs. %d matches, %d agree with one rigid motion (green), the rest are rejected (red). '
                    'OUTPUT 2: camera moved %.2f m between the two frames.' % (len(keep), len(inl), move), 1000)]]
    if paths:
        rows.append([(paths_image(paths, rep['loops']),
                      'OUTPUT 3: keyframes joined in a pose graph. %d keyframes, %d loop closures. Loop closures pull '
                      'the drifting leg-odometry path (blue) onto the real one (green): red is SLAM. '
                      '(Both estimates are aligned to ground truth for plotting.)' % (rep['n_kfs'], len(rep['loops'])))])
    img = figure('2. SLAM (ORB + PnP + keyframes + pose graph + loop closure)',
                 'Problem: where is the robot? Leg odometry drifts. Input: left image, depth, odometry guess. '
                 'Output: camera pose in the map, keyframes, path.', rows)
    cv2.imwrite(os.path.join(out, '02_slam.png'), img)


def fig_mapping(rep, intr, out):
    mid = rep['mid']
    mp = Mapper(intr)
    pts_cam = mp.points_cam(mid['depth'])
    T_bc = T_body_camcv()
    pts_base = pts_cam @ T_bc[:3, :3].T + T_bc[:3, 3]
    # top-down points: x forward = up, y left = left
    S = 40.0
    td = np.full((360, 360, 3), 255, np.uint8)
    h_above = pts_base[:, 2] + 0.48
    for (x, y, z), hh in zip(pts_base, h_above):
        if x > 8 or abs(y) > 4.5:
            continue
        c = cv2.applyColorMap(np.array([[int(np.clip(hh / 2.0, 0, 1) * 255)]], np.uint8), cv2.COLORMAP_TURBO)[0, 0]
        cv2.circle(td, (int(180 - y * S), int(350 - x * S)), 1, tuple(int(v) for v in c), -1)
    cv2.circle(td, (180, 350), 6, (0, 160, 0), -1)
    text(td, 'robot', (192, 352), 0.45, (0, 160, 0))
    ranges = points_to_scan(pts_base, 0.48)
    sc = np.full((360, 360, 3), 255, np.uint8)
    n = len(ranges)
    for i, r in enumerate(ranges):
        ang = -0.72 + (i + 0.5) * 1.44 / n
        if np.isnan(r):
            continue
        L = 4.0 if np.isinf(r) else min(r, 4.0)
        end = (int(180 - np.sin(ang) * L * 80), int(350 - np.cos(ang) * L * 80))
        cv2.line(sc, (180, 350), end, (190, 235, 190) if np.isinf(r) else (200, 200, 200), 2)
        if not np.isinf(r):
            cv2.circle(sc, end, 3, (0, 0, 220), -1)
    cv2.circle(sc, (180, 350), 6, (0, 160, 0), -1)
    text(sc, 'green = floor seen, nothing there (free)', (8, 318), 0.42, (60, 150, 60))
    text(sc, 'red = nearest obstacle in 0.15-1.4 m band', (8, 336), 0.42, (0, 0, 200))
    grid_img, px = grid_image(rep['grid'], scale=3)
    # 3D height map of occupied voxels (colour = height of the tallest occupied voxel)
    top = rep['height']
    hm = np.full(top.shape + (3,), 160, np.uint8)
    hm[rep['grid'] == 0] = (255, 255, 255)
    occ = top >= 0
    hv = np.clip((top * rep['res'] + rep['z0'] - rep['floor_z']) / 2.5, 0, 1)
    col = cv2.applyColorMap((hv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    hm[occ] = col[occ]
    known = np.argwhere(rep['grid'] >= 0)
    r0, c0 = np.maximum(known.min(0) - 4, 0)
    r1, c1 = known.max(0) + 5
    hm = cv2.resize(hm[r0:r1, c0:c1][::-1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    img = figure('3. Mapping (log-odds voxel grid, rebuilt after loop closures)',
                 'Problem: turn depth images plus poses into a map that survives drift and moving objects. '
                 'Input: depth + SLAM pose. Output: 3D voxel map, 2D occupancy map for navigation, pseudo laser scan.',
                 [[(color_depth(mid['depth']), 'INPUT depth image (stage 1) and the SLAM pose for it'),
                   'back-project', (td, 'points seen from above (x forward). Colour = height above the floor, 0-2 m'),
                   'floor / obstacle band', (sc, 'OUTPUT 1: pseudo laser scan used by Nav2 to mark and clear obstacles')],
                  [(hm, 'OUTPUT 2: the 3D voxel map seen from above (10 cm cells). Colour = height of the tallest '
                        'voxel above the floor, white = free, grey = unknown. Built by ray casting: +0.85 where a ray ends, '
                        '-0.4 along the way.'),
                   'flatten 0.15-1.4 m band', (grid_img, 'OUTPUT 3: the 2D /map (white free, black occupied, grey '
                                                         'unknown) that Nav2 plans on')]])
    cv2.imwrite(os.path.join(out, '03_mapping.png'), img)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def fig_detection(data, intr, out):
    files = sorted(glob.glob(os.path.join(data, 'f*.npz')))
    oracle, hog = OracleDetector(intr, T_body_camcv()), HogDetector()
    hit = miss = None
    for f in files[::4]:
        t, left, depth, odom, gtp, objects = load(f)
        gts = [d for d in oracle.detect(gtp, objects, depth) if d.cls == 'person' and d.box[3] - d.box[1] > 70]
        if not gts:
            continue
        dets = hog.detect(left)
        matched = any(iou(d.box, g.box) > 0.3 for d in dets for g in gts)
        if matched and hit is None:
            hit = (left, dets, gts)
        if not matched and miss is None:
            miss = (left, dets, gts)
        if hit and miss:
            break
    rows = []
    for name, case in (('detected', hit), ('missed', miss)):
        if case is None:
            continue
        left, dets, gts = case
        outimg = draw_boxes(left, [g.box for g in gts], ['', ] * len(gts), (255, 255, 255))
        outimg = draw_boxes(outimg, [d.box for d in dets], ['person %.2f' % d.score for d in dets], (0, 160, 0))
        rows.append([(left, 'INPUT left image'), 'HOG + linear SVM',
                     (outimg, 'OUTPUT (%s case): green = HOG boxes with SVM score, white = where the person really '
                              'is (simulator ground truth, for comparison only)' % name)])
    img = figure('4. Detection (HOG people detector)',
                 'Problem: find what is in the image. Input: left image. Output: list of (class, score, box). '
                 'HOG finds people only: about 47 % recall / 88 % precision here, and no cars.', rows)
    cv2.imwrite(os.path.join(out, '04_detection.png'), img)


def fig_tracking(rep, out):
    best = rep['best']
    if best is None:
        return
    inp = draw_boxes(best['left'], [d['box'] for d in best['dets']],
                     ['%s %.1fm' % (d['cls'], d['range']) for d in best['dets']], (200, 120, 0))
    outimg = best['left'].copy()
    for tr in best['tracks']:
        if tr['box'] is None:
            continue
        lab = '#%d %s %.1f m/s %s (truth %.1f)' % (tr['id'], tr['cls'], tr['speed'], 'MOVING' if tr['moving'] else (
            'static' if tr['confirmed'] else 'new'), tr['gt_speed'] if tr['gt_speed'] is not None else float('nan'))
        outimg = draw_boxes(outimg, [tr['box']], [lab], (0, 0, 220) if tr['moving'] else (0, 150, 0))
    ids = {tr['id'] for tr in best['tracks']}
    win = {k: dict(t, pts=[(x, y) for n, x, y in t['pts'] if abs(n - best['n']) <= 150])
           for k, t in rep['trails'].items() if k in ids}
    win = {k: t for k, t in win.items() if t['pts']}
    allp = np.vstack([np.array(t['pts']) for t in win.values()])
    lo, hi = allp.min(0), allp.max(0)
    size, margin = 432, 40
    scale = (size - 2 * margin) / max(hi - lo)
    top = np.full((size, size, 3), 255, np.uint8)
    for tid, t in win.items():
        p = ((np.array(t['pts']) - lo) * scale + margin).astype(np.int32)
        p[:, 1] = size - p[:, 1]
        col = (0, 0, 220) if t['moving'] else (0, 150, 0)
        cv2.polylines(top, [p.reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)
        cv2.circle(top, tuple(p[-1]), 5, col, -1)
        text(top, '#%d %s' % (tid, t['cls']), (int(p[-1][0]) + 7, int(p[-1][1])), 0.45, col)
    text(top, 'red = moving, green = static (map frame, about +-12 s)', (8, 20), 0.45, MUTED)
    ag = rep['agree']
    note = 'Over the whole run the moving/static label matched the simulator truth for %.0f %% of %d confirmed track updates.' % (
        100.0 * ag['ok'] / max(ag['ok'] + ag['bad'], 1), ag['ok'] + ag['bad'])
    img = figure('5. Tracking (3D lift + Kalman filter + nearest-neighbour association)',
                 'Problem: a box in one image is not an object. Input: 2D boxes, depth, SLAM pose. Output: tracks with '
                 'id, 3D position, velocity and moving/static, and boxes of moving objects fed back to SLAM and the '
                 'mapper. NOTE: the input boxes here are the simulator ground truth (a stand-in for a good detector), '
                 'because the HOG detector sees too few cars and people to show the tracker working. ' + note,
                 [[(inp, 'INPUT 2D boxes. Median depth inside each box + camera model + SLAM pose give a world '
                         'position'), 'Kalman filter per track, match within 1.6 m',
                   (outimg, 'OUTPUT tracks: id, speed from the Kalman filter, moving if > 0.35 m/s; "truth" is the simulator\'s real speed of that object'),
                   '', (top, 'OUTPUT trails of all confirmed tracks, seen from above')]])
    cv2.imwrite(os.path.join(out, '05_tracking.png'), img)


def fig_navigation(rep, out):
    grid, res, x0, y0 = rep['grid'], rep['res'], rep['x0'], rep['y0']
    base, px = grid_image(grid, scale=3)
    rx, ry = rep['robot'][:2]
    rr, rc = (ry - y0) / res, (rx - x0) / res
    clusters = find_frontiers(grid, res)
    fr = base.copy()
    best, best_score = None, 1e9
    for cells, size in clusters:
        for r, c in cells:
            cv2.circle(fr, px(r, c), 2, (0, 0, 230), -1)
        xy = np.stack([x0 + (cells[:, 1] + 0.5) * res, y0 + (cells[:, 0] + 0.5) * res], axis=1)
        d = np.hypot(xy[:, 0] - rx, xy[:, 1] - ry)
        d_far = np.where(d >= 1.0, d, np.inf)
        i = int(np.argmin(d_far))
        if np.isfinite(d_far[i]) and d[i] - 0.03 * size < best_score:
            best, best_score = cells[i], d[i] - 0.03 * size
    start = base.copy()
    cv2.circle(start, px(rr, rc), 6, (0, 160, 0), -1)
    goal = fr.copy()
    cv2.circle(goal, px(rr, rc), 6, (0, 160, 0), -1)
    if best is not None:
        cv2.circle(goal, px(*best), 9, (220, 100, 0), 3)
        cv2.line(goal, px(rr, rc), px(*best), (220, 100, 0), 1, cv2.LINE_AA)
    img = figure('6. Navigation (frontier explorer + Nav2)',
                 'Problem: use the map to move without a person steering. Input: the live /map. Output: a goal for '
                 'Nav2, which plans a path (NavFn) and follows it (Regulated Pure Pursuit) to produce /cmd_vel. '
                 'Shown here: the frontier search on the final map of the recorded run.',
                 [[(start, 'INPUT /map: white free, black occupied, grey unknown. Green dot = robot'),
                   'free cell next to unknown = frontier', (fr, 'frontier cells (red); connected ones form a cluster'),
                   'nearest cluster >= 1 m away', (goal, 'OUTPUT goal (blue ring) sent to Nav2 as a navigation goal')]])
    cv2.imwrite(os.path.join(out, '06_navigation.png'), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--samples', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache', default='/tmp/pipeline_figures_cache.pkl')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cam = json.load(open(os.path.join(args.data, 'camera.json')))
    intr = Intrinsics(cam['k'][0], cam['k'][2], cam['k'][5], cam['width'], cam['height'])
    fig_stereo(args.samples, args.out)
    print('stereo done', flush=True)
    rep = replay(args.data, args.cache)
    print('replay done: %d frames, %d keyframes, %d loops' % (rep['n_frames'], rep['n_kfs'], len(rep['loops'])),
          flush=True)
    fig_slam(args.data, rep, args.out)
    fig_mapping(rep, intr, args.out)
    fig_detection(args.data, intr, args.out)
    fig_tracking(rep, args.out)
    fig_navigation(rep, args.out)
    print('figures in', args.out)


if __name__ == '__main__':
    main()
