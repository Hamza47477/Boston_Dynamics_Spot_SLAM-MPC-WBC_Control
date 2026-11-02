"""Live dashboard of the whole pipeline in one OpenCV window.

   +---------------------------+---------------------------+
   | SLAM: tracked features    | detections (2D boxes)     |
   +---------------------------+---------------------------+
   | stereo depth              | top-down map, path, robot,|
   |                           | tracked objects, plan     |
   +---------------------------+---------------------------+
   | every stage: rate (fps in wall time), ms per frame, and its own numbers (ATE, tracks, ...) |

Parameters: `save` (path of a PNG to rewrite once per second, for headless use), `show` (open a window).
"""
import json
import os
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

from .detector import CLASS_COLORS
from .geometry import yaw_of
from .ros_utils import image_to_numpy, msg_to_T, SENSOR_QOS

PANEL_W, PANEL_H = 512, 288
STAGE_ORDER = ['stereo_depth', 'slam', 'mapper', 'detector', 'tracker']


class ViewerNode(Node):

    def __init__(self):
        super().__init__('viewer')
        self.declare_parameter('save', '')
        self.declare_parameter('show', True)
        self.show = self.get_parameter('show').value and bool(os.environ.get('DISPLAY'))
        self.save = self.get_parameter('save').value
        self.imgs = {}
        self.stats = {}
        self.grid = None
        self.path = np.zeros((0, 2))
        self.pose = None
        self.tracks = []
        self.plan = np.zeros((0, 2))
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        for topic, key in (('/slam/debug_image', 'slam'), ('/detections/debug_image', 'det')):
            self.create_subscription(Image, topic, lambda m, k=key: self.imgs.__setitem__(k, image_to_numpy(m).copy()),
                                     SENSOR_QOS)
        self.create_subscription(Image, '/stereo/depth/image_raw', self.depth_cb, SENSOR_QOS)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, latched)
        self.create_subscription(Path, '/slam/path', lambda m: setattr(self, 'path', np.array(
            [[p.pose.position.x, p.pose.position.y] for p in m.poses]).reshape(-1, 2)), 5)
        self.create_subscription(Odometry, '/slam/odom', lambda m: setattr(self, 'pose', msg_to_T(m.pose.pose)), 10)
        self.create_subscription(Detection3DArray, '/tracks', self.tracks_cb, 5)
        self.create_subscription(Path, '/plan', lambda m: setattr(self, 'plan', np.array(
            [[p.pose.position.x, p.pose.position.y] for p in m.poses]).reshape(-1, 2)), 5)
        self.create_subscription(String, '/pipeline/stats', self.stats_cb, 50)
        self.create_timer(0.2, self.draw)
        self.last_save = 0.0

    def depth_cb(self, msg):
        d = image_to_numpy(msg)
        img = cv2.applyColorMap(np.clip(d / 8.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[d <= 0] = 0
        self.imgs['depth'] = img

    def map_cb(self, msg):
        self.grid = (np.array(msg.data, np.int8).reshape(msg.info.height, msg.info.width),
                     msg.info.origin.position.x, msg.info.origin.position.y, msg.info.resolution)

    def tracks_cb(self, msg):
        self.tracks = [(d.results[0].hypothesis.class_id, d.bbox.center.position.x, d.bbox.center.position.y,
                        d.results[0].pose.pose.position.x, d.results[0].pose.pose.position.y, d.id)
                       for d in msg.detections]

    def stats_cb(self, msg):
        d = json.loads(msg.data)
        d['t'] = time.time()
        self.stats[d.pop('stage')] = d

    # ------------------------------------------------------------------
    def panel(self, key, title):
        img = self.imgs.get(key)
        if img is None:
            img = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
            cv2.putText(img, 'waiting for ' + key, (20, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        else:
            img = cv2.resize(img, (PANEL_W, PANEL_H))
        cv2.putText(img, title, (8, PANEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def map_panel(self):
        img = np.full((PANEL_H, PANEL_W, 3), 40, np.uint8)
        if self.grid is None:
            return img
        g, x0, y0, res = self.grid
        known = np.argwhere(g >= 0)
        pts = [self.pose[:2, 3]] if self.pose is not None else []
        if len(known) == 0:
            return img
        lo = np.minimum(known.min(0)[::-1] * res + [x0, y0], np.min(pts, 0) if pts else [1e9, 1e9]) - 1.0
        hi = np.maximum(known.max(0)[::-1] * res + [x0, y0], np.max(pts, 0) if pts else [-1e9, -1e9]) + 1.0
        scale = min(PANEL_W / (hi[0] - lo[0]), PANEL_H / (hi[1] - lo[1]))

        def px(x, y):
            return int((x - lo[0]) * scale), int(PANEL_H - (y - lo[1]) * scale)

        c0, r0 = (int((lo[0] - x0) / res), int((lo[1] - y0) / res))
        c1, r1 = (int((hi[0] - x0) / res) + 1, int((hi[1] - y0) / res) + 1)
        sub = g[max(r0, 0):r1, max(c0, 0):c1]
        view = np.full(sub.shape, 128, np.uint8)
        view[sub == 0] = 255
        view[sub == 100] = 0
        view = cv2.resize(view[::-1], (int(sub.shape[1] * res * scale), int(sub.shape[0] * res * scale)),
                          interpolation=cv2.INTER_NEAREST)
        ox, oy = px(x0 + max(c0, 0) * res, y0 + max(r0, 0) * res)
        h, w = view.shape
        y_top = oy - h
        canvas = img
        ys, xs = max(y_top, 0), max(ox, 0)
        ye, xe = min(y_top + h, PANEL_H), min(ox + w, PANEL_W)
        if ye > ys and xe > xs:
            canvas[ys:ye, xs:xe] = cv2.cvtColor(view[ys - y_top:ye - y_top, xs - ox:xe - ox], cv2.COLOR_GRAY2BGR)
        if len(self.path) > 1:
            cv2.polylines(canvas, [np.array([px(*p) for p in self.path], np.int32).reshape(-1, 1, 2)], False,
                          (0, 0, 220), 1, cv2.LINE_AA)
        if len(self.plan) > 1:
            cv2.polylines(canvas, [np.array([px(*p) for p in self.plan], np.int32).reshape(-1, 1, 2)], False,
                          (255, 255, 0), 2, cv2.LINE_AA)
        for cls, x, y, vx, vy, name in self.tracks:
            col = CLASS_COLORS.get(cls, (200, 200, 200))
            moving = np.hypot(vx, vy) > 0.3
            cv2.circle(canvas, px(x, y), 6 if cls == 'person' else 9, col, -1)
            if moving:
                cv2.arrowedLine(canvas, px(x, y), px(x + vx, y + vy), (0, 255, 255), 2, tipLength=0.3)
        if self.pose is not None:
            yaw = yaw_of(self.pose[:3, :3])
            p = self.pose[:2, 3]
            cv2.arrowedLine(canvas, px(*p), px(p[0] + 0.8 * np.cos(yaw), p[1] + 0.8 * np.sin(yaw)),
                            (0, 200, 0), 3, tipLength=0.4)
        cv2.putText(canvas, 'map (white free, black occupied) | green robot, red path, cyan plan', (8, PANEL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def stats_panel(self):
        img = np.zeros((190, 2 * PANEL_W, 3), np.uint8)
        cv2.putText(img, 'stage          fps(wall)  ms/frame  details', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)
        now = time.time()
        for i, name in enumerate(STAGE_ORDER):
            s = self.stats.get(name)
            y = 48 + 28 * i
            if s is None:
                cv2.putText(img, '{:<14s} (not running)'.format(name), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (90, 90, 90), 1, cv2.LINE_AA)
                continue
            extra = ', '.join('{} {}'.format(k, v) for k, v in s.items() if k not in ('fps', 'ms', 'n', 't') and v is not None)
            fresh = now - s['t'] < 3.0
            cv2.putText(img, '{:<14s} {:>5.1f}   {:>6.1f}   {}'.format(name, s['fps'], s['ms'], extra[:112]),
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255) if fresh else (110, 110, 110), 1,
                        cv2.LINE_AA)
        return img

    def draw(self):
        top = np.hstack([self.panel('slam', 'SLAM: tracked features (green = used, red = rejected)'),
                         self.panel('det', 'detections')])
        mid = np.hstack([self.panel('depth', 'stereo depth (colour = distance, black = no match)'), self.map_panel()])
        frame = np.vstack([top, mid, self.stats_panel()])
        if self.save and time.time() - self.last_save > 1.0:
            self.last_save = time.time()
            cv2.imwrite(self.save, frame)
        if self.show:
            cv2.imshow('Spot perception pipeline', frame)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = ViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
