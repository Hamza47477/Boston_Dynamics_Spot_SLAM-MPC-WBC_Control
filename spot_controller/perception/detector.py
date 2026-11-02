"""2D object detection backends (CPU). All return a list of Detection(cls, score, box=(u0, v0, u1, v1)).

  yolo    YOLOv4-tiny through OpenCV's DNN module (people, cars, trucks, ...; needs the model files).
  hog     OpenCV's HOG + linear SVM pedestrian detector (people only, no download needed).
  oracle  Simulation ground truth projected into the image. A CHEAT for developing the downstream stages
          (tracking, dynamic masking, mapping): it is never selected automatically.
"""
import os
from collections import namedtuple

import cv2
import numpy as np

Detection = namedtuple('Detection', 'cls score box')

DYNAMIC_CLASSES = {'person', 'car', 'truck', 'bus', 'bicycle', 'motorbike', 'motorcycle'}
CLASS_COLORS = {'person': (60, 60, 255), 'car': (255, 160, 40), 'truck': (255, 160, 40)}
MODEL_DIR = os.path.expanduser('~/.spot_models')


class HogDetector:
    name = 'hog'

    def __init__(self, upscale=1.5, min_score=0.5):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.upscale, self.min_score = upscale, min_score

    def detect(self, bgr):
        img = cv2.resize(bgr, None, fx=self.upscale, fy=self.upscale)
        rects, weights = self.hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.06)
        out = []
        for (x, y, w, h), score in zip(rects, np.ravel(weights)):
            if score >= self.min_score:
                out.append(Detection('person', float(score),
                                     tuple(float(v) / self.upscale for v in (x, y, x + w, y + h))))
        return out


class YoloDnn:
    name = 'yolo'

    def __init__(self, cfg, weights, names, size=416, conf=0.35, nms=0.4):
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.names = [n.strip() for n in open(names)]
        self.out_names = self.net.getUnconnectedOutLayersNames()
        self.size, self.conf, self.nms = size, conf, nms

    @staticmethod
    def available():
        d = MODEL_DIR
        return all(os.path.exists(os.path.join(d, f)) for f in ('yolov4-tiny.cfg', 'yolov4-tiny.weights', 'coco.names'))

    @classmethod
    def load(cls, **kw):
        d = MODEL_DIR
        return cls(os.path.join(d, 'yolov4-tiny.cfg'), os.path.join(d, 'yolov4-tiny.weights'),
                   os.path.join(d, 'coco.names'), **kw)

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        boxes, scores, ids = [], [], []
        for out in self.net.forward(self.out_names):
            for row in out:
                cls_scores = row[5:]
                k = int(np.argmax(cls_scores))
                conf = float(cls_scores[k] * row[4])
                if conf < self.conf or self.names[k] not in DYNAMIC_CLASSES | {'traffic light', 'chair', 'bench'}:
                    continue
                cx, cy, bw, bh = row[0] * w, row[1] * h, row[2] * w, row[3] * h
                boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                scores.append(conf)
                ids.append(k)
        keep = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.nms)
        out = []
        for i in np.array(keep).ravel() if len(boxes) else []:
            x, y, bw, bh = boxes[i]
            out.append(Detection(self.names[ids[i]], scores[i], (x, y, x + bw, y + bh)))
        return out


class OracleDetector:
    """Projects ground-truth boxes into the image; skips objects hidden behind something nearer."""
    name = 'oracle'

    def __init__(self, intr, T_base_cam):
        self.intr, self.T_base_cam = intr, T_base_cam

    def detect(self, T_w_base, objects, depth=None):
        from .geometry import inv_T
        T_cw = inv_T(T_w_base @ self.T_base_cam)
        out = []
        for ob in objects:
            c, yaw, size = np.asarray(ob['pos']), ob['yaw'], np.asarray(ob['size'])
            cs, sn = np.cos(yaw), np.sin(yaw)
            corners = np.array([[sx * size[0] / 2, sy * size[1] / 2, sz * size[2] / 2]
                                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            world = c + corners @ np.array([[cs, sn, 0], [-sn, cs, 0], [0, 0, 1]])
            cam = world @ T_cw[:3, :3].T + T_cw[:3, 3]
            if (cam[:, 2] < 0.3).any():
                continue
            uv, z = self.intr.project(cam)
            u0, v0 = uv.min(0)
            u1, v1 = uv.max(0)
            u0, v0 = max(u0, 0.0), max(v0, 0.0)
            u1, v1 = min(u1, self.intr.width - 1.0), min(v1, self.intr.height - 1.0)
            if u1 - u0 < 12 or v1 - v0 < 12:
                continue
            if depth is not None:                      # occlusion: the pixels in the box must be near
                patch = depth[int(v0):int(v1) + 1, int(u0):int(u1) + 1]
                patch = patch[patch > 0]
                dist = np.linalg.norm(cam.mean(0))
                if len(patch) == 0 or np.percentile(patch, 20) > dist + 3.0 or np.median(patch) < dist - 4.0:
                    continue
            out.append(Detection(ob['cls'], 1.0, (u0, v0, u1, v1)))
        return out


def make_detector(backend, intr=None, T_base_cam=None, log=None):
    """backend: 'auto' (YOLO if its files exist, otherwise HOG), 'yolo', 'hog' or 'oracle'."""
    if backend == 'auto':
        backend = 'yolo' if YoloDnn.available() else 'hog'
    if backend == 'yolo':
        return YoloDnn.load()
    if backend == 'hog':
        return HogDetector()
    if backend == 'oracle':
        return OracleDetector(intr, T_base_cam)
    raise ValueError(backend)
