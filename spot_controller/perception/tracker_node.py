"""ROS node: 2D detections + depth + SLAM pose -> 3D tracks with velocity, semantic markers, dynamic masks.

In : /detections, /stereo/depth/image_raw, /slam/odom, /stereo/left/camera_info
Out: /tracks (Detection3DArray in `map`), /tracks/markers (MarkerArray), /perception/dynamic_boxes
     (image boxes of moving things and people; SLAM and the mapper ignore those pixels), /pipeline/stats.
For scoring only: /ground_truth/pose and /ground_truth/objects (never used for tracking).
"""
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import (Detection2D, Detection2DArray, Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)

from .detector import CLASS_COLORS
from .geometry import inv_T, Intrinsics, PoseBuffer, T_body_camcv
from .rigid import kabsch
from .ros_utils import image_to_numpy, msg_to_T, PairSync, SENSOR_QOS, stamp_to_sec
from .stats import StageStats
from .tracker import Tracker

SIZES = {'person': (0.55, 0.55, 1.75), 'car': (4.4, 1.9, 1.5), 'truck': (6.0, 2.4, 2.8)}
CENTRE_SHIFT = {'person': 0.25, 'car': 1.0, 'truck': 1.5}        # front surface -> object centre (m)


class TrackerNode(Node):

    def __init__(self):
        super().__init__('tracker')
        self.tracker = Tracker()
        self.intr = None
        self.sync = PairSync()
        self.T_base_cam = T_body_camcv()
        self.slam = PoseBuffer()
        self.gt, self.pairs = PoseBuffer(), []
        self.gt_objects = []
        self.align = None
        self.last_align = 0.0
        self.score = {}
        self.stats = StageStats(self, 'tracker')
        self.pub_tracks = self.create_publisher(Detection3DArray, '/tracks', 5)
        self.pub_markers = self.create_publisher(MarkerArray, '/tracks/markers', 5)
        self.pub_boxes = self.create_publisher(Detection2DArray, '/perception/dynamic_boxes', 5)
        self.create_subscription(CameraInfo, '/stereo/left/camera_info', self.info_cb, 5)
        self.create_subscription(Detection2DArray, '/detections', self.det_cb, 5)
        self.create_subscription(Image, '/stereo/depth/image_raw', self.depth_cb, SENSOR_QOS)
        self.create_subscription(Odometry, '/slam/odom', self.slam_cb, 50)
        self.create_subscription(PoseStamped, '/ground_truth/pose', self.gt_cb, 50)
        self.create_subscription(Detection3DArray, '/ground_truth/objects', self.gt_objects_cb, 5)
        self.pending = {}
        self.depths = {}

    def info_cb(self, msg):
        if self.intr is None:
            self.intr = Intrinsics.from_info(msg)

    def slam_cb(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        T = msg_to_T(msg.pose.pose)
        self.slam.add(t, T)
        g = self.gt.get(t)
        if g is not None and msg.pose.covariance[1] < 0.5:
            self.pairs.append((T[:3, 3].copy(), g[:3, 3].copy()))
            self.pairs = self.pairs[-400:]

    def gt_cb(self, msg):
        self.gt.add(stamp_to_sec(msg.header.stamp), msg_to_T(msg.pose))

    def gt_objects_cb(self, msg):
        self.gt_objects = [(d.id, d.results[0].hypothesis.class_id,
                            np.array([d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z]))
                           for d in msg.detections]

    def depth_cb(self, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.depths[key] = msg
        while len(self.depths) > 8:
            self.depths.pop(min(self.depths))
        if key in self.pending:
            self.process(self.pending.pop(key), msg)

    def det_cb(self, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if key in self.depths:
            self.process(msg, self.depths[key])
        else:
            self.pending[key] = msg
            while len(self.pending) > 4:
                self.pending.pop(min(self.pending))

    # ------------------------------------------------------------------
    def lift(self, det, depth, T_w_cam):
        """World position of a detection from the depth inside its box (None if too little depth)."""
        cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
        w, h = det.bbox.size_x, det.bbox.size_y
        u0, u1 = int(max(cx - 0.2 * w, 0)), int(min(cx + 0.2 * w, depth.shape[1] - 1))
        v0, v1 = int(max(cy - 0.2 * h, 0)), int(min(cy + 0.2 * h, depth.shape[0] - 1))
        patch = depth[v0:v1 + 1, u0:u1 + 1]
        patch = patch[(patch > 0.4) & (patch < 15.0)]
        if len(patch) < 15:
            return None
        z = float(np.median(patch))
        cls = det.results[0].hypothesis.class_id
        z += CENTRE_SHIFT.get(cls, 0.3)                  # centre lies behind the visible surface
        p_cam = self.intr.backproject(np.array([cx]), np.array([cy]), np.array([z]))[0]
        return (T_w_cam @ np.append(p_cam, 1.0))[:3], z

    def process(self, det_msg, depth_msg):
        if self.intr is None:
            return
        t = stamp_to_sec(det_msg.header.stamp)
        T_w_base = self.slam.get(t)
        if T_w_base is None:
            return
        T_w_cam = T_w_base @ self.T_base_cam
        depth = image_to_numpy(depth_msg)
        with self.stats.measure():
            dets, sources = [], []
            for d in det_msg.detections:
                res = self.lift(d, depth, T_w_cam)
                if res is not None:
                    sources.append(d)
                    dets.append(dict(cls=d.results[0].hypothesis.class_id, pos=res[0], range=res[1],
                                     score=d.results[0].hypothesis.score,
                                     box=(d.bbox.center.position.x, d.bbox.center.position.y,
                                          d.bbox.size_x, d.bbox.size_y)))
            tracks = self.tracker.update(t, dets)
        self.publish(det_msg.header.stamp, tracks)
        self.publish_dynamic_boxes(det_msg, dets, sources, tracks, T_w_cam)
        self.evaluate(t, tracks)
        self.stats.publish(tracks=len([x for x in tracks if x.confirmed]),
                           moving=len([x for x in tracks if x.confirmed and x.moving]), **self.score)

    def publish(self, stamp, tracks):
        arr = Detection3DArray()
        arr.header.stamp, arr.header.frame_id = stamp, 'map'
        markers = MarkerArray()
        for tr in tracks:
            if not tr.confirmed:
                continue
            size = SIZES.get(tr.cls, (1.0, 1.0, 1.0))
            heading = np.arctan2(tr.x[3], tr.x[2]) if tr.speed > 0.3 else 0.0
            det = Detection3D()
            det.header = arr.header
            det.id = '{}{}'.format(tr.cls, tr.id)
            det.bbox.center.position.x, det.bbox.center.position.y = float(tr.x[0]), float(tr.x[1])
            det.bbox.center.position.z = float(tr.z)
            det.bbox.center.orientation.z, det.bbox.center.orientation.w = float(np.sin(heading / 2)), float(np.cos(heading / 2))
            det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = size
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = tr.cls, float(tr.score)
            hyp.pose.pose.position.x, hyp.pose.pose.position.y = float(tr.x[2]), float(tr.x[3])   # velocity
            det.results.append(hyp)
            arr.detections.append(det)
            r, g, b = (c / 255.0 for c in CLASS_COLORS.get(tr.cls, (200, 200, 200))[::-1])
            if tr.moving:
                r, g, b = 1.0, 0.2, 0.2
            box = Marker()
            box.header, box.ns, box.id = arr.header, 'tracks', tr.id
            box.type = Marker.CUBE if tr.cls != 'person' else Marker.CYLINDER
            box.pose = det.bbox.center
            box.scale.x, box.scale.y, box.scale.z = size
            box.color.r, box.color.g, box.color.b, box.color.a = r, g, b, 0.55
            box.lifetime.sec = 2
            markers.markers.append(box)
            label = Marker()
            label.header, label.ns, label.id = arr.header, 'labels', tr.id
            label.type = Marker.TEXT_VIEW_FACING
            label.pose.position.x, label.pose.position.y = float(tr.x[0]), float(tr.x[1])
            label.pose.position.z = size[2] + 0.4
            label.scale.z = 0.45
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = '{} #{} {}'.format(tr.cls, tr.id, '{:.1f} m/s'.format(tr.speed) if tr.moving else 'static')
            label.lifetime.sec = 2
            markers.markers.append(label)
            if tr.moving:
                arrow = Marker()
                arrow.header, arrow.ns, arrow.id = arr.header, 'velocity', tr.id
                arrow.type = Marker.ARROW
                arrow.points = [Point(x=float(tr.x[0]), y=float(tr.x[1]), z=size[2] / 2),
                                Point(x=float(tr.x[0] + tr.x[2]), y=float(tr.x[1] + tr.x[3]), z=size[2] / 2)]
                arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.12, 0.25, 0.3
                arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1.0, 1.0, 0.0, 1.0
                arrow.lifetime.sec = 2
                markers.markers.append(arrow)
        self.pub_tracks.publish(arr)
        self.pub_markers.publish(markers)

    def project_box(self, tr, T_w_cam):
        """Image box of a track from its 3D state (for dynamic tracks the detector missed this frame)."""
        size = np.array(SIZES.get(tr.cls, (1.0, 1.0, 1.0)))
        c = np.array([tr.x[0], tr.x[1], tr.z])
        corners = c + np.array([[sx * size[0] / 2, sy * size[1] / 2, sz * size[2] / 2]
                                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        cam = corners @ inv_T(T_w_cam)[:3, :3].T + inv_T(T_w_cam)[:3, 3]
        if (cam[:, 2] < 0.3).any():
            return None
        uv, _ = self.intr.project(cam)
        u0, v0 = np.maximum(uv.min(0), 0)
        u1, v1 = np.minimum(uv.max(0), [self.intr.width - 1, self.intr.height - 1])
        return (u0, v0, u1, v1) if u1 - u0 > 8 and v1 - v0 > 8 else None

    def publish_dynamic_boxes(self, det_msg, dets, sources, tracks, T_w_cam):
        """Image boxes (enlarged) of things that must stay out of the static map: detections that belong to
        a moving track or a person, plus predicted boxes of such tracks the detector missed this frame."""
        dynamic = self.tracker.dynamic_tracks()
        out = Detection2DArray()
        out.header = det_msg.header

        def add(cx, cy, w, h):
            box = Detection2D()
            box.header = det_msg.header
            box.bbox.center.position.x, box.bbox.center.position.y = float(cx), float(cy)
            box.bbox.size_x, box.bbox.size_y = float(w * 1.3), float(h * 1.2)
            out.detections.append(box)

        seen = set()
        for det, src in zip(dets, sources):
            best, best_d = None, 1.2
            for tr in tracks:
                d = np.hypot(det['pos'][0] - tr.x[0], det['pos'][1] - tr.x[1])
                if tr.cls == det['cls'] and d < best_d:
                    best, best_d = tr, d
            if best is not None and best in dynamic:
                seen.add(id(best))
                add(src.bbox.center.position.x, src.bbox.center.position.y, src.bbox.size_x, src.bbox.size_y)
        for tr in dynamic:
            if id(tr) not in seen and tr.misses <= 6:
                box = self.project_box(tr, T_w_cam)
                if box is not None:
                    add((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1])
        self.pub_boxes.publish(out)

    def evaluate(self, t, tracks):
        """Score against ground truth: align the map frame to the world from the two trajectories."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_align > 2.0 and len(self.pairs) > 30:
            self.last_align = now
            P = np.array([p for p, _ in self.pairs])
            Q = np.array([q for _, q in self.pairs])
            if np.ptp(P[:, 0]) + np.ptp(P[:, 1]) > 1.5:
                self.align = kabsch(P, Q)
        if self.align is None:
            return
        R, tt = self.align
        T_w_base = self.slam.get(t)
        if T_w_base is None:
            return
        base_world = R @ T_w_base[:3, 3] + tt
        errs, visible = [], 0
        confirmed = [tr for tr in tracks if tr.confirmed]
        for name, cls, pos in self.gt_objects:
            if np.linalg.norm(pos[:2] - base_world[:2]) > 7.0:
                continue
            visible += 1
            best = None
            for tr in confirmed:
                if tr.cls != cls:
                    continue
                d = np.linalg.norm((R @ np.array([tr.x[0], tr.x[1], 0.0]) + tt)[:2] - pos[:2])
                best = d if best is None else min(best, d)
            if best is not None and best < 2.5:
                errs.append(best)
        self.score = dict(gt_near=visible, matched=len(errs),
                          err_m=round(float(np.mean(errs)), 2) if errs else None)


def main():
    rclpy.init()
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
