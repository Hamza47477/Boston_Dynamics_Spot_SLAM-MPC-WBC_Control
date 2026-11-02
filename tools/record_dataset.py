"""Record stereo frames + depth + odometry + ground truth to a folder for offline replay.

    python3 tools/record_dataset.py --out /tmp/run1 --max-frames 1500

One .npz per left frame: left (PNG bytes), depth (float16), odom/gt 4x4 poses, objects (json).
Needs the stereo_depth node running.
"""
import argparse
import json
import os

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection3DArray

from spot_controller.perception.geometry import make_T, PoseBuffer, quat_to_R
from spot_controller.perception.ros_utils import image_to_numpy, PairSync, SENSOR_QOS, stamp_to_sec


def pose_T(p):
    q = p.orientation
    return make_T(quat_to_R(q.x, q.y, q.z, q.w), [p.position.x, p.position.y, p.position.z])


class Recorder(Node):

    def __init__(self, out, max_frames):
        super().__init__('recorder')
        self.out, self.max_frames, self.n = out, max_frames, 0
        os.makedirs(out, exist_ok=True)
        self.odom, self.gt = PoseBuffer(), PoseBuffer()
        self.objects = []
        self.sync = PairSync()
        self.create_subscription(CameraInfo, '/stereo/left/camera_info', self.info_cb, 5)
        self.create_subscription(Odometry, '/odom_estimated', lambda m: self.odom.add(
            stamp_to_sec(m.header.stamp), pose_T(m.pose.pose)), 50)
        self.create_subscription(PoseStamped, '/ground_truth/pose', lambda m: self.gt.add(
            stamp_to_sec(m.header.stamp), pose_T(m.pose)), 50)
        self.create_subscription(Detection3DArray, '/ground_truth/objects', self.obj_cb, 5)
        self.create_subscription(Image, '/stereo/left/image_raw', lambda m: self.add('l', m), SENSOR_QOS)
        self.create_subscription(Image, '/stereo/depth/image_raw', lambda m: self.add('d', m), SENSOR_QOS)
        self.info_saved = False

    def info_cb(self, msg):
        if not self.info_saved:
            json.dump(dict(k=list(msg.k), width=msg.width, height=msg.height),
                      open(os.path.join(self.out, 'camera.json'), 'w'))
            self.info_saved = True

    def obj_cb(self, msg):
        self.objects = [dict(id=d.id, cls=d.results[0].hypothesis.class_id,
                             pos=[d.bbox.center.position.x, d.bbox.center.position.y,
                                  d.bbox.center.position.z],
                             yaw=2 * float(np.arctan2(d.bbox.center.orientation.z,
                                                      d.bbox.center.orientation.w)),
                             size=[d.bbox.size.x, d.bbox.size.y, d.bbox.size.z])
                        for d in msg.detections]

    def add(self, kind, msg):
        pair = self.sync.add_left(msg) if kind == 'l' else self.sync.add_right(msg)
        if not pair or self.n >= self.max_frames:
            return
        left, depth = pair
        t = stamp_to_sec(left.header.stamp)
        odom, gt = self.odom.get(t), self.gt.get(t)
        if odom is None or gt is None:
            return
        ok, png = cv2.imencode('.png', image_to_numpy(left))
        np.savez(os.path.join(self.out, 'f%05d.npz' % self.n), t=t, left=png, odom=odom, gt=gt,
                 depth=image_to_numpy(depth).astype(np.float16), objects=json.dumps(self.objects))
        self.n += 1
        if self.n % 50 == 0:
            self.get_logger().info('recorded {} frames (t = {:.1f} s)'.format(self.n, t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-frames', type=int, default=1500)
    args = ap.parse_args()
    rclpy.init()
    node = Recorder(args.out, args.max_frames)
    try:
        while rclpy.ok() and node.n < args.max_frames:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    print('saved', node.n, 'frames to', args.out)


if __name__ == '__main__':
    main()
