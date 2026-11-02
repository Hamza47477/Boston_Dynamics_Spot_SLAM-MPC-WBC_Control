"""Stereo camera pair and the hidden depth sensor, published with simulation-time stamps.

Webots devices (see worlds/spot_slam.wbt): `stereo_left`, `stereo_right` (512x288, 80 deg, 12 cm
baseline, rectified: parallel optical axes) and `stereo_depth_gt` (a range finder at the left
camera, used ONLY to score the stereo depth, never by the perception stack).

Topics: /stereo/left/image_raw (bgr8), /stereo/right/image_raw (mono8),
        /stereo/left/camera_info, /stereo/depth_gt/image_raw (32FC1, metres).
"""
import math

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, Image

BASELINE = 0.12                      # m between the two cameras
LEFT_IN_BODY = (0.556, 0.06, 0.19)   # left camera position in the body frame (FLU), m
CAMERA_PERIOD_MS = 64                # 15.6 Hz (multiple of the 8 ms step)
DEPTH_PERIOD_MS = 192


class StereoSensors:

    def __init__(self, robot, node, timestep):
        self.node = node
        self.timestep = timestep
        self.left = robot.getDevice('stereo_left')
        self.right = robot.getDevice('stereo_right')
        self.depth = robot.getDevice('stereo_depth_gt')
        self.active = self.left is not None and self.right is not None
        if not self.active:
            return
        self.cam_every = max(1, CAMERA_PERIOD_MS // timestep)
        self.depth_every = max(1, DEPTH_PERIOD_MS // timestep)
        self.left.enable(self.cam_every * timestep)
        self.right.enable(self.cam_every * timestep)
        if self.depth is not None:
            self.depth.enable(self.depth_every * timestep)
        self.w, self.h = self.left.getWidth(), self.left.getHeight()
        fx = (self.w / 2.0) / math.tan(self.left.getFov() / 2.0)
        self.info = CameraInfo()
        self.info.width, self.info.height = self.w, self.h
        self.info.distortion_model = 'plumb_bob'
        self.info.d = [0.0] * 5
        self.info.k = [fx, 0.0, self.w / 2.0, 0.0, fx, self.h / 2.0, 0.0, 0.0, 1.0]
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.p = [fx, 0.0, self.w / 2.0, 0.0, 0.0, fx, self.h / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info.header.frame_id = 'stereo_left'
        self.pub_left = node.create_publisher(Image, '/stereo/left/image_raw', 2)
        self.pub_right = node.create_publisher(Image, '/stereo/right/image_raw', 2)
        self.pub_info = node.create_publisher(CameraInfo, '/stereo/left/camera_info', 2)
        self.pub_depth = node.create_publisher(Image, '/stereo/depth_gt/image_raw', 2)

    @staticmethod
    def _image(stamp, frame, encoding, array):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.height, msg.width = array.shape[:2]
        msg.encoding = encoding
        msg.step = array.strides[0]
        msg.data = array.tobytes()
        return msg

    def publish(self, step_count, stamp):
        if not self.active:
            return
        if step_count % self.cam_every == 0:
            left = self.left.getImage()
            right = self.right.getImage()
            if left and right:
                bgra = np.frombuffer(left, np.uint8).reshape(self.h, self.w, 4)
                gray = cv2.cvtColor(np.frombuffer(right, np.uint8).reshape(self.h, self.w, 4),
                                    cv2.COLOR_BGRA2GRAY)
                self.pub_left.publish(self._image(stamp, 'stereo_left', 'bgr8',
                                                  np.ascontiguousarray(bgra[:, :, :3])))
                self.pub_right.publish(self._image(stamp, 'stereo_right', 'mono8', gray))
                self.info.header.stamp = stamp
                self.pub_info.publish(self.info)
        if self.depth is not None and step_count % self.depth_every == 0:
            rng = np.array(self.depth.getRangeImage(), dtype=np.float32).reshape(self.h, self.w)
            rng[~np.isfinite(rng)] = 0.0
            self.pub_depth.publish(self._image(stamp, 'stereo_left', '32FC1', rng))
