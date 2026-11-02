"""Image <-> NumPy conversion without cv_bridge, and stamp-matched stereo pairs."""
import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

SENSOR_QOS = qos_profile_sensor_data

_DTYPES = {'bgr8': (np.uint8, 3), 'mono8': (np.uint8, 1), '32FC1': (np.float32, 1),
           'rgb8': (np.uint8, 3), '8UC1': (np.uint8, 1)}


def image_to_numpy(msg):
    dtype, channels = _DTYPES[msg.encoding]
    arr = np.frombuffer(msg.data, dtype)
    arr = arr.reshape(msg.height, msg.width, channels) if channels > 1 else \
        arr.reshape(msg.height, msg.width)
    return arr


def numpy_to_image(arr, stamp, frame_id, encoding):
    arr = np.ascontiguousarray(arr)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.step = arr.strides[0]
    msg.data = arr.tobytes()
    return msg


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class PairSync:
    """Match left and right images by identical stamp (they are published together)."""

    def __init__(self, size=6):
        self.size = size
        self.left, self.right = {}, {}

    def _trim(self, store):
        while len(store) > self.size:
            store.pop(min(store))

    def add_left(self, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.left[key] = msg
        self._trim(self.left)
        return self._pop(key)

    def add_right(self, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.right[key] = msg
        self._trim(self.right)
        return self._pop(key)

    def _pop(self, key):
        if key in self.left and key in self.right:
            pair = (self.left.pop(key), self.right.pop(key))
            for store in (self.left, self.right):          # drop anything older
                for k in [k for k in store if k < key]:
                    store.pop(k)
            return pair
        return None


def make_cloud(points, frame_id, stamp, colors=None):
    """sensor_msgs/PointCloud2 from (N, 3) float points and optional (N, 3) uint8 RGB colours."""
    from sensor_msgs.msg import PointCloud2, PointField
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = 1, len(pts)
    msg.is_dense = True
    msg.is_bigendian = False
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
              for i, n in enumerate('xyz')]
    if colors is None:
        msg.fields = fields
        msg.point_step = 12
        msg.data = pts.tobytes()
    else:
        rgb = (colors[:, 0].astype(np.uint32) << 16 | colors[:, 1].astype(np.uint32) << 8
               | colors[:, 2].astype(np.uint32))
        arr = np.zeros(len(pts), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32),
                                         ('rgb', np.uint32)])
        arr['x'], arr['y'], arr['z'], arr['rgb'] = pts[:, 0], pts[:, 1], pts[:, 2], rgb
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))
        msg.fields = fields
        msg.point_step = 16
        msg.data = arr.tobytes()
    msg.row_step = msg.point_step * msg.width
    return msg


def pose_to_msg(T, pose_msg):
    """Fill a geometry_msgs/Pose from a 4x4 matrix."""
    from .geometry import R_to_quat
    pose_msg.position.x, pose_msg.position.y, pose_msg.position.z = (float(v) for v in T[:3, 3])
    q = R_to_quat(T[:3, :3])
    pose_msg.orientation.x, pose_msg.orientation.y = float(q[0]), float(q[1])
    pose_msg.orientation.z, pose_msg.orientation.w = float(q[2]), float(q[3])
    return pose_msg


def msg_to_T(pose_msg):
    from .geometry import make_T, quat_to_R
    q = pose_msg.orientation
    p = pose_msg.position
    return make_T(quat_to_R(q.x, q.y, q.z, q.w), [p.x, p.y, p.z])
