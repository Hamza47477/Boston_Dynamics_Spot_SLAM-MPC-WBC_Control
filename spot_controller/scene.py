"""Moving actors and ground-truth object poses for the mapping world (needs a supervisor robot).

Cars in the world are kinematic models (no physics): CAR_MOVING is driven here along a polyline.
Pedestrians walk by themselves (Webots `pedestrian` controller). Everything listed in OBJECTS is
published as ground truth, used only to score the perception stack, never by it.
"""
import math

import numpy as np

# name -> (class, box size length/width/height in m, forward offset of the box centre from the
# node origin in m, box centre height in m).  Webots vehicles have their origin at the rear axle.
OBJECTS = {
    'CAR_PARKED_1': ('car', (4.6, 1.9, 1.45), 1.40, 0.72),
    'CAR_PARKED_2': ('car', (3.5, 1.5, 1.60), 1.00, 0.80),
    'CAR_MOVING': ('car', (4.6, 1.9, 1.45), 1.40, 0.72),
    'PED_1': ('person', (0.5, 0.5, 1.75), 0.0, 0.875),
    'PED_2': ('person', (0.5, 0.5, 1.75), 0.0, 0.875),
    'PED_3': ('person', (0.5, 0.5, 1.75), 0.0, 0.875),
}

# Driven by this module: path (closed polyline, metres) and speed (m/s).
ACTORS = {
    'CAR_MOVING': dict(path=[(8.2, -8.3), (8.2, 0.0), (6.0, 0.0), (6.0, -8.3)], speed=1.4,
                       turn_rate=0.9),
}


def _yaw_of(rotation):
    """Yaw of a Webots axis-angle rotation that is (nearly) about z."""
    ax, ay, az, angle = rotation
    return angle * (1.0 if az >= 0 else -1.0)


class Scene:

    def __init__(self, robot, log=None):
        self.robot = robot
        self.nodes = {}
        self.fields = {}
        for name in OBJECTS:
            node = robot.getFromDef(name)
            if node is not None:
                self.nodes[name] = node
        for name in ACTORS:
            node = self.nodes.get(name)
            if node is not None:
                self.fields[name] = (node.getField('translation'), node.getField('rotation'))
        self.state = {}
        for name, cfg in ACTORS.items():
            if name in self.fields:
                path = np.array(cfg['path'], dtype=float)
                seg = np.diff(np.vstack([path, path[:1]]), axis=0)
                length = np.hypot(seg[:, 0], seg[:, 1])
                self.state[name] = dict(path=path, seg=seg, length=length, total=length.sum(),
                                        yaw=None)
        self.active = bool(self.nodes)
        if log:
            log.info('scene: {} objects, {} driven actors'.format(len(self.nodes), len(self.state)))

    def _pose_on_path(self, st, distance):
        d = distance % st['total']
        for i, length in enumerate(st['length']):
            if d <= length:
                p = st['path'][i] + st['seg'][i] * (d / length)
                heading = math.atan2(st['seg'][i][1], st['seg'][i][0])
                return p, heading
            d -= length
        return st['path'][0], 0.0

    def step(self, t, dt):
        """Move the driven actors to their position at simulation time t."""
        for name, st in self.state.items():
            cfg = ACTORS[name]
            p, heading = self._pose_on_path(st, cfg['speed'] * t)
            if st['yaw'] is None:
                st['yaw'] = heading
            err = math.atan2(math.sin(heading - st['yaw']), math.cos(heading - st['yaw']))
            st['yaw'] += max(-cfg['turn_rate'] * dt, min(cfg['turn_rate'] * dt, err))
            trans, rot = self.fields[name]
            trans.setSFVec3f([float(p[0]), float(p[1]), 0.4])
            rot.setSFRotation([0.0, 0.0, 1.0, float(st['yaw'])])

    def ground_truth(self):
        """List of dict(name, cls, position (3,), yaw, size (3,)) for every known object."""
        out = []
        for name, node in self.nodes.items():
            cls, size, fwd, zc = OBJECTS[name]
            pos = np.array(node.getPosition(), dtype=float)
            yaw = _yaw_of(node.getField('rotation').getSFRotation())
            centre = np.array([pos[0] + fwd * math.cos(yaw), pos[1] + fwd * math.sin(yaw), zc])
            out.append(dict(name=name, cls=cls, position=centre, yaw=yaw, size=np.array(size)))
        return out
