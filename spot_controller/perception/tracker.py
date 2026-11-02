"""Multi-object 3D tracking: constant-velocity Kalman filters, nearest-neighbour association.

Detections come as world positions (x, y, z) with a class. Each track keeps [x y vx vy] on the ground
plane. A track is tentative until it has been seen a few times, then confirmed. A confirmed track whose
speed stays above a threshold is `moving`; people always count as dynamic (they are likely to move). Static
tracks (parked cars, standing people) are kept much longer than moving ones: that is the semantic map.
"""
import numpy as np


class Track:
    _next_id = 1

    def __init__(self, t, cls, pos, score):
        self.id = Track._next_id
        Track._next_id += 1
        self.cls, self.score = cls, score
        self.x = np.array([pos[0], pos[1], 0.0, 0.0])
        self.P = np.diag([0.2, 0.2, 1.0, 1.0])
        self.z = float(pos[2])
        self.hits, self.misses = 1, 0
        self.born = self.last_seen = t
        self.confirmed = False
        self.moving = False
        self.fast_count = 0
        self.box = None

    @property
    def speed(self):
        return float(np.hypot(self.x[2], self.x[3]))

    def predict(self, t, accel_sigma):
        dt = max(t - self.last_update_time, 0.0) if hasattr(self, 'last_update_time') else 0.0
        self.last_update_time = t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = accel_sigma ** 2
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + q * G @ G.T

    def update(self, t, pos, sigma):
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        S = H @ self.P @ H.T + np.eye(2) * sigma ** 2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(pos[:2]) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.z = 0.7 * self.z + 0.3 * float(pos[2])
        self.hits += 1
        self.misses = 0
        self.last_seen = t


class Tracker:

    def __init__(self, gate=1.6, confirm_hits=3, accel_sigma=0.8, move_speed=0.35, move_frames=4,
                 timeout_moving=2.0, timeout_static=120.0, timeout_tentative=1.0):
        self.gate, self.confirm_hits, self.accel_sigma = gate, confirm_hits, accel_sigma
        self.move_speed, self.move_frames = move_speed, move_frames
        self.timeouts = dict(moving=timeout_moving, static=timeout_static, tentative=timeout_tentative)
        self.tracks = []

    @staticmethod
    def _sigma(rng):
        return 0.12 + 0.05 * rng                       # stereo error grows with range

    def update(self, t, detections):
        """detections: list of dict(cls, pos (3,), score, box, range). Returns the live tracks."""
        for tr in self.tracks:
            tr.predict(t, self.accel_sigma)
        pairs = []
        for di, det in enumerate(detections):
            for ti, tr in enumerate(self.tracks):
                if tr.cls != det['cls']:
                    continue
                d = float(np.hypot(det['pos'][0] - tr.x[0], det['pos'][1] - tr.x[1]))
                gate = self.gate + 0.5 * (t - tr.last_seen)
                if d < gate:
                    pairs.append((d, di, ti))
        pairs.sort()
        used_d, used_t = set(), set()
        for d, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            tr, det = self.tracks[ti], detections[di]
            tr.update(t, det['pos'], self._sigma(det.get('range', 5.0)))
            tr.box, tr.score = det.get('box'), det.get('score', 1.0)
            if not tr.confirmed and tr.hits >= self.confirm_hits:
                tr.confirmed = True
            if tr.confirmed:
                tr.fast_count = tr.fast_count + 1 if tr.speed > self.move_speed else max(tr.fast_count - 1, 0)
                tr.moving = tr.fast_count >= self.move_frames if not tr.moving else tr.speed > 0.6 * self.move_speed
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.misses += 1
        for di, det in enumerate(detections):
            if di not in used_d:
                tr = Track(t, det['cls'], det['pos'], det.get('score', 1.0))
                tr.box = det.get('box')
                tr.predict(t, self.accel_sigma)
                self.tracks.append(tr)
        alive = []
        for tr in self.tracks:
            kind = 'tentative' if not tr.confirmed else ('moving' if tr.moving else 'static')
            if t - tr.last_seen <= self.timeouts[kind]:
                alive.append(tr)
        self.tracks = alive
        return self.tracks

    def dynamic_tracks(self):
        """Tracks that must be kept out of the static map: moving things and people."""
        return [tr for tr in self.tracks if tr.confirmed and (tr.moving or tr.cls == 'person')]
