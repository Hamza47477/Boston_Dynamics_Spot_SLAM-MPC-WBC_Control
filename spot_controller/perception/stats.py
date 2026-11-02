"""Per-stage timing and rate, published as JSON on /pipeline/stats for the viewer."""
import json
import time

from std_msgs.msg import String


class StageStats:
    """Call `with stats.measure():` around the work and `stats.publish(**extra)` now and then."""

    def __init__(self, node, name):
        self.node = node
        self.name = name
        self.pub = node.create_publisher(String, '/pipeline/stats', 10)
        self.ms = 0.0
        self.fps = 0.0
        self.count = 0
        self._last = None
        self._last_pub = 0.0
        self._t0 = 0.0

    def measure(self):
        return self

    def __enter__(self):
        self._t0 = time.perf_counter()

    def __exit__(self, *exc):
        now = time.perf_counter()
        dt = now - self._t0
        self.ms = dt * 1e3 if self.count == 0 else 0.9 * self.ms + 0.1 * dt * 1e3
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = inst if self.fps == 0.0 else 0.9 * self.fps + 0.1 * inst
        self._last = now
        self.count += 1

    def publish(self, **extra):
        now = time.time()
        if now - self._last_pub < 0.5:
            return
        self._last_pub = now
        msg = String()
        msg.data = json.dumps(dict(stage=self.name, ms=round(self.ms, 1), fps=round(self.fps, 1),
                                   n=self.count, **extra))
        self.pub.publish(msg)
