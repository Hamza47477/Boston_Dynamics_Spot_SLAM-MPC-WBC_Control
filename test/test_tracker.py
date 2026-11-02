import numpy as np

from spot_controller.perception.tracker import Tracker


def run(tracker, targets, steps=60, dt=0.2, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    for k in range(steps):
        t = k * dt
        dets = [dict(cls=c, pos=np.array([x0 + vx * t, y0 + vy * t, 0.9]) + rng.normal(0, noise, 3),
                     range=5.0, score=0.9) for c, (x0, y0, vx, vy) in targets]
        tracker.update(t, dets)
    return tracker


def test_velocity_is_estimated_and_flagged_moving():
    tr = run(Tracker(), [('car', (0, 0, 1.4, 0.0))])
    car = tr.tracks[0]
    assert car.confirmed and car.moving
    assert abs(car.x[2] - 1.4) < 0.25 and abs(car.x[3]) < 0.25


def test_static_object_stays_static_and_person_is_dynamic_class():
    tr = run(Tracker(), [('car', (3, 2, 0.0, 0.0)), ('person', (-2, 1, 0.0, 0.0))])
    car = [t for t in tr.tracks if t.cls == 'car'][0]
    person = [t for t in tr.tracks if t.cls == 'person'][0]
    assert car.confirmed and not car.moving
    assert person in tr.dynamic_tracks() and car not in tr.dynamic_tracks()


def test_crossing_targets_keep_their_identity():
    tr = Tracker()
    run(tr, [('person', (-3, 0, 0.8, 0.0)), ('person', (3, 0.6, -0.8, 0.0))], steps=40)
    ids = sorted((t.id, np.sign(t.x[2])) for t in tr.tracks)
    assert len(tr.tracks) == 2 and ids[0][1] != ids[1][1]


def test_lost_track_is_removed_when_moving_but_static_ones_are_remembered():
    tr = run(Tracker(timeout_static=60.0), [('car', (0, 0, 1.4, 0.0)), ('car', (5, 5, 0.0, 0.0))], steps=40)
    for k in range(40, 60):                           # nothing is detected any more
        tr.update(k * 0.2, [])
    assert len(tr.tracks) == 1 and not tr.tracks[0].moving
