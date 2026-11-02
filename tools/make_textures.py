"""Generate feature-rich procedural textures for worlds/spot_slam.wbt (run once, needs OpenCV).

    python3 tools/make_textures.py            # writes worlds/textures/*.png

Plain coloured walls give a visual SLAM nothing to track, so each wall gets a different pattern
(text posters, bricks, graffiti shapes, signs) and the floor a patchwork of tiles. Different
walls also make different places look different, which loop closure needs.
"""
import os

import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'worlds', 'textures')
WORDS = ['EXIT', 'STOP', 'LAB 4', 'WARNING', 'ZONE B', 'SPOT', 'ROBOT', 'DOCK', 'A17', 'FIRE',
         'NO ENTRY', 'LEVEL 2', 'STORAGE', 'KEEP OUT', 'TEST', '42', 'CAMERA', 'MAP', 'GATE']
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]


def smooth_noise(h, w, rng, octaves=((8, 0.5), (24, 0.3), (64, 0.2))):
    out = np.zeros((h, w, 3), np.float32)
    for cells, weight in octaves:
        small = rng.random((cells, max(2, cells * h // w), 3)).astype(np.float32)
        out += weight * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


def colour(rng, lo=40, hi=230):
    return tuple(int(v) for v in rng.integers(lo, hi, 3))


def finish(img, seed):
    """Fine grain and small marks, like real concrete or paint: gives a tracker corners at close range."""
    rng = np.random.default_rng(1000 + seed)
    h, w = img.shape[:2]
    out = img.astype(np.float32) + cv2.GaussianBlur(rng.normal(0, 26, img.shape).astype(np.float32), (0, 0), 0.8)
    out = out.clip(0, 255).astype(np.uint8)
    for _ in range(900):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.6:
            cv2.circle(out, c, int(rng.integers(2, 7)), col, -1)
        else:
            cv2.line(out, c, (c[0] + int(rng.integers(-18, 18)), c[1] + int(rng.integers(-18, 18))), col, 1)
    for _ in range(120):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.rectangle(out, c, (c[0] + int(rng.integers(6, 28)), c[1] + int(rng.integers(6, 28))),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), 1)
    return out


def poster_wall(seed, size=(1024, 1024), text_scale=1.0):
    rng = np.random.default_rng(seed)
    h, w = size
    img = (60 + 150 * smooth_noise(h, w, rng)).clip(0, 255).astype(np.uint8)
    for _ in range(45):                                    # coloured panels
        x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        bw, bh = int(rng.integers(50, 260)), int(rng.integers(40, 200))
        cv2.rectangle(img, (x, y), (min(w - 1, x + bw), min(h - 1, y + bh)), colour(rng),
                      -1 if rng.random() < 0.7 else 3)
    for _ in range(22):                                    # shapes
        centre = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(img, centre, int(rng.integers(12, 70)), colour(rng), -1 if rng.random() < 0.5 else 4)
    for _ in range(30):
        cv2.line(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                 (int(rng.integers(0, w)), int(rng.integers(0, h))), colour(rng), int(rng.integers(1, 5)))
    for _ in range(26):                                    # lettering
        word = WORDS[int(rng.integers(len(WORDS)))]
        scale = float(rng.uniform(0.8, 3.0)) * text_scale
        org = (int(rng.integers(0, w - 120)), int(rng.integers(40, h - 10)))
        cv2.putText(img, word, org, FONTS[int(rng.integers(len(FONTS)))], scale,
                    colour(rng, 0, 255), int(rng.integers(2, 5)), cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0.6)


def bricks(seed, size=(1024, 1024)):
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.full((h, w, 3), 200, np.uint8)
    bh, bw = 48, 128
    for row in range(h // bh + 1):
        offset = (bw // 2) * (row % 2)
        for col in range(-1, w // bw + 2):
            x0, y0 = col * bw + offset, row * bh
            base = np.array([60, 70, 150]) + rng.integers(-35, 35, 3)
            cv2.rectangle(img, (x0 + 3, y0 + 3), (x0 + bw - 3, y0 + bh - 3),
                          tuple(int(v) for v in base.clip(0, 255)), -1)
    img = img.astype(np.float32) + rng.normal(0, 12, img.shape)
    img = cv2.GaussianBlur(img, (3, 3), 0.8).clip(0, 255).astype(np.uint8)
    for _ in range(10):                                    # painted markings so rows differ
        word = WORDS[int(rng.integers(len(WORDS)))]
        cv2.putText(img, word, (int(rng.integers(0, w - 200)), int(rng.integers(80, h - 20))),
                    cv2.FONT_HERSHEY_DUPLEX, float(rng.uniform(1.5, 3.5)), (235, 235, 235), 6,
                    cv2.LINE_AA)
    return img


def graffiti(seed, size=(1024, 1024)):
    rng = np.random.default_rng(seed)
    h, w = size
    img = (90 + 120 * smooth_noise(h, w, rng, ((4, 0.7), (16, 0.3)))).clip(0, 255).astype(np.uint8)
    for _ in range(260):                                   # many small polygons, not a few huge ones
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        pts = np.array([[cx + rng.integers(-60, 60), cy + rng.integers(-60, 60)] for _ in range(int(rng.integers(3, 6)))])
        cv2.fillConvexPoly(img, cv2.convexHull(pts.astype(np.int32)), colour(rng))
    for _ in range(18):
        cv2.ellipse(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                    (int(rng.integers(20, 120)), int(rng.integers(20, 90))),
                    float(rng.integers(0, 180)), 0, 360, colour(rng), int(rng.integers(3, 9)))
    return cv2.GaussianBlur(img, (3, 3), 0.7)


def floor_tiles(seed, size=(1024, 1024)):
    rng = np.random.default_rng(seed)
    h, w = size
    n = 8
    img = np.zeros((h, w, 3), np.uint8)
    for i in range(n):
        for j in range(n):
            base = np.array([95, 100, 105]) + rng.integers(-45, 45, 3)
            img[i * h // n:(i + 1) * h // n, j * w // n:(j + 1) * w // n] = base.clip(0, 255)
    img = img.astype(np.float32) + rng.normal(0, 10, img.shape) + 25 * smooth_noise(h, w, rng)
    img = img.clip(0, 255).astype(np.uint8)
    for k in range(n + 1):                                 # grout lines
        cv2.line(img, (0, k * h // n), (w, k * h // n), (35, 35, 35), 3)
        cv2.line(img, (k * w // n, 0), (k * w // n, h), (35, 35, 35), 3)
    for _ in range(40):                                    # stains, stickers, marks
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                   int(rng.integers(5, 30)), colour(rng, 20, 220), -1)
    for _ in range(10):
        word = WORDS[int(rng.integers(len(WORDS)))]
        cv2.putText(img, word, (int(rng.integers(0, w - 180)), int(rng.integers(40, h - 10))),
                    cv2.FONT_HERSHEY_DUPLEX, float(rng.uniform(0.8, 1.6)), (230, 230, 60), 3,
                    cv2.LINE_AA)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    textures = {
        'wall_north.png': poster_wall(1),
        'wall_east.png': bricks(2),
        'wall_south.png': graffiti(3),
        'wall_west.png': poster_wall(4, text_scale=1.3),
        'partition_a.png': poster_wall(5),
        'partition_b.png': bricks(6),
        'floor.png': floor_tiles(7),
    }
    for k, (name, img) in enumerate(textures.items()):
        img = finish(img, k)
        cv2.imwrite(os.path.join(OUT, name), img)
        print('wrote', name, img.shape)


if __name__ == '__main__':
    main()
