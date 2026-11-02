"""3D occupancy map: a dense log-odds voxel grid, updated by ray casting from the depth camera.

For every depth point the voxel it hits gets "occupied" evidence (+l_occ) and every voxel the ray passes
through gets "free" evidence (-l_free). Evidence is clamped, so a voxel that was occupied by something
that later moved away (a walking person) is cleared again by the rays that now pass through it.

occupancy_2d() flattens a height band of the grid into the 2D map used for navigation.
"""
import numpy as np


class VoxelMap:

    def __init__(self, resolution=0.10, size_xy=40.0, z_range=(-0.6, 3.6), l_occ=0.85, l_free=0.4,
                 l_min=-2.0, l_max=2.4):
        self.res = resolution
        self.x0 = self.y0 = -size_xy / 2.0
        self.z0 = z_range[0]
        self.nx = self.ny = int(round(size_xy / resolution))
        self.nz = int(round((z_range[1] - z_range[0]) / resolution))
        self.L = np.zeros((self.nx, self.ny, self.nz), np.float32)
        self.l_occ, self.l_free, self.l_min, self.l_max = l_occ, l_free, l_min, l_max
        self.frames = 0

    def _index(self, pts):
        i = np.floor((pts[:, 0] - self.x0) / self.res).astype(np.int64)
        j = np.floor((pts[:, 1] - self.y0) / self.res).astype(np.int64)
        k = np.floor((pts[:, 2] - self.z0) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny) & (k >= 0) & (k < self.nz)
        return i, j, k, ok

    def _flat(self, pts):
        i, j, k, ok = self._index(pts)
        return (i[ok] * self.ny + j[ok]) * self.nz + k[ok]

    def integrate(self, origin, points, max_range=8.0, min_range=0.3):
        """Add one depth frame: sensor `origin` (3,), hit `points` (N, 3), both in map coordinates."""
        origin = np.asarray(origin, float)
        d = points - origin
        dist = np.linalg.norm(d, axis=1)
        keep = (dist > min_range) & (dist < max_range)
        points, d, dist = points[keep], d[keep], dist[keep]
        if len(points) == 0:
            return
        step = 0.7 * self.res
        n_steps = int(np.ceil(max_range / step)) + 1
        s = (np.arange(n_steps) + 0.5) * step                                  # distances along the rays
        mask = s[None, :] < (dist[:, None] - self.res)                         # stop one voxel short
        rows, cols = np.nonzero(mask)
        free_pts = origin + d[rows] * (s[cols] / dist[rows])[:, None]
        free_idx = np.unique(self._flat(free_pts))
        hit_idx = np.unique(self._flat(points))
        flat = self.L.reshape(-1)
        flat[free_idx] = np.clip(flat[free_idx] - self.l_free, self.l_min, self.l_max)
        flat[hit_idx] = np.clip(flat[hit_idx] + self.l_occ, self.l_min, self.l_max)
        self.frames += 1

    def clear(self):
        self.L[:] = 0.0
        self.frames = 0

    def occupied_points(self, threshold=1.0, z_min=None):
        """Centres (N, 3) of occupied voxels, optionally above z_min."""
        idx = np.argwhere(self.L > threshold)
        pts = np.stack([self.x0 + (idx[:, 0] + 0.5) * self.res, self.y0 + (idx[:, 1] + 0.5) * self.res,
                        self.z0 + (idx[:, 2] + 0.5) * self.res], axis=1)
        if z_min is not None:
            pts = pts[pts[:, 2] > z_min]
        return pts

    def occupancy_2d(self, floor_z, z_lo=0.15, z_hi=1.4, occ_threshold=1.0, free_threshold=-0.4):
        """Flatten a height band into a 2D grid for navigation. Returns int8 (ny, nx):
        100 occupied, 0 free (seen, nothing in the band), -1 unknown. Row = y, column = x."""
        k0 = int(np.floor((floor_z + z_lo - self.z0) / self.res))
        k1 = int(np.ceil((floor_z + z_hi - self.z0) / self.res))
        k0, k1 = max(k0, 0), min(k1, self.nz)
        band = self.L[:, :, k0:k1]
        occupied = (band > occ_threshold).any(axis=2)
        freed = (band < free_threshold).any(axis=2) | (self.L[:, :, max(k0 - 1, 0)] < free_threshold)
        grid = np.full((self.nx, self.ny), -1, np.int8)
        grid[freed] = 0
        grid[occupied] = 100
        return grid.T.copy()                                                    # (y, x)
