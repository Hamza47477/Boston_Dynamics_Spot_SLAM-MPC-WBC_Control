import numpy as np

from spot_controller.perception.voxel_map import VoxelMap


def wall_points(x, y_range=(-1, 1), z_range=(0.2, 1.2), step=0.05):
    ys = np.arange(y_range[0], y_range[1], step)
    zs = np.arange(z_range[0], z_range[1], step)
    Y, Z = np.meshgrid(ys, zs)
    return np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], axis=1)


def test_wall_becomes_occupied_and_space_in_front_free():
    vm = VoxelMap(resolution=0.1, size_xy=20.0)
    origin = np.array([0.0, 0.0, 0.7])
    for _ in range(2):
        vm.integrate(origin, wall_points(3.0))
    grid = vm.occupancy_2d(floor_z=0.0)
    col = lambda x: int((x - vm.x0) / vm.res)
    row = lambda y: int((y - vm.y0) / vm.res)
    assert grid[row(0.0), col(3.05)] == 100                 # the wall
    assert grid[row(0.0), col(1.5)] == 0                    # free space on the way
    assert grid[row(0.0), col(5.0)] == -1                   # behind the wall: never seen
    assert grid[row(6.0), col(1.0)] == -1


def test_moving_object_is_cleared_when_it_leaves():
    vm = VoxelMap(resolution=0.1, size_xy=20.0)
    origin = np.array([0.0, 0.0, 0.7])
    far, near = wall_points(5.0), wall_points(2.0, y_range=(-0.3, 0.3), z_range=(0.5, 0.9))
    for _ in range(3):
        vm.integrate(origin, np.vstack([far, near]))        # a person at 2 m in front of a wall
    row0, col2 = int((0 - vm.y0) / vm.res), int((2.05 - vm.x0) / vm.res)
    assert vm.occupancy_2d(0.0)[row0, col2] == 100
    for _ in range(4):
        vm.integrate(origin, far)                           # the person has gone
    assert vm.occupancy_2d(0.0)[row0, col2] == 0


def test_points_outside_the_grid_are_ignored():
    vm = VoxelMap(resolution=0.1, size_xy=4.0)
    vm.integrate(np.zeros(3), np.array([[7.0, 0.0, 0.5], [1.0, 0.0, 0.5]]))
    assert (vm.L > 0.5).sum() == 1


def test_pseudo_scan_marks_obstacles_and_clears_free_floor():
    from spot_controller.perception.mapper import points_to_scan
    h = 0.48                                              # base height above floor
    floor = np.array([[x, y, -h] for x in (1.0, 2.0, 3.0) for y in np.linspace(-1.5, 1.5, 61)])
    person = np.array([[2.0, 0.0, -h + z] for z in np.linspace(0.2, 1.7, 15)])
    scan = points_to_scan(np.vstack([floor, person]), h, n_beams=116)
    centre = 58
    assert abs(scan[centre] - 2.0) < 0.05                 # the person is a hit at 2 m
    assert np.isinf(scan[centre + 20])                    # floor but nothing above it: free
    assert np.isnan(points_to_scan(np.zeros((0, 3)), h)[0])
