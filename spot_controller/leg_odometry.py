"""Leg odometry: foot position / velocity in the body frame, and body velocity from a stance foot."""
import numpy as np

from .constants import LEG_ORDER, LEG_ROS_PREFIX


def joints_to_leg_dict(names, values):
    """Map {ros joint name: value} to {leg: (hip_x, hip_y, knee)}; None if a joint is missing."""
    table = dict(zip(names, values))
    try:
        return {leg: tuple(table['{}_{}'.format(LEG_ROS_PREFIX[leg], s)]
                           for s in ('hip_x', 'hip_y', 'knee'))
                for leg in LEG_ORDER}
    except KeyError:
        return None


def foot_states(kin, q, qdot, h=1e-3):
    """Foot positions r_i and velocities rdot_i (relative to the body, in the body frame).

    q, qdot: {leg: (hip_x, hip_y, knee)} in Webots convention. rdot = J(q) qdot is computed
    with a central difference of the forward kinematics along the joint-velocity direction
    (exact for a linear map, accurate to O(h^2) here).
    """
    r = kin.compute_body_fk(q)
    q_plus = {leg: tuple(q[leg][k] + h * qdot[leg][k] for k in range(3)) for leg in LEG_ORDER}
    q_minus = {leg: tuple(q[leg][k] - h * qdot[leg][k] for k in range(3)) for leg in LEG_ORDER}
    f_plus = kin.compute_body_fk(q_plus)
    f_minus = kin.compute_body_fk(q_minus)
    positions = [np.array(r[leg]) for leg in LEG_ORDER]
    velocities = [(np.array(f_plus[leg]) - np.array(f_minus[leg])) / (2.0 * h)
                  for leg in LEG_ORDER]
    return positions, velocities


def body_velocity_from_foot(r, rdot, omega):
    """Body velocity (body frame) assuming this foot is stationary in the world.

    foot_world = p + R r is constant  =>  v + R (rdot + omega x r) = 0
    =>  v_body = R^T v = -(rdot + omega x r)
    """
    return -(np.asarray(rdot) + np.cross(omega, r))
