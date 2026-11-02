"""Shared constants for the Webots Spot: geometry, joint map, limits, conventions.

Numbers come from Webots R2025a Spot.proto / SpotLeg.proto (measured, about +-1 cm).

Two angle conventions are used:
  * IK convention:    x forward, z up. theta1 = abduction about +x (foot toward +y is
                      positive), theta2 = thigh angle from vertical (forward positive),
                      theta3 = knee flexion (0 = leg straight, positive = knee bent).
  * Webots convention: raw motor/sensor values. Zero is NOT a straight leg: the thigh
                      points down and 35.6 deg backward, the shin down and 39.9 deg forward.
Use ik_to_webots() / webots_to_ik() to convert.
"""
import math

LEG_ORDER = ('FL', 'FR', 'RL', 'RR')
LEG_NAMES = {'FL': 'front left', 'FR': 'front right', 'RL': 'rear left', 'RR': 'rear right'}
LEG_ROS_PREFIX = {'FL': 'front_left', 'FR': 'front_right',
                  'RL': 'rear_left', 'RR': 'rear_right'}

# (ROS suffix, Webots device part, (min, max) in Webots radians)
_JOINTS = (
    ('hip_x', 'shoulder abduction', (-0.6, 0.5)),
    ('hip_y', 'shoulder rotation', (-1.7, 1.7)),
    ('knee', 'elbow', (-0.45, 1.6)),
)

JOINT_NAMES = []
MOTOR_DEVICE = {}
SENSOR_DEVICE = {}
JOINT_LIMITS = {}
for _leg in LEG_ORDER:
    for _suffix, _part, _limits in _JOINTS:
        _name = '{}_{}'.format(LEG_ROS_PREFIX[_leg], _suffix)
        JOINT_NAMES.append(_name)
        MOTOR_DEVICE[_name] = '{} {} motor'.format(LEG_NAMES[_leg], _part)
        SENSOR_DEVICE[_name] = '{} {} sensor'.format(LEG_NAMES[_leg], _part)
        JOINT_LIMITS[_name] = _limits

# Leg geometry (metres)
THIGH_LENGTH = math.hypot(0.29077, 0.2083)   # hip pitch axis -> knee
CALF_LENGTH = math.hypot(0.3006, 0.2510)     # knee -> foot centre
HIP_OFFSET_Y = 0.1172                        # lateral offset of the thigh from the abduction axis

# Abduction axis position in the body frame (x differs front/rear in the Webots model)
HIP_MOUNT = {
    'FL': (0.3635, 0.0528, 0.0118),
    'FR': (0.3635, -0.0528, 0.0118),
    'RL': (-0.3084, 0.0528, 0.0117),
    'RR': (-0.3084, -0.0528, 0.0117),
}
LEG_SIDE = {'FL': 1.0, 'FR': -1.0, 'RL': 1.0, 'RR': -1.0}   # +1 left, -1 right

# Webots zero pose expressed in the IK convention
THIGH_ZERO_ANGLE = -math.atan2(0.2083, 0.29077)           # thigh 35.6 deg backward
_CALF_ZERO_ABS = math.atan2(0.2510, 0.3006)               # shin 39.9 deg forward
KNEE_ZERO_ANGLE = _CALF_ZERO_ABS - THIGH_ZERO_ANGLE       # theta3 at Webots zero


def ik_to_webots(theta1, theta2, theta3):
    """Convert one leg's IK angles to Webots motor angles."""
    return (-theta1, theta2 - THIGH_ZERO_ANGLE, theta3 - KNEE_ZERO_ANGLE)


def webots_to_ik(hip_x, hip_y, knee):
    """Convert one leg's Webots sensor angles to IK angles."""
    return (-hip_x, hip_y + THIGH_ZERO_ANGLE, knee + KNEE_ZERO_ANGLE)


def clamp_joint(name, value):
    """Clamp a Webots-convention angle to the motor limits."""
    low, high = JOINT_LIMITS[name]
    return max(low, min(high, value))


# Simulated IMU imperfections (Webots sensors are perfect by default). Set to False for ideal data.
IMU_NOISE_ENABLED = True
GYRO_BIAS = (0.010, -0.008, 0.005)       # rad/s, constant
GYRO_SIGMA = 0.003                       # rad/s per sample
ACCEL_BIAS = (0.05, -0.04, 0.06)         # m/s^2, constant
ACCEL_SIGMA = 0.05                       # m/s^2 per sample

# --- Torque control (stance controller) ---
ROBOT_MASS = 27.2                        # kg: body 14.41 + 4 legs x 3.204 (from the PROTO files)
COM_OFFSET = (0.025, 0.0, 0.02)          # approximate centre of mass in the body frame (m)
TORQUE_LIMIT = 50.0                      # N m, Webots motor maxTorque
# Joint-level PD run inside the plugin every simulation step (Webots convention angles)
JOINT_KP = {'hip_x': 90.0, 'hip_y': 90.0, 'knee': 90.0}
JOINT_KD = {'hip_x': 2.0, 'hip_y': 2.0, 'knee': 2.0}

ROBOT_INERTIA = (0.55, 2.4, 2.5)          # kg m^2 about the CoM (roll, pitch, yaw), estimated from link masses
