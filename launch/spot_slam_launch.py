"""Spot + stereo SLAM + mapping + detection/tracking (+ viewer, RViz, Nav2, exploration).

    ros2 launch spot_controller spot_slam_launch.py                       # sim, perception, dashboard
    ros2 launch spot_controller spot_slam_launch.py nav:=true             # + Nav2 (send goals with `goto`)
    ros2 launch spot_controller spot_slam_launch.py nav:=true explore:=true   # explores by itself
    ros2 launch spot_controller spot_slam_launch.py rviz:=true detector:=oracle

Drive it by hand with `ros2 run spot_controller teleop`, or let Nav2 / the explorer do it.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    cfg = context.launch_configurations
    pkg = get_package_share_directory('spot_controller')
    sim_time = {'use_sim_time': True}
    nodes = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'spot_launch.py')),
        launch_arguments={'control': 'torque', 'controller': cfg['controller'], 'world': 'spot_slam.wbt',
                          'gui': cfg['gui'], 'mode': cfg['mode'], 'port': cfg['port']}.items())]

    def node(executable, **extra):
        # NumPy would start one BLAS thread per core in every node; the stages already share the CPU
        nodes.append(Node(package='spot_controller', executable=executable, output='screen',
                          parameters=[sim_time, extra.pop('parameters', {})],
                          additional_env={'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, **extra))

    node('tf_bridge')
    node('stereo_depth')
    node('stereo_slam')
    node('mapper')
    node('detector', parameters={'backend': cfg['detector'], 'every': int(cfg['detector_every'])})
    node('tracker')
    if cfg['viewer'] == 'true':
        node('viewer', parameters={'save': cfg['save_view'], 'show': True})
    if cfg['rviz'] == 'true':
        nodes.append(Node(package='rviz2', executable='rviz2', output='screen', parameters=[sim_time],
                          arguments=['-d', os.path.join(pkg, 'config', 'spot_slam.rviz')]))
    if cfg['nav'] == 'true':
        params = os.path.join(pkg, 'config', 'nav2_spot.yaml')
        servers = [('nav2_controller', 'controller_server', [('cmd_vel', 'cmd_vel_nav')]),
                   ('nav2_planner', 'planner_server', []),
                   ('nav2_behaviors', 'behavior_server', [('cmd_vel', 'cmd_vel_nav')]),
                   ('nav2_bt_navigator', 'bt_navigator', []),
                   ('nav2_velocity_smoother', 'velocity_smoother',
                    [('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', '/cmd_vel')])]
        for package, executable, remap in servers:
            nodes.append(Node(package=package, executable=executable, name=executable, output='screen',
                              parameters=[params, sim_time], remappings=remap))
        nodes.append(Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                          name='lifecycle_manager_navigation', output='screen',
                          parameters=[sim_time, {'autostart': True, 'node_names': [s[1] for s in servers]}]))
        if cfg['explore'] == 'true':
            node('explorer')
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('controller', default_value='vmc', description="'vmc', 'mpc' or 'mpc_wbc'"),
        DeclareLaunchArgument('gui', default_value='true', description='Webots window (cameras need it)'),
        DeclareLaunchArgument('mode', default_value='realtime', description="'realtime' or 'fast'"),
        DeclareLaunchArgument('port', default_value='1234', description='Webots extern port'),
        DeclareLaunchArgument('detector', default_value='auto',
                              description="'auto' (YOLO if installed, else HOG), 'yolo', 'hog' or 'oracle' (ground truth, cheat)"),
        DeclareLaunchArgument('detector_every', default_value='3', description='run the detector on every Nth frame'),
        DeclareLaunchArgument('viewer', default_value='true', description='OpenCV dashboard window'),
        DeclareLaunchArgument('save_view', default_value='', description='also save the dashboard to this PNG'),
        DeclareLaunchArgument('rviz', default_value='false', description='open RViz'),
        DeclareLaunchArgument('nav', default_value='false', description='start Nav2'),
        DeclareLaunchArgument('explore', default_value='false', description='autonomous frontier exploration (needs nav)'),
        OpaqueFunction(function=launch_setup),
    ])
