import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, OpaqueFunction,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.webots_launcher import WebotsLauncher


def launch_setup(context, *args, **kwargs):
    control = context.launch_configurations['control']
    demo = context.launch_configurations['demo'] == 'true'
    motion = context.launch_configurations['motion']
    height = float(context.launch_configurations['height'])
    controller = context.launch_configurations['controller']
    gui = context.launch_configurations['gui'] == 'true'
    mode = context.launch_configurations['mode']
    port = context.launch_configurations['port']

    package_dir = get_package_share_directory('spot_controller')
    world_path = os.path.join(package_dir, 'worlds', context.launch_configurations['world'])
    with open(world_path) as handle:
        world_text = handle.read()
    if '@TEX@' in world_text:       # textures are referenced by absolute path
        resolved = os.path.join(tempfile.gettempdir(), 'spot_world_resolved_{}.wbt'.format(
            os.path.basename(world_path)))
        with open(resolved, 'w') as handle:
            handle.write(world_text.replace('@TEX@', os.path.join(package_dir, 'worlds',
                                                                  'textures')))
        world_path = resolved

    world_name = context.launch_configurations['world']
    stereo = context.launch_configurations['stereo']
    stereo = (world_name == 'spot_slam.wbt') if stereo == 'auto' else stereo == 'true'
    # The driver would publish every Webots camera (with wall-clock stamps and at full
    # resolution). Switch them all off: the plugin publishes the stereo pair itself.
    devices = ''
    names = ['left_head_camera', 'right_head_camera', 'left_flank_camera', 'right_flank_camera',
             'rear_camera']
    if stereo:
        names += ['stereo_left', 'stereo_right', 'stereo_depth_gt']
    for name in names:
        kind = 'RangeFinder' if name == 'stereo_depth_gt' else 'Camera'
        devices += ('<device reference="{}" type="{}"><ros><enabled>false</enabled></ros>'
                    '</device>').format(name, kind)

    # The XML declaration must be the very first thing in the string.
    urdf = ('<?xml version="1.0" ?><robot name="Spot"><webots>{}'
            '<plugin type="spot_controller.webots_plugin.SpotMotorsPlugin">'
            '<control_mode>{}</control_mode><height>{}</height><controller>{}</controller>'
            '<stereo>{}</stereo></plugin></webots></robot>'.format(
                devices, control, height, controller, 'true' if stereo else 'false'))

    webots = WebotsLauncher(world=world_path, gui=gui, mode=mode, port=port)
    spot_driver = WebotsController(robot_name='Spot', port=port,
                                   parameters=[{'robot_description': urdf}])

    actions = [
        webots,
        spot_driver,
        Node(package='spot_controller', executable='state_estimator', output='screen'),
    ]
    if control == 'torque':
        pass        # estimator + stance controller run inside the plugin (no latency)
    else:
        actions.append(Node(package='spot_controller', executable='ekf_estimator',
                            output='screen'))
        if demo:
            actions.append(Node(package='spot_controller', executable='spot_publisher',
                                output='screen', parameters=[{'motion': motion}]))

    actions.append(RegisterEventHandler(
        event_handler=OnProcessExit(target_action=webots,
                                    on_exit=[EmitEvent(event=Shutdown())])))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('control', default_value='position',
                              description="'position' (Webots servo + demo sway) or 'torque'"),
        DeclareLaunchArgument('height', default_value='0.50',
                              description='stance height in torque mode (m)'),
        DeclareLaunchArgument('controller', default_value='vmc',
                              description="torque mode controller: 'vmc', 'mpc' or 'mpc_wbc'"),
        DeclareLaunchArgument('world', default_value='my_spot.wbt',
                              description="world file: 'my_spot.wbt' or the light 'spot_flat.wbt'"),
        DeclareLaunchArgument('stereo', default_value='auto',
                              description="publish the stereo cameras ('auto': only in spot_slam.wbt)"),
        DeclareLaunchArgument('gui', default_value='true', description='show the Webots window'),
        DeclareLaunchArgument('mode', default_value='realtime',
                              description="Webots mode: 'realtime' or 'fast'"),
        DeclareLaunchArgument('port', default_value='1234', description='Webots extern port'),
        DeclareLaunchArgument('demo', default_value='true',
                              description='position mode: start the demo command publisher'),
        DeclareLaunchArgument('motion', default_value='sway',
                              description="position mode demo: 'sway' or 'stand'"),
        OpaqueFunction(function=launch_setup),
    ])
