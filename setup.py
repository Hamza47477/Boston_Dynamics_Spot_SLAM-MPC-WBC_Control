import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'spot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Tell ROS 2 to install your launch and worlds folders
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.wbt')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'worlds', 'textures'), glob('worlds/textures/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hamza Masud',
    maintainer_email='hamzamasood2250@gmail.com',
    description='SpotSense: Spot control, stereo SLAM and navigation in Webots',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_estimator = spot_controller.state_estimator:main',
            'ekf_estimator = spot_controller.ekf_node:main',
            'push_test = spot_controller.push_test:main',
            'spot_publisher = spot_controller.spot_publisher_node:main',
            'teleop = spot_controller.teleop_keyboard:main',
            'stereo_depth = spot_controller.perception.stereo_depth_node:main',
        ],
    },
)
