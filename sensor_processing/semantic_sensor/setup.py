from setuptools import setup
import os
from glob import glob

package_name = 'semantic_sensor'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml'))
    ],
    install_requires=['setuptools', 'torch', 'torchvision'],
    zip_safe=True,
    maintainer='Gian Erni',
    maintainer_email='gerni@ethz.ch',
    description='The semantic_sensor package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointcloud_node = semantic_sensor.pointcloud_node:main',
            'image_node = semantic_sensor.image_node:main',
        ],
    },
)
