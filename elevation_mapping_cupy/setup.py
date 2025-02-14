from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'elevation_mapping_cupy'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(include=[package_name, f'{package_name}.*']),
    install_requires=['setuptools'],
    zip_safe=True,
    author='Your Name',
    author_email='your.email@example.com',
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Elevation mapping on GPU',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'elevation_mapping_node.py = elevation_mapping_cupy.elevation_mapping_node:main',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        *[(os.path.join('share', package_name, os.path.dirname(yaml_file)), [yaml_file]) for yaml_file in glob('config/**/*.yaml', recursive=True)],
        # also the .*dat files
        *[(os.path.join('share', package_name, os.path.dirname(dat_file)), [dat_file]) for dat_file in glob('config/**/*.dat', recursive=True)],
        # add rviz files
        *[(os.path.join('share', package_name, os.path.dirname(rviz_file)), [rviz_file]) for rviz_file in glob('rviz/**/*.rviz', recursive=True)],
        (os.path.join('share', package_name), ['package.xml']),
    ],
)
