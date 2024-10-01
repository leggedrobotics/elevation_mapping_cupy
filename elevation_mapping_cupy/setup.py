from setuptools import setup, find_packages
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
            'elevation_mapping_node = elevation_mapping_cupy.elevation_mapping_ros:main',
        ],
    },
    # Remove data_files for package.xml
    data_files=[
        # Add the launch file
        (os.path.join('share', package_name, 'launch'), ['launch/elevation_mapping_launch.py']),
    ],
)
