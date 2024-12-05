import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_name = 'elevation_mapping_cupy'
    share_dir = get_package_share_directory(package_name)

    config_file = os.path.join(share_dir, 'config', 'core_param.yaml')
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} does not exist")

    return LaunchDescription([
        Node(
            package=package_name,
            executable='elevation_mapping_node',
            name='elevation_mapping_node',
            output='screen',
            parameters=[config_file]
        )
    ])