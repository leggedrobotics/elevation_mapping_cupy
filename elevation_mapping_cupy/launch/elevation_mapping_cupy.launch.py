import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
import launch_ros.actions

def generate_launch_description():
    core_param_file = os.path.join(
        get_package_share_directory('elevation_mapping_cupy'),
        'config',
        'core',
        'core_param.yaml'
    )

    
    return LaunchDescription([
        launch_ros.actions.SetParameter(name='use_sim_time', value=True),
        Node(
            package='elevation_mapping_cupy',
            executable='elevation_mapping_node',
            name='elevation_mapping',
            output='screen',
            parameters=[core_param_file],
            # arguments=['--ros-args', '--log-level', 'DEBUG']
        )
    ])