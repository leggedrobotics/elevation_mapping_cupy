from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='elevation_mapping_cupy',
            executable='elevation_mapping_node',
            name='elevation_mapping_node',
            output='screen',
            parameters=[
                ParameterFile('config/core/core_param.yaml', allow_substs=True)
            ]
        )
    ])