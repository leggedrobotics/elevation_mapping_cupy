from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    elevation_mapping_cupy_dir = get_package_share_directory('elevation_mapping_cupy')

    return LaunchDescription([
        # Elevation Mapping Node
        Node(
            package='elevation_mapping_cupy',
            executable='elevation_mapping_node',
            name='elevation_mapping',
            parameters=[
                PathJoinSubstitution([
                    elevation_mapping_cupy_dir,
                    'config',
                    'setups',
                    'anymal',
                    'anymal_parameters.yaml'
                ]),
                PathJoinSubstitution([
                    elevation_mapping_cupy_dir,
                    'config',
                    'setups',
                    'anymal',
                    'anymal_sensor_parameter.yaml'
                ])
            ],
            output='screen'
        )
    ])