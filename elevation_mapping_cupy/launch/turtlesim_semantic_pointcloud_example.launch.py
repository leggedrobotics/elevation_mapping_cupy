from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    elevation_mapping_cupy_dir = get_package_share_directory('elevation_mapping_cupy')
    semantic_sensor_dir = get_package_share_directory('semantic_sensor')

    return LaunchDescription([
        # Include the turtlesim_init launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    elevation_mapping_cupy_dir,
                    'launch',
                    'turtlesim_init.launch.py'
                ])
            ),
            launch_arguments={
                'rviz_config': PathJoinSubstitution([
                    elevation_mapping_cupy_dir,
                    'rviz',
                    'turtle_segmentation_example.rviz'
                ])
            }.items()
        ),

        # Semantic Sensor Node
        Node(
            package='semantic_sensor',
            executable='pointcloud_node.py',
            name='front_cam',
            arguments=['front_cam'],
            parameters=[PathJoinSubstitution([
                semantic_sensor_dir,
                'config',
                'sensor_parameter.yaml'
            ])],
            output='screen'
        ),

        # Elevation Mapping Node
        Node(
            package='elevation_mapping_cupy',
            executable='elevation_mapping_node',
            name='elevation_mapping',
            parameters=[
                PathJoinSubstitution([
                    elevation_mapping_cupy_dir,
                    'config',
                    'core',
                    'core_param.yaml'
                ]),
                PathJoinSubstitution([
                    elevation_mapping_cupy_dir,
                    'config',
                    'setups',
                    'turtle_bot',
                    'turtle_bot_semantics_pointcloud.yaml'
                ])
            ],
            output='screen'
        )
    ]) 