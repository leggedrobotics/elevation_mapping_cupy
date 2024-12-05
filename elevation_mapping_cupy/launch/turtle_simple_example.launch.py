# elevation_mapping_launch.py

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, FindExecutable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare the 'rviz_config' launch argument with a default value
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(
            get_package_share_directory('elevation_mapping_cupy'),
            'rviz',
            'turtle_example.rviz'
        ),
        description='Path to the RViz config file'
    )

    # Define the path to the turtlesim_init launch file
    turtlesim_init_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('elevation_mapping_cupy'),
                'launch',
                'turtlesim_init.launch.py'
            )
        ),
        launch_arguments={
            'rviz_config': LaunchConfiguration('rviz_config'),
            'use_sim_time': 'true'  # Set simulation time to true
        }.items()
    )

    # Define the elevation_mapping node
    elevation_mapping_node = Node(
        package='elevation_mapping_cupy',
        executable='elevation_mapping_node',  # Ensure this is the correct executable name
        name='elevation_mapping',
        output='screen',
        parameters=[
            os.path.join(
                get_package_share_directory('elevation_mapping_cupy'),
                'config',
                'core',
                'core_param.yaml'
            ),
            os.path.join(
                get_package_share_directory('elevation_mapping_cupy'),
                'config',
                'setups',
                'turtle_bot',
                'turtle_bot_simple.yaml'
            )
        ]
    )

    # Create and return the LaunchDescription object
    return LaunchDescription([
        rviz_config_arg,
        turtlesim_init_launch,
        # elevation_mapping_node,
    ])
