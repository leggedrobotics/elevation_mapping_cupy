import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    package_name = 'elevation_mapping_cupy'
    share_dir = get_package_share_directory(package_name)
    
    core_param_path = os.path.join(share_dir, 'config', 'core', 'core_param.yaml')
    turtle_param_path = os.path.join(share_dir, 'config', 'setups', 'turtle_bot', 'turtle_bot_simple.yaml')
    
    # Add verification
    if not os.path.exists(core_param_path):
        raise FileNotFoundError(f"Core param file not found: {core_param_path}")
    if not os.path.exists(turtle_param_path):
        raise FileNotFoundError(f"Turtle param file not found: {turtle_param_path}")

    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            share_dir, 'rviz', 'turtle_sim_laser.rviz'
        ]),
        description='Path to the RViz config file'
    )
    rviz_config = LaunchConfiguration('rviz_config')

    # RViz Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    elevation_mapping_node = Node(
        package='elevation_mapping_cupy',
        executable='elevation_mapping_node',
        name='elevation_mapping_node',
        output='screen',
        parameters=[
            core_param_path, 
            turtle_param_path,
            {'use_sim_time': use_sim_time}
        ]
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        rviz_config_arg,
        elevation_mapping_node,
        rviz_node
    ])
