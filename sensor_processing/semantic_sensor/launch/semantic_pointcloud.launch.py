from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='semantic_sensor',
            executable='pointcloud_node',
            name='semantic_pointcloud',
            arguments=['front_cam'],
            output='screen',
            parameters=[{'ros__parameters': 'config/sensor_parameter.yaml'}]
        )
    ])