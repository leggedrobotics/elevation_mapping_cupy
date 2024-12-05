from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='semantic_sensor',
            executable='image_node',
            name='semantic_image',
            arguments=['front_cam_image'],
            output='screen',
            parameters=[{'ros__parameters': 'config/sensor_parameter.yaml'}]
        )
    ])