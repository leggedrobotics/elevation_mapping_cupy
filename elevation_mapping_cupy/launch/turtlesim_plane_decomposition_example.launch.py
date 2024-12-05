from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    elevation_mapping_cupy_dir = get_package_share_directory('elevation_mapping_cupy')
    convex_plane_decomposition_dir = get_package_share_directory('convex_plane_decomposition_ros')

    return LaunchDescription([
        # Launch elevation mapping turtle sim
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    elevation_mapping_cupy_dir,
                    'launch',
                    'turtlesim_simple_example.launch.py'
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

        # Launch the plane decomposition node
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    convex_plane_decomposition_dir,
                    'launch',
                    'convex_plane_decomposition.launch.py'
                ])
            )
        )
    ])