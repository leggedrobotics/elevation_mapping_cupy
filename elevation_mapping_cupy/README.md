## Example 1: Turtle Simple Example
Install the realsense wrapper for [TurtleBot3]([ros2_turtlebot3_waffle_intel_realsense](https://github.com/mlherd/ros2_turtlebot3_waffle_intel_realsense)) and place the model in your home .gazebo/models/ folder.
You can launch the turtlebot3 in Gazebo with the following command:
```bash
ros2 launch elevation_mapping_cupy turtle_simple_example.launch.py
``` 
You can also maunally spawn the robot in Gazebo.


Launch the elevation mapping node:
```bash
ros2 launch elevation_mapping_cupy turtle_simple_example.launch.py```