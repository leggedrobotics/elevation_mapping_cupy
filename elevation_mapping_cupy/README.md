## Example 1: Turtle Simple Example
Install the realsense wrapper for [TurtleBot3]([ros2_turtlebot3_waffle_intel_realsense](https://github.com/mlherd/ros2_turtlebot3_waffle_intel_realsense)) and place the model in your home .gazebo/models/ folder.
You can launch the turtlebot3 in Gazebo with the following command:
```bash
ros2 launch elevation_mapping_cupy turtle_simple_example.launch.py
``` 
You can also maunally spawn the robot in Gazebo.


Launch the elevation mapping node:
```bash
ros2 launch elevation_mapping_cupy turtle_simple_example.launch.py
```


## (Corrected) Example 1: Turtle Simple Example
Set the env in the Dockerfile
```dockerfile
ENV TURTLEBOT3_MODEL=waffle_realsense_depth
```
or in the terminal
```bash
export TURTLEBOT3_MODEL=waffle_realsense_depth
```
If using zenoh as rmw then start one terminal up and run the router
```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

Now launch the turtlebot3 in Gazebo with the following command:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
``` 

Launch the elevation mapping node with the configs for the turtle:
```bash
ros2 launch elevation_mapping_cupy elevation_mapping_turtle.launch.py 
```