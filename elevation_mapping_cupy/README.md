# Example 1: Turtle Simple Example
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

Launch the elevation mapping node with the configs for the turtle. Set use_python_node to true to use it instead of the cpp node:
```bash
ros2 launch elevation_mapping_cupy elevation_mapping_turtle.launch.py use_python_node:=false
```

If you want to drive the turtlebot around using the keyboard then run:
```bash
ros2 run turtlebot3_teleop teleop_keyboard 
```