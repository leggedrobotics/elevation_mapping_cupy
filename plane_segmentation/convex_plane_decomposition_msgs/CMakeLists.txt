cmake_minimum_required(VERSION 3.5)
project(convex_plane_decomposition_msgs)

# Find ament and necessary packages
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)

# Add message files for ROS2
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox2d.msg"
  "msg/PlanarRegion.msg"
  "msg/PlanarTerrain.msg"
  "msg/Point2d.msg"
  "msg/Polygon2d.msg"
  "msg/PolygonWithHoles2d.msg"
  DEPENDENCIES std_msgs geometry_msgs grid_map_msgs
)

# Replace catkin_package with ament_package
ament_package(
  # ... existing export settings if any ...
)
