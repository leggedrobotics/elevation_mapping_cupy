# %Tag(FULLTEXT)%
cmake_minimum_required(VERSION 2.8.3)
project(elevation_mapping_cupy)

find_package(catkin REQUIRED
  rospy
  ros_numpy
  sensor_msgs
  std_msgs
  geometry_msgs
  nav_msgs
  grid_map_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    ros_numpy
    sensor_msgs
    std_msgs
    geometry_msgs
    nav_msgs
    grid_map_msgs
)

install(
  PROGRAMS
    scripts/elevation_mapping.py
    scripts/elevation_mapping_node.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION})
