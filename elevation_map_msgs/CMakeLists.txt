cmake_minimum_required(VERSION 3.0.2)
project(elevation_map_msgs)

find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  Statistics.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  CheckSafety.srv
  Initialize.srv
)


## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
)
