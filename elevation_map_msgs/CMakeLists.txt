cmake_minimum_required(VERSION 3.5)
project(elevation_map_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED) # Added dependency

set(msg_files
  "msg/Statistics.msg"
  "msg/ChannelInfo.msg"
)

set(srv_files
  "srv/CheckSafety.srv"
  "srv/Initialize.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES geometry_msgs std_msgs # Combined dependencies
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
