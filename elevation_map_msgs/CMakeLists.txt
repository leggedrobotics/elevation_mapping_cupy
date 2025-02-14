cmake_minimum_required(VERSION 3.5)
project(elevation_map_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)  # Added std_msgs

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
  DEPENDENCIES geometry_msgs std_msgs  std_msgs # Added std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()

# cmake_minimum_required(VERSION 3.8)
# project(elevation_map_msgs)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# # find dependencies
# find_package(ament_cmake REQUIRED)
# find_package(rosidl_default_generators REQUIRED)
# find_package(builtin_interfaces REQUIRED)
# find_package(std_msgs REQUIRED)
# find_package(geometry_msgs REQUIRED)
# find_package(action_msgs REQUIRED)

# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/Statistics.msg"
#   "msg/ChannelInfo.msg"
#   "srv/CheckSafety.srv"
#   "srv/Initialize.srv"
#   DEPENDENCIES builtin_interfaces geometry_msgs std_msgs action_msgs
# )

# ament_export_dependencies(rosidl_default_runtime)

# ament_package()

