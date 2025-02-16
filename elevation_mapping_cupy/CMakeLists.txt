cmake_minimum_required(VERSION 3.8)
project(elevation_mapping_cupy)

# # Enable C++11 (or higher if needed for ROS 2 and pybind11)
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Additional dependencies
# find_package(Python COMPONENTS Interpreter Development)
find_package(PythonInterp 3 REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Find pybind11
message([MAIN] "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")
message([MAIN] "pybind11_INCLUDE_DIRS = ${pybind11_INCLUDE_DIRS}")
message([MAIN] "pybind11_LIBRARIES = ${pybind11_LIBRARIES}")

# Find ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(message_filters REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(elevation_map_msgs REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(python_cmake_module REQUIRED)
find_package(point_cloud_transport REQUIRED)

# List dependencies for ament_target_dependencies
set(dependencies
  rclcpp
  rclpy
  std_msgs
  std_srvs
  builtin_interfaces
  geometry_msgs
  sensor_msgs
  elevation_map_msgs
  grid_map_msgs
  grid_map_ros
  image_transport
  pcl_ros
  message_filters
  tf2_eigen
  point_cloud_transport
)

# Include directories
include_directories(
  include    
  ${PYTHON_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
  ${elevation_map_msgs_INCLUDE_DIRS}
)

# Declare C++ library
add_library(elevation_mapping_ros
    src/elevation_mapping_wrapper.cpp
    src/elevation_mapping_ros.cpp
)

# Link the library with necessary dependencies
target_link_libraries(elevation_mapping_ros ${PYTHON_LIBRARIES} ${OpenCV_LIBRARIES} pybind11::embed)
ament_target_dependencies(elevation_mapping_ros ${dependencies})

# Declare C++ executable
add_executable(elevation_mapping_node src/elevation_mapping_node.cpp)

# Link the executable with the library and dependencies
target_link_libraries(elevation_mapping_node elevation_mapping_ros ${OpenCV_LIBRARIES} pybind11::embed)
ament_target_dependencies(elevation_mapping_node ${dependencies})


# Install targets Not sure if these other argrs are necessary
# install(
#   TARGETS elevation_mapping_ros elevation_mapping_node 
#   DESTINATION lib/${PROJECT_NAME}
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin
# )

install(TARGETS elevation_mapping_ros elevation_mapping_node 
  DESTINATION lib/${PROJECT_NAME}
)

# install(PROGRAMS  
#   DESTINATION lib/${PROJECT_NAME}
# )

# Install launch, config, and rviz directories
install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

# TODO: Understand if this line is necessary
# _ament_cmake_python_register_environment_hook()
# For use as a Python module outside of ROS 2
ament_python_install_package(${PROJECT_NAME})

# Install the Python ROS 2 modules
install(PROGRAMS
  scripts/elevation_mapping_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Ament package declaration
ament_package()
