cmake_minimum_required(VERSION 3.5)
project(elevation_mapping_cupy)

find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(elevation_map_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pybind11 REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install launch and config directories
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
