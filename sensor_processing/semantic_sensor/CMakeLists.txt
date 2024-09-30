cmake_minimum_required(VERSION 3.8)
project(semantic_sensor)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR script/semantic_sensor)

# Install Python executables
install(PROGRAMS
  script/semantic_sensor/pointcloud_node.py
  script/semantic_sensor/image_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()