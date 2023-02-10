cmake_minimum_required(VERSION 3.0.2)
project(semantic_sensor)

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

if(PYTHONLIBS_FOUND)
  message(STATUS "Using Python Libraries at: " ${PYTHON_LIBRARIES})
  message(STATUS "Using Python include directories at: " ${PYTHON_INCLUDE_DIRS})
else()
  message(WARNING "Could not find Python Libraries")
endif()


find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  tf
  tf_conversions
  sensor_msgs
  std_msgs
  geometry_msgs
)

catkin_package()

catkin_python_setup()



catkin_install_python(PROGRAMS script/semantic_sensor/pointcloud_node.py script/semantic_sensor/image_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

