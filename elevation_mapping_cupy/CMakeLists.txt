# %Tag(FULLTEXT)%
cmake_minimum_required(VERSION 3.0.2)
project(elevation_mapping_cupy)

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

if(PYTHONLIBS_FOUND)
  message(STATUS "Using Python Libraries at: " ${PYTHON_LIBRARIES})
  message(STATUS "Using Python include directories at: " ${PYTHON_INCLUDE_DIRS})
else()
  message(WARNING "Could not find Python Libraries")
endif()

find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED
  roscpp
  rospy
  tf
  tf_conversions
  sensor_msgs
  std_msgs
  geometry_msgs
  elevation_map_msgs
  grid_map_msgs
  grid_map_ros
  pcl_ros
  pybind11_catkin
)

catkin_package(
  INCLUDE_DIRS
    include
    ${EIGEN3_INCLUDE_DIRS}
  CATKIN_DEPENDS
    roscpp
    rospy
    tf
    tf_conversions
    sensor_msgs
    std_msgs
    geometry_msgs
    elevation_map_msgs
    grid_map_ros
    pcl_ros
    pybind11_catkin
)

include_directories(
  include
  ${PYTHON_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(elevation_mapping_ros
    src/elevation_mapping_wrapper.cpp
    src/elevation_mapping_ros.cpp)

target_link_libraries(elevation_mapping_ros ${PYTHON_LIBRARIES} ${catkin_LIBRARIES})

add_executable(elevation_mapping_node src/elevation_mapping_node.cpp)
target_link_libraries(elevation_mapping_node elevation_mapping_ros)

catkin_python_setup()

install(
  TARGETS elevation_mapping_ros elevation_mapping_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(
  DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
