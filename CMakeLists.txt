# %Tag(FULLTEXT)%
cmake_minimum_required(VERSION 2.8.12)
project(elevation_mapping_cupy)

# find_package(pybind11 REQUIRED)  # or add_subdirectory(pybind11)
find_package(pybind11 REQUIRED)  # or add_subdirectory(pybind11)
find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED
  roscpp
  rospy
  tf
  tf_conversions
  sensor_msgs
  std_msgs
  geometry_msgs
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
    grid_map_ros
    pcl_ros
    pybind11_catkin
)

include_directories(
  include
  # ${PYBIND11_INCLUDE_DIR} 
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(elevation_mapping_ros
    src/elevation_mapping_wrapper.cpp
    src/elevation_mapping_ros.cpp)

# pybind_add_module(elevation_mapping_ros SHARED 
#     src/elevation_mapping_wrapper.cpp
#     src/elevation_mapping_ros.cpp)
target_link_libraries(elevation_mapping_ros ${catkin_LIBRARIES} pybind11::embed)
# target_link_libraries(elevation_mapping_ros ${catkin_LIBRARIES})

add_executable(elevation_mapping_node src/elevation_mapping_node.cpp)
target_link_libraries(elevation_mapping_node elevation_mapping_ros)

install(
  PROGRAMS
    scripts/elevation_mapping.py
    scripts/elevation_mapping_node.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION})

