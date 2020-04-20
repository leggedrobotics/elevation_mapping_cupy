cmake_minimum_required(VERSION 3.1...3.15)
project(convex_plane_decomposition_ros)

# Catkin dependencies
find_package(catkin REQUIRED COMPONENTS
	roscpp
	grid_map_core
	grid_map_ros
	grid_map_cv
	grid_map_filters
	grid_map_loader
	grid_map_msgs
	grid_map_octomap
	grid_map_rviz_plugin
	grid_map_visualization
	geometry_msgs
	sensor_msgs
	cv_bridge
	jsk_recognition_msgs
	convex_plane_decomposition
)

# OpenCv
find_package(OpenCV REQUIRED)

# CGAL
find_package(GMP QUIET)

# glog
find_package(PkgConfig REQUIRED)
pkg_check_modules(glog libglog REQUIRED)

# Cpp standard version
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###################################
## catkin specific configuration ##
###################################
catkin_package()

###########
## Build ##
###########

include_directories(
	include
	${catkin_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
	src/convex_plane_decomposition_node.cpp
	src/convex_plane_decomposition_ros.cpp
	src/grid_map_preprocessing.cpp
	src/pipeline_ros.cpp
	src/ros_visualizations.cpp
)
target_link_libraries(${PROJECT_NAME}
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	glog
	gmp
)

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY config data launch rviz
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############
