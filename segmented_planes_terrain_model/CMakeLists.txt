cmake_minimum_required(VERSION 3.1...3.15)
project(segmented_planes_terrain_model)

# Catkin dependencies
find_package(catkin REQUIRED COMPONENTS
	ocs2_switched_model_interface
	ocs2_quadruped_interface
	convex_plane_decomposition
	convex_plane_decomposition_msgs
	convex_plane_decomposition_ros
	roscpp
)

# OpenCv
find_package(OpenCV REQUIRED)

# CGAL
find_package(CGAL COMPONENTS Core)

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

add_library(${PROJECT_NAME}
	src/SegmentedPlanesTerrainModel.cpp
	src/SegmentedPlanesTerrainModelRos.cpp
	)
add_dependencies(${PROJECT_NAME}
	${catkin_EXPORTED_TARGETS}
	)
target_link_libraries(${PROJECT_NAME}
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	glog
	)
target_compile_options(${PROJECT_NAME}
	PUBLIC -DCGAL_HAS_THREADS
	)

add_executable(${PROJECT_NAME}_demo_node
	src/DemoNode.cpp
)
target_link_libraries(${PROJECT_NAME}_demo_node
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	glog
	${PROJECT_NAME}
)

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
	)
install(DIRECTORY include/${PROJECT_NAME}/
	DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
	)

install(TARGETS ${PROJECT_NAME}_demo_node
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############
