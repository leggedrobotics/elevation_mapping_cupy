cmake_minimum_required(VERSION 3.1...3.15)
project(convex_plane_decomposition_ros)

# Catkin dependencies
set(CATKIN_PACKAGE_DEPENDENCIES
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
	convex_plane_decomposition_msgs
)

find_package(catkin REQUIRED COMPONENTS
	${CATKIN_PACKAGE_DEPENDENCIES}
)

# OpenCv
find_package(OpenCV REQUIRED)

# Qt
find_package(Qt5 COMPONENTS Xml Script OpenGL Svg)

# CGAL
find_package(CGAL COMPONENTS Core Qt5)

# glog
find_package(PkgConfig REQUIRED)
pkg_check_modules(glog libglog REQUIRED)

# Cpp standard version
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###################################
## catkin specific configuration ##
###################################
catkin_package(
	INCLUDE_DIRS include
	LIBRARIES ${PROJECT_NAME}
	CATKIN_DEPENDS ${CATKIN_PACKAGE_DEPENDENCIES}
	DEPENDS
)

###########
## Build ##
###########

include_directories(
	include
	${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
	src/ConvexPlaneDecompositionRos.cpp
	src/MessageConversion.cpp
	src/ParameterLoading.cpp
	src/RosVisualizations.cpp
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

add_executable(${PROJECT_NAME}_node
	src/ConvexPlaneDecompositionNode.cpp
)
target_link_libraries(${PROJECT_NAME}_node
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	glog
	${PROJECT_NAME}
)

add_executable(${PROJECT_NAME}_test
	test/test.cpp
	)
target_link_libraries(${PROJECT_NAME}_test
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	glog
	Qt5::Xml Qt5::Script Qt5::OpenGL Qt5::Svg
	CGAL::CGAL CGAL::CGAL_Core CGAL::CGAL_Qt5
	)
target_compile_options(${PROJECT_NAME}_test
	PUBLIC -DCGAL_HAS_THREADS -DCGAL_USE_BASIC_VIEWER
	)

add_executable(${PROJECT_NAME}_shapegrowing
	test/shapeGrowing.cpp
	)
target_link_libraries(${PROJECT_NAME}_shapegrowing
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	glog
	Qt5::Xml Qt5::Script Qt5::OpenGL Qt5::Svg
	CGAL::CGAL CGAL::CGAL_Core CGAL::CGAL_Qt5
	)
target_compile_options(${PROJECT_NAME}_shapegrowing
	PUBLIC -DCGAL_HAS_THREADS -DCGAL_USE_BASIC_VIEWER
	)

add_executable(${PROJECT_NAME}_save_elevationmap
	src/SaveElevationMapAsImageNode.cpp
	)
target_link_libraries(${PROJECT_NAME}_save_elevationmap
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
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

install(TARGETS ${PROJECT_NAME}_node
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config data launch rviz
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############
