cmake_minimum_required(VERSION 3.10)
project(convex_plane_decomposition_ros)

# Catkin dependencies
set(CATKIN_PACKAGE_DEPENDENCIES
	roscpp
	grid_map_core
	grid_map_ros
	grid_map_cv
	grid_map_msgs
	geometry_msgs
	convex_plane_decomposition
	convex_plane_decomposition_msgs
	tf2_ros
	visualization_msgs
)

find_package(catkin REQUIRED COMPONENTS
	${CATKIN_PACKAGE_DEPENDENCIES}
)

# OpenCv
find_package(OpenCV REQUIRED)

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Cpp standard version
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###################################
## catkin specific configuration ##
###################################
catkin_package(
	INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIRS}
	LIBRARIES ${PROJECT_NAME}
	CATKIN_DEPENDS ${CATKIN_PACKAGE_DEPENDENCIES}
	DEPENDS OpenCV
)

###########
## Build ##
###########

include_directories(
	include
	${EIGEN3_INCLUDE_DIRS}
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
	)

add_executable(${PROJECT_NAME}_node
	src/ConvexPlaneDecompositionNode.cpp
)
target_link_libraries(${PROJECT_NAME}_node
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PROJECT_NAME}
)

add_executable(${PROJECT_NAME}_add_noise
	src/noiseNode.cpp
)
target_link_libraries(${PROJECT_NAME}_add_noise
	${catkin_LIBRARIES}
)

add_executable(${PROJECT_NAME}_save_elevationmap
	src/SaveElevationMapAsImageNode.cpp
	)
target_link_libraries(${PROJECT_NAME}_save_elevationmap
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	)

add_executable(${PROJECT_NAME}_approximation_demo_node
	src/ConvexApproximationDemoNode.cpp
	)
target_link_libraries(${PROJECT_NAME}_approximation_demo_node
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
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

install(TARGETS ${PROJECT_NAME}_node
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config data launch rviz
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############
add_executable(${PROJECT_NAME}_TestShapeGrowing
	test/TestShapeGrowing.cpp
)
target_link_libraries(${PROJECT_NAME}_TestShapeGrowing
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
)