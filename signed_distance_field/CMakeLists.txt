cmake_minimum_required(VERSION 3.10)
project(signed_distance_field)

# Catkin dependencies
set(CATKIN_PACKAGE_DEPENDENCIES
	grid_map_core
	pcl_ros
)

find_package(catkin REQUIRED COMPONENTS
	${CATKIN_PACKAGE_DEPENDENCIES}
)

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
  src/GridmapSignedDistanceField.cpp
  src/SignedDistance2d.cpp
)
add_dependencies(${PROJECT_NAME}
	${catkin_EXPORTED_TARGETS}
)
target_link_libraries(${PROJECT_NAME}
	${catkin_LIBRARIES}
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

#############
## Testing ##
#############

add_executable(${PROJECT_NAME}_test
	test/test3dLookup.cpp
	test/testDerivatives.cpp
	test/testPixelBorderDistance.cpp
	test/testSignedDistance2d.cpp
	test/testSignedDistance3d.cpp
)
target_link_libraries(${PROJECT_NAME}_test
	${catkin_LIBRARIES}
	${PROJECT_NAME}
	gtest_main
)