cmake_minimum_required(VERSION 3.1...3.15)
project(convex_plane_decomposition)

# Catkin dependencies
set(CATKIN_PACKAGE_DEPENDENCIES
	grid_map_core
)

find_package(catkin REQUIRED COMPONENTS
  ${CATKIN_PACKAGE_DEPENDENCIES}
)

# CGAL
find_package(CGAL QUIET)
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)

# OpenCv
find_package(OpenCV REQUIRED
  COMPONENTS
  	opencv_highgui
  CONFIG
)

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
	DEPENDS OpenCV CGAL GMP MPFR
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_library(${PROJECT_NAME}
	src/ransac_plane_extractor.cpp
	src/sliding_window_plane_extractor.cpp
	src/plane.cpp
	src/polygon.cpp
	src/geometry_utils.cpp
	src/export_utils.cpp
	src/plane_factory.cpp
	src/polygonizer.cpp
	src/convex_decomposer.cpp
	src/pipeline.cpp
)
add_dependencies(${PROJECT_NAME}
	${catkin_EXPORTED_TARGETS}
)
target_link_libraries(${PROJECT_NAME}
	${catkin_LIBRARIES}
	${CGAL_LIBRARIES}
	${OpenCV_LIBRARIES}
	glog
	gmp
	mpfr
)
target_compile_options(${PROJECT_NAME}
	PUBLIC -DCGAL_HAS_NO_THREADS
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
