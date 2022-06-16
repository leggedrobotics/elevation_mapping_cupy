cmake_minimum_required(VERSION 3.10)
project(convex_plane_decomposition)

# Catkin dependencies
set(CATKIN_PACKAGE_DEPENDENCIES
	cgal5_catkin
	grid_map_core
	grid_map_cv
	grid_map_filters_rsl
)

find_package(catkin REQUIRED COMPONENTS
  ${CATKIN_PACKAGE_DEPENDENCIES}
)

# CGAL dependencies (QUIET because they cannot be found by Clion)
find_package(GMP 4.2 QUIET)
find_package(MPFR 2.2.1 QUIET)
find_package(Boost 1.57 QUIET)

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
  INCLUDE_DIRS
		include
		${EIGEN3_INCLUDE_DIRS}
  LIBRARIES
		${PROJECT_NAME}
  CATKIN_DEPENDS
		${CATKIN_PACKAGE_DEPENDENCIES}
	DEPENDS
		OpenCV GMP MPFR Boost
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
  src/contour_extraction/ContourExtraction.cpp
  src/contour_extraction/Upsampling.cpp
  src/ransac/RansacPlaneExtractor.cpp
  src/sliding_window_plane_extraction/SlidingWindowPlaneExtractor.cpp
  src/ConvexRegionGrowing.cpp
  src/Draw.cpp
  src/GridMapPreprocessing.cpp
  src/LoadGridmapFromImage.cpp
  src/PlanarRegion.cpp
  src/PlaneDecompositionPipeline.cpp
  src/Postprocessing.cpp
  src/SegmentedPlaneProjection.cpp
)
add_dependencies(${PROJECT_NAME}
	${catkin_EXPORTED_TARGETS}
)
target_link_libraries(${PROJECT_NAME}
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	gmp
	mpfr
)
target_compile_options(${PROJECT_NAME}
	PUBLIC -DCGAL_HAS_THREADS
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

catkin_add_gtest(test_${PROJECT_NAME}
	test/testConvexApproximation.cpp
	test/testPipeline.cpp
	test/testPlanarRegion.cpp
	test/testRegionGrowing.cpp
	test/testUpsampling.cpp
)
target_link_libraries(test_${PROJECT_NAME}
	${PROJECT_NAME}
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	gmp
	mpfr
	gtest_main
)
