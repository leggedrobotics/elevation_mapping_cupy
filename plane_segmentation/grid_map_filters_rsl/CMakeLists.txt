cmake_minimum_required(VERSION 3.5)
project(grid_map_filters_rsl)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Enable warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(grid_map_cv REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(sensor_msgs REQUIRED)          # Added dependency
find_package(rosidl_runtime_cpp REQUIRED)   # Added dependency

# Debugging output
message(STATUS "grid_map_cv_FOUND: ${grid_map_cv_FOUND}")
message(STATUS "grid_map_cv_INCLUDE_DIRS: ${grid_map_cv_INCLUDE_DIRS}")
message(STATUS "grid_map_cv_LIBRARIES: ${grid_map_cv_LIBRARIES}")

# Include directories
include_directories(
  include
  ${grid_map_cv_INCLUDE_DIRS}
  ${grid_map_core_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}            # Ensure sensor_msgs include dirs are added
  ${rosidl_runtime_cpp_INCLUDE_DIRS}     # Ensure rosidl_runtime_cpp include dirs are added
)

# Add library
add_library(${PROJECT_NAME} SHARED
  src/GridMapDerivative.cpp
  src/inpainting.cpp
  src/lookup.cpp
  src/smoothing.cpp
  src/processing.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Replace target_link_libraries with ament_target_dependencies
ament_target_dependencies(${PROJECT_NAME}
  grid_map_cv
  grid_map_core
  sensor_msgs
  rosidl_runtime_cpp
)

# Install library
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}/
)


# Clang tools (optional)
# find_package(cmake_clang_tools QUIET)
# if(cmake_clang_tools_FOUND)
#   add_default_clang_tooling()
# endif()

ament_package()