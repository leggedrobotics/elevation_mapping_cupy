cmake_minimum_required(VERSION 3.14)
project(cgal5_ament)

# Use ament for ROS2
find_package(ament_cmake REQUIRED)

include(ExternalProject)

set(VERSION 5.3)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

set(CGAL_VERSION 5.3)
ExternalProject_Add(cgal
  URL https://github.com/CGAL/cgal/archive/refs/tags/v${CGAL_VERSION}.tar.gz
  UPDATE_COMMAND ""
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
    -DCMAKE_BUILD_TYPE:STRING=Release
  BUILD_COMMAND $(MAKE)
  INSTALL_COMMAND $(MAKE) install
)

ament_export_include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/CGAL/
  DESTINATION include/CGAL/
)

ament_package()
