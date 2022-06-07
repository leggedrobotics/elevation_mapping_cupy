cmake_minimum_required(VERSION 3.14)
project(cgal5_catkin)

find_package(catkin REQUIRED)

include(ExternalProject)

set(VERSION 5.3)

file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/include)

set(CGAL_VERSION 5.3)
ExternalProject_Add(cgal
  URL https://github.com/CGAL/cgal/archive/refs/tags/v${CGAL_VERSION}.tar.gz
  UPDATE_COMMAND ""
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CATKIN_DEVEL_PREFIX}
    -DCMAKE_BUILD_TYPE:STRING=Release
  BUILD_COMMAND $(MAKE)
  INSTALL_COMMAND $(MAKE) install
)

catkin_package(
  INCLUDE_DIRS ${CATKIN_DEVEL_PREFIX}/include
  CFG_EXTRAS cgal-extras.cmake
)

install(DIRECTORY ${CATKIN_DEVEL_PREFIX}/include/CGAL/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}/CGAL/
)

