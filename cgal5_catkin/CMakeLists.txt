cmake_minimum_required(VERSION 3.10)
project(cgal5_catkin)

find_package(catkin REQUIRED)

include(ExternalProject)

set(VERSION 5.0.2)

file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/include)

# Cpp standard version
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CGAL_VERSION 5.0.2)
ExternalProject_Add(cgal
  URL https://github.com/CGAL/cgal/archive/releases/CGAL-${CGAL_VERSION}.tar.gz
  UPDATE_COMMAND ""
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CATKIN_DEVEL_PREFIX}
    -DWITH_CGAL_Qt5:BOOL=OFF
    -DWITH_CGAL_ImageIO:BOOL=OFF
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

