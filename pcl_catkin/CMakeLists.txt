cmake_minimum_required(VERSION 3.10)
project(pcl_catkin)

find_package(catkin REQUIRED)

include(ExternalProject)

set(VERSION 1.10.0)

file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/include)

set(PCL_VERSION 1.10.0)
ExternalProject_Add(pcl
  URL https://github.com/PointCloudLibrary/pcl/archive/pcl-${PCL_VERSION}.tar.gz
  UPDATE_COMMAND ""
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CATKIN_DEVEL_PREFIX}
    -DCMAKE_BUILD_TYPE:STRING=Release
  BUILD_COMMAND $(MAKE)
  INSTALL_COMMAND $(MAKE) install
)

catkin_package(
  INCLUDE_DIRS ${CATKIN_DEVEL_PREFIX}/include
  CFG_EXTRAS pcl-extras.cmake
)

install(DIRECTORY ${CATKIN_DEVEL_PREFIX}/include/pcl-1.10/pcl/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}/pcl/
)

