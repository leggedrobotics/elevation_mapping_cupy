cmake_minimum_required(VERSION 3.10)
project(pcl_catkin)

find_package(catkin REQUIRED)

file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/lib/libpcl_visualization.so DESTINATION ${CATKIN_DEVEL_PREFIX}/lib)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/lib/libpcl_visualization.so.1.10 DESTINATION ${CATKIN_DEVEL_PREFIX}/lib)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/lib/libpcl_visualization.so.1.10.0 DESTINATION ${CATKIN_DEVEL_PREFIX}/lib)

file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/pkgconfig)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/lib/pkgconfig/pcl_visualization-1.10.pc DESTINATION ${CATKIN_DEVEL_PREFIX}/lib/pkgconfig)

catkin_package(
  INCLUDE_DIRS
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

