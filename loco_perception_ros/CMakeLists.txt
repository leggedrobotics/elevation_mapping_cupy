# Project configuration
cmake_minimum_required (VERSION 2.8)
project(loco_perception_ros)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    loco_ros
    loco_perception_utils
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES loco_perception_ros
  CATKIN_DEPENDS
    loco_ros
    loco_perception_utils
#  DEPENDS system_lib
)

find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS timer REQUIRED)

add_definitions(--std=c++11)

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_library(loco_perception_ros SHARED
   src/ConvexSetSequence.cpp
)

add_dependencies(loco_perception_ros ${catkin_EXPORTED_TARGETS})

target_link_libraries(loco_perception_ros
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}/${PROJECT_NAME}
        PATTERN ".svn" EXCLUDE
)
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
