cmake_minimum_required(VERSION 3.10)
project(convex_plane_decomposition_msgs)

set(CATKIN_PACKAGE_DEPENDENCIES
  std_msgs
  geometry_msgs
  grid_map_msgs
)

find_package(catkin REQUIRED
  COMPONENTS
    ${CATKIN_PACKAGE_DEPENDENCIES}
    message_generation
)

add_message_files(
  FILES
    BoundingBox2d.msg
    PlanarRegion.msg
    PlanarTerrain.msg
    Point2d.msg
    Polygon2d.msg
    PolygonWithHoles2d.msg
)

generate_messages(
  DEPENDENCIES
    ${CATKIN_PACKAGE_DEPENDENCIES}
)

catkin_package(
  CATKIN_DEPENDS
    ${CATKIN_PACKAGE_DEPENDENCIES}
    message_runtime
)
