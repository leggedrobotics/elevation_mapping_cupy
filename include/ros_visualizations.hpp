#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_ROS_VISUALIZATIONS_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_ROS_VISUALIZATIONS_HPP_
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"

#include  "geometry_msgs/Point32.h"
#include  "geometry_msgs/PolygonStamped.h"
#include "jsk_recognition_msgs/PolygonArray.h"

#include "plane.hpp"

namespace convex_plane_extraction{

  void addRosPolygons(const Polygon3dVectorContainer& input_polygons, jsk_recognition_msgs::PolygonArray* polygon_array);

}

#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_ROS_VISUALIZATIONS_HPP_
