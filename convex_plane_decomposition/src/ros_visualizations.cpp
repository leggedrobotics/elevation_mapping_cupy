#include "ros_visualizations.hpp"

namespace convex_plane_extraction {

  void addRosPolygons(const Polygon3dVectorContainer &input_polygons, jsk_recognition_msgs::PolygonArray *polygon_array) {

    polygon_array->header.stamp = ros::Time::now();
    polygon_array->header.frame_id = "odom";
    for (const auto &polygon : input_polygons) {
      if (polygon.empty()) {
        continue;
      }
      geometry_msgs::PolygonStamped polygon_stamped;
      polygon_stamped.header.stamp = ros::Time::now();
      polygon_stamped.header.frame_id = "odom";
      for (const auto &point : polygon) {
        geometry_msgs::Point32 point_ros;
        point_ros.x = point.x();
        point_ros.y = point.y();
        point_ros.z = point(2);
        polygon_stamped.polygon.points.push_back(point_ros);
      }
      polygon_array->polygons.push_back(polygon_stamped);
    }
  }
}

