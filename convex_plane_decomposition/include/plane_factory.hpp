#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_PLANE_FACTORY_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_PLANE_FACTORY_HPP_

#include <vector>

#include "plane.hpp"

namespace convex_plane_extraction {

class PlaneFactory {
 public:

  PlaneFactory(grid_map::GridMap& map)
  :map_(map){
    computeMapTransformation();
  };

  void computeMapTransformation();

  void createPlanesFromLabeledImageAndPlaneParameters(const cv::Mat& labeled_image,
      const std::map<int, convex_plane_extraction::PlaneParameters>& parameters);

 private:

  // Grid map related members.
  grid_map::GridMap& map_;
  Eigen::Matrix2d transformation_xy_to_world_frame_;
  Eigen::Vector2d map_offset_;


  std::vector<Plane> planes_;

};
} // namespace convex_plane_extraction
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_PLANE_FACTORY_HPP_
