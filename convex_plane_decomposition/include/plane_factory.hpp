#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_PLANE_FACTORY_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_PLANE_FACTORY_HPP_

#include <cmath>
#include <vector>

#include "convex_decomposer.hpp"
#include "plane.hpp"
#include "polygonizer.hpp"

namespace convex_plane_extraction {

struct PlaneFactoryParameters{
  PolygonizerParameters polygonizer_parameters = PolygonizerParameters();
  ConvexDecomposerParameters convex_decomposer_parameters = ConvexDecomposerParameters();
  double plane_inclination_threshold_degrees = 70;
};

class PlaneFactory {
 public:

  PlaneFactory(grid_map::GridMap& map, const PlaneFactoryParameters& parameters)
  :map_(map),
   parameters_(parameters){
    computeMapTransformation();
  };

  void createPlanesFromLabeledImageAndPlaneParameters(const cv::Mat& labeled_image, const int number_of_labels,
      const std::map<int, convex_plane_extraction::PlaneParameters>& plane_parameters);

  void decomposePlanesInConvexPolygons();

  Polygon3dVectorContainer getConvexPolygonsInWorldFrame() const;

  Polygon3dVectorContainer getPlaneContoursInWorldFrame() const;

 private:

  void computeMapTransformation();

  Eigen::Vector3d convertPlanePointToWorldFrame(const CgalPoint2d& point, const PlaneParameters& plane_parameters) const;

  Polygon3dVectorContainer convertPlanePolygonsToWorldFrame(const CgalPolygon2dContainer& polygons,
                                                            const PlaneParameters& plane_parameters) const;

  Polygon3dVectorContainer convertPlanePolygonToWorldFrame(const CgalPolygon2d& polygon, const PlaneParameters& plane_parameters) const;

  bool isPlaneInclinationBelowThreshold(const Eigen::Vector3d& plane_normal_vector) const;

  // Grid map related members.
  grid_map::GridMap& map_;
  Eigen::Matrix2d transformation_xy_to_world_frame_;
  Eigen::Vector2d map_offset_;

  // Parameters.
  PlaneFactoryParameters parameters_;

  std::vector<Plane> planes_;

};
} // namespace convex_plane_extraction
#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_PLANE_FACTORY_HPP_
