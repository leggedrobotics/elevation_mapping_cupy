#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_CONVEX_DECOMPOSER_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_CONVEX_DECOMPOSER_HPP_

#include "polygon.hpp"

namespace convex_plane_extraction {

enum class ConvexDecompositionType{
  kGreeneOptimalDecomposition = 1,
  kInnerConvexApproximation = 2
};

struct ConvexDecomposerParameters{
  ConvexDecompositionType decomposition_type = ConvexDecompositionType::kGreeneOptimalDecomposition;
  double dent_angle_threshold_rad = 0.001; // 0.05 deg in rad.
};

class ConvexDecomposer {
 public:
  explicit ConvexDecomposer(const ConvexDecomposerParameters& parameters);

  CgalPolygon2dContainer performConvexDecomposition(const CgalPolygon2d& polygon) const;

 private:
  std::multimap<double, int> detectDentLocations(const CgalPolygon2d& polygon) const;

  CgalPolygon2dContainer performInnerConvexApproximation(const CgalPolygon2d& polygon) const;

  CgalPolygon2dContainer performOptimalConvexDecomposition(const CgalPolygon2d& polygon) const;

  ConvexDecomposerParameters parameters_;
};

} // namespace convex_plane_extraction

#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_CONVEX_DECOMPOSER_HPP_
