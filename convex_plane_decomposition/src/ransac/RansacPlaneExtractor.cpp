#include "convex_plane_decomposition/ransac/RansacPlaneExtractor.hpp"

namespace ransac_plane_extractor {

RansacPlaneExtractor::RansacPlaneExtractor(const RansacPlaneExtractorParameters& parameters) {
  setParameters(parameters);
  ransac_.add_shape_factory<Plane>();
}

void RansacPlaneExtractor::setParameters(const RansacPlaneExtractorParameters& parameters) {
  cgalRansacParameters_.probability = parameters.probability;
  cgalRansacParameters_.min_points = parameters.min_points;
  cgalRansacParameters_.epsilon = parameters.epsilon;
  cgalRansacParameters_.cluster_epsilon = parameters.cluster_epsilon;
  cgalRansacParameters_.normal_threshold = std::cos(parameters.normal_threshold * M_PI / 180.0);
}

void RansacPlaneExtractor::detectPlanes(std::vector<PointWithNormal>& points_with_normal) {
  ransac_.set_input(points_with_normal);
  ransac_.detect(cgalRansacParameters_);
}

}  // namespace ransac_plane_extractor
