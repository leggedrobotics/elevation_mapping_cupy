#include "convex_plane_decomposition/ransac_plane_extractor.hpp"

using namespace ransac_plane_extractor;

RansacPlaneExtractor::RansacPlaneExtractor(std::vector<PointWithNormal>& points_with_normal, const RansacPlaneExtractorParameters& parameters){
  ransac_.set_input(points_with_normal);
  ransac_.add_shape_factory<Plane>();
  parameters_.probability = parameters.probability;
  parameters_.min_points = parameters.min_points;
  parameters_.epsilon = parameters.epsilon;
  parameters_.cluster_epsilon = parameters.cluster_epsilon;
  parameters_.normal_threshold = parameters.normal_threshold;
}

void RansacPlaneExtractor::runDetection(){
  // Detect shapes.
  ransac_.detect(parameters_);
}

void RansacPlaneExtractor::setParameters(const RansacPlaneExtractorParameters& parameters){
  parameters_.probability = parameters.probability;
  parameters_.min_points = parameters.min_points;
  parameters_.epsilon = parameters.epsilon;
  parameters_.cluster_epsilon = parameters.cluster_epsilon;
  parameters_.normal_threshold = parameters.normal_threshold;
}


