//
// Created by rgrandia on 07.06.20.
//

#pragma once

namespace ransac_plane_extractor {

struct RansacPlaneExtractorParameters {
  /// Set probability to miss the largest primitive at each iteration.
  double probability = 0.01;
  /// Detect shapes with at least 200 points.
  double min_points = 200;
  /// Set maximum Euclidean distance between a point and a shape.
  double epsilon = 0.004;
  /// Set maximum Euclidean distance between points to be clustered. Two points are connected if separated by a distance of at most 2*sqrt(2)*cluster_epsilon = 2.828 * cluster_epsilon
  double cluster_epsilon = 0.03;
  /// Set maximum normal deviation. normal_threshold < dot(surface_normal, point_normal);
  double normal_threshold = 0.98;
};

}  // namespace ransac_plane_extractor
