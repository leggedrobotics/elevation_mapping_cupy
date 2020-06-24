//
// Created by rgrandia on 07.06.20.
//

#pragma once

namespace convex_plane_decomposition {
namespace sliding_window_plane_extractor {

struct SlidingWindowPlaneExtractorParameters {
  /// Size of the sliding window patch used for normal vector calculation and planarity detection
  /// Should be an odd number and at least 3.
  int kernel_size = 3;

  /// [deg] Maximum allowed angle between the surface normal and the world-z direction for a patch
  double plane_inclination_threshold_degrees = 70.0;

  /// [m] The allowed root-mean-squared deviation from the plane fitted to the patch. Higher -> not planar
  double plane_patch_error_threshold = 0.005;

  /// [#] Labels with less points assigned to them are discarded
  int min_number_points_per_label = 10;

  /// Label kernel connectivity. 4 or 8 (cross or box)
  int connectivity = 4;

  /// Enable RANSAC refinement if the plane is not globally fitting to the assigned points.
  bool include_ransac_refinement = true;

  /// [m] Allowed maximum distance from a point to the plane. If any point violates this, RANSAC is triggered
  double global_plane_fit_distance_error_threshold = 0.04;

  /// [deg] Allowed normal vector deviation for a point w.r.t. the plane normal. If any point violates this, RANSAC is triggered
  double global_plane_fit_angle_error_threshold_degrees = 5.0;
};

}  // namespace sliding_window_plane_extractor
}  // namespace convex_plane_decomposition
