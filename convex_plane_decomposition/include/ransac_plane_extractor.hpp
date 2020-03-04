//
// Created by andrej on 11/21/19.
//

#ifndef CONVEX_PLANE_EXTRACTION_RANSACPLANEEXTRACTOR_HPP_
#define CONVEX_PLANE_EXTRACTION_RANSACPLANEEXTRACTOR_HPP_

#if defined (_MSC_VER) && !defined (_WIN64)
#pragma warning(disable:4244) // boost::number_distance::distance()
                              // converts 64 to 32 bits integers
#endif

#include <iostream>
#include <vector>

#include "CGAL/property_map.h"
#include "CGAL/Point_with_normal_3.h"
#include "CGAL/Exact_predicates_inexact_constructions_kernel.h"
#include "CGAL/Shape_detection/Efficient_RANSAC.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include <glog/logging.h>


namespace ransac_plane_extractor {

  // Point with normal related type declarations.
  using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
  using Point3D = Kernel::Point_3;
  using Vector3D = Kernel::Vector_3;
  using PointWithNormal = std::pair<Kernel::Point_3, Kernel::Vector_3>;
  using PwnVector =  std::vector<PointWithNormal>;

  // RANSAC plane extractor related type declarations.
  using PointMap = CGAL::First_of_pair_property_map<PointWithNormal>;
  using NormalMap = CGAL::Second_of_pair_property_map<PointWithNormal>;
  using Traits = CGAL::Shape_detection::Efficient_RANSAC_traits
          <Kernel, PwnVector, PointMap, NormalMap>;
  using EfficientRansac = CGAL::Shape_detection::Efficient_RANSAC<Traits>;
  using Plane = CGAL::Shape_detection::Plane<Traits>;

    struct RansacPlaneExtractorParameters{
      // Set probability to miss the largest primitive at each iteration.
      double probability = 0.01;
      // Detect shapes with at least 200 points.
      double min_points = 200;
      // Set maximum Euclidean distance between a point and a shape.
      double epsilon = 0.004;
      // Set maximum Euclidean distance between points to be clustered.
      double cluster_epsilon = 0.0282842712475;
      // Set maximum normal deviation. 0.98 < dot(surface_normal, point_normal);
      double normal_threshold = 0.98;
    };

    class RansacPlaneExtractor {
    public:

        RansacPlaneExtractor(std::vector<PointWithNormal>& points_with_normal, const RansacPlaneExtractorParameters& parameters);

        void setParameters(const RansacPlaneExtractorParameters& parameters);

        void runDetection();

      auto getDetectedPlanes() const{
        return ransac_.shapes();
      };

//        void ransacPlaneVisualization();

    private:

        EfficientRansac ransac_;
        EfficientRansac::Parameters parameters_;

    };

}

#endif //GRID_MAP_DEMOS_RANSACPLANEEXTRACTOR_HPP_
