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

#include "grid_map_ros/grid_map_ros.hpp"

#include "point_with_normal_container.hpp"




namespace ransac_plane_extractor {

    using namespace point_with_normal_container;
    // Type declarations.
    typedef CGAL::First_of_pair_property_map<PointWithNormal> PointMap;
    typedef CGAL::Second_of_pair_property_map<PointWithNormal> NormalMap;
    typedef CGAL::Shape_detection::Efficient_RANSAC_traits
            <Kernel, PwnVector, PointMap, NormalMap> Traits;
    typedef CGAL::Shape_detection::Efficient_RANSAC<Traits> EfficientRansac;
    typedef CGAL::Shape_detection::Plane<Traits> Plane;


    struct RansacParameters{
      double probability;
      double min_points;
      double epsilon;
      double cluster_epsilon;
      double normal_threshold;
    };

    class RansacPlaneExtractor {
    public:

        /*!
        * Constructor.
        */
        explicit RansacPlaneExtractor(grid_map::GridMap &map, double resolution, const std::string &normals_layer_prefix,
                             const std::string &layer_height);

        /*!
        * Destructor.
        */
        virtual ~RansacPlaneExtractor();

        void setParameters(const RansacParameters& parameters);

        void runDetection();

        void ransacPlaneVisualization();

    private:

        grid_map::GridMap& map_;
        double resolution_;

        PointWithNormalContainer points_with_normal_;

        EfficientRansac ransac_;
        EfficientRansac::Parameters parameters_;

        Eigen::MatrixXf ransac_map_;

    };

}

#endif //GRID_MAP_DEMOS_RANSACPLANEEXTRACTOR_HPP_
