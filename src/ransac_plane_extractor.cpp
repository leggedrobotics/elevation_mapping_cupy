//
// Created by andrej on 11/21/19.
//

#include "ransac_plane_extractor.hpp"

using namespace ransac_plane_extractor;

RansacPlaneExtractor::RansacPlaneExtractor(grid_map::GridMap &map, double resolution, const std::string &normals_layer_prefix,
          const std::string &layer_height)
          : map_(map){
  int map_rows = (map.getSize())(0);
  int map_cols = (map.getSize())(1);
  double resolution_ = resolution;
  ransac_map_ = Eigen::MatrixXf::Zero(map_rows, map_cols);
  for (grid_map::GridMapIterator iterator(map_);
       !iterator.isPastEnd(); ++iterator) {
      grid_map::Position3 position3;
      Eigen::Vector3d point;
      Eigen::Vector3d normal;
      map.getPosition3(layer_height, *iterator, point);
      map.getVector(normals_layer_prefix, *iterator, normal);
      points_with_normal_.addPointNormalPair(point, normal);
  }
  std::cout << "Before reference handover!" << std::endl;
  ransac_.set_input(points_with_normal_.getReference());
  ransac_.add_shape_factory<Plane>();
  // Set parameters for shape detection.
  // Set probability to miss the largest primitive at each iteration.
  parameters_.probability = 0.01;

  // Detect shapes with at least 500 points.
  parameters_.min_points = 200;
  // Set maximum Euclidean distance between a point and a shape.
  parameters_.epsilon = 0.004;

  // Set maximum Euclidean distance between points to be clustered.
  parameters_.cluster_epsilon = 0.0282842712475 ;

  // Set maximum normal deviation.
  // 0.9 < dot(surface_normal, point_normal);
  parameters_.normal_threshold = 0.98;
}

RansacPlaneExtractor::~RansacPlaneExtractor(){}

void RansacPlaneExtractor::runDetection(){
  // Detect shapes.
  ransac_.detect(parameters_);
}

void RansacPlaneExtractor::setParameters(const RansacParameters& parameters){
  // Set probability to miss the largest primitive at each iteration.
  parameters_.probability = parameters.probability;
  // Detect shapes with at least min_points number of points.
  parameters_.min_points = parameters.min_points;
  // Set maximum Euclidean distance between a point and a shape.
  parameters_.epsilon = parameters.epsilon;

  // Set maximum Euclidean distance between points to be clustered.
  parameters_.cluster_epsilon = parameters.cluster_epsilon;

  // Set maximum normal deviation.
  // Example: 0.9 < dot(surface_normal, point_normal).
  parameters_.normal_threshold = parameters.normal_threshold;
}

void RansacPlaneExtractor::ransacPlaneVisualization(){
  Eigen::MatrixXf plane_map = Eigen::MatrixXf::Zero(map_.getSize().x(), map_.getSize().y());
  int32_t plane_label_iterator = 1;
  std::cout << "Extracted " << ransac_.shapes().size() << " planes." << std::endl;
  for (auto plane : ransac_.shapes()){
    const std::vector<std::size_t>& plane_points =  (*plane.get()).indices_of_assigned_points();
    auto plane_points_it = plane_points.begin();
    for (; plane_points_it != plane_points.end(); ++plane_points_it){
      Point_3D& point = points_with_normal_.getPoint(*plane_points_it);
      Eigen::Array2i map_indices;
      map_.getIndex(Eigen::Vector2d(point.x(), point.y()), map_indices);
      ransac_map_(map_indices(0), map_indices(1)) = plane_label_iterator;
    }
    ++plane_label_iterator;
  }
  map_.add("ransac_planes", ransac_map_);
  std::cout << "Added ransac plane layer!" << std::endl;
}


