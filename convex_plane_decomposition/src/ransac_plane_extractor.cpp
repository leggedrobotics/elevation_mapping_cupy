#include "ransac_plane_extractor.hpp"

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

//void RansacPlaneExtractor::ransacPlaneVisualization(){
//  Eigen::MatrixXf plane_map = Eigen::MatrixXf::Zero(map_.getSize().x(), map_.getSize().y());
//  int32_t plane_label_iterator = 1;
//  for (auto plane : ransac_.shapes()){
//    const std::vector<std::size_t>& plane_points =  (*plane.get()).indices_of_assigned_points();
//    auto plane_points_it = plane_points.begin();
//    for (; plane_points_it != plane_points.end(); ++plane_points_it){
//      Point_3D& point = points_with_normal_.getPoint(*plane_points_it);
//      Eigen::Array2i map_indices;
//      map_.getIndex(Eigen::Vector2d(point.x(), point.y()), map_indices);
//      ransac_map_(map_indices(0), map_indices(1)) = 1;//plane_label_iterator;
//    }
//    ++plane_label_iterator;
//  }
//  std::ofstream output_file;
//  output_file.open("/home/andrej/Desktop/number_of_planes_ransac_12.txt", std::ofstream::app);;
//  output_file << plane_label_iterator << "\n";
//  output_file.close();
//  map_.add("ransac_planes", ransac_map_);
//  std::cout << "Added ransac plane layer!" << std::endl;
//}


