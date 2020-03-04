#include "pipeline_ros.hpp"

using namespace convex_plane_extraction;

PipelineParameters loadPipelineParameters(ros::NodeHandle& nodeHandle, grid_map::GridMap& map){

  // Grid map parameters.
  GridMapParameters grid_map_parameters = loadGridMapParameters(nodeHandle, map);

  // Sliding Window Plane Extractor parameters.
  sliding_window_plane_extractor::SlidingWindowPlaneExtractorParameters sliding_window_plane_extractor_parameters;
  const std::string kSlidingWindowParametersPrefix = "/sliding_window_plane_extractor/";
  if (!nodeHandle.getParam(kSlidingWindowParametersPrefix + "kernel_size", sliding_window_plane_extractor_parameters.kernel_size)) {
    ROS_ERROR("Could not read parameter `kernel_size`. Setting parameter to default value.");
  }
  if (!nodeHandle.getParam(kSlidingWindowParametersPrefix + "plane_patch_error_threshold", sliding_window_plane_extractor_parameters.plane_patch_error_threshold)) {
    ROS_ERROR("Could not read parameter `plane_patch_error_threshold`. Setting parameter to default value.");
  }
  if (!nodeHandle.getParam(kSlidingWindowParametersPrefix + "include_curvature_detection", sliding_window_plane_extractor_parameters.include_curvature_detection)) {
    ROS_ERROR("Could not read parameter `include_curvature_detection`. Setting parameter to default value.");
  }
  if (!nodeHandle.getParam(kSlidingWindowParametersPrefix + "include_ransac_refinement", sliding_window_plane_extractor_parameters.include_ransac_refinement)) {
    ROS_ERROR("Could not read parameter `include_ransac_refinement`. Setting parameter to default value.");
  }
  if (!nodeHandle.getParam(kSlidingWindowParametersPrefix + "global_plane_fit_error_threshold", sliding_window_plane_extractor_parameters.global_plane_fit_error_threshold)) {
    ROS_ERROR("Could not read parameter `global_plane_fit_error_threshold`. Setting parameter to default value.");
  }
  if(sliding_window_plane_extractor_parameters.include_ransac_refinement){
    // RASNAC refinement related parameters.
    ransac_plane_extractor::RansacPlaneExtractorParameters ransac_plane_extractor_parameters;
    const std::string kRansacRefinementParameterPrefix = "/ransac_plane_refinement/";
    if (!nodeHandle.getParam(kRansacRefinementParameterPrefix + "probability", ransac_plane_extractor_parameters.probability)) {
      LOG(WARNING) << "Could not read parameter" << kRansacRefinementParameterPrefix << "probability. Setting parameter to default value.";
    }
    if (!nodeHandle.getParam(kRansacRefinementParameterPrefix + "min_points", ransac_plane_extractor_parameters.min_points)) {
      LOG(WARNING) << "Could not read parameter" << kRansacRefinementParameterPrefix << "ransac_min_points. Setting parameter to default value.";
    }
    if (!nodeHandle.getParam(kRansacRefinementParameterPrefix + "epsilon", ransac_plane_extractor_parameters.epsilon)) {
      ROS_ERROR("Could not read parameter `epsilon`. Setting parameter to default value.");
    }
    if (!nodeHandle.getParam(kRansacRefinementParameterPrefix + "cluster_epsilon", ransac_plane_extractor_parameters.cluster_epsilon)) {
      ROS_ERROR("Could not read parameter `cluster_epsilon`. Setting parameter to default value.");
    }
    if (!nodeHandle.getParam(kRansacRefinementParameterPrefix + "normal_threshold", ransac_plane_extractor_parameters.normal_threshold)) {
      ROS_ERROR("Could not read parameter `normal_threshold`. Setting parameter to default value.");
    }
    sliding_window_plane_extractor_parameters.ransac_parameters = ransac_plane_extractor_parameters;
  }

  // Polygonizer parameters.
  PolygonizerParameters polygonizer_parameters;
  polygonizer_parameters.resolution = grid_map_parameters.resolution;
  const std::string kPolygonizerParametersPrefix = "/polygonizer/";
  if (!nodeHandle.getParam(kPolygonizerParametersPrefix + "upsampling_factor", polygonizer_parameters.resolution)) {
    ROS_ERROR("Could not read parameter `normal_threshold`. Setting parameter to default value.");
  }
  if (!nodeHandle.getParam(kPolygonizerParametersPrefix + "activate_long_edge_upsampling", polygonizer_parameters.activate_long_edge_upsampling)) {
    ROS_ERROR("Could not read parameter `activate_long_edge_upsampling`. Setting parameter to default value.");
  }
  if (!nodeHandle.getParam(kPolygonizerParametersPrefix + "activate_contour_approximation", polygonizer_parameters.activate_contour_approximation)) {
    ROS_ERROR("Could not read parameter `activate_contour_approximation`. Setting parameter to default value.");
  }
  if (polygonizer_parameters.activate_contour_approximation){
    if (!nodeHandle.getParam(kPolygonizerParametersPrefix + "contour_approximation_deviation_threshold", polygonizer_parameters.contour_approximation_deviation_threshold)) {
      ROS_ERROR("Could not read parameter `contour_approximation_deviation_threshold`. Setting parameter to default value.");
    }
  }
  if (!nodeHandle.getParam(kPolygonizerParametersPrefix + "hole_area_threshold_squared_meters", polygonizer_parameters.hole_area_threshold_squared_meters)) {
    ROS_ERROR("Could not read parameter `hole_area_threshold_squared_meters`. Setting parameter to default value.");
  }

  // Convex decomposer parameters.
  ConvexDecomposerParameters convex_decomposer_parameters;
  const std::string kConvexDecomposerParametersPrefix = "/convex_decomposer/";
  std::string decomposer_type;
  if (!nodeHandle.getParam(kConvexDecomposerParametersPrefix + "decomposer_type", decomposer_type)) {
    ROS_ERROR("Could not read parameter `decomposer_type`. Setting parameter to default value.");
  }
  if(decomposer_type == "optimal_decompostion"){
    convex_decomposer_parameters.decomposition_type = ConvexDecompositionType::kGreeneOptimalDecomposition;
  } else if (decomposer_type == "inner_convex_decomposition"){
    convex_decomposer_parameters.decomposition_type = ConvexDecompositionType::kInnerConvexApproximation;
  }
  if (!nodeHandle.getParam(kConvexDecomposerParametersPrefix + "dent_angle_threshold_rad", convex_decomposer_parameters.dent_angle_threshold_rad)) {
    ROS_ERROR("Could not read parameter `dent_angle_threshold_rad`. Setting parameter to default value.");
  }

  // Plane Factory parameters.
  PlaneFactoryParameters plane_factory_parameters;
  const std::string kPlaneFactoryParametersPrefix = "/plane_factory/";
  if (!nodeHandle.getParam(kPlaneFactoryParametersPrefix + "plane_inclination_threshold_degrees", plane_factory_parameters.plane_inclination_threshold_degrees)) {
    ROS_ERROR("Could not read parameter `plane_inclination_threshold_degrees`. Setting parameter to default value.");
  }
  plane_factory_parameters.polygonizer_parameters = polygonizer_parameters;
  plane_factory_parameters.convex_decomposer_parameters = convex_decomposer_parameters;

  // Pipeline parameters.
  PipelineParameters pipeline_parameters;
  pipeline_parameters.sliding_window_plane_extractor_parameters = sliding_window_plane_extractor_parameters;
  pipeline_parameters.plane_factory_parameters = plane_factory_parameters;

  return pipeline_parameters;
}

jsk_recognition_msgs::PolygonArray PipelineROS::getConvexPolygons() const{
  const Polygon3dVectorContainer convex_polygon_buffer = pipeline_.getConvexPolygons();
  return convertToRosPolygons(convex_polygon_buffer);
}

jsk_recognition_msgs::PolygonArray PipelineROS::getOuterPlaneContours() const{
  const Polygon3dVectorContainer plane_contour_buffer = pipeline_.getPlaneContours();
  return convertToRosPolygons(plane_contour_buffer);
}

void PipelineROS::augmentGridMapWithSegmentation(grid_map::GridMap& map){
  Eigen::MatrixXf new_layer = Eigen::MatrixXf::Zero(map.getSize().x(), map.getSize().y());
  cv::cv2eigen(pipeline_.getSegmentationImage(), new_layer);
  map.add("plane_segmentation", new_layer);
  VLOG(1) << "Added plane segmentation to Grid Map!";
}

