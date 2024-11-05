//
// Copyright (c) 2022, Takahiro Miki. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//


#include "elevation_mapping_cupy/elevation_mapping_ros.hpp"

// Pybind
#include <pybind11/eigen.h>

// ROS2
#include <geometry_msgs/msg/point32.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

// PCL
#include <pcl/common/projection_matrix.h>

#include <elevation_map_msgs/msg/statistics.hpp>

namespace elevation_mapping_cupy {

std::vector<std::string> extract_unique_names(const std::map<std::string, rclcpp::Parameter>& subscriber_params) {
    std::set<std::string> unique_names_set;
    for (const auto& param : subscriber_params) {
        std::size_t pos = param.first.find('.');
        if (pos != std::string::npos) {
            std::string name = param.first.substr(0, pos);
            unique_names_set.insert(name);
        }
    }
    return std::vector<std::string>(unique_names_set.begin(), unique_names_set.end());
}


ElevationMappingNode::ElevationMappingNode()
    : rclcpp::Node("elevation_mapping_node", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)),     
      node_(std::shared_ptr<ElevationMappingNode>(this, [](auto *) {})),
      // it_(node_),
      lowpassPosition_(0, 0, 0),
      lowpassOrientation_(0, 0, 0, 1),
      positionError_(0),
      orientationError_(0),
      positionAlpha_(0.1),
      orientationAlpha_(0.1),
      enablePointCloudPublishing_(false),
      isGridmapUpdated_(false){
      RCLCPP_INFO(this->get_logger(), "Initializing ElevationMappingNode...");

  tfBroadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);// ROS2构造TransformBroadcaster
  tfBuffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tfListener_ = std::make_shared<tf2_ros::TransformListener>(*tfBuffer_);
  
  
  std::string pose_topic, map_frame;
  std::vector<std::string> map_topics;
  double recordableFps, updateVarianceFps, timeInterval, updatePoseFps, updateGridMapFps, publishStatisticsFps;
  bool enablePointCloudPublishing(false);
  
  py::gil_scoped_acquire acquire;
  auto math = py::module::import("math");
  double root_two = math.attr("sqrt")(2.0).cast<double>();  
  RCLCPP_INFO(this->get_logger(), "The square root of 2 is: %f", root_two);

  this->get_parameter("initialize_frame_id", initialize_frame_id_);
  this->get_parameter("initialize_tf_offset", initialize_tf_offset_);  
  this->get_parameter("pose_topic", pose_topic);
  this->get_parameter("map_frame", mapFrameId_);
  this->get_parameter("base_frame", baseFrameId_);
  this->get_parameter("corrected_map_frame", correctedMapFrameId_);
  this->get_parameter("initialize_method", initializeMethod_);
  this->get_parameter("position_lowpass_alpha", positionAlpha_);
  this->get_parameter("orientation_lowpass_alpha", orientationAlpha_);
  this->get_parameter("recordable_fps", recordableFps);
  this->get_parameter("update_variance_fps", updateVarianceFps);
  this->get_parameter("time_interval", timeInterval);
  this->get_parameter("update_pose_fps", updatePoseFps);
  this->get_parameter("initialize_tf_grid_size", initializeTfGridSize_);
  this->get_parameter("map_acquire_fps", updateGridMapFps);
  this->get_parameter("publish_statistics_fps", publishStatisticsFps);
  this->get_parameter("enable_pointcloud_publishing", enablePointCloudPublishing);
  this->get_parameter("enable_normal_arrow_publishing", enableNormalArrowPublishing_);
  this->get_parameter("enable_drift_corrected_TF_publishing", enableDriftCorrectedTFPublishing_);
  this->get_parameter("use_initializer_at_start", useInitializerAtStart_);
  this->get_parameter("always_clear_with_initializer", alwaysClearWithInitializer_);

  RCLCPP_INFO(this->get_logger(), "initialize_frame_id: %s", initialize_frame_id_.empty() ? "[]" : initialize_frame_id_[0].c_str());
  RCLCPP_INFO(this->get_logger(), "initialize_tf_offset: [%f, %f, %f, %f]", initialize_tf_offset_[0], initialize_tf_offset_[1], initialize_tf_offset_[2], initialize_tf_offset_[3]);
  RCLCPP_INFO(this->get_logger(), "pose_topic: %s", pose_topic.c_str());
  RCLCPP_INFO(this->get_logger(), "map_frame: %s", mapFrameId_.c_str());
  RCLCPP_INFO(this->get_logger(), "base_frame: %s", baseFrameId_.c_str());
  RCLCPP_INFO(this->get_logger(), "corrected_map_frame: %s", correctedMapFrameId_.c_str());
  RCLCPP_INFO(this->get_logger(), "initialize_method: %s", initializeMethod_.c_str());
  RCLCPP_INFO(this->get_logger(), "position_lowpass_alpha: %f", positionAlpha_);
  RCLCPP_INFO(this->get_logger(), "orientation_lowpass_alpha: %f", orientationAlpha_);
  RCLCPP_INFO(this->get_logger(), "recordable_fps: %f", recordableFps);
  RCLCPP_INFO(this->get_logger(), "update_variance_fps: %f", updateVarianceFps);
  RCLCPP_INFO(this->get_logger(), "time_interval: %f", timeInterval);
  RCLCPP_INFO(this->get_logger(), "update_pose_fps: %f", updatePoseFps);
  RCLCPP_INFO(this->get_logger(), "initialize_tf_grid_size: %f", initializeTfGridSize_);
  RCLCPP_INFO(this->get_logger(), "map_acquire_fps: %f", updateGridMapFps);
  RCLCPP_INFO(this->get_logger(), "publish_statistics_fps: %f", publishStatisticsFps);
  RCLCPP_INFO(this->get_logger(), "enable_pointcloud_publishing: %s", enablePointCloudPublishing ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "enable_normal_arrow_publishing: %s", enableNormalArrowPublishing_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "enable_drift_corrected_TF_publishing: %s", enableDriftCorrectedTFPublishing_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "use_initializer_at_start: %s", useInitializerAtStart_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "always_clear_with_initializer: %s", alwaysClearWithInitializer_ ? "true" : "false");

  enablePointCloudPublishing_ = enablePointCloudPublishing;
     
  // map_ = std::make_shared<ElevationMappingWrapper>();
  // map_->initialize(node_);



  std::map<std::string, rclcpp::Parameter> subscriber_params, publisher_params;  
  if (!this->get_parameters("subscribers", subscriber_params)) {
    RCLCPP_FATAL(this->get_logger(), "There aren't any subscribers to be configured, the elevation mapping cannot be configured. Exit");
    rclcpp::shutdown();
  }
  if (!this->get_parameters("publishers", publisher_params)) {
    RCLCPP_FATAL(this->get_logger(), "There aren't any publishers to be configured, the elevation mapping cannot be configured. Exit");
    rclcpp::shutdown();
  }
    
   
    auto unique_sub_names = extract_unique_names(subscriber_params);
    for (const auto& sub_name : unique_sub_names) {  
        std::string data_type;
      if(this->get_parameter("subscribers." + sub_name + ".data_type", data_type)){
          // image           
          if(data_type == "image"){
            std::string camera_topic;
            std::string info_topic;
            this->get_parameter("subscribers." + sub_name + ".topic_name", camera_topic);
            this->get_parameter("subscribers." + sub_name + ".info_name", info_topic);
            RCLCPP_INFO(this->get_logger(), "camera_topic  %s: %s", sub_name.c_str(), camera_topic.c_str());
            RCLCPP_INFO(this->get_logger(), "info_name  %s: %s", sub_name.c_str(), info_topic.c_str());

            // std::string transport_hint = "compressed";
            // std::size_t ind = camera_topic.find(transport_hint);  // Find if compressed is in the topic name
            // if (ind != std::string::npos) {
            //   transport_hint = camera_topic.substr(ind, camera_topic.length());  // Get the hint as the last part
            //   camera_topic.erase(ind - 1, camera_topic.length());                // We remove the hint from the topic
            // } else {
            //   transport_hint = "raw";  // In the default case we assume raw topic
            // }                        
            
            ImageSubscriberPtr image_sub = std::make_shared<ImageSubscriber>(this, camera_topic, rmw_qos_profile_sensor_data);                    
            imageSubs_.push_back(image_sub);                                

            CameraInfoSubscriberPtr cam_info_sub = std::make_shared<CameraInfoSubscriber>(this, info_topic, rmw_qos_profile_sensor_data);                                    
            cameraInfoSubs_.push_back(cam_info_sub);  

            std::string channel_info_topic; 
            if (this->get_parameter("subscribers." + sub_name + ".channel_name", channel_info_topic)) {
              ChannelInfoSubscriberPtr channel_info_sub = std::make_shared<ChannelInfoSubscriber>(this, channel_info_topic, rmw_qos_profile_sensor_data);                                       
              channelInfoSubs_.push_back(channel_info_sub);
              CameraChannelSyncPtr sync = std::make_shared<CameraChannelSync>(CameraChannelPolicy(10), *image_sub, *cam_info_sub, *channel_info_sub);
              sync->registerCallback(std::bind(&ElevationMappingNode::imageChannelCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
              cameraChannelSyncs_.push_back(sync);
              RCLCPP_INFO(this->get_logger(), "Subscribed to Image topic: %s, Camera info topic: %s, Channel info topic: %s", camera_topic.c_str(), info_topic.c_str(), channel_info_topic.c_str());
            }else{
                std::string key = sub_name;
                channels_[key].push_back("rgb");
                // RCLCPP_INFO(this->get_logger(), "Subscribed to Image topic: %s, Camera info topic: %s. Channel info topic: %s", camera_topic.c_str(), info_topic.c_str(), (channel_info_topic.empty() ? ("Not found. Using channels: " + boost::algorithm::join(channels_[key], ", ")).c_str() : channel_info_topic.c_str()));
                CameraSyncPtr sync = std::make_shared<CameraSync>(CameraPolicy(10), *image_sub, *cam_info_sub);
                sync->registerCallback(std::bind(&ElevationMappingNode::imageCallback, this, std::placeholders::_1, std::placeholders::_2, key));
                cameraSyncs_.push_back(sync);
            }
          }else if(data_type == "pointcloud"){
            std::string pointcloud_topic;
            this->get_parameter("subscribers." + sub_name + ".topic_name", pointcloud_topic);                                    
            std::string key = sub_name;
            channels_[key].push_back("x");
            channels_[key].push_back("y");
            channels_[key].push_back("z");
            auto callback = [this, key](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                  this->pointcloudCallback(msg, key);
              };
              auto sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(pointcloud_topic, 1, callback);
              pointcloudSubs_.push_back(sub);
                RCLCPP_INFO(this->get_logger(), "Subscribed to PointCloud2 topic: %s", pointcloud_topic.c_str());
          }
      }
    }
 
    
    
    auto unique_pub_names = extract_unique_names(publisher_params);

    // for (const auto& pub_name : unique_pub_names) {  
    //   std::string topic_name = pub_name;      
    //   std::vector<std::string> layers_list;
    //   std::vector<std::string> basic_layers_list;      
    //   double fps;
    //   this->get_parameter("publishers." + pub_name + ".layers", layers_list)
    //   this->get_parameter("publishers." + pub_name + ".basic_layers", basic_layers_list)
    //   this->get_parameter("publishers." + pub_name + ".fps", fps)
    // }

    for (const auto& pub_name : unique_pub_names) {  
    std::string topic_name = pub_name;
    double fps;
    std::vector<std::string> layers_list;
    std::vector<std::string> basic_layers_list;      

    this->get_parameter("publishers." + pub_name + ".layers", layers_list);
    this->get_parameter("publishers." + pub_name + ".basic_layers", basic_layers_list);
    this->get_parameter("publishers." + pub_name + ".fps", fps);

    if (fps > updateGridMapFps) {
      RCLCPP_WARN(
        this->get_logger(),
        "[ElevationMappingCupy] fps for topic %s is larger than map_acquire_fps (%f > %f). The topic data will be only updated at %f "
        "fps.",
        topic_name.c_str(), fps, updateGridMapFps, updateGridMapFps);
    }

    // Make publishers
    auto pub = this->create_publisher<grid_map_msgs::msg::GridMap>(topic_name, 1);
    RCLCPP_INFO(this->get_logger(), "Publishing map to topic %s", topic_name.c_str());
    mapPubs_.push_back(pub);

    // Register map layers
    map_layers_.push_back(layers_list);
    map_basic_layers_.push_back(basic_layers_list);

    // Register map fps
    map_fps_.push_back(fps);
    map_fps_unique_.insert(fps);

    }


  setupMapPublishers();

//   pointPub_ = nh_.advertise<sensor_msgs::PointCloud2>("elevation_map_points", 1);
//   alivePub_ = nh_.advertise<std_msgs::Empty>("alive", 1);
//   normalPub_ = nh_.advertise<visualization_msgs::MarkerArray>("normal", 1);
//   statisticsPub_ = nh_.advertise<elevation_map_msgs::Statistics>("statistics", 1);

//   gridMap_.setFrameId(mapFrameId_);
//   rawSubmapService_ = nh_.advertiseService("get_raw_submap", &ElevationMappingNode::getSubmap, this);
//   clearMapService_ = nh_.advertiseService("clear_map", &ElevationMappingNode::clearMap, this);
//   initializeMapService_ = nh_.advertiseService("initialize", &ElevationMappingNode::initializeMap, this);
//   clearMapWithInitializerService_ =
//       nh_.advertiseService("clear_map_with_initializer", &ElevationMappingNode::clearMapWithInitializer, this);
//   setPublishPointService_ = nh_.advertiseService("set_publish_points", &ElevationMappingNode::setPublishPoint, this);
//   checkSafetyService_ = nh_.advertiseService("check_safety", &ElevationMappingNode::checkSafety, this);

//   if (updateVarianceFps > 0) {
//     double duration = 1.0 / (updateVarianceFps + 0.00001);
//     updateVarianceTimer_ = nh_.createTimer(ros::Duration(duration), &ElevationMappingNode::updateVariance, this, false, true);
//   }
//   if (timeInterval > 0) {
//     double duration = timeInterval;
//     updateTimeTimer_ = nh_.createTimer(ros::Duration(duration), &ElevationMappingNode::updateTime, this, false, true);
//   }
//   if (updatePoseFps > 0) {
//     double duration = 1.0 / (updatePoseFps + 0.00001);
//     updatePoseTimer_ = nh_.createTimer(ros::Duration(duration), &ElevationMappingNode::updatePose, this, false, true);
//   }
//   if (updateGridMapFps > 0) {
//     double duration = 1.0 / (updateGridMapFps + 0.00001);
//     updateGridMapTimer_ = nh_.createTimer(ros::Duration(duration), &ElevationMappingNode::updateGridMap, this, false, true);
//   }
//   if (publishStatisticsFps > 0) {
//     double duration = 1.0 / (publishStatisticsFps + 0.00001);
//     publishStatisticsTimer_ = nh_.createTimer(ros::Duration(duration), &ElevationMappingNode::publishStatistics, this, false, true);
//   }
//   lastStatisticsPublishedTime_ = ros::Time::now();
//   ROS_INFO("[ElevationMappingCupy] finish initialization");
}

  // namespace elevation_mapping_cupy


// // Setup map publishers
void ElevationMappingNode::setupMapPublishers() {
  // Find the layers with highest fps.
  float max_fps = -1;
  // Create timers for each unique map frequencies
  for (auto fps : map_fps_unique_) {
    // Which publisher to call in the timer callback
    std::vector<int> indices;
    // If this fps is max, update the map layers.
    if (fps >= max_fps) {
      max_fps = fps;
      map_layers_all_.clear();
    }
    for (int i = 0; i < map_fps_.size(); i++) {
      if (map_fps_[i] == fps) {
        indices.push_back(i);
        // If this fps is max, add layers
        if (fps >= max_fps) {
          for (const auto layer : map_layers_[i]) {
            map_layers_all_.insert(layer);
          }
        }
      }
    }
    // Callback funtion.
    // It publishes to specific topics.
    auto cb = [this, indices]() -> void {
      for (int i : indices) {
      publishMapOfIndex(i);
      }
    };
    double duration = 1.0 / (fps + 0.00001);
    mapTimers_.push_back(this->create_wall_timer(std::chrono::duration<double>(duration), cb));
  }
}


void ElevationMappingNode::publishMapOfIndex(int index) {
  // publish the map layers of index
  if (!isGridmapUpdated_) {
    return;
  }
  grid_map_msgs::msg::GridMap msg;
  std::vector<std::string> layers;

  // {  // need continuous lock between adding layers and converting to message. Otherwise updateGridmap can reset the data not in
  //    // map_layers_all_
  //   std::lock_guard<std::mutex> lock(mapMutex_);
  //   for (const auto& layer : map_layers_[index]) {
  //     const bool is_layer_in_all = map_layers_all_.find(layer) != map_layers_all_.end();
  //     if (is_layer_in_all && gridMap_.exists(layer)) {
  //       layers.push_back(layer);
  //     } else if (map_.exists_layer(layer)) {
  //       // if there are layers which is not in the syncing layer.
  //       ElevationMappingWrapper::RowMatrixXf map_data;
  //       map_.get_layer_data(layer, map_data);
  //       gridMap_.add(layer, map_data);
  //       layers.push_back(layer);
  //     }
  //   }
  //   if (layers.empty()) {
  //     return;
  //   }

  //   grid_map::GridMapRosConverter::toMessage(gridMap_, layers, msg);
  // }

  msg.basic_layers = map_basic_layers_[index];
  mapPubs_[index]->publish(msg);
}


void ElevationMappingNode::pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud, const std::string& key) {
  //  get channels
  auto fields = cloud->fields;
  std::vector<std::string> channels;

  for (size_t it = 0; it < fields.size(); it++) {
      auto& field = fields[it];
      channels.push_back(field.name);
  }
  inputPointCloud(cloud, channels);

  // This is used for publishing as statistics.
  pointCloudProcessCounter_++;
}

void ElevationMappingNode::inputPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr cloud,
                                           const std::vector<std::string>& channels) {
    auto start = this->now();
    auto* pcl_pc = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(pcl_pc);
    pcl_conversions::toPCL(*cloud, *pcl_pc);

    // Get channels
    auto fields = cloud->fields;
    uint array_dim = channels.size();

    RowMatrixXd points = RowMatrixXd(pcl_pc->width * pcl_pc->height, array_dim);

    for (unsigned int i = 0; i < pcl_pc->width * pcl_pc->height; ++i) {
        for (unsigned int j = 0; j < channels.size(); ++j) {
            float temp;
            uint point_idx = i * pcl_pc->point_step + pcl_pc->fields[j].offset;
            memcpy(&temp, &pcl_pc->data[point_idx], sizeof(float));
            points(i, j) = static_cast<double>(temp);
        }
    }

    // Get pose of sensor in map frame
    geometry_msgs::msg::TransformStamped transformStamped;
    std::string sensorFrameId = cloud->header.frame_id;
    auto timeStamp = cloud->header.stamp;
    Eigen::Affine3d transformationSensorToMap;
    try {
        transformStamped = tfBuffer_->lookupTransform(mapFrameId_, sensorFrameId, tf2::TimePointZero);
        transformationSensorToMap = tf2::transformToEigen(transformStamped);
    } catch (tf2::TransformException& ex) {
        RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
        return;
    }

    double positionError{0.0};
    double orientationError{0.0};
    {
        std::lock_guard<std::mutex> lock(errorMutex_);
        positionError = positionError_;
        orientationError = orientationError_;
    }

    // map_.input(points, channels, transformationSensorToMap.rotation(), transformationSensorToMap.translation(), positionError,
    //            orientationError);

    // if (enableDriftCorrectedTFPublishing_) {
    //     publishMapToOdom(map_.get_additive_mean_error());
    // }

    RCLCPP_DEBUG(this->get_logger(), "ElevationMap processed a point cloud (%i points) in %f sec.", static_cast<int>(points.size()),
                 (this->now() - start).seconds());
    RCLCPP_DEBUG(this->get_logger(), "positionError: %f ", positionError);
    RCLCPP_DEBUG(this->get_logger(), "orientationError: %f ", orientationError);
}




// void ElevationMappingNode::inputImage(const sensor_msgs::ImageConstPtr& image_msg,
//                                       const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
//                                       const std::vector<std::string>& channels) {
//   // Get image
//   cv::Mat image = cv_bridge::toCvShare(image_msg, image_msg->encoding)->image;

//   // Change encoding to RGB/RGBA
//   if (image_msg->encoding == "bgr8") {
//     cv::cvtColor(image, image, CV_BGR2RGB);
//   } else if (image_msg->encoding == "bgra8") {
//     cv::cvtColor(image, image, CV_BGRA2RGBA);
//   }

//   // Extract camera matrix
//   Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> cameraMatrix(&camera_info_msg->K[0]);

//   // Extract distortion coefficients
//   Eigen::VectorXd distortionCoeffs;
//   if (!camera_info_msg->D.empty()) {
//     distortionCoeffs = Eigen::Map<const Eigen::VectorXd>(camera_info_msg->D.data(), camera_info_msg->D.size());
//   } else {
//     ROS_WARN("Distortion coefficients are empty.");
//     distortionCoeffs = Eigen::VectorXd::Zero(5);
//     // return;
//   }

//   // distortion model
//   std::string distortion_model = camera_info_msg->distortion_model;
  
//   // Get pose of sensor in map frame
//   tf::StampedTransform transformTf;
//   std::string sensorFrameId = image_msg->header.frame_id;
//   auto timeStamp = image_msg->header.stamp;
//   Eigen::Affine3d transformationMapToSensor;
//   try {
//     transformListener_.waitForTransform(sensorFrameId, mapFrameId_, timeStamp, ros::Duration(1.0));
//     transformListener_.lookupTransform(sensorFrameId, mapFrameId_, timeStamp, transformTf);
//     poseTFToEigen(transformTf, transformationMapToSensor);
//   } catch (tf::TransformException& ex) {
//     ROS_ERROR("%s", ex.what());
//     return;
//   }

//   // Transform image to vector of Eigen matrices for easy pybind conversion
//   std::vector<cv::Mat> image_split;
//   std::vector<ColMatrixXf> multichannel_image;
//   cv::split(image, image_split);
//   for (auto img : image_split) {
//     ColMatrixXf eigen_img;
//     cv::cv2eigen(img, eigen_img);
//     multichannel_image.push_back(eigen_img);
//   }

//   // Check if the size of multichannel_image and channels and channel_methods matches. "rgb" counts for 3 layers.
//   int total_channels = 0;
//   for (const auto& channel : channels) {
//     if (channel == "rgb") {
//       total_channels += 3;
//     } else {
//       total_channels += 1;
//     }
//   }
//   if (total_channels != multichannel_image.size()) {
//     ROS_ERROR("Mismatch in the size of multichannel_image (%d), channels (%d). Please check the input.", multichannel_image.size(), channels.size());
//     ROS_ERROR_STREAM("Current Channels: " << boost::algorithm::join(channels, ", "));
//     return;
//   }

//   // Pass image to pipeline
//   map_.input_image(multichannel_image, channels, transformationMapToSensor.rotation(), transformationMapToSensor.translation(), cameraMatrix, 
//                    distortionCoeffs, distortion_model, image.rows, image.cols);
// }

void ElevationMappingNode::imageCallback(const std::shared_ptr<const sensor_msgs::msg::Image>& image_msg,
                                         const std::shared_ptr<const sensor_msgs::msg::CameraInfo>& camera_info_msg,
                                         const std::string& key) {
  auto start = this->now();
  // inputImage(image_msg, camera_info_msg, channels_[key]);
  RCLCPP_DEBUG(this->get_logger(), "ElevationMap imageCallback processed an image in %f sec.", (this->now() - start).seconds());
  
}


void ElevationMappingNode::imageChannelCallback(const std::shared_ptr<const sensor_msgs::msg::Image>& image_msg,
                                                const std::shared_ptr<const sensor_msgs::msg::CameraInfo>& camera_info_msg, 
                                                const std::shared_ptr<const elevation_map_msgs::msg::ChannelInfo>& channel_info_msg) {
auto start = this->now();
// Default channels and fusion methods for image is rgb and image_color
// std::vector<std::string> channels;
// channels = channel_info_msg->channels;
// inputImage(image_msg, camera_info_msg, channels);
RCLCPP_DEBUG(this->get_logger(), "ElevationMap imageChannelCallback processed an image in %f sec.", (this->now() - start).seconds());
}

// void ElevationMappingNode::updatePose(const ros::TimerEvent&) {
//   tf::StampedTransform transformTf;
//   const auto& timeStamp = ros::Time::now();
//   Eigen::Affine3d transformationBaseToMap;
//   try {
//     transformListener_.waitForTransform(mapFrameId_, baseFrameId_, timeStamp, ros::Duration(1.0));
//     transformListener_.lookupTransform(mapFrameId_, baseFrameId_, timeStamp, transformTf);
//     poseTFToEigen(transformTf, transformationBaseToMap);
//   } catch (tf::TransformException& ex) {
//     ROS_ERROR("%s", ex.what());
//     return;
//   }

//   // This is to check if the robot is moving. If the robot is not moving, drift compensation is disabled to avoid creating artifacts.
//   Eigen::Vector3d position(transformTf.getOrigin().x(), transformTf.getOrigin().y(), transformTf.getOrigin().z());
//   map_.move_to(position, transformationBaseToMap.rotation().transpose());
//   Eigen::Vector3d position3(transformTf.getOrigin().x(), transformTf.getOrigin().y(), transformTf.getOrigin().z());
//   Eigen::Vector4d orientation(transformTf.getRotation().x(), transformTf.getRotation().y(), transformTf.getRotation().z(),
//                               transformTf.getRotation().w());
//   lowpassPosition_ = positionAlpha_ * position3 + (1 - positionAlpha_) * lowpassPosition_;
//   lowpassOrientation_ = orientationAlpha_ * orientation + (1 - orientationAlpha_) * lowpassOrientation_;
//   {
//     std::lock_guard<std::mutex> lock(errorMutex_);
//     positionError_ = (position3 - lowpassPosition_).norm();
//     orientationError_ = (orientation - lowpassOrientation_).norm();
//   }

//   if (useInitializerAtStart_) {
//     ROS_INFO("Clearing map with initializer.");
//     initializeWithTF();
//     useInitializerAtStart_ = false;
//   }
// }

// void ElevationMappingNode::publishAsPointCloud(const grid_map::GridMap& map) const {
//   sensor_msgs::PointCloud2 msg;
//   grid_map::GridMapRosConverter::toPointCloud(map, "elevation", msg);
//   pointPub_.publish(msg);
// }

// bool ElevationMappingNode::getSubmap(grid_map_msgs::GetGridMap::Request& request, grid_map_msgs::GetGridMap::Response& response) {
//   std::string requestedFrameId = request.frame_id;
//   Eigen::Isometry3d transformationOdomToMap;
//   grid_map::Position requestedSubmapPosition(request.position_x, request.position_y);
//   if (requestedFrameId != mapFrameId_) {
//     tf::StampedTransform transformTf;
//     const auto& timeStamp = ros::Time::now();
//     try {
//       transformListener_.waitForTransform(requestedFrameId, mapFrameId_, timeStamp, ros::Duration(1.0));
//       transformListener_.lookupTransform(requestedFrameId, mapFrameId_, timeStamp, transformTf);
//       tf::poseTFToEigen(transformTf, transformationOdomToMap);
//     } catch (tf::TransformException& ex) {
//       ROS_ERROR("%s", ex.what());
//       return false;
//     }
//     Eigen::Vector3d p(request.position_x, request.position_y, 0);
//     Eigen::Vector3d mapP = transformationOdomToMap.inverse() * p;
//     requestedSubmapPosition.x() = mapP.x();
//     requestedSubmapPosition.y() = mapP.y();
//   }
//   grid_map::Length requestedSubmapLength(request.length_x, request.length_y);
//   ROS_DEBUG("Elevation submap request: Position x=%f, y=%f, Length x=%f, y=%f.", requestedSubmapPosition.x(), requestedSubmapPosition.y(),
//             requestedSubmapLength(0), requestedSubmapLength(1));

//   bool isSuccess;
//   grid_map::Index index;
//   grid_map::GridMap subMap;
//   {
//     std::lock_guard<std::mutex> lock(mapMutex_);
//     subMap = gridMap_.getSubmap(requestedSubmapPosition, requestedSubmapLength, index, isSuccess);
//   }
//   const auto& length = subMap.getLength();
//   if (requestedFrameId != mapFrameId_) {
//     subMap = subMap.getTransformedMap(transformationOdomToMap, "elevation", requestedFrameId);
//   }

//   if (request.layers.empty()) {
//     grid_map::GridMapRosConverter::toMessage(subMap, response.map);
//   } else {
//     std::vector<std::string> layers;
//     for (const auto& layer : request.layers) {
//       layers.push_back(layer);
//     }
//     grid_map::GridMapRosConverter::toMessage(subMap, layers, response.map);
//   }
//   return isSuccess;
// }

// bool ElevationMappingNode::clearMap(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response) {
//   ROS_INFO("Clearing map.");
//   map_.clear();
//   if (alwaysClearWithInitializer_) {
//     initializeWithTF();
//   }
//   return true;
// }

// bool ElevationMappingNode::clearMapWithInitializer(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response) {
//   ROS_INFO("Clearing map with initializer.");
//   map_.clear();
//   initializeWithTF();
//   return true;
// }

// void ElevationMappingNode::initializeWithTF() {
//   std::vector<Eigen::Vector3d> points;
//   const auto& timeStamp = ros::Time::now();
//   int i = 0;
//   Eigen::Vector3d p;
//   for (const auto& frame_id : initialize_frame_id_) {
//     // Get tf from map frame to tf frame
//     Eigen::Affine3d transformationBaseToMap;
//     tf::StampedTransform transformTf;
//     try {
//       transformListener_.waitForTransform(mapFrameId_, frame_id, timeStamp, ros::Duration(1.0));
//       transformListener_.lookupTransform(mapFrameId_, frame_id, timeStamp, transformTf);
//       poseTFToEigen(transformTf, transformationBaseToMap);
//     } catch (tf::TransformException& ex) {
//       ROS_ERROR("%s", ex.what());
//       return;
//     }
//     p = transformationBaseToMap.translation();
//     p.z() += initialize_tf_offset_[i];
//     points.push_back(p);
//     i++;
//   }
//   if (!points.empty() && points.size() < 3) {
//     points.emplace_back(p + Eigen::Vector3d(initializeTfGridSize_, initializeTfGridSize_, 0));
//     points.emplace_back(p + Eigen::Vector3d(-initializeTfGridSize_, initializeTfGridSize_, 0));
//     points.emplace_back(p + Eigen::Vector3d(initializeTfGridSize_, -initializeTfGridSize_, 0));
//     points.emplace_back(p + Eigen::Vector3d(-initializeTfGridSize_, -initializeTfGridSize_, 0));
//   }
//   ROS_INFO_STREAM("Initializing map with points using " << initializeMethod_);
//   map_.initializeWithPoints(points, initializeMethod_);
// }

// bool ElevationMappingNode::checkSafety(elevation_map_msgs::CheckSafety::Request& request,
//                                        elevation_map_msgs::CheckSafety::Response& response) {
//   for (const auto& polygonstamped : request.polygons) {
//     if (polygonstamped.polygon.points.empty()) {
//       continue;
//     }
//     std::vector<Eigen::Vector2d> polygon;
//     std::vector<Eigen::Vector2d> untraversable_polygon;
//     Eigen::Vector3d result;
//     result.setZero();
//     const auto& polygonFrameId = polygonstamped.header.frame_id;
//     const auto& timeStamp = polygonstamped.header.stamp;
//     double polygon_z = polygonstamped.polygon.points[0].z;

//     // Get tf from map frame to polygon frame
//     if (mapFrameId_ != polygonFrameId) {
//       Eigen::Affine3d transformationBaseToMap;
//       tf::StampedTransform transformTf;
//       try {
//         transformListener_.waitForTransform(mapFrameId_, polygonFrameId, timeStamp, ros::Duration(1.0));
//         transformListener_.lookupTransform(mapFrameId_, polygonFrameId, timeStamp, transformTf);
//         poseTFToEigen(transformTf, transformationBaseToMap);
//       } catch (tf::TransformException& ex) {
//         ROS_ERROR("%s", ex.what());
//         return false;
//       }
//       for (const auto& p : polygonstamped.polygon.points) {
//         const auto& pvector = Eigen::Vector3d(p.x, p.y, p.z);
//         const auto transformed_p = transformationBaseToMap * pvector;
//         polygon.emplace_back(Eigen::Vector2d(transformed_p.x(), transformed_p.y()));
//       }
//     } else {
//       for (const auto& p : polygonstamped.polygon.points) {
//         polygon.emplace_back(Eigen::Vector2d(p.x, p.y));
//       }
//     }

//     map_.get_polygon_traversability(polygon, result, untraversable_polygon);

//     geometry_msgs::PolygonStamped untraversable_polygonstamped;
//     untraversable_polygonstamped.header.stamp = ros::Time::now();
//     untraversable_polygonstamped.header.frame_id = mapFrameId_;
//     for (const auto& p : untraversable_polygon) {
//       geometry_msgs::Point32 point;
//       point.x = static_cast<float>(p.x());
//       point.y = static_cast<float>(p.y());
//       point.z = static_cast<float>(polygon_z);
//       untraversable_polygonstamped.polygon.points.push_back(point);
//     }
//     // traversability_result;
//     response.is_safe.push_back(bool(result[0] > 0.5));
//     response.traversability.push_back(result[1]);
//     response.untraversable_polygons.push_back(untraversable_polygonstamped);
//   }
//   return true;
// }

// bool ElevationMappingNode::setPublishPoint(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& response) {
//   enablePointCloudPublishing_ = request.data;
//   response.success = true;
//   return true;
// }

// void ElevationMappingNode::updateVariance(const ros::TimerEvent&) {
//   map_.update_variance();
// }

// void ElevationMappingNode::updateTime(const ros::TimerEvent&) {
//   map_.update_time();
// }

// void ElevationMappingNode::publishStatistics(const ros::TimerEvent&) {
//   ros::Time now = ros::Time::now();
//   double dt = (now - lastStatisticsPublishedTime_).toSec();
//   lastStatisticsPublishedTime_ = now;
//   elevation_map_msgs::Statistics msg;
//   msg.header.stamp = now;
//   if (dt > 0.0) {
//     msg.pointcloud_process_fps = pointCloudProcessCounter_ / dt;
//   }
//   pointCloudProcessCounter_ = 0;
//   statisticsPub_.publish(msg);
// }

// void ElevationMappingNode::updateGridMap(const ros::TimerEvent&) {
//   std::vector<std::string> layers(map_layers_all_.begin(), map_layers_all_.end());
//   std::lock_guard<std::mutex> lock(mapMutex_);
//   map_.get_grid_map(gridMap_, layers);
//   gridMap_.setTimestamp(ros::Time::now().toNSec());
//   alivePub_.publish(std_msgs::Empty());

//   // Mostly debug purpose
//   if (enablePointCloudPublishing_) {
//     publishAsPointCloud(gridMap_);
//   }
//   if (enableNormalArrowPublishing_) {
//     publishNormalAsArrow(gridMap_);
//   }
//   isGridmapUpdated_ = true;
// }

// bool ElevationMappingNode::initializeMap(elevation_map_msgs::Initialize::Request& request,
//                                          elevation_map_msgs::Initialize::Response& response) {
//   // If initialize method is points
//   if (request.type == request.POINTS) {
//     std::vector<Eigen::Vector3d> points;
//     for (const auto& point : request.points) {
//       const auto& pointFrameId = point.header.frame_id;
//       const auto& timeStamp = point.header.stamp;
//       const auto& pvector = Eigen::Vector3d(point.point.x, point.point.y, point.point.z);

//       // Get tf from map frame to points' frame
//       if (mapFrameId_ != pointFrameId) {
//         Eigen::Affine3d transformationBaseToMap;
//         tf::StampedTransform transformTf;
//         try {
//           transformListener_.waitForTransform(mapFrameId_, pointFrameId, timeStamp, ros::Duration(1.0));
//           transformListener_.lookupTransform(mapFrameId_, pointFrameId, timeStamp, transformTf);
//           poseTFToEigen(transformTf, transformationBaseToMap);
//         } catch (tf::TransformException& ex) {
//           ROS_ERROR("%s", ex.what());
//           return false;
//         }
//         const auto transformed_p = transformationBaseToMap * pvector;
//         points.push_back(transformed_p);
//       } else {
//         points.push_back(pvector);
//       }
//     }
//     std::string method;
//     switch (request.method) {
//       case request.NEAREST:
//         method = "nearest";
//         break;
//       case request.LINEAR:
//         method = "linear";
//         break;
//       case request.CUBIC:
//         method = "cubic";
//         break;
//     }
//     ROS_INFO_STREAM("Initializing map with points using " << method);
//     map_.initializeWithPoints(points, method);
//   }
//   response.success = true;
//   return true;
// }

// void ElevationMappingNode::publishNormalAsArrow(const grid_map::GridMap& map) const {
//   auto startTime = ros::Time::now();

//   const auto& normalX = map["normal_x"];
//   const auto& normalY = map["normal_y"];
//   const auto& normalZ = map["normal_z"];
//   double scale = 0.1;

//   visualization_msgs::MarkerArray markerArray;
//   // For each cell in map.
//   for (grid_map::GridMapIterator iterator(map); !iterator.isPastEnd(); ++iterator) {
//     if (!map.isValid(*iterator, "elevation")) {
//       continue;
//     }
//     grid_map::Position3 p;
//     map.getPosition3("elevation", *iterator, p);
//     Eigen::Vector3d start = p;
//     const auto i = iterator.getLinearIndex();
//     Eigen::Vector3d normal(normalX(i), normalY(i), normalZ(i));
//     Eigen::Vector3d end = start + normal * scale;
//     if (normal.norm() < 0.1) {
//       continue;
//     }
//     markerArray.markers.push_back(vectorToArrowMarker(start, end, i));
//   }
//   normalPub_.publish(markerArray);
//   ROS_INFO_THROTTLE(1.0, "publish as normal in %f sec.", (ros::Time::now() - startTime).toSec());
// }

// visualization_msgs::Marker ElevationMappingNode::vectorToArrowMarker(const Eigen::Vector3d& start, const Eigen::Vector3d& end,
//                                                                      const int id) const {
//   visualization_msgs::Marker marker;
//   marker.header.frame_id = mapFrameId_;
//   marker.header.stamp = ros::Time::now();
//   marker.ns = "normal";
//   marker.id = id;
//   marker.type = visualization_msgs::Marker::ARROW;
//   marker.action = visualization_msgs::Marker::ADD;
//   marker.points.resize(2);
//   marker.points[0].x = start.x();
//   marker.points[0].y = start.y();
//   marker.points[0].z = start.z();
//   marker.points[1].x = end.x();
//   marker.points[1].y = end.y();
//   marker.points[1].z = end.z();
//   marker.pose.orientation.x = 0.0;
//   marker.pose.orientation.y = 0.0;
//   marker.pose.orientation.z = 0.0;
//   marker.pose.orientation.w = 1.0;
//   marker.scale.x = 0.01;
//   marker.scale.y = 0.01;
//   marker.scale.z = 0.01;
//   marker.color.a = 1.0;  // Don't forget to set the alpha!
//   marker.color.r = 0.0;
//   marker.color.g = 1.0;
//   marker.color.b = 0.0;

//   return marker;
// }

// void ElevationMappingNode::publishMapToOdom(double error) {
//   tf::Transform transform;
//   transform.setOrigin(tf::Vector3(0.0, 0.0, error));
//   tf::Quaternion q;
//   q.setRPY(0, 0, 0);
//   transform.setRotation(q);
//   tfBroadcaster_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), mapFrameId_, correctedMapFrameId_));
// }

}  // namespace elevation_mapping_cupy
