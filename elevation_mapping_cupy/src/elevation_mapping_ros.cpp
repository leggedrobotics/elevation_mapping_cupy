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


ElevationMappingNode::ElevationMappingNode(const rclcpp::NodeOptions& options)
    : Node("elevation_mapping_node", options),
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
  this->get_parameter("voxel_filter_size", voxel_filter_size_);

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
  RCLCPP_INFO(this->get_logger(), "voxel_filter_size: %f", voxel_filter_size_);

  enablePointCloudPublishing_ = enablePointCloudPublishing;
     
  map_ = std::make_shared<ElevationMappingWrapper>();
  map_->initialize(node_);



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
            
            std::string key = sub_name;


            sensor_msgs::msg::CameraInfo img_info;
            elevation_map_msgs::msg::ChannelInfo channel_info;
            imageInfoReady_[key] = std::make_pair(img_info, false);
            imageChannelReady_[key] = std::make_pair(channel_info, false);
              // Image subscriber init
              
            rmw_qos_profile_t sensor_qos_profile = rmw_qos_profile_sensor_data;
            auto sensor_qos = rclcpp::QoS(rclcpp::QoSInitialization(sensor_qos_profile.history, sensor_qos_profile.depth), sensor_qos_profile);

              auto img_callback = [this, key](const sensor_msgs::msg::Image::SharedPtr msg) {
                  this->imageCallback(msg, key);
              };
              auto img_sub = this->create_subscription<sensor_msgs::msg::Image>(camera_topic, sensor_qos, img_callback);
              imageSubs_.push_back(img_sub);
              RCLCPP_INFO(this->get_logger(), "Subscribed to Image topic: %s", camera_topic.c_str());
              // Camera Info subscriber init
              auto img_info_callback = [this, key](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
                  this->imageInfoCallback(msg, key);
              };
              auto img_info_sub = this->create_subscription<sensor_msgs::msg::CameraInfo>(info_topic, sensor_qos, img_info_callback);
              cameraInfoSubs_.push_back(img_info_sub);
              RCLCPP_INFO(this->get_logger(), "Subscribed to ImageInfo topic: %s", info_topic.c_str());

            std::string channel_info_topic; 
            if (this->get_parameter("subscribers." + sub_name + ".channel_name", channel_info_topic)) {
              // channel subscriber init
              imageChannelReady_[key].second = false;
              auto img_channel_callback = [this, key](const elevation_map_msgs::msg::ChannelInfo::SharedPtr msg) {
                  this->imageChannelCallback(msg, key);
              };
              auto channel_info_sub = this->create_subscription<elevation_map_msgs::msg::ChannelInfo>(channel_info_topic, sensor_qos, img_channel_callback);
              channelInfoSubs_.push_back(channel_info_sub);
              RCLCPP_INFO(this->get_logger(), "Subscribed to ChannelInfo topic: %s", channel_info_topic.c_str());
            }else{
                imageChannelReady_[key].second = true;
                channels_[key].push_back("rgb");
            }
          }else if(data_type == "pointcloud"){            
            std::string pointcloud_topic;
            this->get_parameter("subscribers." + sub_name + ".topic_name", pointcloud_topic);                                    
            std::string key = sub_name;
            channels_[key].push_back("x");
            channels_[key].push_back("y");
            channels_[key].push_back("z");

            // rmw_qos_profile_t qos_profile = rmw_qos_profile_default;
            // auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, qos_profile.depth), qos_profile);

            rmw_qos_profile_t sensor_qos_profile = rmw_qos_profile_sensor_data;
            auto sensor_qos = rclcpp::QoS(rclcpp::QoSInitialization(sensor_qos_profile.history, sensor_qos_profile.depth), sensor_qos_profile);


            // point_cloud_transport::Subscriber pct_sub = pct.subscribe(
            //     "pct/point_cloud", 100,
            //     [node](const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
            //     {
            //       RCLCPP_INFO_STREAM(
            //         node->get_logger(),
            //         "Message received, number of points is: " << msg->width * msg->height);
            //     }, {});
            // point_cloud_transport::Subscriber sub = pct.subscribe(pointcloud_topic, 100, callback, {}, transport_hint.get())


            auto callback_transport = [this, key](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
            this->pointcloudtransportCallback(msg, key);
            };

            // Create transport hints (e.g., "draco")
            // auto transport_hint = std::make_shared<point_cloud_transport::TransportHints>("draco");
            

            // Use PointCloudTransport to create a subscriber
            point_cloud_transport::PointCloudTransport pct(node_);
            auto sub_transport = pct.subscribe(pointcloud_topic, 100, callback_transport);

            // Add the subscriber to the vector to manage its lifetime
            pointcloudtransportSubs_.push_back(sub_transport);


            // auto callback = [this, key](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            //       this->pointcloudCallback(msg, key);
            //   };
              
            //   auto sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(pointcloud_topic, sensor_qos, callback);
            
            //   pointcloudSubs_.push_back(sub);
                RCLCPP_INFO(this->get_logger(), "Subscribed to PointCloud2 topic: %s", pointcloud_topic.c_str());
          }
      }
    }
 
    
    
    auto unique_pub_names = extract_unique_names(publisher_params);

    
    std::string node_name = this->get_name();

    for (const auto& pub_name : unique_pub_names) {  
        // Namespacing published topics under node_name
        std::string topic_name = node_name + "/" + pub_name;  
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

  pointPub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("elevation_map_points", 1);
  alivePub_ = this->create_publisher<std_msgs::msg::Empty>("alive", 1);  
  normalPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("normal", 1);
  statisticsPub_ = this->create_publisher<elevation_map_msgs::msg::Statistics>("statistics", 1);

  gridMap_.setFrameId(mapFrameId_);

rawSubmapService_ = this->create_service<grid_map_msgs::srv::GetGridMap>(
        "get_raw_submap", std::bind(&ElevationMappingNode::getSubmap, this, std::placeholders::_1, std::placeholders::_2));

clearMapService_ = this->create_service<std_srvs::srv::Empty>(
        "clear_map", std::bind(&ElevationMappingNode::clearMap, this, std::placeholders::_1, std::placeholders::_2));

clearMapWithInitializerService_ = this->create_service<std_srvs::srv::Empty>(
        "clear_map_with_initializer", std::bind(&ElevationMappingNode::clearMapWithInitializer, this, std::placeholders::_1, std::placeholders::_2));


initializeMapService_ = this->create_service<elevation_map_msgs::srv::Initialize>(
        "initialize", std::bind(&ElevationMappingNode::initializeMap, this, std::placeholders::_1, std::placeholders::_2));

setPublishPointService_ = this->create_service<std_srvs::srv::SetBool>(
      "set_publish_points", std::bind(&ElevationMappingNode::setPublishPoint, this, std::placeholders::_1, std::placeholders::_2));

// checkSafetyService_ = this->create_service<std_srvs::srv::Empty>(
//     "check_safety", std::bind(&ElevationMappingNode::checkSafety, this, std::placeholders::_1, std::placeholders::_2));


  if (updateVarianceFps > 0) {
    double duration = 1.0 / (updateVarianceFps + 0.00001);    
    updateVarianceTimer_ = this->create_wall_timer(std::chrono::duration<double>(duration),
            std::bind(&ElevationMappingNode::updateVariance, this));
  }
 if (timeInterval > 0) {
        double duration = timeInterval;
        updateTimeTimer_ = this->create_wall_timer(std::chrono::duration<double>(duration),
            std::bind(&ElevationMappingNode::updateTime, this));
    }
    if (updatePoseFps > 0) {
        double duration = 1.0 / (updatePoseFps + 0.00001);
        updatePoseTimer_ = this->create_wall_timer(std::chrono::duration<double>(duration),
            std::bind(&ElevationMappingNode::updatePose, this));
    }
    if (updateGridMapFps > 0) {
        double duration = 1.0 / (updateGridMapFps + 0.00001);
        updateGridMapTimer_ = this->create_wall_timer(std::chrono::duration<double>(duration),
            std::bind(&ElevationMappingNode::updateGridMap, this));
    }
    if (publishStatisticsFps > 0) {
        double duration = 1.0 / (publishStatisticsFps + 0.00001);
        publishStatisticsTimer_ = this->create_wall_timer(std::chrono::duration<double>(duration),
            std::bind(&ElevationMappingNode::publishStatistics, this));
    }
    lastStatisticsPublishedTime_ = this->now();
    RCLCPP_INFO(this->get_logger(), "[ElevationMappingCupy] finish initialization");
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
  
  
  std::unique_ptr<grid_map_msgs::msg::GridMap> msg_ptr;
  grid_map_msgs::msg::GridMap msg;

  std::vector<std::string> layers;

   {  // need continuous lock between adding layers and converting to message. Otherwise updateGridmap can reset the data not in
     // map_layers_all_
    std::lock_guard<std::mutex> lock(mapMutex_);
    for (const auto& layer : map_layers_[index]) {
      const bool is_layer_in_all = map_layers_all_.find(layer) != map_layers_all_.end();
      if (is_layer_in_all && gridMap_.exists(layer)) {
        layers.push_back(layer);
      } else if (map_->exists_layer(layer)) {
        // if there are layers which is not in the syncing layer.
        ElevationMappingWrapper::RowMatrixXf map_data;
        map_->get_layer_data(layer, map_data);
        gridMap_.add(layer, map_data);
        layers.push_back(layer);
      }
    }
    if (layers.empty()) {
      return;
    }

    msg_ptr = grid_map::GridMapRosConverter::toMessage(gridMap_, layers);
    msg= *msg_ptr;
  }
   
  msg.basic_layers = map_basic_layers_[index];
  mapPubs_[index]->publish(msg);
}

void ElevationMappingNode::imageInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr image_info, const std::string& key) {
imageInfoReady_[key] = std::make_pair(*image_info, true);
    // Find and remove the subscription for this key
    auto it = std::find_if(cameraInfoSubs_.begin(), cameraInfoSubs_.end(),
                           [key](const rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr& sub) {
                               return sub->get_topic_name() == key;
                           });
    if (it != cameraInfoSubs_.end()) {
        cameraInfoSubs_.erase(it);        
    }

}

void ElevationMappingNode::imageChannelCallback(const elevation_map_msgs::msg::ChannelInfo::SharedPtr channel_info, const std::string& key) {
imageChannelReady_[key] = std::make_pair(*channel_info, true);
 auto it = std::find_if(channelInfoSubs_.begin(), channelInfoSubs_.end(),
                           [key](const rclcpp::Subscription<elevation_map_msgs::msg::ChannelInfo>::SharedPtr& sub) {
                               return sub->get_topic_name() == key;
                           });
    if (it != channelInfoSubs_.end()) {
        channelInfoSubs_.erase(it);        
    }
}


void ElevationMappingNode::pointcloudtransportCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud, const std::string& key) {
  
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

void ElevationMappingNode::inputPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud,
                                           const std::vector<std::string>& channels) {
    auto start = this->now();
    // auto* raw_pcl_pc = new pcl::PCLPointCloud2;
    // pcl::PCLPointCloud2ConstPtr cloudPtr(raw_pcl_pc);    
    pcl::PCLPointCloud2::Ptr raw_pcl_pc(new pcl::PCLPointCloud2());
    pcl_conversions::toPCL(*cloud, *raw_pcl_pc);
    
    // apply the voxel filtering     
    pcl::PCLPointCloud2::Ptr pcl_pc (new pcl::PCLPointCloud2());
    // pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_filter;
    voxel_filter.setInputCloud(raw_pcl_pc);
    voxel_filter.setLeafSize(voxel_filter_size_,voxel_filter_size_,voxel_filter_size_);
    voxel_filter.filter(*pcl_pc);   
    
    RCLCPP_DEBUG(this->get_logger(), "Voxel grid filtered point cloud from %d points to %d points.", static_cast<int>(raw_pcl_pc->width * raw_pcl_pc->height), static_cast<int>(pcl_pc->width * pcl_pc->height));

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

    map_->input(points, channels, transformationSensorToMap.rotation(), transformationSensorToMap.translation(), positionError,
               orientationError);

    if (enableDriftCorrectedTFPublishing_) {
        publishMapToOdom(map_->get_additive_mean_error());
    }

    RCLCPP_DEBUG(this->get_logger(), "ElevationMap processed a point cloud (%i points) in %f sec.", static_cast<int>(points.size()),
                 (this->now() - start).seconds());
    RCLCPP_DEBUG(this->get_logger(), "positionError: %f ", positionError);
    RCLCPP_DEBUG(this->get_logger(), "orientationError: %f ", orientationError);
}




void ElevationMappingNode::inputImage(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                                      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg,
                                      const std::vector<std::string>& channels) {            
  // Get image
  cv::Mat image = cv_bridge::toCvShare(image_msg, image_msg->encoding)->image;

  // Change encoding to RGB/RGBA
  if (image_msg->encoding == "bgr8") {
    cv::cvtColor(image, image, CV_BGR2RGB);
  } else if (image_msg->encoding == "bgra8") {
    cv::cvtColor(image, image, CV_BGRA2RGBA);
  }

  // Extract camera matrix
  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> cameraMatrix(&camera_info_msg->k[0]);

  // Extract distortion coefficients
  Eigen::VectorXd distortionCoeffs;
  if (!camera_info_msg->d.empty()) {
    distortionCoeffs = Eigen::Map<const Eigen::VectorXd>(camera_info_msg->d.data(), camera_info_msg->d.size());
  } else {
    RCLCPP_WARN(this->get_logger(), "Distortion coefficients are empty.");    
    distortionCoeffs = Eigen::VectorXd::Zero(5);
    // return;
  }

  // distortion model
  std::string distortion_model = camera_info_msg->distortion_model;
  
 // Get pose of sensor in map frame
  geometry_msgs::msg::TransformStamped transformStamped;
  std::string sensorFrameId = image_msg->header.frame_id;
  auto timeStamp = image_msg->header.stamp;
  Eigen::Isometry3d transformationMapToSensor;
  try {
    transformStamped = tfBuffer_->lookupTransform(sensorFrameId, mapFrameId_, tf2::TimePointZero);
    transformationMapToSensor = tf2::transformToEigen(transformStamped);
  } catch (tf2::TransformException& ex) {
    RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    return;
  }

  // Transform image to vector of Eigen matrices for easy pybind conversion
  std::vector<cv::Mat> image_split;
  std::vector<ColMatrixXf> multichannel_image;
  cv::split(image, image_split);
  for (auto img : image_split) {
    ColMatrixXf eigen_img;
    cv::cv2eigen(img, eigen_img);
    multichannel_image.push_back(eigen_img);
  }

  // Check if the size of multichannel_image and channels and channel_methods matches. "rgb" counts for 3 layers.
  int total_channels = 0;
  for (const auto& channel : channels) {
    if (channel == "rgb") {
      total_channels += 3;
    } else {
      total_channels += 1;
    }
  }
  if (total_channels != multichannel_image.size()) {
    RCLCPP_ERROR(this->get_logger(), "Mismatch in the size of multichannel_image (%d), channels (%d). Please check the input.", multichannel_image.size(), channels.size());
    for (const auto& channel : channels) {
      RCLCPP_INFO(this->get_logger(), "Channel: %s", channel.c_str());
    }
    return;
  }

  // Pass image to pipeline
  map_->input_image(multichannel_image, channels, transformationMapToSensor.rotation(), transformationMapToSensor.translation(), cameraMatrix, 
                   distortionCoeffs, distortion_model, image.rows, image.cols);
}



void ElevationMappingNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr image_msg, const std::string& key){                                              
  auto start = this->now();
  
  if (!imageInfoReady_[key].second){
    RCLCPP_WARN(this->get_logger(), "CameraInfo for key %s is not available yet.", key.c_str());
    return;
  } 
  
  
  auto camera_info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>(imageInfoReady_[key].first);    
  if (std::find(channels_[key].begin(), channels_[key].end(), "rgb") != channels_[key].end()){
    inputImage(image_msg, camera_info_msg, channels_[key]);
    }
  else{
    if (!imageChannelReady_[key].second){
      RCLCPP_WARN(this->get_logger(), "ChannelInfo for key %s is not available yet.", key.c_str());
      return;    
    }     
    // Default channels and fusion methods for image is rgb and image_color
    std::vector<std::string> channels;    
    channels = imageChannelReady_[key].first.channels;
    inputImage(image_msg, camera_info_msg, channels);
  }  
    RCLCPP_DEBUG(this->get_logger(), "ElevationMap imageChannelCallback processed an image in %f sec.", (this->now() - start).seconds());       
}


void ElevationMappingNode::updatePose() {
  geometry_msgs::msg::TransformStamped transformStamped;
  const auto& timeStamp = this->now();
  Eigen::Affine3d transformationBaseToMap;
  try {
    transformStamped = tfBuffer_->lookupTransform(mapFrameId_, baseFrameId_, timeStamp, tf2::durationFromSec(1.0));
    transformationBaseToMap = tf2::transformToEigen(transformStamped);
  } catch (tf2::TransformException& ex) {
    RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    return;
  }

  // This is to check if the robot is moving. If the robot is not moving, drift compensation is disabled to avoid creating artifacts.
  Eigen::Vector3d position(transformStamped.transform.translation.x,
                           transformStamped.transform.translation.y,
                           transformStamped.transform.translation.z);
  map_->move_to(position, transformationBaseToMap.rotation().transpose());
  Eigen::Vector3d position3(transformStamped.transform.translation.x,
                            transformStamped.transform.translation.y,
                            transformStamped.transform.translation.z);
  Eigen::Vector4d orientation(transformStamped.transform.rotation.x,
                              transformStamped.transform.rotation.y,
                              transformStamped.transform.rotation.z,
                              transformStamped.transform.rotation.w);
  lowpassPosition_ = positionAlpha_ * position3 + (1 - positionAlpha_) * lowpassPosition_;
  lowpassOrientation_ = orientationAlpha_ * orientation + (1 - orientationAlpha_) * lowpassOrientation_;
  {
    std::lock_guard<std::mutex> lock(errorMutex_);
    positionError_ = (position3 - lowpassPosition_).norm();
    orientationError_ = (orientation - lowpassOrientation_).norm();
  }

  if (useInitializerAtStart_) {
    RCLCPP_INFO(this->get_logger(), "Clearing map with initializer.");
    initializeWithTF();
    useInitializerAtStart_ = false;
  }
}

void ElevationMappingNode::publishAsPointCloud(const grid_map::GridMap& map) const {
  sensor_msgs::msg::PointCloud2 msg;
  grid_map::GridMapRosConverter::toPointCloud(map, "elevation", msg);
  pointPub_->publish(msg);
}


bool ElevationMappingNode::getSubmap(
    const std::shared_ptr<grid_map_msgs::srv::GetGridMap::Request> request,
    std::shared_ptr<grid_map_msgs::srv::GetGridMap::Response> response) {
  std::string requestedFrameId = request->frame_id;
  Eigen::Isometry3d transformationOdomToMap;
  geometry_msgs::msg::Pose pose;          
  grid_map::Position requestedSubmapPosition(request->position_x, request->position_y);
  
  
  if (requestedFrameId != mapFrameId_) {
    geometry_msgs::msg::TransformStamped transformStamped;
    
    try {
      const auto& timeStamp = this->now();
      transformStamped = tfBuffer_->lookupTransform(requestedFrameId, mapFrameId_, timeStamp, tf2::durationFromSec(1.0));      
    
    
    pose.position.x = transformStamped.transform.translation.x;
    pose.position.y = transformStamped.transform.translation.y;
    pose.position.z = transformStamped.transform.translation.z;
    pose.orientation = transformStamped.transform.rotation;

    tf2::fromMsg(pose, transformationOdomToMap);
    } catch (tf2::TransformException& ex) {
      RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
      return false;
    }
    Eigen::Vector3d p(request->position_x, request->position_y, 0);
    Eigen::Vector3d mapP = transformationOdomToMap.inverse() * p;
    requestedSubmapPosition.x() = mapP.x();
    requestedSubmapPosition.y() = mapP.y();
  }
  grid_map::Length requestedSubmapLength(request->length_x, request->length_y);
  RCLCPP_DEBUG(this->get_logger(), "Elevation submap request: Position x=%f, y=%f, Length x=%f, y=%f.",
               requestedSubmapPosition.x(), requestedSubmapPosition.y(),
               requestedSubmapLength(0), requestedSubmapLength(1));

  bool isSuccess;
  grid_map::Index index;
  grid_map::GridMap subMap;
  {
    std::lock_guard<std::mutex> lock(mapMutex_);
    subMap = gridMap_.getSubmap(requestedSubmapPosition, requestedSubmapLength, isSuccess);    
  }
  const auto& length = subMap.getLength();
  if (requestedFrameId != mapFrameId_) {
    subMap = subMap.getTransformedMap(transformationOdomToMap, "elevation", requestedFrameId);
  }

  if (request->layers.empty()) {    
    auto msg_ptr = grid_map::GridMapRosConverter::toMessage(subMap);
    response->map = *msg_ptr;
  } else {
    std::vector<std::string> layers;
    for (const auto& layer : request->layers) {
      layers.push_back(layer);
    }
    auto msg_ptr = grid_map::GridMapRosConverter::toMessage(subMap, layers);    
    response->map = *msg_ptr;

  }
  
  return isSuccess;
}



void ElevationMappingNode::clearMap(const std::shared_ptr<std_srvs::srv::Empty::Request> request,
                                    std::shared_ptr<std_srvs::srv::Empty::Response> response) {
   
    std::lock_guard<std::mutex> lock(mapMutex_);
    RCLCPP_INFO(this->get_logger(), "Clearing map.");
    map_->clear();
    if (alwaysClearWithInitializer_) {
        initializeWithTF();
    }       
}

void ElevationMappingNode::clearMapWithInitializer(const std::shared_ptr<std_srvs::srv::Empty::Request> request, std::shared_ptr<std_srvs::srv::Empty::Response> response){
  RCLCPP_INFO(this->get_logger(), "Clearing map with initializer");
  map_->clear();
  initializeWithTF();  
  
}

void ElevationMappingNode::initializeWithTF() {
  std::vector<Eigen::Vector3d> points;
  const auto& timeStamp = this->now();
  int i = 0;
  Eigen::Vector3d p;
  for (const auto& frame_id : initialize_frame_id_) {
    // Get tf from map frame to tf frame
    Eigen::Affine3d transformationBaseToMap;
    geometry_msgs::msg::TransformStamped transformStamped;
    try {
      transformStamped = tfBuffer_->lookupTransform(mapFrameId_, frame_id, timeStamp, tf2::durationFromSec(1.0));
      transformationBaseToMap = tf2::transformToEigen(transformStamped);
    } catch (tf2::TransformException& ex) {
      RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
      return;
    }
    p = transformationBaseToMap.translation();
    p.z() += initialize_tf_offset_[i];
    points.push_back(p);
    i++;
  }
  if (!points.empty() && points.size() < 3) {
    points.emplace_back(p + Eigen::Vector3d(initializeTfGridSize_, initializeTfGridSize_, 0));
    points.emplace_back(p + Eigen::Vector3d(-initializeTfGridSize_, initializeTfGridSize_, 0));
    points.emplace_back(p + Eigen::Vector3d(initializeTfGridSize_, -initializeTfGridSize_, 0));
    points.emplace_back(p + Eigen::Vector3d(-initializeTfGridSize_, -initializeTfGridSize_, 0));
  }
  RCLCPP_INFO(this->get_logger(), "Initializing map with points using %s", initializeMethod_.c_str());
  map_->initializeWithPoints(points, initializeMethod_);
}

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



void ElevationMappingNode::setPublishPoint(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                                           std::shared_ptr<std_srvs::srv::SetBool::Response> response) {    
    enablePointCloudPublishing_ = request->data;
    response->success = true;    
}


void ElevationMappingNode::updateVariance() {
  map_->update_variance();
}

void ElevationMappingNode::updateTime() {
  map_->update_time();
}

void ElevationMappingNode::publishStatistics() {
  auto now = this->now();
  double dt = (now - lastStatisticsPublishedTime_).seconds();
  lastStatisticsPublishedTime_ = now;
  elevation_map_msgs::msg::Statistics msg;
  msg.header.stamp = now;
  if (dt > 0.0) {
    msg.pointcloud_process_fps = pointCloudProcessCounter_ / dt;
  }
  pointCloudProcessCounter_ = 0;
  statisticsPub_->publish(msg);
}

void ElevationMappingNode::updateGridMap() {
  std::vector<std::string> layers(map_layers_all_.begin(), map_layers_all_.end());
  std::lock_guard<std::mutex> lock(mapMutex_);
  map_->get_grid_map(gridMap_, layers);
  gridMap_.setTimestamp(this->now().nanoseconds());
  alivePub_->publish(std_msgs::msg::Empty());

  // Mostly debug purpose
  if (enablePointCloudPublishing_) {
    publishAsPointCloud(gridMap_);
  }
  if (enableNormalArrowPublishing_) {
    publishNormalAsArrow(gridMap_);
  }
  isGridmapUpdated_ = true;
}

void ElevationMappingNode::initializeMap(const std::shared_ptr<elevation_map_msgs::srv::Initialize::Request> request,
                                         std::shared_ptr<elevation_map_msgs::srv::Initialize::Response> response) {
  // If initialize method is points
  if (request->type == elevation_map_msgs::srv::Initialize::Request::POINTS) {
    std::vector<Eigen::Vector3d> points;
    for (const auto& point : request->points) {
      const auto& pointFrameId = point.header.frame_id;
      const auto& timeStamp = point.header.stamp;
      const auto& pvector = Eigen::Vector3d(point.point.x, point.point.y, point.point.z);

      // Get tf from map frame to points' frame
      if (mapFrameId_ != pointFrameId) {
        Eigen::Affine3d transformationBaseToMap;
        geometry_msgs::msg::TransformStamped transformStamped;
        try {
          transformStamped = tfBuffer_->lookupTransform(mapFrameId_, pointFrameId, timeStamp, tf2::durationFromSec(1.0));
          transformationBaseToMap = tf2::transformToEigen(transformStamped);
        } catch (tf2::TransformException& ex) {
          RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
          response->success = false;
          return;
        }
        const auto transformed_p = transformationBaseToMap * pvector;
        points.push_back(transformed_p);
      } else {
        points.push_back(pvector);
      }
    }
    std::string method;
    switch (request->method) {
      case elevation_map_msgs::srv::Initialize::Request::NEAREST:
        method = "nearest";
        break;
      case elevation_map_msgs::srv::Initialize::Request::LINEAR:
        method = "linear";
        break;
      case elevation_map_msgs::srv::Initialize::Request::CUBIC:
        method = "cubic";
        break;
    }
    RCLCPP_INFO(this->get_logger(), "Initializing map with points using %s", method.c_str());
    map_->initializeWithPoints(points, method);
  }
  response->success = true;
}

void ElevationMappingNode::publishNormalAsArrow(const grid_map::GridMap& map) const {
  auto startTime = this->now();

  const auto& normalX = map["normal_x"];
  const auto& normalY = map["normal_y"];
  const auto& normalZ = map["normal_z"];
  double scale = 0.1;

  visualization_msgs::msg::MarkerArray markerArray;
  // For each cell in map.
  for (grid_map::GridMapIterator iterator(map); !iterator.isPastEnd(); ++iterator) {
    if (!map.isValid(*iterator, "elevation")) {
      continue;
    }
    grid_map::Position3 p;
    map.getPosition3("elevation", *iterator, p);
    Eigen::Vector3d start = p;
    const auto i = iterator.getLinearIndex();
    Eigen::Vector3d normal(normalX(i), normalY(i), normalZ(i));
    Eigen::Vector3d end = start + normal * scale;
    if (normal.norm() < 0.1) {
      continue;
    }
    markerArray.markers.push_back(vectorToArrowMarker(start, end, i));
  }
  normalPub_->publish(markerArray);
  
}



visualization_msgs::msg::Marker ElevationMappingNode::vectorToArrowMarker(const Eigen::Vector3d& start, const Eigen::Vector3d& end,
                                                                         const int id) const {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = mapFrameId_;
  marker.header.stamp = this->now();
  marker.ns = "normal";
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::ARROW;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.points.resize(2);
  marker.points[0].x = start.x();
  marker.points[0].y = start.y();
  marker.points[0].z = start.z();
  marker.points[1].x = end.x();
  marker.points[1].y = end.y();
  marker.points[1].z = end.z();
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.01;
  marker.scale.y = 0.01;
  marker.scale.z = 0.01;
  marker.color.a = 1.0;  // Don't forget to set the alpha!
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;

  return marker;
}


void ElevationMappingNode::publishMapToOdom(double error) {
  geometry_msgs::msg::TransformStamped transform_stamped;
  transform_stamped.header.stamp = this->now();
  transform_stamped.header.frame_id = mapFrameId_;
  transform_stamped.child_frame_id = correctedMapFrameId_;
  transform_stamped.transform.translation.x = 0.0;
  transform_stamped.transform.translation.y = 0.0;
  transform_stamped.transform.translation.z = error;

  tf2::Quaternion q;
  q.setRPY(0, 0, 0);
  transform_stamped.transform.rotation.x = q.x();
  transform_stamped.transform.rotation.y = q.y();
  transform_stamped.transform.rotation.z = q.z();
  transform_stamped.transform.rotation.w = q.w();

  tfBroadcaster_->sendTransform(transform_stamped);
}



}  // namespace elevation_mapping_cupy
