//
// Copyright (c) 2022, Takahiro Miki. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#pragma once

// STL
#include <iostream>
#include <mutex>

// Eigen
#include <Eigen/Dense>

// Pybind
#include <pybind11/embed.h>  // everything needed for embedding
#include <rclcpp/qos.hpp>
// ROS2
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <point_cloud_transport/point_cloud_transport.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/empty.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// Grid Map
#include <grid_map_msgs/srv/get_grid_map.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <grid_map_ros/grid_map_ros.hpp>

// PCL
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <elevation_map_msgs/srv/check_safety.hpp>
#include <elevation_map_msgs/srv/initialize.hpp>
#include <elevation_map_msgs/msg/channel_info.hpp>
#include <elevation_map_msgs/msg/statistics.hpp>

#include "elevation_mapping_cupy/elevation_mapping_wrapper.hpp"

namespace py = pybind11;

namespace elevation_mapping_cupy {

class ElevationMappingNode : public rclcpp::Node {
 public:
  ElevationMappingNode(const rclcpp::NodeOptions& options);
  using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ColMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

  // using ImageSubscriber = image_transport::SubscriberFilter;
  using ImageSubscriber = message_filters::Subscriber<sensor_msgs::msg::Image>;  
  using ImageSubscriberPtr = std::shared_ptr<ImageSubscriber>;

  // Subscriber and Synchronizer for CameraInfo messages
  using CameraInfoSubscriber = message_filters::Subscriber<sensor_msgs::msg::CameraInfo>;
  using CameraInfoSubscriberPtr = std::shared_ptr<CameraInfoSubscriber>;
  using CameraPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>;
  using CameraSync = message_filters::Synchronizer<CameraPolicy>;
  using CameraSyncPtr = std::shared_ptr<CameraSync>;

  // Subscriber and Synchronizer for ChannelInfo messages
  using ChannelInfoSubscriber = message_filters::Subscriber<elevation_map_msgs::msg::ChannelInfo>;
  using ChannelInfoSubscriberPtr = std::shared_ptr<ChannelInfoSubscriber>;
  using CameraChannelPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo, elevation_map_msgs::msg::ChannelInfo>;
  using CameraChannelSync = message_filters::Synchronizer<CameraChannelPolicy>;
  using CameraChannelSyncPtr = std::shared_ptr<CameraChannelSync>;

  // Subscriber and Synchronizer for Pointcloud messages
  // using PointCloudSubscriber = message_filters::Subscriber<sensor_msgs::msg::PointCloud2>;
  // using PointCloudSubscriberPtr = std::shared_ptr<PointCloudSubscriber>;
  // using PointCloudPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, elevation_map_msgs::msg::ChannelInfo>;
  // using PointCloudSync = message_filters::Synchronizer<PointCloudPolicy>;
  // using PointCloudSyncPtr = std::shared_ptr<PointCloudSync>;

 private:
void readParameters();
void setupMapPublishers();
void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud, const std::string& key);    
void pointcloudtransportCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud, const std::string& key);    

void inputPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud, const std::vector<std::string>& channels);
  
void inputImage(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                  const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg,
                  const std::vector<std::string>& channels);


// void imageCallback(const sensor_msgs::msg::Image::SharedPtr image_msg, const sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg, const std::string& key);
// void imageChannelCallback(const sensor_msgs::msg::Image::SharedPtr image_msg, const sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg, const elevation_map_msgs::msg::ChannelInfo::SharedPtr channel_info_msg);
void imageCallback(const sensor_msgs::msg::Image::SharedPtr cloud, const std::string& key);
void imageChannelCallback(const elevation_map_msgs::msg::ChannelInfo::SharedPtr chennel_info, const std::string& key);
void imageInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr image_info, const std::string& key);
//   void pointCloudChannelCallback(const sensor_msgs::PointCloud2& cloud, const elevation_map_msgs::ChannelInfoConstPtr& channel_info_msg);
//   // void multiLayerImageCallback(const elevation_map_msgs::MultiLayerImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& camera_info_msg);
void publishAsPointCloud(const grid_map::GridMap& map) const;
bool getSubmap( const std::shared_ptr<grid_map_msgs::srv::GetGridMap::Request> request, std::shared_ptr<grid_map_msgs::srv::GetGridMap::Response> response);


//   bool checkSafety(elevation_map_msgs::CheckSafety::Request& request, elevation_map_msgs::CheckSafety::Response& response);
    
void initializeMap(const std::shared_ptr<elevation_map_msgs::srv::Initialize::Request> request, std::shared_ptr<elevation_map_msgs::srv::Initialize::Response> response);  
void clearMap(const std::shared_ptr<std_srvs::srv::Empty::Request> request, std::shared_ptr<std_srvs::srv::Empty::Response> response);
void clearMapWithInitializer(const std::shared_ptr<std_srvs::srv::Empty::Request> request, std::shared_ptr<std_srvs::srv::Empty::Response> response);



void setPublishPoint(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                        std::shared_ptr<std_srvs::srv::SetBool::Response> response);


void updateVariance();
void updateTime();
void updatePose();
void updateGridMap();
void publishStatistics();

void publishNormalAsArrow(const grid_map::GridMap& map) const;
void initializeWithTF();
void publishMapToOdom(double error);
void publishMapOfIndex(int index);
visualization_msgs::msg::Marker vectorToArrowMarker(const Eigen::Vector3d& start, const Eigen::Vector3d& end, const int id) const;
  
  rclcpp::Node::SharedPtr node_;
  // image_transport::ImageTransport it_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> pointcloudSubs_;

  // std::vector<point_cloud_transport::Subscriber>::SharedPtr> pointcloudtransportSubs_;
  std::vector<point_cloud_transport::Subscriber> pointcloudtransportSubs_;

  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> imageSubs_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> cameraInfoSubs_;
  std::vector<rclcpp::Subscription<elevation_map_msgs::msg::ChannelInfo>::SharedPtr> channelInfoSubs_;
  

  // std::vector<ImageSubscriberPtr> imageSubs_;
  // std::vector<CameraInfoSubscriberPtr> cameraInfoSubs_;
  // std::vector<ChannelInfoSubscriberPtr> channelInfoSubs_;
  // std::vector<CameraSyncPtr> cameraSyncs_;
  // std::vector<CameraChannelSyncPtr> cameraChannelSyncs_;
  // std::vector<PointCloudSyncPtr> pointCloudSyncs_;
  std::vector<rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr> mapPubs_;
  

  std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;
  std::shared_ptr<tf2_ros::TransformListener> tfListener_;
  std::shared_ptr<tf2_ros::Buffer> tfBuffer_;


  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr alivePub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointPub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr normalPub_;
  rclcpp::Publisher<elevation_map_msgs::msg::Statistics>::SharedPtr statisticsPub_;
  rclcpp::Service<grid_map_msgs::srv::GetGridMap>::SharedPtr rawSubmapService_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr clearMapService_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr clearMapWithInitializerService_;
  rclcpp::Service<elevation_map_msgs::srv::Initialize>::SharedPtr initializeMapService_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr setPublishPointService_;
  rclcpp::Service<elevation_map_msgs::srv::CheckSafety>::SharedPtr checkSafetyService_;
  rclcpp::TimerBase::SharedPtr updateVarianceTimer_;
  rclcpp::TimerBase::SharedPtr updateTimeTimer_;
  rclcpp::TimerBase::SharedPtr updatePoseTimer_;
  rclcpp::TimerBase::SharedPtr updateGridMapTimer_;
  rclcpp::TimerBase::SharedPtr publishStatisticsTimer_;
  rclcpp::Time lastStatisticsPublishedTime_;
  
  
  std::shared_ptr<ElevationMappingWrapper> map_;
  // ElevationMappingWrapper map_;
  std::string mapFrameId_;
  std::string correctedMapFrameId_;
  std::string baseFrameId_;


    // map topics info
    std::vector<std::vector<std::string>> map_topics_;
    std::vector<std::vector<std::string>> map_layers_;
    std::vector<std::vector<std::string>> map_basic_layers_;
    std::set<std::string> map_layers_all_;
    std::set<std::string> map_layers_sync_;
    std::vector<double> map_fps_;
    std::set<double> map_fps_unique_;
    std::vector<rclcpp::TimerBase::SharedPtr> mapTimers_;
    std::map<std::string, std::vector<std::string>> channels_;

    std::vector<std::string> initialize_frame_id_;
    std::vector<double> initialize_tf_offset_;
    std::string initializeMethod_;

    Eigen::Vector3d lowpassPosition_;
    Eigen::Vector4d lowpassOrientation_;

    std::mutex mapMutex_;  // protects gridMap_
    grid_map::GridMap gridMap_;
    std::atomic_bool isGridmapUpdated_;  // needs to be atomic (read is not protected by mapMutex_)

    std::mutex errorMutex_;  // protects positionError_, and orientationError_
    double positionError_;
    double orientationError_;

    double positionAlpha_;
    double orientationAlpha_;
    double voxel_filter_size_;
    pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_filter;

    double recordableFps_;
    std::atomic_bool enablePointCloudPublishing_;
    bool enableNormalArrowPublishing_;
    bool enableDriftCorrectedTFPublishing_;
    bool useInitializerAtStart_;
    double initializeTfGridSize_;
    bool alwaysClearWithInitializer_;
    std::atomic_int pointCloudProcessCounter_;

    std::map<std::string, std::pair<sensor_msgs::msg::CameraInfo, bool>> imageInfoReady_;
    std::map<std::string, std::pair<elevation_map_msgs::msg::ChannelInfo, bool>> imageChannelReady_;
};

}  // namespace elevation_mapping_cupy
