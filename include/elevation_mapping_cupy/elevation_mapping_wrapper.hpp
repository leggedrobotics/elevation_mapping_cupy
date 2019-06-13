#include <pybind11/embed.h> // everything needed for embedding
#include <iostream>
#include <Eigen/Dense>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/transform_listener.h>
// Grid Map
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
// PCL
// #include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

namespace py = pybind11;


namespace elevation_mapping_cupy{

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class ElevationMappingWrapper {
  public:
    ElevationMappingWrapper();
    ~ElevationMappingWrapper()=default;

    void input(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud, const RowMatrixXd& R, const Eigen::VectorXd& t);
    void move_to(const Eigen::VectorXd& p);
    void get_maps(std::vector<Eigen::MatrixXd>& maps);
    void get_grid_map(grid_map::GridMap& gridMap);

    void pointCloudToMatrix(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud, RowMatrixXd& points);
  private:
    py::object map_;
    double resolution_;
    // grid_map::Length length_;
    double map_length_;
};


class ElevationMappingNode {
  public:
    ElevationMappingNode(ros::NodeHandle& nh);
    ~ElevationMappingNode() = default;

  private:
    void readParameters();
    void pointcloudCallback(const sensor_msgs::PointCloud2& cloud);
    void poseCallback(const geometry_msgs::PoseWithCovarianceStamped& pose);
    ros::NodeHandle nh_;
    ros::Subscriber pointcloudSub_;
    ros::Subscriber poseSub_;
    ros::Publisher mapPub_;
    tf::TransformListener transformListener_;
    ElevationMappingWrapper map_;
    std::string mapFrameId_;
    grid_map::GridMap gridMap_;
};

}
