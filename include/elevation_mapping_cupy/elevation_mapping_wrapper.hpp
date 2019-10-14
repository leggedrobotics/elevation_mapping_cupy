#include <pybind11_catkin/pybind11/embed.h> // everything needed for embedding
#include <iostream>
#include <Eigen/Dense>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <std_srvs/Empty.h>
#include <tf/transform_listener.h>
// Grid Map
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_msgs/GetGridMap.h>
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
    void initialize(ros::NodeHandle& nh);

    void input(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud, const RowMatrixXd& R, const Eigen::VectorXd& t,
               const double positionNoise, const double orientationNoise);
    void move_to(const Eigen::VectorXd& p);
    void clear();
    void get_maps(std::vector<Eigen::MatrixXd>& maps);
    void get_grid_map(grid_map::GridMap& gridMap);
    double get_polygon_traversability(std::vector<Eigen::Vector2d>& polygon, Eigen::Vector3d& result);

    void pointCloudToMatrix(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud, RowMatrixXd& points);
  private:
    void setParameters(ros::NodeHandle& nh);
    py::object map_;
    py::object param_;
    double resolution_;
    double map_length_;
    int map_n_;
};

}
