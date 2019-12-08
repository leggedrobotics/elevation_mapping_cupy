//
// Created by andrej on 12/6/19.
//

#ifndef CONVEX_PLANE_EXTRACTION_INCLUDE_TYPES_HPP_
#define CONVEX_PLANE_EXTRACTION_INCLUDE_TYPES_HPP_

#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/mat.hpp"

namespace convex_plane_extraction {

  typedef Eigen::VectorXd VectorXd;
  typedef Eigen::MatrixXd MatrixXd;
  typedef Eigen::Vector3d Vector3d;
  typedef Eigen::Matrix3d Matrix3d;
  typedef Eigen::Matrix2d Matrix2d;
  typedef Eigen::Vector2d Vector2d;
  typedef Eigen::VectorXf VectorXf;
  typedef Eigen::MatrixXf MatrixXf;
  typedef Eigen::Vector3f Vector3f;
  typedef Eigen::Matrix3f Matrix3f;
  typedef Eigen::Matrix2f Matrix2f;
  typedef Eigen::Vector2f Vector2f;
  typedef Eigen::VectorXi VectorXi;
  typedef Eigen::MatrixXi MatrixXi;
  typedef Eigen::Vector3i Vector3i;
  typedef Eigen::Matrix3i Matrix3i;
  typedef Eigen::Matrix2i Matrix2i;
  typedef Eigen::Vector2i Vector2i;
  typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;

  typedef cv::Point2f   PointCV;

}

#endif //CONVEX_PLANE_EXTRACTION_INCLUDE_TYPES_HPP_
