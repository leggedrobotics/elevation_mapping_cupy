/**
 * @file        GridMapDerivative.hpp
 * @authors     Fabian Jenelten
 * @date        20.05, 2021
 * @affiliation ETH RSL
 * @brief       Computes first and second order derivatives for a grid map. Intended to be used for online applications (i.e, derivatives
 *              are not precomputed but computed at the time and location where they are needed).
 */

#pragma once

// grid_map_core
#include <grid_map_core/grid_map_core.hpp>

namespace grid_map {
namespace derivative {
class GridMapDerivative {
 private:
  using Gradient = Eigen::Vector2d;
  using Curvature = Eigen::Matrix2d;

  //! Grid map resolution (hard coded for efficiency).
  static constexpr double res_ = 0.04;

  //! Kernel size for derivatives.
  static constexpr int kernelSize_ = 5;

 public:
  GridMapDerivative();
  ~GridMapDerivative() = default;

  /**
   * Compute local gradient using grid map. Gradient is set to zero if the index is outside of the grid map.
   * @param gridMap     The grid map
   * @param gradient    gradient vector in world frame
   * @param index       grid map index
   * @param H           grid map corresponding to layer (Eigen implementation)
   */
  void estimateGradient(const grid_map::GridMap& gridMap, Gradient& gradient, const grid_map::Index& index,
                        const grid_map::Matrix& H) const;

  /**
   * Compute local height gradient and curvature using grid map. Gradient and curvature are set to zero if the index is outside of the
   * grid map.
   * @param gridMap         The grid map
   * @param gradient        gradient vector in world frame
   * @param curvature       curvature matrix in world frame
   * @param index           grid map index
   * @param H               grid map corresponding to layer (Eigen implementation)
   */
  void estimateGradientAndCurvature(const grid_map::GridMap& gridMap, Gradient& gradient, Curvature& curvature,
                                    const grid_map::Index& index, const grid_map::Matrix& H) const;

  //! Returns resolution used to compute derivative kernels.
  constexpr static double getRes() { return res_; }

 private:
  /**
   * If the index is not within the grid map, then it will clip it back s.t. the index distance is the closest.
   * @param gridMap     The grid map
   * @param index       Grid map index
   */
  void mapIndexToGrid(const grid_map::GridMap& gridMap, grid_map::Index& index) const;

  //! First order derivative kernel (https://en.wikipedia.org/wiki/Finite_difference_coefficient).
  static constexpr std::array<double, kernelSize_> kernelD1_{-1.0 / (12.0 * res_), 2.0 / (3.0 * res_), 0.0, -2.0 / (3.0 * res_),
                                                             1.0 / (12.0 * res_)};

  // static constexpr std::array<double, kernelSize_> kernelD1_{1.0 / (2.0 * res_), 0.0, -1.0 / (2.0 * res_)};

  //! Second order derivative kernel (https://en.wikipedia.org/wiki/Finite_difference_coefficient).
  static constexpr std::array<double, kernelSize_> kernelD2_{-1.0 / (12.0 * res_ * res_), 4.0 / (3.0 * res_ * res_),
                                                             -5.0 / (2.0 * res_ * res_), 4.0 / (3.0 * res_ * res_),
                                                             -1.0 / (12.0 * res_ * res_)};

  // static constexpr std::array<double, kernelSize_> kernelD2_{1.0 / (res_ * res_), -2.0 / (res_ * res_), 1.0 / (res_ * res_)};
};
}  // namespace derivative
}  // namespace grid_map
