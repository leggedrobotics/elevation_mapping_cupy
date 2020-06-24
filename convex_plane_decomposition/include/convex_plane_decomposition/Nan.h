//
// Created by rgrandia on 11.06.20.
//

#pragma once

#include <limits>

namespace convex_plane_decomposition {

constexpr float nanfPatch = std::numeric_limits<float>::max();

inline bool isNan(float val) {
  return val == nanfPatch;
}

inline bool isNan(double val) {
  throw std::runtime_error("Should not call the custom isNan with a double");
}

inline void patchNans(Eigen::MatrixXf& matrix) {
  matrix = matrix.unaryExpr([](float val) { return (std::isnan(val)) ? nanfPatch : val; });
}

inline void reapplyNans(Eigen::MatrixXf& matrix) {
  matrix = matrix.unaryExpr([](float val) { return (isNan(val)) ? std::nanf("") : val; });
}

}  // namespace convex_plane_decomposition
