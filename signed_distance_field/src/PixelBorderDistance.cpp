//
// Created by rgrandia on 07.08.20.
//

#include "signed_distance_field/PixelBorderDistance.h"

#include <cmath>

namespace signed_distance_field {

namespace internal {

/**
 * Return equidistancepoint between origin and pixel p (with p > 0) with offset fp
 */
inline float intersectionPointRightSideOfOrigin(int p, float fp) {
  auto pFloat = static_cast<float>(p);
  auto pSquared = pFloat * pFloat;
  auto fpAbs = std::abs(fp);
  if (fpAbs >= pSquared) {
    float s = (pSquared + fp) / (2.0F * pFloat);
    return (fp > 0.0F) ? s + 0.5F : s - 0.5F;
  } else {
    float boundary = (pSquared - 2 * pFloat + 1);
    if (fpAbs < boundary) {
      return (pSquared - pFloat + fp) / (2.0F * pFloat - 2.0F);
    } else {
      float s = 0.5F + std::sqrt(fpAbs);
      return (fp > 0.0F) ? s : pFloat - s;
    }
  }
}

/**
 * Return equidistancepoint between origin and pixel p with offset fp
 */
inline float intersectionOffsetFromOrigin(int p, float fp) {
  float intersectionOffset = intersectionPointRightSideOfOrigin(std::abs(p), fp);
  return (p > 0) ? intersectionOffset : -intersectionOffset;
}

}  // namespace internal

float equidistancePoint(int q, float fq, int p, float fp) {
  if (q != p) {
    return internal::intersectionOffsetFromOrigin(p - q, fp - fq) + static_cast<float>(q);
  } else {
    return static_cast<float>(q);
  }
}

}