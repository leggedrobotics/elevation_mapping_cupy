//
// Created by rgrandia on 07.08.20.
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cassert>

namespace signed_distance_field {

/**
 * Returns distance between the center of a pixel and the border of an other pixel.
 * Returns zero if the center is inside the other pixel.
 * Pixels are assumed to have size 1.0F
 * @param i : location of pixel 1
 * @param j : location of pixel 2
 * @return : absolute distance between center of pixel 1 and the border of pixel 2
 */
inline float pixelBorderDistance(float i, float j) {
  return std::max(std::abs(i-j) - 0.5F, 0.0F);
}

/**
 * Returns square pixelBorderDistance, adding offset f.
 * ! only works when i and j are cast from integers.
 */
inline float squarePixelBorderDistance(float i, float j, float f) {
  assert(i == j || std::abs(i - j) >= 1.0F); // Check that i and j are proper pixel locations: either the same pixel or non-overlapping pixels
  if (i == j) {
    return f;
  } else {
    float diff = std::abs(i-j) - 0.5F;
    return diff * diff + f;
  }
}

namespace internal {

/**
 * Return equidistancepoint between origin and pixel p (with p > 0) with offset fp
 */
inline float intersectionPointRightSideOfOrigin(float p, float fp) {
  auto pSquared = p * p;
  auto fpAbs = std::abs(fp);
  if (fpAbs >= pSquared) {
    float s = (pSquared + fp) / (2.0F * p);
    return (fp > 0.0F) ? s + 0.5F : s - 0.5F;
  } else {
    float boundary = pSquared - 2.0F * p + 1.0F;
    if (fpAbs < boundary) {
      return (pSquared - p + fp) / (2.0F * p - 2.0F);
    } else {
      float s = 0.5F + std::sqrt(fpAbs);
      return (fp > 0.0F) ? s : p - s;
    }
  }
}

/**
 * Return equidistancepoint between origin and pixel p with offset fp
 */
inline float intersectionOffsetFromOrigin(float p, float fp) {
  float intersectionOffset = intersectionPointRightSideOfOrigin(std::abs(p), fp);
  return (p > 0) ? intersectionOffset : -intersectionOffset;
}

}  // namespace internal

/**
 * Return the point s in pixel space that is equally far from p and q (taking into account offsets fp, and fq)
 * It is the solution to the following equation:
 *      squarePixelBorderDistance(s, q, fq) == squarePixelBorderDistance(s, p, fp)
 */
inline float equidistancePoint(float q, float fq, float p, float fp) {
  assert(q == p || std::abs(q-p) >= 1.0F); // Check that q and p are proper pixel locations: either the same pixel or non-overlapping pixels
  assert((q == p) ? fp == fq : true); // Check when q and p are equal, the offsets are also equal

  if (fp == fq) { // quite common case when both pixels are of the same class (occupied / free)
    return 0.5F * (p + q);
  } else {
    float df = fp - fq;
    float dr = p - q;
    return internal::intersectionOffsetFromOrigin(dr, df) + q;
  }
}

}  // namespace signed_distance_field
