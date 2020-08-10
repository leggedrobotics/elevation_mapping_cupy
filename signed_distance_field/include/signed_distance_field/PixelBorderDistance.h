//
// Created by rgrandia on 07.08.20.
//

#pragma once

#include <algorithm>

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
 */
inline float squarePixelBorderDistance(float i, float j, float f) {
  float distance = pixelBorderDistance(i, j);
  return distance * distance + f;
}

/**
 * Return the point s in pixel space that is equally far from p and q (taking into account offsets fp, and fq)
 * It is the solution to the following equation:
 *      squarePixelBorderDistance(s, q, fq) == squarePixelBorderDistance(s, p, fp)
 */
float equidistancePoint(int q, float fq, int p, float fp);

}  // namespace signed_distance_field
