//
// Created by rgrandia on 07.08.20.
//

#include <gtest/gtest.h>

#include "signed_distance_field/PixelBorderDistance.h"

namespace {
/*
 * Specialization of the pixel border distance to test for the correctness of the equidistant point.
 * The function contained in the library can only be called for casts from integer i, and j
 */
inline float squarePixelBorderDistanceTest(float i, float j, float f) {
  if (i == j) {
    return f;
  } else {
    float diff = std::abs(i - j) - 0.5F;
    diff = std::max(diff, 0.0F);  // ! adaptation needed to test for non integer i, j
    return diff * diff + f;
  }
}
}  // namespace

TEST(testPixelBorderDistance, distanceFunction) {
  using signed_distance_field::pixelBorderDistance;
  // Basic properties of the distance function
  ASSERT_TRUE(pixelBorderDistance(0, 0) == 0.0F);
  ASSERT_FLOAT_EQ(pixelBorderDistance(0, 1), 0.5);
  ASSERT_FLOAT_EQ(pixelBorderDistance(0, 2), 1.5);
  ASSERT_TRUE(pixelBorderDistance(0, 1) == pixelBorderDistance(1, 0));
  ASSERT_TRUE(pixelBorderDistance(-10, 42) == pixelBorderDistance(42, -10));
}

TEST(testPixelBorderDistance, equidistantPoint) {
  using signed_distance_field::equidistancePoint;

  int pixelRange = 10;
  float offsetRange = 20.0;
  float offsetStep = 0.25;
  float tol = 1e-4;

  for (int p = -pixelRange; p < pixelRange; ++p) {
    for (float fp = -offsetRange; fp < offsetRange; fp += offsetStep) {
      for (int q = -pixelRange; q < pixelRange; ++q) {
        for (float fq = -offsetRange; fq < offsetRange; fq += offsetStep) {
          // Fix that offset is the same if pixels are the same
          if (p == q) {
            fp = fq;
          }
          // Check symmetry of the equidistant point computation
          float s0 = equidistancePoint(q, fq, p, fp);
          float s1 = equidistancePoint(p, fp, q, fq);
          ASSERT_LT(std::abs(s0 - s1), tol);

          // Check that the distance from s0 to p and q is indeed equal
          float dp = squarePixelBorderDistanceTest(s0, p, fp);
          float dq = squarePixelBorderDistanceTest(s0, q, fq);
          ASSERT_LT(std::abs(dp - dq), tol) << "p: " << p << ", q: " << q << ", fp: " << fp << ", fq: " << fq;
        }
      }
    }
  }
}
