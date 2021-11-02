//
// Created by rgrandia on 07.08.20.
//

#include <gtest/gtest.h>

#include "signed_distance_field/PixelBorderDistance.h"

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
  using signed_distance_field::squarePixelBorderDistance;

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
          float dp = squarePixelBorderDistance(s0, p, fp);
          float dq = squarePixelBorderDistance(s0, q, fq);
          ASSERT_LT(std::abs(dp - dq), tol) << "p: " << p << ", q: " << q << ", fp: " << fp << ", fq: " << fq;
        }
      }
    }
  }
}
