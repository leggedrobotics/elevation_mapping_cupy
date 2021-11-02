//
// Created by rgrandia on 10.08.20.
//

#include <gtest/gtest.h>

#include "signed_distance_field/GridmapSignedDistanceField.h"
#include "signed_distance_field/PixelBorderDistance.h"
#include "signed_distance_field/SignedDistance2d.h"

#include "naiveSignedDistance.h"

TEST(testSignedDistance3d, flatTerrain) {
  const int n = 3;
  const int m = 4;
  const float resolution = 0.1;
  const float terrainHeight = 0.5;
  const grid_map::Matrix map = grid_map::Matrix::Constant(n, m, terrainHeight);
  const float minHeight = map.minCoeff();
  const float maxHeight = map.maxCoeff();

  const float testHeightAboveTerrain = 3.0;
  const auto naiveSignedDistanceAbove = signed_distance_field::naiveSignedDistanceAtHeight(map, testHeightAboveTerrain, resolution);
  const auto signedDistanceAbove =
      signed_distance_field::signedDistanceAtHeight(map, testHeightAboveTerrain, resolution, minHeight, maxHeight);
  ASSERT_TRUE(signed_distance_field::isEqualSdf(signedDistanceAbove, naiveSignedDistanceAbove, 1e-4));

  const float testHeightBelowTerrain = -3.0;
  const auto naiveSignedDistanceBelow = signed_distance_field::naiveSignedDistanceAtHeight(map, testHeightBelowTerrain, resolution);
  const auto signedDistanceBelow =
      signed_distance_field::signedDistanceAtHeight(map, testHeightBelowTerrain, resolution, minHeight, maxHeight);
  ASSERT_TRUE(signed_distance_field::isEqualSdf(signedDistanceBelow, naiveSignedDistanceBelow, 1e-4));
}

TEST(testSignedDistance3d, randomTerrain) {
  const int n = 20;
  const int m = 30;
  const float resolution = 0.1;
  grid_map::Matrix map = grid_map::Matrix::Random(n, m);  // random [-1.0, 1.0]
  const float minHeight = map.minCoeff();
  const float maxHeight = map.maxCoeff();

  // Check at different heights, resulting in different levels of sparsity.
  float heightStep = 0.1;
  for (float height = -1.0 - heightStep; height < 1.0 + heightStep; height += heightStep) {
    const auto naiveSignedDistance = signed_distance_field::naiveSignedDistanceAtHeight(map, height, resolution);
    const auto signedDistance = signed_distance_field::signedDistanceAtHeight(map, height, resolution, minHeight, maxHeight);

    ASSERT_TRUE(signed_distance_field::isEqualSdf(signedDistance, naiveSignedDistance, 1e-4)) << "height: " << height;
  }
}

TEST(testSignedDistance3d, randomTerrainInterpolation) {
  const int n = 20;
  const int m = 30;
  const float resolution = 0.1;
  grid_map::GridMap map;
  map.setGeometry({n * resolution, m * resolution}, resolution);
  map.add("elevation");
  map.get("elevation").setRandom();  // random [-1.0, 1.0]
  const grid_map::Matrix mapData = map.get("elevation");
  const float minHeight = mapData.minCoeff();
  const float maxHeight = mapData.maxCoeff();

  signed_distance_field::GridmapSignedDistanceField sdf(map, "elevation", minHeight, maxHeight);

  // Check at different heights/
  for (float height = minHeight; height < maxHeight; height += resolution) {
    const auto naiveSignedDistance = signed_distance_field::naiveSignedDistanceAtHeight(mapData, height, resolution);

    for (int i = 0; i < mapData.rows(); ++i) {
      for (int j = 0; j < mapData.rows(); ++j) {
        grid_map::Position position2d;
        map.getPosition({i, j}, position2d);

        const auto sdfValue = sdf.value({position2d.x(), position2d.y(), height});
        const auto sdfCheck = naiveSignedDistance(i, j);
        ASSERT_LT(std::abs(sdfValue - sdfCheck), 1e-4);
      }
    }
  }
}