//
// Created by rgrandia on 18.08.20.
//

#include <gtest/gtest.h>

#include "signed_distance_field/Gridmap3dLookup.h"

using signed_distance_field::Gridmap3dLookup;
using size_t_3d = signed_distance_field::Gridmap3dLookup::size_t_3d;

TEST(testGridmap3dLookup, nearestNode) {
  const size_t_3d gridsize{8, 9, 10};
  const Eigen::Vector3d gridOrigin{-0.1, -0.2, -0.4};
  const double resolution = 0.1;

  signed_distance_field::Gridmap3dLookup gridmap3DLookup(gridsize, gridOrigin, resolution);

  // Retreive origin
  const size_t_3d originNode = gridmap3DLookup.nearestNode(gridOrigin);
  ASSERT_EQ(originNode.x, 0);
  ASSERT_EQ(originNode.y, 0);
  ASSERT_EQ(originNode.z, 0);

  // test underflow
  const size_t_3d originNodeProjected = gridmap3DLookup.nearestNode(gridOrigin + Eigen::Vector3d{1e20, 1e20, -1e20});
  ASSERT_EQ(originNodeProjected.x, 0);
  ASSERT_EQ(originNodeProjected.y, 0);
  ASSERT_EQ(originNodeProjected.z, 0);

  // test overflow
  const size_t_3d maxNodeProjected = gridmap3DLookup.nearestNode(gridOrigin + Eigen::Vector3d{-1e20, -1e20, +1e20});
  ASSERT_EQ(maxNodeProjected.x, gridsize.x - 1);
  ASSERT_EQ(maxNodeProjected.y, gridsize.y - 1);
  ASSERT_EQ(maxNodeProjected.z, gridsize.z - 1);

  // Nearest neighbour
  const size_t_3d nodeIndex{3, 4, 5};
  const Eigen::Vector3d nodePosition = gridmap3DLookup.nodePosition(nodeIndex);
  const size_t_3d closestNodeIndex = gridmap3DLookup.nearestNode(nodePosition + 0.49 * Eigen::Vector3d::Constant(resolution));
  ASSERT_EQ(closestNodeIndex.x, nodeIndex.x);
  ASSERT_EQ(closestNodeIndex.y, nodeIndex.y);
  ASSERT_EQ(closestNodeIndex.z, nodeIndex.z);
}

TEST(testGridmap3dLookup, linearIndex) {
  const size_t_3d gridsize{8, 9, 10};
  const Eigen::Vector3d gridOrigin{-0.1, -0.2, -0.4};
  const double resolution = 0.1;

  signed_distance_field::Gridmap3dLookup gridmap3DLookup(gridsize, gridOrigin, resolution);
  ASSERT_EQ(gridmap3DLookup.linearIndex({0, 0, 0}), 0);
  ASSERT_EQ(gridmap3DLookup.linearIndex({gridsize.x - 1, gridsize.y - 1, gridsize.z - 1}), gridmap3DLookup.linearSize() - 1);
}