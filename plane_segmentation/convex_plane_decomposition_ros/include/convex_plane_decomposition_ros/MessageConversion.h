//
// Created by rgrandia on 14.06.20.
//

#pragma once

#include <geometry_msgs/Pose.h>

#include <convex_plane_decomposition_msgs/BoundingBox2d.h>
#include <convex_plane_decomposition_msgs/PlanarRegion.h>
#include <convex_plane_decomposition_msgs/PlanarTerrain.h>
#include <convex_plane_decomposition_msgs/Point2d.h>
#include <convex_plane_decomposition_msgs/Polygon2d.h>
#include <convex_plane_decomposition_msgs/PolygonWithHoles2d.h>

#include <convex_plane_decomposition/PlanarRegion.h>
#include <convex_plane_decomposition/PolygonTypes.h>

namespace convex_plane_decomposition {

CgalBbox2d fromMessage(const convex_plane_decomposition_msgs::BoundingBox2d& msg);
convex_plane_decomposition_msgs::BoundingBox2d toMessage(const CgalBbox2d& bbox2d);

PlanarRegion fromMessage(const convex_plane_decomposition_msgs::PlanarRegion& msg);
convex_plane_decomposition_msgs::PlanarRegion toMessage(const PlanarRegion& planarRegion);

PlanarTerrain fromMessage(const convex_plane_decomposition_msgs::PlanarTerrain& msg);
convex_plane_decomposition_msgs::PlanarTerrain toMessage(const PlanarTerrain& planarTerrain);

Eigen::Isometry3d fromMessage(const geometry_msgs::Pose& msg);
geometry_msgs::Pose toMessage(const Eigen::Isometry3d& transform);

CgalPoint2d fromMessage(const convex_plane_decomposition_msgs::Point2d& msg);
convex_plane_decomposition_msgs::Point2d toMessage(const CgalPoint2d& point2d);

CgalPolygon2d fromMessage(const convex_plane_decomposition_msgs::Polygon2d& msg);
convex_plane_decomposition_msgs::Polygon2d toMessage(const CgalPolygon2d& polygon2d);

CgalPolygonWithHoles2d fromMessage(const convex_plane_decomposition_msgs::PolygonWithHoles2d& msg);
convex_plane_decomposition_msgs::PolygonWithHoles2d toMessage(const CgalPolygonWithHoles2d& polygonWithHoles2d);

}  // namespace convex_plane_decomposition
