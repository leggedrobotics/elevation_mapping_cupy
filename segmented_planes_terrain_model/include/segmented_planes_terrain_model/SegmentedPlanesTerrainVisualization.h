//
// Created by rgrandia on 24.06.20.
//

#pragma once

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <ocs2_switched_model_interface/terrain/ConvexTerrain.h>

#include <ocs2_quadruped_interface/QuadrupedVisualizationHelpers.h>

namespace switched_model {

visualization_msgs::MarkerArray getConvexTerrainMarkers(const ConvexTerrain& convexTerrain, Color color, double diameter, double linewidth,
                                                        double normalLength);

}  // namespace switched_model
