/**
 * @file        lookup.cpp
 * @authors     Fabian Jenelten, Ruben Grandia
 * @date        04.08, 2021
 * @affiliation ETH RSL
 * @brief       implementation of lookup
 */

// grid map filters rsl.
#include <grid_map_filters_rsl/lookup.hpp>

// stl.
#include <limits>

namespace grid_map {
namespace lookup {

LookupResult maxValueBetweenLocations(const grid_map::Position& position1, const grid_map::Position& position2,
                                      const grid_map::GridMap& gridMap, const grid_map::Matrix& data) {
  // Map corner points into grid map. The line iteration doesn't account for the case where the line does not intersect the map.
  const grid_map::Position startPos = gridMap.getClosestPositionInMap(position1);
  const grid_map::Position endPos = gridMap.getClosestPositionInMap(position2);

  // Line iteration
  float searchMaxValue = std::numeric_limits<float>::lowest();
  grid_map::Index maxIndex(0, 0);
  for (grid_map::LineIterator iterator(gridMap, startPos, endPos); !iterator.isPastEnd(); ++iterator) {
    const grid_map::Index index = *iterator;
    const auto value = data(index(0), index(1));
    if (std::isfinite(value)) {
      searchMaxValue = std::max(searchMaxValue, value);
      maxIndex = index;
    }
  }

  // Get position of max
  grid_map::Position maxPosition;
  gridMap.getPosition(maxIndex, maxPosition);

  const bool maxValueFound = searchMaxValue > std::numeric_limits<float>::lowest();
  return {maxValueFound, searchMaxValue, maxPosition};
}

std::vector<grid_map::Position3> valuesBetweenLocations(const grid_map::Position& position1, const grid_map::Position& position2,
                                                        const grid_map::GridMap& gridMap, const grid_map::Matrix& data) {
  // Map corner points into grid map. The line iteration doesn't account for the case where the line does not intersect the map.
  const grid_map::Position startPos = gridMap.getClosestPositionInMap(position1);
  const grid_map::Position endPos = gridMap.getClosestPositionInMap(position2);

  // Approximate amount of points to reserve memory
  const auto manhattanDistance = std::max(std::abs(endPos.x() - startPos.x()), std::abs(endPos.y() - startPos.y()));
  const int manhattanPixels = std::ceil(manhattanDistance / gridMap.getResolution()) + 1;

  // Container for results
  std::vector<grid_map::Position3> lineValues;
  lineValues.reserve(manhattanPixels);

  // Line iteration
  for (grid_map::LineIterator iterator(gridMap, startPos, endPos); !iterator.isPastEnd(); ++iterator) {
    const grid_map::Index index = *iterator;
    const auto value = data(index(0), index(1));

    if (std::isfinite(value)) {
      grid_map::Position position;
      gridMap.getPosition(index, position);
      lineValues.push_back({position.x(), position.y(), value});
    }
  }

  return lineValues;
}

}  // namespace lookup
}  // namespace grid_map