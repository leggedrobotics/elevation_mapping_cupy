//
// Created by rgrandia on 09.06.20.
//

#include "convex_plane_decomposition/ConvexRegionGrowing.h"

#include "convex_plane_decomposition/GeometryUtils.h"

namespace convex_plane_decomposition {

CgalPolygon2d createRegularPolygon(const CgalPoint2d& center, double radius, int numberOfVertices) {
  assert(numberOfVertices > 2);
  CgalPolygon2d polygon;
  double angle = (2. * M_PI) / numberOfVertices;
  for (int i = 0; i < numberOfVertices; ++i) {
    double phi = i * angle;
    double px = radius * std::cos(phi) + center.x();
    double py = radius * std::sin(phi) + center.y();
    // Counter clockwise
    polygon.push_back({px, py});
  }
  return polygon;
}

int getCounterClockWiseNeighbour(int i, int lastVertex) {
  return (i > 0) ? i - 1 : lastVertex;
}

int getClockWiseNeighbour(int i, int lastVertex) {
  return (i < lastVertex) ? i + 1 : 0;
}

void updateMean(CgalPoint2d & mean, const CgalPoint2d& oldValue, const CgalPoint2d& updatedValue, int N) {
  // old_mean = 1/N * ( others + old_value); -> others = N*old_mean - old_value
  // new_mean = 1/N * ( others + new_value); -> new_mean = old_mean - 1/N * oldValue + 1/N * updatedValue
  mean += 1.0 / N * (updatedValue - oldValue);
}

std::array<CgalPoint2d, 2> getNeighbours(const CgalPolygon2d& polygon, int i) {
  assert(i < polygon.size());
  assert(polygon.size() > 1);
  int lastVertex = static_cast<int>(polygon.size()) - 1;
  int cwNeighbour = getClockWiseNeighbour(i, lastVertex);
  int ccwNeighbour = getCounterClockWiseNeighbour(i, lastVertex);
  return {polygon.vertex(cwNeighbour), polygon.vertex(ccwNeighbour)};
}

std::array<CgalPoint2d, 4> get2ndNeighbours(const CgalPolygon2d& polygon, int i) {
  assert(i < polygon.size());
  assert(polygon.size() > 1);
  int lastVertex = static_cast<int>(polygon.size()) - 1;
  int cwNeighbour1 = getClockWiseNeighbour(i, lastVertex);
  int cwNeighbour2 = getClockWiseNeighbour(cwNeighbour1, lastVertex);
  int ccwNeighbour1 = getCounterClockWiseNeighbour(i, lastVertex);
  int ccwNeighbour2 = getCounterClockWiseNeighbour(ccwNeighbour1, lastVertex);
  return {polygon.vertex(cwNeighbour2), polygon.vertex(cwNeighbour1), polygon.vertex(ccwNeighbour1), polygon.vertex(ccwNeighbour2)};
}

bool remainsConvexWhenMovingPoint(const CgalPolygon2d& polygon, int i, const CgalPoint2d& point) {
  auto secondNeighbours = get2ndNeighbours(polygon, i);
  CgalPolygon2d subPolygon;
  subPolygon.container().reserve(5);
  subPolygon.push_back(secondNeighbours[0]);
  subPolygon.push_back(secondNeighbours[1]);
  subPolygon.push_back(point);
  subPolygon.push_back(secondNeighbours[2]);
  subPolygon.push_back(secondNeighbours[3]);
  return subPolygon.is_convex();
}

bool pointAndNeighboursAreWithinFreeSphere(const std::array<CgalPoint2d, 2>& neighbours, const CgalPoint2d& point,
                                           const CgalCircle2d& circle) {
  return isInside(neighbours[0], circle) && isInside(point, circle) && isInside(neighbours[1], circle);
}

template <typename T>
bool pointCanBeMoved(const CgalPolygon2d& growthShape, int i, const CgalPoint2d& candidatePoint, CgalCircle2d& freeSphere,
                     const T& parentShape) {
  if (remainsConvexWhenMovingPoint(growthShape, i, candidatePoint)) {
    auto neighbours = getNeighbours(growthShape, i);
    if (pointAndNeighboursAreWithinFreeSphere(neighbours, candidatePoint, freeSphere)) {
      return true;
    } else {
      // Update free sphere around new point
      freeSphere = CgalCircle2d(candidatePoint, squaredDistance(candidatePoint, parentShape));
      CgalSegment2d segment0{neighbours[0], candidatePoint};
      CgalSegment2d segment1{neighbours[1], candidatePoint};

      bool segment0IsFree = isInside(neighbours[0], freeSphere) || !doEdgesIntersect(segment0, parentShape);
      if (segment0IsFree) {
        bool segment1IsFree = isInside(neighbours[1], freeSphere) || !doEdgesIntersect(segment1, parentShape);
        return segment1IsFree;
      } else {
        return false;
      }
    }
  } else {
    return false;
  }
}

template <typename T>
CgalPolygon2d growConvexPolygonInsideShape_impl(const T& parentShape, CgalPoint2d center, int numberOfVertices, double growthFactor) {
  constexpr double initialRadiusFactor = 0.999;
  double radius = initialRadiusFactor * distance(center, parentShape);
  CgalPolygon2d growthShape = createRegularPolygon(center, radius, numberOfVertices);

  // Cached values per vertex
  std::vector<bool> blocked(numberOfVertices, false);
  std::vector<CgalCircle2d> freeSpheres(numberOfVertices, CgalCircle2d(center, radius*radius));

  int Nblocked = 0;
  while (Nblocked < numberOfVertices) {
    for (int i = 0; i < numberOfVertices; i++) {
      if (!blocked[i]) {
        const auto candidatePoint = getPointOnLine(center, growthShape.vertex(i), growthFactor);
        if (pointCanBeMoved(growthShape, i, candidatePoint, freeSpheres[i], parentShape)) {
          updateMean(center, growthShape.vertex(i), candidatePoint, numberOfVertices);
          growthShape.vertex(i) = candidatePoint;
        } else {
          blocked[i] = true;
          ++Nblocked;
        }
      }
    }
  }
  return growthShape;
}

CgalPolygon2d growConvexPolygonInsideShape(const CgalPolygon2d & parentShape, CgalPoint2d center, int numberOfVertices, double growthFactor) {
  return growConvexPolygonInsideShape_impl(parentShape, center, numberOfVertices, growthFactor);
}

CgalPolygon2d growConvexPolygonInsideShape(const CgalPolygonWithHoles2d& parentShape, CgalPoint2d center, int numberOfVertices,
                                           double growthFactor) {
  return growConvexPolygonInsideShape_impl(parentShape, center, numberOfVertices, growthFactor);
}
}  // namespace convex_plane_decomposition
