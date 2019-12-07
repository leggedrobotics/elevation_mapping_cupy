//
// Created by andrej on 11/21/19.
//

#include "point_with_normal_container.hpp"

namespace point_with_normal_container{

    PointWithNormalContainer::PointWithNormalContainer() = default;

    PointWithNormalContainer::~PointWithNormalContainer() = default;

    const Point_3D& PointWithNormalContainer::getPoint(size_t index) const{
      return std::get<0>(container.at(index));
    }

    const Vector_3D& PointWithNormalContainer::getNormalVector(size_t index) const{
      return std::get<1>(container.at(index));
    }

    void PointWithNormalContainer::addPointNormalPair(Eigen::Vector3d& position, Eigen::Vector3d& normal){
      container.push_back(std::make_pair(*reinterpret_cast<Point_3D*>(const_cast<double*>(position.data())),
              *reinterpret_cast<Vector_3D*>(const_cast<double*>(normal.data()))));
    }

    void PointWithNormalContainer::addPointNormalPair(Point_3D& position, Vector_3D& normal){
      container.push_back(std::make_pair(position, normal));
    }

    auto PointWithNormalContainer::begin(){
      return container.begin();
    }

    auto PointWithNormalContainer::end(){
      return container.end();
    }
}