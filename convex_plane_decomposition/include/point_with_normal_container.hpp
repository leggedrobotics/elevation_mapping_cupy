//
// Created by andrej on 11/21/19.
//

#ifndef CONVEX_PLANE_EXTRACTION_POINTWITHNORMALCONTAINER_HPP_
#define CONVEX_PLANE_EXTRACTION_POINTWITHNORMALCONTAINER_HPP_

#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <CGAL/property_map.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

namespace point_with_normal_container {
    // Typedefs
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef Kernel::Point_3 Point_3D;
    typedef Kernel::Vector_3 Vector_3D;
    typedef std::pair<Kernel::Point_3, Kernel::Vector_3> PointWithNormal;
    typedef std::vector<PointWithNormal> PwnVector;

    struct PointIterator{

        explicit PointIterator(std::vector<PointWithNormal>::iterator c){
          it = c;
        }

        Point_3D& operator* (){
          return std::get<0>(*it);
        }

        PointIterator operator++ (){
          it++;
          return *this;
        }

        std::vector<PointWithNormal>::iterator it;
    };

    class PointWithNormalContainer {
    public:

        PointWithNormalContainer();

        virtual ~PointWithNormalContainer();

        const Point_3D& getPoint(size_t index) const;
        const Vector_3D& getNormalVector(size_t index) const;
        void addPointNormalPair(Eigen::Vector3d& position, Eigen::Vector3d& normal);
        void addPointNormalPair(Point_3D& position, Vector_3D& normal);

        std::vector<PointWithNormal>& getReference(){
          return container;
        }

        // Pointer to first element in container
        auto begin();

        auto end();

        PointIterator pointsBegin(){
          return PointIterator(container.begin());
        };

        PointIterator pointsEnd(){
          return PointIterator(container.end());
        };

        Point_3D& getPoint(size_t index){
          return std::get<0>(container.at(index));
        }

    private:
        std::vector<PointWithNormal> container;
    };

}

#endif //GRID_MAP_DEMOS_POINTWITHNORMALCONTAINER_HPP_
