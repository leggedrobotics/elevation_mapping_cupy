/*
 * Utils.h
 *
 *  Created on: Jul 10, 2020
 *      Author: Ruben Grandia
 *   Institute: ETH Zurich
 */

#pragma once

namespace grid_map {
namespace signed_distance_field {

//! Distance value that is considered infinite. Needs to be well below numeric_limits::max to avoid overflow in computation involving ^2
constexpr float INF = 1e15;

}  // namespace signed_distance_field
}  // namespace grid_map