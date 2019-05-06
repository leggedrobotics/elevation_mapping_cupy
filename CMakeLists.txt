# %Tag(FULLTEXT)%
cmake_minimum_required(VERSION 2.8.3)
project(elevation_mapping_cupy)
find_package(catkin_simple REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)

cs_install()
cs_export()
