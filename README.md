# Elevation Mapping cupy

![python tests](https://github.com/leggedrobotics/elevation_mapping_semantic_cupy/actions/workflows/python-tests.yml/badge.svg)

## Overview

This is a ROS package for elevation mapping on GPU. The elevation mapping code is written in python and uses cupy for
GPU computation. The
plane segmentation is done independently and runs on CPU. When the plane segmentation is generated, local convex
approximations of the
terrain can be efficiently generated.
![screenshot](doc/main_repo.png)
![gif](doc/convex_approximation.gif)

## Citing

> Takahiro Miki, Lorenz Wellhausen, Ruben Grandia, Fabian Jenelten, Timon Homberger, Marco Hutter  
> Elevation Mapping for Locomotion and Navigation using GPU  [arXiv](https://arxiv.org/abs/2204.12876)

```
@misc{https://doi.org/10.48550/arxiv.2204.12876,
  doi = {10.48550/ARXIV.2204.12876},
  url = {https://arxiv.org/abs/2204.12876},
  author = {Miki, Takahiro and Wellhausen, Lorenz and Grandia, Ruben and Jenelten, Fabian and Homberger, Timon and Hutter, Marco},
  keywords = {Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Elevation Mapping for Locomotion and Navigation using GPU},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Quick instructions to run:

For the lonomy bag:

CPP wrapper:

```zsh
    roslaunch elevation_mapping_cupy lonomy_semantic_elevation_single.launch use_sim_true:=true
    rosbag play --clock ~/bags/good_working_wine_field_zed_topcom_rtk_person_9_2022-07-15-14-37-05.bag 
```

Python wrapper:

````zsh
    python -m elevation_mapping_cupy.elevation_mapping_ros
    roslaunch elevation_mapping_cupy pointcloud.launch use_sim_time:=true
    rosbag play --clock ~/bags/good_working_wine_field_zed_topcom_rtk_person_9_2022-07-15-14-37-05.bag
````

For the anymal bag:

```zsh
    roslaunch elevation_mapping_cupy anymal_semantic_elevation_single.launch use_sim_time:=true
    rosbag play --clock ~/bags/anymal_coyote_2022-12-11-20-01-46.bag
```


