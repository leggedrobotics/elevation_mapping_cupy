# Elevation Mapping cupy

![python tests](https://github.com/leggedrobotics/elevation_mapping_semantic_cupy/actions/workflows/python-tests.yml/badge.svg)

[Documentation](https://leggedrobotics.github.io/elevation_mapping_cupy/)


## Overview

The Elevaton Mapping CuPy software package represents an advancement in robotic navigation and locomotion. 
Integrating with the Robot Operating System (ROS) and utilizing GPU acceleration, this framework enhances point cloud registration and ray casting,
crucial for efficient and accurate robotic movement, particularly in legged robots.
![screenshot](doc/media/main_repo.png)
![screenshot](doc/media/main_mem.png)
![gif](doc/media/convex_approximation.gif)

## Key Features

- **Height Drift Compensation**: Tackles state estimation drifts that can create mapping artifacts, ensuring more accurate terrain representation.

- **Visibility Cleanup and Artifact Removal**: Raycasting methods and an exclusion zone feature are designed to remove virtual artifacts and correctly interpret overhanging obstacles, preventing misidentification as walls.

- **Learning-based Traversability Filter**: Assesses terrain traversability using local geometry, improving path planning and navigation.

- **Versatile Locomotion Tools**: Incorporates smoothing filters and plane segmentation, optimizing movement across various terrains.

- **Multi-Modal Elevation Map (MEM) Framework**: Allows seamless integration of diverse data like geometry, semantics, and RGB information, enhancing multi-modal robotic perception.

- **GPU-Enhanced Efficiency**: Facilitates rapid processing of large data structures, crucial for real-time applications.

## Overview

![Overview of multi-modal elevation map structure](doc/media/overview.png)

Overview of our multi-modal elevation map structure. The framework takes multi-modal images (purple) and multi-modal (blue) point clouds as
input. This data is input into the elevation map by first associating the data to the cells and then fused with different fusion algorithms into the various
layers of the map. Finally the map can be post-processed with various custom plugins to generate new layers (e.g. traversability) or process layer for
external components (e.g. line detection).

## Citing
If you use the Elevation Mapping CuPy, please cite the following paper:
Elevation Mapping for Locomotion and Navigation using GPU

[Elevation Mapping for Locomotion and Navigation using GPU](https://arxiv.org/abs/2204.12876)

Takahiro Miki, Lorenz Wellhausen, Ruben Grandia, Fabian Jenelten, Timon Homberger, Marco Hutter  

```bibtex
@misc{mikielevation2022,
    doi = {10.48550/ARXIV.2204.12876},
    author = {Miki, Takahiro and Wellhausen, Lorenz and Grandia, Ruben and Jenelten, Fabian and Homberger, Timon and Hutter, Marco},
    keywords = {Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Elevation Mapping for Locomotion and Navigation using GPU},
    publisher = {International Conference on Intelligent Robots and Systems (IROS)},
    year = {2022},
}
```

If you use the Multi-modal Elevation Mapping for color or semantic layers, please cite the following paper:

[MEM: Multi-Modal Elevation Mapping for Robotics and Learning](https://arxiv.org/abs/2309.16818v1)

Gian Erni, Jonas Frey, Takahiro Miki, Matias Mattamala, Marco Hutter

```bibtex
@misc{Erni2023-bs,
    title = "{MEM}: {Multi-Modal} Elevation Mapping for Robotics and Learning",
    author = "Erni, Gian and Frey, Jonas and Miki, Takahiro and Mattamala, Matias and Hutter, Marco",
    publisher = {International Conference on Intelligent Robots and Systems (IROS)},
    year = {2023},
}
```

## Quick instructions to run:

### Installation

First, clone to your catkin_ws

```zsh
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone https://github.com/leggedrobotics/elevation_mapping_cupy.git
```

Then install dependencies.
You can also use docker which already install all dependencies.

```zsh
cd docker
./build.sh
./run.sh
```

### Build package

Inside docker container.
```zsh
cd $HOME/catkin_ws
catkin build elevation_mapping_cupy
```

### Run turtlebot example.
![Elevation map examples](doc/media/turtlebot.png)

```bash
export TURTLEBOT3_MODEL=waffle
roslaunch elevation_mapping_cupy turtlesim_simple_example.launch
```

For fusing semantics into the map such as rgb from a multi modal pointcloud:

```bash
export TURTLEBOT3_MODEL=waffle
roslaunch elevation_mapping_cupy turtlesim_semantic_pointcloud_example.launch
```

For fusing semantics into the map such as rgb semantics or features from an image:

```bash
export TURTLEBOT3_MODEL=waffle
roslaunch elevation_mapping_cupy turtlesim_semantic_image_example.launch
```

For plane segmentation:

```bash
catkin build convex_plane_decomposition_ros
export TURTLEBOT3_MODEL=waffle
roslaunch elevation_mapping_cupy turtlesim_plane_decomposition_example.launch
```

To control the robot with a keyboard, a new terminal window needs to be opened.
Then run

```bash
export TURTLEBOT3_MODEL=waffle
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

Velocity inputs can be sent to the robot by pressing the keys `a`, `w`, `d`, `x`. To stop the robot completely, press `s`.
