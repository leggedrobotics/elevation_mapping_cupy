##################################################
Multi-modal elevation mapping's documentation
##################################################
Welcome to elevation mapping documentation

.. image:: https://github.com/leggedrobotics/elevation_mapping_semantic_cupy/actions/workflows/python-tests.yml/badge.svg
    :target: https://github.com/leggedrobotics/elevation_mapping_semantic_cupy/actions/workflows/python-tests.yml/badge.svg
    :alt: python tests

.. image:: https://github.com/leggedrobotics/elevation_mapping_semantic_cupy/actions/workflows/documentation.yml/badge.svg
    :target: https://github.com/leggedrobotics/elevation_mapping_semantic_cupy/actions/workflows/documentation.yml/badge.svg
    :alt: documentation

Index
---------------

| :doc:`getting_started/introduction` - What is elevation mapping cupy
| :doc:`getting_started/installation` - How to install the elevation map
| :doc:`getting_started/tutorial` - How to launch the first elevation map


This is a ROS package for elevation mapping on GPU. The elevation mapping code is written in python and uses cupy for GPU computation. The
plane segmentation is done independently and runs on CPU. When the plane segmentation is generated, local convex approximations of the
terrain can be efficiently generated.

.. image:: ../media/main_repo.png
    :alt: Elevation map examples
.. image:: ../media/main_mem.png
    :alt: Overview of the project


Citing
---------------
If you use the elevation mapping cupy, please cite the following paper:
Elevation Mapping for Locomotion and Navigation using GPU

.. hint:: 

    Elevation Mapping for Locomotion and Navigation using GPU  `Link <https://arxiv.org/abs/2204.12876>`_

    Takahiro Miki, Lorenz Wellhausen, Ruben Grandia, Fabian Jenelten, Timon Homberger, Marco Hutter  

.. code-block::

    @misc{mikielevation2022,
        doi = {10.48550/ARXIV.2204.12876},
        author = {Miki, Takahiro and Wellhausen, Lorenz and Grandia, Ruben and Jenelten, Fabian and Homberger, Timon and Hutter, Marco},
        keywords = {Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
        title = {Elevation Mapping for Locomotion and Navigation using GPU},
        publisher = {International Conference on Intelligent Robots and Systems (IROS)},
        year = {2022},
    }

Multi-modal elevation mapping if you use color or semantic layers

.. hint::

    MEM: Multi-Modal Elevation Mapping for Robotics and Learning  `Link <https://arxiv.org/abs/2309.16818v1>`_

    Gian Erni, Jonas Frey, Takahiro Miki, Matias Mattamala, Marco Hutter

.. code-block::

    @misc{Erni2023-bs,
        title = "{MEM}: {Multi-Modal} Elevation Mapping for Robotics and Learning",
        author = "Erni, Gian and Frey, Jonas and Miki, Takahiro and Mattamala, Matias and Hutter, Marco",
        publisher = {International Conference on Intelligent Robots and Systems (IROS)},
        year = {2023},
    }
