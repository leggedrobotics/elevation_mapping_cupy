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

| :doc:`getting_started/index` - What is elevation mapping cupy
| :doc:`getting_started/installation` - How to install the elevation map
| :doc:`getting_started/tutorial` - How to launch the first elevation map




This is a ROS package for elevation mapping on GPU. The elevation mapping code is written in python and uses cupy for GPU computation. The
plane segmentation is done independently and runs on CPU. When the plane segmentation is generated, local convex approximations of the
terrain can be efficiently generated.

.. image:: ../media/main_repo.png
    :alt: Elevation map examples


Citing
---------------
.. hint:: 

    Takahiro Miki, Lorenz Wellhausen, Ruben Grandia, Fabian Jenelten, Timon Homberger, Marco Hutter  

    Elevation Mapping for Locomotion and Navigation using GPU  `Link <https://arxiv.org/abs/2204.12876>`_



.. code-block:: none

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

