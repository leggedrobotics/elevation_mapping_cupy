.. _installation:


.. toctree::
    :hidden:
    :maxdepth: 2

    Cuda installation <cuda_installation>



Installation
******************************************************************

CUDA & cuDNN
==================================================================

The tested versions are CUDA10.2, 11.6

`CUDA <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation>`_
`cuDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux>`_


Check how to install :ref:`here<cuda_installation>`.



Python dependencies
-------------------------------------------------------------------

You will need

* `cupy <https://cupy.chainer.org/>`_
* `numpy <https://www.numpy.org/>`_
* `scipy <https://www.scipy.org/>`_
* `shapely==1.7.1 <https://github.com/Toblerity/Shapely>`_

For traversability filter, either of

* `torch <https://pytorch.org/>`_
* `chainer <https://chainer.org/>`_

Optionally, OpenCV for inpainting filter.

* `opencv-python <https://opencv.org/>`_

Install `numpy`, `scipy`, `shapely`, `opencv-python` with the following command.

.. code-block:: bash

  pip3 install -r requirements.txt


Cupy
-------------------------------------------------------------------


cupy can be installed with specific CUDA versions. (On jetson, only "from source" i.e. `pip install cupy` could work)
For CUDA 10.2

.. code-block:: bash

  pip install cupy-cuda102

For CUDA 11.0

.. code-block:: bash

  pip install cupy-cuda110

For CUDA 11.1

.. code-block:: bash

  pip install cupy-cuda111

For CUDA 11.2

.. code-block:: bash

  pip install cupy-cuda112

For CUDA 11.3

.. code-block:: bash

  pip install cupy-cuda113

For CUDA 11.4

.. code-block:: bash

  pip install cupy-cuda114

For CUDA 11.5

.. code-block:: bash

  pip install cupy-cuda115

For CUDA 11.6

.. code-block:: bash

  pip install cupy-cuda116

(Install CuPy from source)

.. code-block:: bash

  pip install cupy

Traversability filter
-------------------------------------------------------------------

You can choose either pytorch, or chainer to run the CNN based traversability filter.  
Install by following the official documents.

* `torch <https://pytorch.org/>`_
* `chainer <https://chainer.org/>`_

Pytorch uses ~2GB more GPU memory than Chainer, but runs a bit faster.  
Use parameter `use_chainer` to select which backend to use.

ROS package dependencies
-------------------------------------------------------------------

* `pybind11_catkin <https://github.com/ipab-slmc/pybind11_catkin>`_
* `grid_map <https://github.com/ANYbotics/grid_map>`_

.. code-block:: bash

  sudo apt install ros-noetic-pybind11-catkin
  sudo apt install ros-noetic-grid-map-core ros-noetic-grid-map-msgs


On Jetson
==================================================================

CUDA CuDNN
-------------------------------------------------------------------

`CUDA` and `cuDNN` can be installed via apt. It comes with nvidia-jetpack. The tested version is jetpack 4.5 with L4T 32.5.0.

Python dependencies
-------------------------------------------------------------------

On jetson, you need the version for its CPU arch:

.. code-block:: bash
    
    wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
    pip3 install Cython
    pip3 install numpy==1.19.5 torch-1.8.0-cp36-cp36m-linux_aarch64.whl


Also, you need to install cupy with

.. code-block:: bash

  pip3 install cupy


This builds the packages from source so it would take time.

ROS dependencies
-----------------------

* `pybind11_catkin <https://github.com/ipab-slmc/pybind11_catkin>`_
* `grid_map <https://github.com/ANYbotics/grid_map>`_

.. code-block:: bash

  sudo apt install ros-melodic-pybind11-catkin
  sudo apt install ros-melodic-grid-map-core ros-melodic-grid-map-msgs


Also, on jetson you need fortran (should already be installed).

.. code-block:: bash

  sudo apt install gfortran


If the Jetson is set up with Jetpack 4.5 with ROS Melodic the following package is additionally required:

.. code-block:: bash

  git clone git@github.com:ros/filters.git -b noetic-devel


Plane segmentation dependencies
==================================================================

OpenCV
-------------------------------------------------------------------

.. code-block:: bash

  sudo apt install libopencv-dev

Eigen
-------------------------------------------------------------------

.. code-block:: bash

  sudo apt install libeigen3-dev

CGAL
-------------------------------------------------------------------

CGAL5 is required. It will be automatically downloaded and installed into the catkin workspace by the cgal5_catkin package. Make sure you
have the third-party libaries installed on you machine:

.. code-block:: bash

  sudo apt install libgmp-dev
  sudo apt install libmpfr-dev
  sudo apt install libboost-all-dev


