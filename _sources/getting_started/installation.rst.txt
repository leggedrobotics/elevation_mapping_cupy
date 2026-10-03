.. _installation:


Installation
******************************************************************
This page describes the supported installation path for the official
``ros2`` branch.

Supported Platform
==================================================================

The release surface documented here is:

* Ubuntu 24.04
* ROS2 Jazzy
* NVIDIA GPU with a working host driver
* CUDA 12.x userspace for CuPy

The ``main`` branch documents the legacy ROS1 line. Do not use those
instructions for this branch.

Workspace Setup
==================================================================

Create a ROS2 workspace and clone the ``ros2`` branch explicitly.

.. code-block:: bash

  mkdir -p ~/ros2_ws/src
  cd ~/ros2_ws/src
  git clone -b ros2 https://github.com/leggedrobotics/elevation_mapping_cupy.git

Source ROS2 Jazzy before building:

.. code-block:: bash

  source /opt/ros/jazzy/setup.bash

System Dependencies
==================================================================

Install ROS dependencies with ``rosdep``.

.. code-block:: bash

  cd ~/ros2_ws
  rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y \
    --skip-keys "cupy-cuda12x numpy_lessthan_2 simple-parsing"

The package manifest deliberately depends on a CUDA-specific CuPy wheel and a
``numpy<2`` constraint. If your local ``rosdep`` database does not provide
those keys, install them directly with ``pip``:

.. code-block:: bash

  python3 -m pip install --upgrade pip
  python3 -m pip install "numpy<2.0.0" simple-parsing cupy-cuda12x

If you want to run the restored semantic demos, install a matching
``torchvision`` build for your local PyTorch/CUDA stack:

.. code-block:: bash

  python3 -m pip install torchvision

Build
==================================================================

Build both the core mapping package and the in-repo semantic sensor package.

.. code-block:: bash

  cd ~/ros2_ws
  colcon build \
    --symlink-install \
    --packages-up-to semantic_sensor elevation_mapping_cupy \
    --cmake-args -DBUILD_TESTING=ON

Overlay the workspace:

.. code-block:: bash

  source ~/ros2_ws/install/setup.bash

Quick Validation
==================================================================

First confirm that CuPy can see the GPU:

.. code-block:: bash

  python3 -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"

Then run the package tests:

.. code-block:: bash

  cd ~/ros2_ws
  colcon test --packages-select elevation_mapping_cupy --event-handlers console_direct+

For the semantic sensor package unit tests:

.. code-block:: bash

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
    src/elevation_mapping_cupy/sensor_processing/semantic_sensor/test -q

For the ROS2 integration test with the DDS workarounds used in CI:

.. code-block:: bash

  ros2 daemon stop
  FASTDDS_BUILTIN_TRANSPORTS=UDPv4 python3 -m launch_testing.launch_test \
    src/elevation_mapping_cupy/elevation_mapping_cupy/test/test_tf_gridmap_integration.py

Container Workflow
==================================================================

The repository still ships a Docker workflow under ``docker/``. That path is
useful when you want a pinned ROS2/CUDA userspace, but the container still
needs GPU access from the host driver.

.. code-block:: bash

  cd ~/ros2_ws/src/elevation_mapping_cupy/docker
  ./run.sh

Release Validation Status
==================================================================

The ``v2.2.0`` ROS 2/Jazzy release validation used the pinned GPU CI container
on an NVIDIA RTX 4090. See :doc:`../release_notes/jazzy_release` for the exact
test surface and deployment note.
