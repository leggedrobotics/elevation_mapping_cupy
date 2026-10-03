.. _semantics:

Semantics
******************************************************************

The elevation map can also include semantic information. On the ROS2 branch the
workflow consists of two pieces:

* semantic fusion inside ``elevation_mapping_cupy``
* semantic inputs produced by the in-repo ``semantic_sensor`` package or an
  external vision model

Semantic Fusion in the Elevation Map
==========================================

Semantic inputs are configured in the robot sensor YAML under the
``subscribers`` section. Each subscriber can define channel names and a fusion
mode per channel.

The channel list maps incoming semantic channels to map layers. The fusion list
defines how each channel is fused into the persistent map.

Supported fusion modes include:

``average``
  Computes a weighted average between the current cell value and the newly
  observed value.

  Use case: dense semantic features

``bayesian_inference``
  Applies Gaussian Bayesian fusion, using the previous posterior as the next
  prior.

  Use case: continuous semantic features

``class_average``
  Averages per-cell class scores while ignoring uninitialized cells.

  Use case: class probabilities

``class_bayesian``
  Applies Bayesian fusion on a categorical distribution with a Dirichlet
  prior.

  Use case: class probabilities

``color``
  Fuses packed RGB color channels.

  Use case: RGB appearance layers

Semantic Input Producers
=======================================

Sensors do not always publish semantic channels directly. The in-repo
``semantic_sensor`` package provides two launchable producers:

* ``semantic_pointcloud.launch.py`` generates a multi-channel pointcloud.
* ``semantic_image.launch.py`` generates semantic image channels plus
  ``ChannelInfo`` metadata.

The supported end-to-end ROS2 examples are:

* ``turtlesim_semantic_pointcloud_example.launch.py``
* ``turtlesim_semantic_image_example.launch.py``

The intended architecture is simple: a vision model produces semantic channels,
and ``elevation_mapping_cupy`` fuses those channels into map layers.
