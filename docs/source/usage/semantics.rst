.. _semantics:

Semantics
******************************************************************

The elevation map is also able to include semantic information.
The workflow consists in two elements:

* semantic extension of the elevation map
* semantic pointcloud

semantic extension of the elevation map
==========================================

The semantics of the elevation map can be configured in the sensor parameter file.
The sensor parameter file contains all the topics the map subscribes to.

The channels and fusion lists are the parameters that define the semantics of the elevation map.
The channel is a list that contains the name of the semantic layers as well as the name of the channel
of the data that the elevation map node subscribes to.
The fusion list indicates the fusion algorithm type which is applied to the data to fuse the
sensor data into the map.

There are different fusion algorithms types:

* average
""""""""""""""""""""""""""

    Takes the average of all the points that fall within the same cell and average them
    and then takes a weighted average of the existing value.

  use case: semantic features

* bayesian_inference:
""""""""""""""""""""""""""

  Employs a gaussian bayesian inference at each iteration. Where we use the psoterior
  of the previous iteration as the prior to the new iteration.

  use case: semantic features


* class_average
""""""""""""""""""""""""""

    Takes the average of all the points that fall within the same cell and average them
    and then takes a weighted average of the existing value. If the previous value is zero
  it weights the previous value with a zero weight.

  use case: class probabilities

* class_bayesian
""""""""""""""""""""""""""

  Employs a bayesian inference of a categorical distribution with a dirichlet prior.
  The alpha hyperparameters of the dirichlet prior are updated at every iteration,
  such that the posterior of iteration t-1 is the prior of t.

  use case: class probabilities


* color
""""""""""""""""""""""""""

  The color is subscribed as a uint32. The color of the cell is the result of the average of
  all the points within that cell.

  use case: rgb color information

Semantic pointcloud
=======================================

Sensors do not always directly provide all semantic information.
The semantic pointcloud is a ROS node that subscribes to stereo cameras and generates a
multichannel pointcloud containing semantic information additionally to the goemteric position of
the points.
The pointcloud is also configured from the sensor_parameter file.


