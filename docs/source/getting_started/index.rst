.. _introduction:



Introduction
******************************************************************



Subscribed Topics
-------------------------------------------------------------------
* topics specified in **`pointcloud_topics`** in **`elevation_mapping_cupy/config/parameters.yaml`** ([sensor_msgs/PointCloud2])

  The distance measurements.

* **`/tf`** ([tf/tfMessage])

  The transformation tree.

* The plane segmentation node subscribes to an elevation map topic ([grid_map_msg/GridMap]). This can be configured in
  **`convex_plane_decomposition_ros/config/parameters.yaml`**

Published Topics
-------------------------------------------------------------------
For elevation_mapping_cupy, topics are published as set in the rosparam.
You can specify which layers to publish in which fps.

Under `publishers`, you can specify the `topic_name`, `layers` `basic_layers` and `fps`.

.. code-block:: yaml

  publishers:
      your_topic_name:
        layers: [ 'list_of_layer_names', 'layer1', 'layer2' ] # Choose from 'elevation', 'variance', 'traversability', 'time' + plugin layers
        basic_layers: [ 'list of basic layers', 'layer1' ] # basic_layers for valid cell computation (e.g. Rviz): Choose a subset of `layers`.
        fps: 5.0  # Publish rate. Use smaller value than `map_acquire_fps`.


Example setting in `config/parameters.yaml`.

* **`elevation_map_raw`** ([grid_map_msg/GridMap])

  The entire elevation map.

* **`elevation_map_recordable`** ([grid_map_msg/GridMap])

  The entire elevation map with slower update rate for visualization and logging.

* **`elevation_map_filter`** ([grid_map_msg/GridMap])

  The filtered maps using plugins.

The plane segmentation node publishes the following:

* **`planar_terrain`**  ([convex_plane_decomposition_msgs/PlanarTerrain])

  A custom message that contains the full segmentation as a map together with the boundary information.

* **`filtered_map`**  ([grid_map_msg/GridMap])

  A grid map message to visualize the segmentation and some intermediate results. This information is also part of **`planar_terrain`**.

* **`boundaries`**  ([visualization_msgs/MarkerArray])

  A set of polygons that trace the boundaries of the segmented region. Holes and boundaries of a single region are published as separate
  markers with the same color.

* **`insets`**  ([visualization_msgs/PolygonArray])

  A set of polygons that are at a slight inward offset from **`boundaries`**. There might be more insets than boundaries since the inward
  shift can cause a single region to break down into multiple when narrow passages exist.
