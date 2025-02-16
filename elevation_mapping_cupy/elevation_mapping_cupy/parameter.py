#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from dataclasses import dataclass, field
import pickle
import numpy as np
from simple_parsing.helpers import Serializable
from dataclasses import field
from typing import Tuple


@dataclass
class Parameter(Serializable):
    """
    This class holds the parameters for the elevation mapping algorithm.
    
    Attributes:
        resolution: The resolution in meters.
                    (Default: ``0.04``)
        subscriber_cfg: The configuration for the subscriber.
                        (Default: ``{ "front_cam": { "channels": ["rgb", "person"], "topic_name": "/elevation_mapping/pointcloud_semantic", "data_type": "pointcloud", } }``)
        additional_layers: The additional layers for the map.  
                           (Default: ``["color"]``)
        fusion_algorithms: The list of fusion algorithms.  
                           (Default: ``[ "image_color", "image_exponential", "pointcloud_average", "pointcloud_bayesian_inference", "pointcloud_class_average", "pointcloud_class_bayesian", "pointcloud_class_max", "pointcloud_color", ]``)
        pointcloud_channel_fusions: The fusion for pointcloud channels.  
                                   (Default: ``{"rgb": "color", "default": "class_average"}``)
        image_channel_fusions: The fusion for image channels.  
                               (Default: ``{"rgb": "color", "default": "exponential"}``)
        data_type: The data type for the map.  
                   (Default: ``np.float32``)
        average_weight: The weight for the average fusion.  
                        (Default: ``0.5``)
        map_length: The map's size in meters.  
                    (Default: ``8.0``)
        sensor_noise_factor: The point's noise is sensor_noise_factor*z^2 (z is distance from sensor).  
                            (Default: ``0.05``)
        mahalanobis_thresh: Points outside this distance is outlier.  
                            (Default: ``2.0``)
        outlier_variance: If point is outlier, add this value to the cell.  
                          (Default: ``0.01``)
        drift_compensation_variance_inlier: Cells under this value is used for drift compensation.  
                                           (Default: ``0.1``)
        time_variance: Add this value when update_variance is called.  
                       (Default: ``0.01``)
        time_interval: Time layer is updated with this interval.  
                       (Default: ``0.1``)
        max_variance: The maximum variance for each cell.  
                       (Default: ``1.0``)
        dilation_size: The dilation filter size before traversability filter.  
                       (Default: ``2``)
        dilation_size_initialize: The dilation size after the init.  
                                  (Default: ``10``)
        drift_compensation_alpha: The drift compensation alpha for smoother update of drift compensation.  
                                  (Default: ``1.0``)
        traversability_inlier: Cells with higher traversability are used for drift compensation.  
                               (Default: ``0.1``)
        wall_num_thresh: If there are more points than this value, only higher points than the current height are used to make the wall more sharp.  
                         (Default: ``100``)
        min_height_drift_cnt: Drift compensation only happens if the valid cells are more than this number.  
                              (Default: ``100``)
        max_ray_length: The maximum length for ray tracing.  
                        (Default: ``2.0``)
        cleanup_step: Substitute this value from validity layer at visibility cleanup.  
                      (Default: ``0.01``)
        cleanup_cos_thresh: Substitute this value from validity layer at visibility cleanup.  
                            (Default: ``0.5``)
        min_valid_distance: Points with shorter distance will be filtered out.  
                            (Default: ``0.3``)
        max_height_range: Points higher than this value from sensor will be filtered out to disable ceiling.  
                           (Default: ``1.0``)
        ramped_height_range_a: If z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.  
                               (Default: ``0.3``)
        ramped_height_range_b: If z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.  
                               (Default: ``1.0``)
        ramped_height_range_c: If z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.  
                               (Default: ``0.2``)
        safe_thresh: If traversability is smaller, it is counted as unsafe cell.  
                     (Default: ``0.5``)
        safe_min_thresh: Polygon is unsafe if there exists lower traversability than this.  
                          (Default: ``0.5``)
        max_unsafe_n: If the number of cells under safe_thresh exceeds this value, polygon is unsafe.  
                      (Default: ``20``)
        checker_layer: Layer used for checking safety.  
                       (Default: ``"traversability"``)
        max_drift: The maximum drift for the compensation.  
                   (Default: ``0.10``)
        overlap_clear_range_xy: XY range [m] for clearing overlapped area. This defines the valid area for overlap clearance. (used for multi floor setting)  
                               (Default: ``4.0``)
        overlap_clear_range_z: Z range [m] for clearing overlapped area. Cells outside this range will be cleared. (used for multi floor setting)  
                              (Default: ``2.0``)
        enable_edge_sharpen: Enable edge sharpening.  
                             (Default: ``True``)
        enable_drift_compensation: Enable drift compensation.  
                                   (Default: ``True``)
        enable_visibility_cleanup: Enable visibility cleanup.  
                                   (Default: ``True``)
        enable_overlap_clearance: Enable overlap clearance.  
                                  (Default: ``True``)
        use_only_above_for_upper_bound: Use only above for upper bound.  
                                        (Default: ``True``)
        use_chainer: Use chainer as a backend of traversability filter or pytorch. If false, it uses pytorch. Pytorch requires ~2GB more GPU memory compared to chainer but runs faster.  
                     (Default: ``True``)
        position_noise_thresh: If the position change is bigger than this value, the drift compensation happens.  
                              (Default: ``0.1``)
        orientation_noise_thresh: If the orientation change is bigger than this value, the drift compensation happens.  
                                  (Default: ``0.1``)
        plugin_config_file: Configuration file for the plugin.  
                            (Default: ``"config/plugin_config.yaml"``)
        weight_file: Weight file for traversability filter.  
                     (Default: ``"config/weights.dat"``)
        initial_variance: Initial variance for each cell.  
                          (Default: ``10.0``)
        initialized_variance: Initialized variance for each cell.  
                              (Default: ``10.0``)
        w1: Weights for the first layer.  
            (Default: ``np.zeros((4, 1, 3, 3))``)
        w2: Weights for the second layer.  
            (Default: ``np.zeros((4, 1, 3, 3))``)
        w3: Weights for the third layer.  
            (Default: ``np.zeros((4, 1, 3, 3))``)
        w_out: Weights for the output layer.  
               (Default: ``np.zeros((1, 12, 1, 1))``)
        true_map_length: True length of the map.  
                         (Default: ``None``)
        cell_n: Number of cells in the map.  
                (Default: ``None``)
        true_cell_n: True number of cells in the map.  
                     (Default: ``None``)
        
    """
    resolution: float = 0.04  # resolution in m.
    subscriber_cfg: dict = field(
        default_factory=lambda: {
            "front_cam": {
                "channels": ["rgb", "person"],
                "topic_name": "/elevation_mapping/pointcloud_semantic",
                "data_type": "pointcloud",
            }
        }
    )  # configuration for the subscriber
    additional_layers: list = field(default_factory=lambda: ["color"])  # additional layers for the map
    fusion_algorithms: list = field(
        default_factory=lambda: [
            "image_color",
            "image_exponential",
            "pointcloud_average",
            "pointcloud_bayesian_inference",
            "pointcloud_class_average",
            "pointcloud_class_bayesian",
            "pointcloud_class_max",
            "pointcloud_color",
        ]
    )  # list of fusion algorithms
    pointcloud_channel_fusions: dict = field(default_factory=lambda: {"rgb": "color", "default": "class_average"})  # fusion for pointcloud channels
    image_channel_fusions: dict = field(default_factory=lambda: {"rgb": "color", "default": "exponential"})  # fusion for image channels
    data_type: str = np.float32  # data type for the map
    average_weight: float = 0.5  # weight for the average fusion

    map_length: float = 8.0  # map's size in m.
    sensor_noise_factor: float = 0.05  # point's noise is sensor_noise_factor*z^2 (z is distance from sensor).
    mahalanobis_thresh: float = 2.0  # points outside this distance is outlier.
    outlier_variance: float = 0.01  # if point is outlier, add this value to the cell.
    drift_compensation_variance_inlier: float = 0.1  # cells under this value is used for drift compensation.
    time_variance: float = 0.01  # add this value when update_variance is called.
    time_interval: float = 0.1  # Time layer is updated with this interval.

    max_variance: float = 1.0  # maximum variance for each cell.
    dilation_size: int = 2  # dilation filter size before traversability filter.
    dilation_size_initialize: int = 10  # dilation size after the init.
    drift_compensation_alpha: float = 1.0  # drift compensation alpha for smoother update of drift compensation.

    traversability_inlier: float = 0.1  # cells with higher traversability are used for drift compensation.
    wall_num_thresh: int = 100  # if there are more points than this value, only higher points than the current height are used to make the wall more sharp.
    min_height_drift_cnt: int = 100  # drift compensation only happens if the valid cells are more than this number.

    max_ray_length: float = 2.0  # maximum length for ray tracing.
    cleanup_step: float = 0.01  # substitute this value from validity layer at visibility cleanup.
    cleanup_cos_thresh: float = 0.5  # substitute this value from validity layer at visibility cleanup.
    min_valid_distance: float = 0.3  # points with shorter distance will be filtered out.
    max_height_range: float = 1.0  # points higher than this value from sensor will be filtered out to disable ceiling.
    ramped_height_range_a: float = 0.3  # if z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.
    ramped_height_range_b: float = 1.0  # if z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.
    ramped_height_range_c: float = 0.2  # if z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.

    safe_thresh: float = 0.5  # if traversability is smaller, it is counted as unsafe cell.
    safe_min_thresh: float = 0.5  # polygon is unsafe if there exists lower traversability than this.
    max_unsafe_n: int = 20  # if the number of cells under safe_thresh exceeds this value, polygon is unsafe.
    checker_layer: str = "traversability"  # layer used for checking safety

    max_drift: float = 0.10  # maximum drift for the compensation

    overlap_clear_range_xy: float = 4.0  # xy range [m] for clearing overlapped area. this defines the valid area for overlap clearance. (used for multi floor setting)
    overlap_clear_range_z: float = 2.0  # z range [m] for clearing overlapped area. cells outside this range will be cleared. (used for multi floor setting)

    enable_edge_sharpen: bool = True  # enable edge sharpening
    enable_drift_compensation: bool = True  # enable drift compensation
    enable_visibility_cleanup: bool = True  # enable visibility cleanup
    enable_overlap_clearance: bool = True  # enable overlap clearance
    use_only_above_for_upper_bound: bool = True  # use only above for upper bound
    use_chainer: bool = True  # use chainer as a backend of traversability filter or pytorch. If false, it uses pytorch. pytorch requires ~2GB more GPU memory compared to chainer but runs faster.
    position_noise_thresh: float = 0.1  # if the position change is bigger than this value, the drift compensation happens.
    orientation_noise_thresh: float = 0.1  # if the orientation change is bigger than this value, the drift compensation happens.

    plugin_config_file: str = "config/plugin_config.yaml"  # configuration file for the plugin
    weight_file: str = "config/weights.dat"  # weight file for traversability filter

    initial_variance: float = 10.0  # initial variance for each cell.
    initialized_variance: float = 10.0  # initialized variance for each cell.
    w1: np.ndarray = field(default_factory=lambda: np.zeros((4, 1, 3, 3)))  # weights for the first layer
    w2: np.ndarray = field(default_factory=lambda: np.zeros((4, 1, 3, 3)))  # weights for the second layer
    w3: np.ndarray = field(default_factory=lambda: np.zeros((4, 1, 3, 3)))  # weights for the third layer
    w_out: np.ndarray = field(default_factory=lambda: np.zeros((1, 12, 1, 1)))  # weights for the output layer

    # # not configurable params
    true_map_length: float = None  # true length of the map
    cell_n: int = None  # number of cells in the map
    true_cell_n: int = None  # true number of cells in the map

    def load_weights(self, filename: str):
        """
        Load weights from a file into the model's parameters.
        
        Args:
            filename (str): The path to the file containing the weights.
        """
        with open(filename, "rb") as file:
            weights = pickle.load(file)
            self.w1 = weights["conv1.weight"]
            self.w2 = weights["conv2.weight"]
            self.w3 = weights["conv3.weight"]
            self.w_out = weights["conv_final.weight"]

    def get_names(self):
        """
        Get the names of the parameters.
        
        Returns:
            list: A list of parameter names.
        """
        return list(self.__annotations__.keys())

    def get_types(self):
        """
        Get the types of the parameters.
        
        Returns:
            list: A list of parameter types.
        """
        return [v.__name__ for v in self.__annotations__.values()]

    def set_value(self, name, value):
        """
        Set the value of a parameter.
        
        Args:
            name (str): The name of the parameter.
            value (any): The new value for the parameter.
        """
        setattr(self, name, value)

    def get_value(self, name):
        """
        Get the value of a parameter.
        
        Args:
            name (str): The name of the parameter.
        
        Returns:
            any: The value of the parameter.
        """
        return getattr(self, name)

    def update(self):
        """
        Update the parameters related to the map size and resolution.
        """
        # +2 is a border for outside map
        self.cell_n = int(round(self.map_length / self.resolution)) + 2
        self.true_cell_n = round(self.map_length / self.resolution)
        self.true_map_length = self.true_cell_n * self.resolution


if __name__ == "__main__":
    param = Parameter()
    print(param)
    print(param.resolution)
    param.set_value("resolution", 0.1)
    print(param.resolution)

    print("names ", param.get_names())
    print("types ", param.get_types())
