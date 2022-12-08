#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from dataclasses import dataclass
import pickle
import numpy as np
from simple_parsing.helpers import Serializable
from dataclasses import field

# todo add this for the future
@dataclass
class Subscriber(Serializable):
    sensor_name: str = "front_cam"
    channels: list = field(default_factory=lambda: ["feat_0"])
    fusion: list = field(default_factory=lambda: ["average"])
    topic_name: str = '/elevation_mapping/pointcloud_semantic'

@dataclass
class Subscribers(Serializable):
    front_cam: Subscriber = Subscriber

@dataclass
class Parameter(Serializable):
    # todo: fix subscriber
    subscriber_cfg: dict = field(default_factory=lambda: {})

    resolution: float = 0.04
    subscribers: dict = field(default_factory=lambda: {'front_cam': {'cam_frame': 'zed2i_right_camera_optical_frame', 'cam_info_topic': '/zed2i/zed_node/depth/camera_info', 'channels': ['rgb', 'person'], 'confidence': True, 'confidence_threshold': 10, 'confidence_topic': '/zed2i/zed_node/confidence/confidence_map', 'depth_topic': '/zed2i/zed_node/depth/depth_registered', 'feature_extractor': False, 'fusion': ['color', 'class_average'], 'image_topic': '/zed2i/zed_node/left/image_rect_color', 'segmentation_model': 'lraspp_mobilenet_v3_large', 'semantic_segmentation': True, 'show_label_legend': True, 'topic_name': '/elvation_mapping/pointcloud_semantic'}})
    additional_layers: list = field(default_factory=lambda: ["feat_0"]) 
    fusion_algorithms: list = field(default_factory=lambda: ["average"])
    data_type: str = np.float32
    average_weight: float = 0.5

    map_length: float = 8.0
    sensor_noise_factor: float = 0.05
    mahalanobis_thresh: float = 2.0
    outlier_variance: float = 0.01
    drift_compensation_variance_inlier: float = 0.1
    time_variance: float = 0.01
    time_interval: float = 0.1

    max_variance: float = 1.0
    dilation_size: float = 2
    dilation_size_initialize: float = 10
    drift_compensation_alpha: float = 1.0

    traversability_inlier: float = 0.1
    wall_num_thresh: float = 100
    min_height_drift_cnt: float = 100

    max_ray_length: float = 2.0
    cleanup_step: float = 0.01
    cleanup_cos_thresh: float = 0.5
    min_valid_distance: float = 0.3
    max_height_range: float = 1.0
    ramped_height_range_a: float = 0.3
    ramped_height_range_b: float = 1.0
    ramped_height_range_c: float = 0.2

    safe_thresh: float = 0.5
    safe_min_thresh: float = 0.5
    max_unsafe_n: int = 20
    checker_layer: str = "traversability"

    min_filter_size: int = 5
    min_filter_iteration: int = 3

    max_drift: float = 0.10

    overlap_clear_range_xy: float = 4.0
    overlap_clear_range_z: float = 2.0

    enable_edge_sharpen: bool = True
    enable_drift_compensation: bool = True
    enable_visibility_cleanup: bool = True
    enable_overlap_clearance: bool = True
    use_only_above_for_upper_bound: bool = True
    use_chainer: bool = True
    position_noise_thresh: float = 0.1
    orientation_noise_thresh: float = 0.1

    plugin_config_file: str = "config/plugin_config.yaml"
    weight_file: str = "config/weights.dat"

    initial_variance: float = 10.0
    initialized_variance: float = 10.0
    w1: np.ndarray = np.zeros((4, 1, 3, 3))
    w2: np.ndarray = np.zeros((4, 1, 3, 3))
    w3: np.ndarray = np.zeros((4, 1, 3, 3))
    w_out: np.ndarray = np.zeros((1, 12, 1, 1))

    # # not configurable params
    true_map_length: float = None
    cell_n: int = None
    true_cell_n: int = None

    def load_weights(self, filename):
        with open(filename, "rb") as file:
            weights = pickle.load(file)
            self.w1 = weights["conv1.weight"]
            self.w2 = weights["conv2.weight"]
            self.w3 = weights["conv3.weight"]
            self.w_out = weights["conv_final.weight"]

    def get_names(self):
        return list(self.__annotations__.keys())

    def get_types(self):
        return [v.__name__ for v in self.__annotations__.values()]

    def set_value(self, name, value):
        setattr(self, name, value)

    def get_value(self, name):
        return getattr(self, name)

    def update(self):
        # +2 is a border for outside map
        self.cell_n = int(round(self.map_length / self.resolution)) + 2
        self.true_cell_n = round(self.map_length / self.resolution)
        self.true_map_length = self.true_cell_n * self.resolution
        semantic_layers = []
        fusion_algorithms = []
        for subscriber,sub_val in self.subscribers.items():
            channels = sub_val["channels"]
            fusion = sub_val["fusion"]
            for i in range(len(channels)):
                name = channels[i]
                if name not in semantic_layers:
                    semantic_layers.append(name)
                    fusion_algorithms.append(fusion[i])
        self.additional_layers = semantic_layers
        self.fusion_algorithms = fusion_algorithms



if __name__ == "__main__":
    param = Parameter()
    print(param)
    print(param.resolution)
    param.set_value("resolution", 0.1)
    print(param.resolution)

    print("names ", param.get_names())
    print("types ", param.get_types())
