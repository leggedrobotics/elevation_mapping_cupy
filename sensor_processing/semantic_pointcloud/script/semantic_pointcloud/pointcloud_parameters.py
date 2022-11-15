from dataclasses import dataclass, field
from simple_parsing.helpers import Serializable
from typing import Tuple, List


@dataclass
class FeatureExtractorParameter(Serializable):
    name: str = "DINO"
    interpolation: str = "bilinear"
    model: str = "vit_small"
    patch_size: int = 16
    dim: int = 5
    dropout: bool = False
    dino_feat_type: str = "feat"
    projection_type: str = "nonlinear"
    input_size: List[int] = field(default_factory=[80, 160].copy)


@dataclass
class PointcloudParameter(Serializable):
    sensor_name: str = "camera"
    topic_name: str = "/elvation_mapping/pointcloud_semantic"
    channels: list = ["rgb", "feat_0", "feat_1", "c_prob_0"].copy
    fusion: list = ["color", "average", "average", "class_average"].copy

    semantic_segmentation: bool = True
    segmentation_model: str = "detectron_coco_panoptic_fpn_R_101_3x"
    publish_segmentation_image: bool = True
    segmentation_image_topic: str = "/semantic_pointcloud/sem_seg"

    cam_info_topic: str = "/zed2i/zed_node/depth/camera_info"
    image_topic: str = "/zed2i/zed_node/left/image_rect_color"
    depth_topic: str = "/zed2i/zed_node/depth/depth_registered"
    cam_frame: str = "zed2i_right_camera_optical_frame"
    confidence: bool = True
    confidence_topic: str = "/zed2i/zed_node/confidence/confidence_map"
    confidence_threshold: int = 10

    feature_extractor: bool = True
    feature_config: FeatureExtractorParameter = FeatureExtractorParameter
