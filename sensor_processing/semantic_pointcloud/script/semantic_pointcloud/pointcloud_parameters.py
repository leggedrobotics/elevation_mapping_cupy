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
    # input_size: list = field(default_factory=[80, 160].copy)
    input_size: list = field(
        default_factory=lambda: [80, 160]
    )


@dataclass
class PointcloudParameter(Serializable):
    sensor_name: str = "camera"
    topic_name: str = "/elvation_mapping/pointcloud_semantic"
    channels: list = field(
        default_factory=lambda: ["rgb", "person", "grass", "tree", "max"]
    )
    fusion: list = field(
        default_factory=lambda: [
            "color",
            "class_average",
            "class_average",
            "class_average",
            "class_max",
        ]
    )

    semantic_segmentation: bool = True
    segmentation_model: str = "detectron_coco_panoptic_fpn_R_101_3x"
    publish_segmentation_image: bool = True
    segmentation_image_topic: str = "/semantic_pointcloud/sem_seg"
    pub_all: bool = False
    show_label_legend: bool = False

    cam_info_topic: str = "/zed2i/zed_node/depth/camera_info"
    image_topic: str = "/zed2i/zed_node/left/image_rect_color"
    depth_topic: str = "/zed2i/zed_node/depth/depth_registered"
    cam_frame: str = "zed2i_right_camera_optical_frame"
    confidence: bool = True
    confidence_topic: str = "/zed2i/zed_node/confidence/confidence_map"
    confidence_threshold: int = 10

    feature_extractor: bool = False
    feature_config: FeatureExtractorParameter = FeatureExtractorParameter
