from dataclasses import dataclass, field
from simple_parsing.helpers import Serializable


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
    input_size: list = field(default_factory=lambda: [80, 160])
    feature_image_topic: str = "/semantic_sensor/feature_image"
    pcl: bool = True


@dataclass
class PointcloudParameter(Serializable):
    sensor_name: str = "camera"
    topic_name: str = "/elvation_mapping/pointcloud_semantic"
    channels: list = field(default_factory=lambda: ["rgb", "person", "grass", "tree", "max"])
    fusion: list = field(
        default_factory=lambda: ["color", "class_average", "class_average", "class_average", "class_max",]
    )

    semantic_segmentation: bool = False
    segmentation_model: str = "lraspp_mobilenet_v3_large"
    publish_segmentation_image: bool = False
    segmentation_image_topic: str = "/semantic_sensor/sem_seg"
    pub_all: bool = False
    show_label_legend: bool = False

    cam_info_topic: str = "/zed2i/zed_node/depth/camera_info"
    image_topic: str = "/zed2i/zed_node/left/image_rect_color"
    depth_topic: str = "/zed2i/zed_node/depth/depth_registered"
    cam_frame: str = "zed2i_right_camera_optical_frame"
    confidence: bool = False
    confidence_topic: str = "/zed2i/zed_node/confidence/confidence_map"
    confidence_threshold: int = 10

    feature_extractor: bool = False
    publish_feature_image: bool = False
    feature_config: FeatureExtractorParameter = FeatureExtractorParameter
    feature_config.input_size: list = field(default_factory=lambda: [80, 160])
