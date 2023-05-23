from dataclasses import dataclass, field
from simple_parsing.helpers import Serializable


@dataclass
class FeatureExtractorParameter(Serializable):
    name: str = "DINO"
    interpolation: str = "bilinear"
    model: str = "vit_small"
    patch_size: int = 16
    dim: int = 10
    dropout: bool = False
    dino_feat_type: str = "feat"
    projection_type: str = "nonlinear"
    input_size: list = field(default_factory=lambda: [80, 160])
    pcl: bool = False


@dataclass
class ImageParameter(Serializable):
    image_topic: str = "/alphasense_driver_ros/cam4/debayered"

    semantic_segmentation: bool = True
    sem_seg_topic: str = "/elevation_mapping/semantic_seg"
    sem_seg_image_topic: str = "/elevation_mapping/semantic_seg_im"
    segmentation_model: str = "detectron_coco_panoptic_fpn_R_101_3x"
    show_label_legend: bool = False
    channels: list = field(default_factory=lambda: ["grass", "road", "tree", "sky"])
    fusion: list = field(default_factory=lambda: ["class_average", "class_average", "class_average", "class_average"])

    feature_extractor: bool = False
    feature_config: FeatureExtractorParameter = FeatureExtractorParameter
    # feature_config.input_size: list = field(default_factory=lambda: [80, 160])
    feature_topic: str = "/elevation_mapping/semantic_seg_feat"
    feat_image_topic: str = "/elevation_mapping/semantic_seg_feat_im"
    resize: float = None
    image_info_topic: str = "/elevation_mapping/image_info"

