from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize
import torch.nn.functional as NF
import numpy as np

# Setup detectron2 logger
# import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

from semantic_pointcloud.DINO.modules import DinoFeaturizer

# from .DINO.modules import DinoFeaturizer
from semantic_pointcloud.pointcloud_parameters import (
    PointcloudParameter,
    FeatureExtractorParameter,
)

# from .pointcloud_parameters import PointcloudParameter, FeatureExtractorParameter


def resolve_model(name, config: FeatureExtractorParameter = None):
    """Get the model class based on the name of the pretrained model.

    Args:
        name (str): Name of pretrained model
        [fcn_resnet50,lraspp_mobilenet_v3_large,detectron_coco_panoptic_fpn_R_101_3x]

    Returns:
        Dict[str, str]:
    """
    if name == "fcn_resnet50":
        weights = FCN_ResNet50_Weights.DEFAULT
        net = fcn_resnet50
        model = PytorchModel(net, weights)
        return {
            "name": "fcn_resnet50",
            "model": model,
        }
    elif name == "lraspp_mobilenet_v3_large":
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        net = lraspp_mobilenet_v3_large
        model = PytorchModel(net, weights)
        return {
            "name": "lraspp_mobilenet_v3_large",
            "model": model,
        }
    elif name == "detectron_coco_panoptic_fpn_R_101_3x":
        net = ""
        # "Cityscapes/mask_rcnn_R_50_FPN.yaml"
        # "Misc/semantic_R_50_FPN_1x.yaml"
        # "Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"
        # "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
        weights = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
        model = DetectronModel(net, weights)
        return {
            "name": "detectron_coco_panoptic_fpn_R_101_3x",
            "model": model,
        }
    elif name == "DINO":
        weights = config.model + str(config.patch_size)
        model = STEGOModel(weights, config)
        return {
            "name": config.model + str(config.patch_size),
            "model": model,
        }
    else:
        raise NotImplementedError


class PytorchModel:
    def __init__(self, net, weights):
        self.model = net(weights)
        self.weights = weights
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device=device)

    def __call__(self, image, *args, **kwargs):
        """Feewforward image through model.

        Args:
            image (cupy._core.core.ndarray):
            *args ():
            **kwargs ():

        Returns:
            torch.Tensor:
        """
        batch = torch.as_tensor(image, device="cuda").permute(2, 0, 1).unsqueeze(0)
        batch = TF.convert_image_dtype(batch, torch.float32)
        prediction = self.model(batch)["out"]
        normalized_masks = torch.squeeze(prediction.softmax(dim=1), dim=0)
        return normalized_masks

    def get_classes(self):
        """Get list of strings containing all the classes.

        Returns:
            List[str]: List of classes
        """
        return self.weights.meta["categories"]


class DetectronModel:
    def __init__(self, net, weights):
        self.cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file(weights))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights)
        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, image, *args, **kwargs):
        # TODO: there are some instruction on how to change input type
        prediction = self.predictor(image.get())
        # panoptic_seg, segments_info = self.predictor(image.get())["panoptic_seg"]
        normalized_masks = prediction["sem_seg"].softmax(dim=1)
        return normalized_masks

    def get_classes(self):
        meta = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        return meta.get("stuff_classes")


class STEGOModel:
    def __init__(self, weights, cfg):
        self.cfg: FeatureExtractorParameter = cfg
        self.model = DinoFeaturizer(weights, cfg=self.cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.shrink = Resize(size=(self.cfg.input_size[0], self.cfg.input_size[1]))

    def to_tensor(self, data):
        data = data.astype(np.float32)
        if len(data.shape) == 3:  # transpose image-like data
            data = data.transpose(2, 0, 1)
        elif len(data.shape) == 2:
            data = data.reshape((1,) + data.shape)
        if len(data.shape) == 3 and data.shape[0] == 3:  # normalization of rgb images
            data = data / 255.0
        tens = torch.as_tensor(data, device="cuda")
        return tens

    def __call__(self, image, *args, **kwargs):
        # image = torch.as_tensor(image, device="cuda").permute(2, 0, 1).unsqueeze(0)
        image = self.to_tensor(image).unsqueeze(0)
        reset_size = Resize(image.shape[-2:])
        image = self.shrink(image)
        image = TF.normalize(
            image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

        feat1, code1 = self.model(image)
        feat2, code2 = self.model(image.flip(dims=[3]))
        code = (code1 + code2.flip(dims=[3])) / 2
        code = NF.interpolate(
            code, image.shape[-2:], mode=self.cfg.interpolation, align_corners=False
        ).detach()
        code = torch.squeeze(reset_size(code), dim=0)
        return code
