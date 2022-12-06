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
import cupy as cp

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
def encode_max(maxim, index):
    maxim, index = cp.asarray(maxim, dtype=cp.float32), cp.asarray(
        index, dtype=cp.uint32
    )
    # fuse them
    maxim = maxim.astype(cp.float16)
    maxim = maxim.view(cp.uint16)
    maxim = maxim.astype(cp.uint32)
    index = index.astype(cp.uint32)
    mer = cp.array(cp.left_shift(index, 16) | maxim, dtype=cp.uint32)
    mer = mer.view(cp.float32)
    return mer


def decode_max(mer):
    mer = mer.astype(cp.float32)
    mer = mer.view(dtype=cp.uint32)
    ma = cp.bitwise_and(mer, 0xFFFF, dtype=np.uint16)
    ma = ma.view(np.float16)
    ma = ma.astype(np.float32)
    ind = cp.right_shift(mer, 16)
    return ma, ind


def resolve_model(name, config=None):
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
        model = PytorchModel(net, weights, config)
        return {
            "name": "fcn_resnet50",
            "model": model,
        }
    elif name == "lraspp_mobilenet_v3_large":
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        net = lraspp_mobilenet_v3_large
        model = PytorchModel(net, weights, config)
        return {
            "name": "lraspp_mobilenet_v3_large",
            "model": model,
        }
    elif name == "detectron_coco_panoptic_fpn_R_101_3x":
        # "Cityscapes/mask_rcnn_R_50_FPN.yaml"
        # "Misc/semantic_R_50_FPN_1x.yaml"
        # "Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"
        # "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
        weights = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
        model = DetectronModel(weights, config)
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
    def __init__(self, net, weights, param):
        self.model = net(weights)
        self.weights = weights
        self.param = param
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device=device)
        self.resolve_cateories()

    def resolve_cateories(self):
        class_to_idx = {cls: idx for (idx, cls) in enumerate(self.get_classes())}
        print(
            "Semantic Segmentation possible channels: ",
            self.get_classes(),
        )
        indices = []
        channels = []
        for it, chan in enumerate(self.param.channels):
            if chan in [cls for cls in list(class_to_idx.keys())]:
                indices.append(class_to_idx[chan])
                channels.append(chan)
            elif self.param.fusion[it] in ["class_average", "class_bayesian"]:
                print(chan, " is not in the semantic segmentation model.")
        # for argamax
        for it, chan in enumerate(self.param.channels):
            if chan in [cls for cls in list(class_to_idx.keys())]:
                indices.append(class_to_idx[chan])
                channels.append(chan)
            elif self.param.fusion[it] in ["class_max"]:
                print(chan, " is not in the semantic segmentation model.")
        self.stuff_categories = dict(zip(channels, indices))
        self.segmentation_channels = self.stuff_categories

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
        with torch.no_grad():
            prediction = self.model(batch)["out"]
            normalized_masks = torch.squeeze(prediction.softmax(dim=1), dim=0)
            # get masks of fix classes
            selected_masks = cp.asarray(
                normalized_masks[list(self.stuff_categories.values())]
            )

            # get values of max, first remove the ones we already have
            normalized_masks[list(self.stuff_categories.values())] = 0
            for i in range(self.param.fusion.count("class_max")):
                maxim, index = torch.max(normalized_masks, dim=0)
                mer = encode_max(maxim, index)
                selected_masks = cp.concatenate(
                    (selected_masks, cp.expand_dims(mer, axis=0)), axis=0
                )
                # TODO set to zero the max
                # normalized_masks[index] = 0
        return cp.asarray(selected_masks)

    def get_classes(self):
        """Get list of strings containing all the classes.

        Returns:
            List[str]: List of classes
        """
        return self.weights.meta["categories"]


class DetectronModel:
    def __init__(self, weights, param):
        self.cfg = get_cfg()
        self.param = param

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file(weights))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights)
        self.predictor = DefaultPredictor(self.cfg)
        self.meta = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.stuff_categories, self.is_stuff = self.resolve_cateories("stuff_classes")
        self.thing_categories, self.is_thing = self.resolve_cateories("thing_classes")
        self.segmentation_channels = {}
        for chan in self.param.channels:
            if chan in self.stuff_categories.keys():
                self.segmentation_channels[chan] = self.stuff_categories[chan]
            elif chan in self.thing_categories.keys():
                self.segmentation_channels[chan] = self.thing_categories[chan]
            else:
                # remove it
                pass

    def resolve_cateories(self, name):
        classes = self.get_cat(name)
        class_to_idx = {cls: idx for (idx, cls) in enumerate(classes)}
        print(
            "Semantic Segmentation possible channels: ",
            classes,
        )
        indices = []
        channels = []
        is_thing = []
        for it, chan in enumerate(self.param.channels):
            if chan in [cls for cls in list(class_to_idx.keys())]:
                indices.append(class_to_idx[chan])
                channels.append(chan)
                is_thing.append(True)
            elif self.param.fusion[it] in ["class_average", "class_bayesian"]:
                is_thing.append(False)
                print(chan, " is not in the semantic segmentation model.")
        categories = dict(zip(channels, indices))
        cat_isthing = dict(zip(self.param.channels, is_thing))
        return categories, cat_isthing

    def __call__(self, image, *args, **kwargs):
        # TODO: there are some instruction on how to change input type
        image = cp.flip(image, axis=2)
        prediction = self.predictor(image.get())
        probabilities = cp.asarray(torch.softmax(prediction["sem_seg"], dim=0))
        output = cp.zeros(
            (
                len(self.segmentation_channels),
                probabilities.shape[1],
                probabilities.shape[2],
            )
        )
        # add semseg
        output[cp.array(list(self.is_stuff.values()))] = probabilities[
            list(self.stuff_categories.values())
        ]
        # add instances
        indices, insta_info = prediction["panoptic_seg"]
        # TODO dont know why i need temp, look into how to avoid
        temp = output[cp.array(list(self.is_thing.values()))]
        for i, insta in enumerate(insta_info):
            if insta is None or not insta["isthing"]:
                continue
            mask = cp.asarray((indices == insta["id"]).int())
            if insta["instance_id"] in self.thing_categories.values():
                temp[i] = mask * insta["score"]
        output[cp.array(list(self.is_thing.values()))] = temp
        return output

    def get_cat(self, name):
        return self.meta.get(name)

    def get_classes(self):
        return self.get_cat("thing_classes") + self.get_cat("stuff_classes")


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
