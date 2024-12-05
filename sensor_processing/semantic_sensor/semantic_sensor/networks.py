from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize
import torch.nn.functional as NF
import numpy as np
import cupy as cp

from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

from semantic_sensor.DINO.modules import DinoFeaturizer

from semantic_sensor.pointcloud_parameters import (
    PointcloudParameter,
    FeatureExtractorParameter,
)
from semantic_sensor.utils import encode_max


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
    elif name == "deeplabv3_mobilenet_v3_large":
        weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        net = deeplabv3_mobilenet_v3_large
        model = PytorchModel(net, weights, config)
        return {
            "name": "deeplabv3_mobilenet_v3_large",
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
    """

    segemntation_cahnnels contains only classes
    actual_channels: array of all channels of output
    """

    def __init__(self, net, weights, param):
        self.model = net(weights=weights)
        self.weights = weights
        self.param = param
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device=device)
        self.resolve_categories()

    def resolve_categories(self):
        """Create a segmentation_channels containing the actual values that are processed."""
        # get all classes
        class_to_idx = {cls: idx for (idx, cls) in enumerate(self.get_classes())}
        print(
            "Semantic Segmentation possible channels: ", self.get_classes(),
        )
        indices = []
        channels = []
        self.actual_channels = []
        # check correspondence of semseg and paramter classes, class_max
        for it, chan in enumerate(self.param.channels):
            if chan in [cls for cls in list(class_to_idx.keys())]:
                indices.append(class_to_idx[chan])
                channels.append(chan)
                self.actual_channels.append(chan)
            else:
                self.actual_channels.append(chan)
            # elif self.param.fusion_methods[it] in ["class_average", "class_bayesian"]:
            #     print(chan, " is not in the semantic segmentation model.")
        # for it, chan in enumerate(self.param.channels):
        #     self.actual_channels.append(chan)
        # if self.param.fusion_methods[it] in ["class_max"]:
        #     self.actual_channels.append(chan)
        #     print(
        #         chan,
        #         " is not in the semantic segmentation model but is a max channel.",
        #     )
        # else:
        #     pass
        self.stuff_categories = dict(zip(channels, indices))
        self.segmentation_channels = dict(zip(channels, indices))

    def __call__(self, image, *args, **kwargs):
        """Feedforward image through model and then create channels

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
            selected_masks = cp.asarray(normalized_masks[list(self.stuff_categories.values())])
            # get values of max, first remove the ones we already have
            normalized_masks[list(self.stuff_categories.values())] = 0
            # for i in range(self.param.fusion_methods.count("class_max")):
            #     maxim, index = torch.max(normalized_masks, dim=0)
            #     mer = encode_max(maxim, index)
            #     selected_masks = cp.concatenate((selected_masks, cp.expand_dims(mer, axis=0)), axis=0)
            #     x = torch.arange(0, index.shape[0])
            #     y = torch.arange(0, index.shape[1])
            #     c = torch.meshgrid(x, y, indexing="ij")
            #     normalized_masks[index, c[0], c[1]] = 0
            assert len(self.actual_channels) == selected_masks.shape[0]
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
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.stuff_categories, self.is_stuff = self.resolve_categories("stuff_classes")
        self.thing_categories, self.is_thing = self.resolve_categories("thing_classes")
        self.segmentation_channels = {}
        for chan in self.param.channels:
            if chan in self.stuff_categories.keys():
                self.segmentation_channels[chan] = self.stuff_categories[chan]
            elif chan in self.thing_categories.keys():
                self.segmentation_channels[chan] = self.thing_categories[chan]
            else:
                # remove it
                pass
        self.actual_channels = self.segmentation_channels.keys()

    def resolve_categories(self, name):
        classes = self.get_category(name)
        class_to_idx = {cls: idx for (idx, cls) in enumerate(classes)}
        print(
            "Semantic Segmentation possible channels: ", classes,
        )
        indices = []
        channels = []
        is_thing = []
        for it, channel in enumerate(self.param.channels):
            if channel in [cls for cls in list(class_to_idx.keys())]:
                indices.append(class_to_idx[channel])
                channels.append(channel)
                is_thing.append(True)
            elif self.param.fusion_methods[it] in ["class_average", "class_bayesian"]:
                is_thing.append(False)
                print(channel, " is not in the semantic segmentation model.")
        categories = dict(zip(channels, indices))
        category_isthing = dict(zip(self.param.channels, is_thing))
        return categories, category_isthing

    def __call__(self, image, *args, **kwargs):
        # TODO: there are some instruction on how to change input type
        image = cp.flip(image, axis=2)
        prediction = self.predictor(image.get())
        probabilities = cp.asarray(torch.softmax(prediction["sem_seg"], dim=0))
        output = cp.zeros((len(self.segmentation_channels), probabilities.shape[1], probabilities.shape[2],))
        # add semseg
        output[cp.array(list(self.is_stuff.values()))] = probabilities[list(self.stuff_categories.values())]
        # add instances
        indices, instance_info = prediction["panoptic_seg"]
        # TODO dont know why i need temp, look into how to avoid
        temp = output[cp.array(list(self.is_thing.values()))]
        for i, instance in enumerate(instance_info):
            if instance is None or not instance["isthing"]:
                continue
            mask = cp.asarray((indices == instance["id"]).int())
            if instance["instance_id"] in self.thing_categories.values():
                temp[i] = mask * instance["score"]
        output[cp.array(list(self.is_thing.values()))] = temp
        return output

    def get_category(self, name):
        return self.metadata.get(name)

    def get_classes(self):
        return self.get_category("thing_classes") + self.get_category("stuff_classes")


class STEGOModel:
    def __init__(self, weights, cfg):
        self.cfg: FeatureExtractorParameter = cfg
        self.model = DinoFeaturizer(weights, cfg=self.cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.shrink = Resize(size=(self.cfg.input_size[0], self.cfg.input_size[1]), antialias=True)

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
        # if self.cfg.pcl:
        reset_size = Resize(image.shape[-2:], interpolation=TF.InterpolationMode.NEAREST, antialias=True)
        im_size = image.shape[-2:]
        image = self.shrink(image)
        image = TF.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        feat1, code1 = self.model(image)
        feat2, code2 = self.model(image.flip(dims=[3]))

        code = (code1 + code2.flip(dims=[3])) / 2
        # code = NF.interpolate(code, image.shape[-2:], mode=self.cfg.interpolation, align_corners=False).detach()
        # if we just use first ten
        # code = (feat1[:,:10] + feat2[:,:10].flip(dims=[3])) / 2

        # if self.cfg.pcl:
        code = NF.interpolate(code, im_size, mode=self.cfg.interpolation, align_corners=False).detach()
        code = torch.squeeze(code, dim=0)
        return code
