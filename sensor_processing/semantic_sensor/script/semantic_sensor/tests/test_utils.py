import pytest
from semantic_sensor.networks import resolve_model
import cupy as cp
import torch
from semantic_sensor.pointcloud_parameters import (
    FeatureExtractorParameter,
    PointcloudParameter,
)


@pytest.mark.parametrize(
    "model_name,channels, fusion",
    [
        ("fcn_resnet50", [], []),
        ("lraspp_mobilenet_v3_large", [], []),
        ("detectron_coco_panoptic_fpn_R_101_3x", [], []),
    ],
)
def test_semantic_segmentation(model_name, channels, fusion):
    param = PointcloudParameter()
    param.segmentation_model = model_name
    m = resolve_model(model_name, param)
    im_sz = [360, 640]
    image = cp.random.random((im_sz[0], im_sz[1], 3))
    classes = m["model"].get_classes()
    assert type(classes) is list
    prediction = m["model"](image)
    # assert prediction.shape == torch.Size([len(param.channels), im_sz[0], im_sz[1]])
    # assert (prediction <= 1).all()
    # assert (0 <= prediction).all()


@pytest.mark.parametrize(
    "model_name", ["DINO",],
)
def test_feature_extractor(model_name):
    param = FeatureExtractorParameter()
    param.name = model_name
    m = resolve_model(model_name, param)
    im_sz = [320, 640]
    image = cp.random.random((im_sz[0], im_sz[1], 3))
    prediction = m["model"](image.get())
    assert prediction.shape == torch.Size([param.dim, im_sz[0], im_sz[1]])
    assert prediction.shape[-2:] == torch.Size([im_sz[0], im_sz[1]])
