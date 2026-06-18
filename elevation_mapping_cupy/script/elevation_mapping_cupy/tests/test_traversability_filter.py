import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from elevation_mapping_cupy.traversability_filter import get_filter_torch


def _dilated_conv2d(image, weights, dilation):
    out_channels = weights.shape[0]
    height, width = image.shape
    out_height = height - 2 * dilation
    out_width = width - 2 * dilation
    output = np.zeros((out_channels, out_height, out_width), dtype=np.float32)

    for channel in range(out_channels):
        for row in range(out_height):
            for col in range(out_width):
                patch = image[
                    row : row + 2 * dilation + 1 : dilation,
                    col : col + 2 * dilation + 1 : dilation,
                ]
                output[channel, row, col] = np.sum(patch * weights[channel, 0])

    return output


def _reference_traversability(elevation, w1, w2, w3, w_out):
    out1 = _dilated_conv2d(elevation, w1, dilation=1)[:, 2:-2, 2:-2]
    out2 = _dilated_conv2d(elevation, w2, dilation=2)[:, 1:-1, 1:-1]
    out3 = _dilated_conv2d(elevation, w3, dilation=3)
    features = np.concatenate((out1, out2, out3), axis=0)

    cost = np.zeros((features.shape[1], features.shape[2]), dtype=np.float32)
    for channel in range(features.shape[0]):
        cost += np.abs(features[channel]) * w_out[0, channel, 0, 0]

    return np.exp(-cost).reshape(1, 1, features.shape[1], features.shape[2])


def test_torch_filter_matches_reference_convolution():
    elevation = np.linspace(-0.4, 0.6, 110, dtype=np.float32).reshape(10, 11)
    w1 = np.arange(36, dtype=np.float32).reshape(4, 1, 3, 3) * 0.001
    w2 = np.arange(36, 72, dtype=np.float32).reshape(4, 1, 3, 3) * 0.001
    w3 = np.arange(72, 108, dtype=np.float32).reshape(4, 1, 3, 3) * 0.001
    w_out = np.linspace(0.01, 0.12, 12, dtype=np.float32).reshape(1, 12, 1, 1)

    traversability_filter = get_filter_torch(w1, w2, w3, w_out)
    actual = cp.asnumpy(traversability_filter(cp.asarray(elevation)))
    expected = _reference_traversability(elevation, w1, w2, w3, w_out)

    assert actual.shape == (1, 1, 4, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

