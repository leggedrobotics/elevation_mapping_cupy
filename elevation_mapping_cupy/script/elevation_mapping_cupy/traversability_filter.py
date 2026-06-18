#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp


class TraversabilityFilterCupy:
    def __init__(self, w1, w2, w3, w_out, device="cuda", use_bias=False, use_cupy=True):
        if use_bias:
            raise NotImplementedError("TraversabilityFilterCupy does not support bias weights")
        self.w1 = cp.asarray(w1, dtype=cp.float32)
        self.w2 = cp.asarray(w2, dtype=cp.float32)
        self.w3 = cp.asarray(w3, dtype=cp.float32)
        self.w_out = cp.asarray(w_out, dtype=cp.float32)
        self._validate_weights()

    def _validate_weights(self):
        expected = {
            "w1": (4, 1, 3, 3),
            "w2": (4, 1, 3, 3),
            "w3": (4, 1, 3, 3),
            "w_out": (1, 12, 1, 1),
        }
        actual = {
            "w1": self.w1.shape,
            "w2": self.w2.shape,
            "w3": self.w3.shape,
            "w_out": self.w_out.shape,
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} must have shape {shape}, got {actual[name]}")

    @staticmethod
    def _dilated_conv2d(image, weights, dilation):
        height, width = image.shape
        out_channels = weights.shape[0]
        out_height = height - 2 * dilation
        out_width = width - 2 * dilation
        if out_height <= 0 or out_width <= 0:
            raise ValueError(
                f"Input elevation map is too small for dilation={dilation}: "
                f"shape={image.shape}"
            )

        output = cp.zeros((1, out_channels, out_height, out_width), dtype=cp.float32)
        for row in range(3):
            row_start = row * dilation
            for col in range(3):
                col_start = col * dilation
                image_window = image[
                    row_start : row_start + out_height,
                    col_start : col_start + out_width,
                ]
                kernel = weights[:, 0, row, col].reshape(1, out_channels, 1, 1)
                output += image_window.reshape(1, 1, out_height, out_width) * kernel
        return output

    def __call__(self, elevation_cupy):
        elevation = cp.asarray(elevation_cupy, dtype=cp.float32)
        if elevation.ndim != 2:
            raise ValueError(f"elevation_cupy must be a 2D array, got shape={elevation.shape}")

        out1 = self._dilated_conv2d(elevation, self.w1, dilation=1)[:, :, 2:-2, 2:-2]
        out2 = self._dilated_conv2d(elevation, self.w2, dilation=2)[:, :, 1:-1, 1:-1]
        out3 = self._dilated_conv2d(elevation, self.w3, dilation=3)
        out = cp.concatenate((out1, out2, out3), axis=1)
        cost = cp.sum(cp.abs(out) * self.w_out, axis=1, keepdims=True)
        return cp.exp(-cost)


def get_filter_torch(*args, **kwargs):
    return TraversabilityFilterCupy(*args, **kwargs)


def get_filter_chainer(*args, **kwargs):
    return TraversabilityFilterCupy(*args, **kwargs)


if __name__ == "__main__":
    import cupy as cp
    from parameter import Parameter

    elevation = cp.random.randn(202, 202, dtype=cp.float32)
    print("elevation ", elevation.shape)
    param = Parameter()
    fc = get_filter_chainer(param.w1, param.w2, param.w3, param.w_out)
    print("chainer ", fc(elevation))

    ft = get_filter_torch(param.w1, param.w2, param.w3, param.w_out)
    print("torch ", ft(elevation))
