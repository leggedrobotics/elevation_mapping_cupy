#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp


def get_filter_torch(*args, **kwargs):
    import torch
    import torch.nn as nn

    class TraversabilityFilter(nn.Module):
        def __init__(self, w1, w2, w3, w_out, device="cuda", use_bias=False):
            super(TraversabilityFilter, self).__init__()
            self.conv1 = nn.Conv2d(1, 4, 3, dilation=1, padding=0, bias=use_bias)
            self.conv2 = nn.Conv2d(1, 4, 3, dilation=2, padding=0, bias=use_bias)
            self.conv3 = nn.Conv2d(1, 4, 3, dilation=3, padding=0, bias=use_bias)
            self.conv_out = nn.Conv2d(12, 1, 1, bias=use_bias)

            # Set weights.
            self.conv1.weight = nn.Parameter(torch.from_numpy(w1).float())
            self.conv2.weight = nn.Parameter(torch.from_numpy(w2).float())
            self.conv3.weight = nn.Parameter(torch.from_numpy(w3).float())
            self.conv_out.weight = nn.Parameter(torch.from_numpy(w_out).float())

        def __call__(self, elevation_cupy):
            # Convert cupy tensor to pytorch.
            elevation_cupy = elevation_cupy.astype(cp.float32)
            elevation = torch.as_tensor(elevation_cupy, device=self.conv1.weight.device)

            with torch.no_grad():
                out1 = self.conv1(elevation.view(-1, 1, elevation.shape[0], elevation.shape[1]))
                out2 = self.conv2(elevation.view(-1, 1, elevation.shape[0], elevation.shape[1]))
                out3 = self.conv3(elevation.view(-1, 1, elevation.shape[0], elevation.shape[1]))

                out1 = out1[:, :, 2:-2, 2:-2]
                out2 = out2[:, :, 1:-1, 1:-1]
                out = torch.cat((out1, out2, out3), dim=1)
                # out = F.concat((out1, out2, out3), axis=1)
                out = self.conv_out(out.abs())
                out = torch.exp(-out)
                out_cupy = cp.asarray(out)

            return out_cupy

    traversability_filter = TraversabilityFilter(*args, **kwargs).cuda().eval()
    return traversability_filter


def get_filter_chainer(*args, **kwargs):
    import os

    os.environ["CHAINER_WARN_VERSION_MISMATCH"] = "0"
    import chainer
    import chainer.links as L
    import chainer.functions as F

    class TraversabilityFilter(chainer.Chain):
        def __init__(self, w1, w2, w3, w_out, use_cupy=True):
            super(TraversabilityFilter, self).__init__()
            self.conv1 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=1, nobias=True, initialW=w1)
            self.conv2 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=2, nobias=True, initialW=w2)
            self.conv3 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=3, nobias=True, initialW=w3)
            self.conv_out = L.Convolution2D(12, 1, ksize=1, nobias=True, initialW=w_out)

            if use_cupy:
                self.conv1.to_gpu()
                self.conv2.to_gpu()
                self.conv3.to_gpu()
                self.conv_out.to_gpu()
            chainer.config.train = False
            chainer.config.enable_backprop = False

        def __call__(self, elevation):
            out1 = self.conv1(elevation.reshape(-1, 1, elevation.shape[0], elevation.shape[1]))
            out2 = self.conv2(elevation.reshape(-1, 1, elevation.shape[0], elevation.shape[1]))
            out3 = self.conv3(elevation.reshape(-1, 1, elevation.shape[0], elevation.shape[1]))

            out1 = out1[:, :, 2:-2, 2:-2]
            out2 = out2[:, :, 1:-1, 1:-1]
            out = F.concat((out1, out2, out3), axis=1)
            out = self.conv_out(F.absolute(out))
            return F.exp(-out).array

    traversability_filter = TraversabilityFilter(*args, **kwargs)
    return traversability_filter


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
