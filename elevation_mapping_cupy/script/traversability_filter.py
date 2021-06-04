import cupy as cp
import torch
import torch.nn as nn


class TraversabilityFilter(nn.Module):
    def __init__(self, w1, w2, w3, w_out, device='cuda', use_bias=False):
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
        # Just for debug check if memory is the same.
        assert elevation.__cuda_array_interface__['data'][0] == elevation_cupy.__cuda_array_interface__['data'][0]

        with torch.no_grad():
        # elevation = elevation_map[0]
            out1 = self.conv1(elevation.view(-1, 1,
                                             elevation.shape[0],
                                             elevation.shape[1]))
            out2 = self.conv2(elevation.view(-1, 1,
                                             elevation.shape[0],
                                             elevation.shape[1]))
            out3 = self.conv3(elevation.view(-1, 1,
                                             elevation.shape[0],
                                             elevation.shape[1]))

            out1 = out1[:, :, 2:-2, 2:-2]
            out2 = out2[:, :, 1:-1, 1:-1]
            out = torch.cat((out1, out2, out3), dim=1)
            # out = F.concat((out1, out2, out3), axis=1)
            out = self.conv_out(out.abs())
            out = torch.exp(-out)
            out_cupy = cp.asarray(out)
            # Just for debug check if memory is the same.
            assert out_cupy.__cuda_array_interface__['data'][0] == out.__cuda_array_interface__['data'][0]

        return out_cupy


if __name__ == '__main__':
    import cupy as cp
    from parameter import Parameter
    elevation = cp.random.randn(202, 202, dtype=cp.float32)
    print('elevation ', elevation.shape)
    param = Parameter()
    f = TraversabilityFilter(param.w1,
                             param.w2,
                             param.w3,
                             param.w_out).cuda()
    f.eval()
    f(elevation)
