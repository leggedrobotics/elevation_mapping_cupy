import chainer
import chainer.links as L
import chainer.functions as F


class TraversabilityFilter(chainer.Chain):
    def __init__(self, w1, w2, w3, w_out, use_cupy=True):
        super(TraversabilityFilter, self).__init__()
        self.conv1 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=1,
                                     nobias=True, initialW=w1)
        self.conv2 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=2,
                                     nobias=True, initialW=w2)
        self.conv3 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=3,
                                     nobias=True, initialW=w3)
        self.conv_out = L.Convolution2D(12, 1, ksize=1,
                                        nobias=True, initialW=w_out)

        if use_cupy:
            self.conv1.to_gpu()
            self.conv2.to_gpu()
            self.conv3.to_gpu()
            self.conv_out.to_gpu()
        chainer.config.train = False
        chainer.config.enable_backprop = False

    def __call__(self, elevation):
        # elevation = elevation_map[0]
        out1 = self.conv1(elevation.reshape(-1, 1,
                                            elevation.shape[0],
                                            elevation.shape[1]))
        out2 = self.conv2(elevation.reshape(-1, 1,
                                            elevation.shape[0],
                                            elevation.shape[1]))
        out3 = self.conv3(elevation.reshape(-1, 1,
                                            elevation.shape[0],
                                            elevation.shape[1]))

        out1 = out1[:, :, 2:-2, 2:-2]
        out2 = out2[:, :, 1:-1, 1:-1]
        out = F.concat((out1, out2, out3), axis=1)
        out = self.conv_out(F.absolute(out))
        return F.exp(-out).array


if __name__ == '__main__':
    import cupy as cp
    from parameter import Parameter
    elevation = cp.random.randn(202, 202)
    print('elevation ', elevation.shape)
    param = Parameter()
    f = TraversabilityFilter(param.w1,
                             param.w2,
                             param.w3,
                             param.w_out)
    f(elevation)
