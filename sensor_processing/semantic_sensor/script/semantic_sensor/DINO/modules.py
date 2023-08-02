import semantic_sensor.DINO.vision_transformer as vits
import torch.nn as nn
import torch


class DinoFeaturizer(nn.Module):
    def __init__(self, weights, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim

        self.feat_type = self.cfg.dino_feat_type
        arch = self.cfg.model
        self.model = vits.__dict__[arch](patch_size=self.cfg.patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=0.1)

        if arch == "vit_small" and self.cfg.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and self.cfg.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and self.cfg.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and self.cfg.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

        self.model.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768
        self.cluster1 = self.make_clusterer(self.n_feats)
        self.proj_type = cfg.projection_type
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

    def make_clusterer(self, in_channels):
        """
        Function to make the clusterer model which consists of a single Conv2d layer.
        The input channels are taken as the argument, and the output channels are equal to the `dim` of the model.

        Parameters:
        in_channels (int): The number of input channels.

        Returns:
        nn.Sequential: A sequential model consisting of a single Conv2d layer."""

        return torch.nn.Sequential(torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    def make_nonlinear_clusterer(self, in_channels):
        """
        Function to make the nonlinear clusterer model which consists of a series of Conv2d and ReLU layers.
        The input channels are taken as the argument, and the output channels are equal to the `dim` of the model.

        Parameters:
        in_channels (int): The number of input channels.

        Returns:
        nn.Sequential: A sequential model consisting of a Conv2d layer, a ReLU layer, and another Conv2d layer.
        """
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)),
        )

    def forward(self, img, n=1, return_class_feat=False):
        """
        Forward pass of the model.
        The input image is passed through the `model` to get the intermediate features.
        The intermediate features are then processed to get the final image feature and code.
        If `return_class_feat` is set to True, the class features are returned instead.

        Parameters:
        img (torch.Tensor): The input image tensor.
        n (int, optional): The number of intermediate features to get. Default value is 1. 1 means only the features of the last block.
        return_class_feat (bool, optional): Whether to return the class features. Default value is False.

        Returns:
        tuple: If `cfg.dropout` is True, a tuple containing the final image feature and code is returned.
               Otherwise, only the final image feature is returned.
               If `return_class_feat` is True, the class features are returned instead.
        """
        self.model.eval()
        with torch.no_grad():
            assert img.shape[2] % self.cfg.patch_size == 0
            assert img.shape[3] % self.cfg.patch_size == 0

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.cfg.patch_size
            feat_w = img.shape[3] // self.cfg.patch_size

            if self.feat_type == "feat":
                # deflatten
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
        else:
            code = image_feat

        if self.cfg.dropout:
            return self.dropout(image_feat), code
        else:
            return image_feat, code
