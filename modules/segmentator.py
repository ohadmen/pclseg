#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F




class Segmentator(nn.Module):
    def __init__(self, arch, nclasses, path):
        super().__init__()

        self.nclasses = nclasses

        self.sensor_img_means = arch["dataset"]["sensor"]["img_means"]

        self.sensor_img_stds = arch["dataset"]["sensor"]["img_stds"]

        bboneModule = importlib.import_module('backbones.{}'.format(arch["backbone"]["name"]))
        self.backbone = bboneModule.Backbone(params=arch["backbone"])

        # do a pass of the backbone to initialize the skip connections
        stub = torch.zeros((1,
                            self.backbone.get_input_depth(),
                            arch["dataset"]["sensor"]["img_prop"]["height"],
                            arch["dataset"]["sensor"]["img_prop"]["width"]))

        if torch.cuda.is_available():
            stub = stub.cuda()
            self.backbone.cuda()
        _, stub_skips = self.backbone(stub)

        decoderModule = importlib.import_module('decoders.{}'.format(arch["decoder"]["name"]))
        self.decoder = decoderModule.Decoder(params=arch["decoder"],
                                             stub_skips=stub_skips,
                                             OS=arch["backbone"]["OS"],
                                             feature_depth=self.backbone.get_last_depth())

        self.head = nn.Sequential(nn.Dropout2d(p=arch["head"]["dropout"]),
                                  nn.Conv2d(self.decoder.get_last_depth(),
                                            self.nclasses, kernel_size=3,
                                            stride=1, padding=1))

        # get weights
        w_dict = torch.load("{}/backbone".format(path), map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        w_dict = torch.load("{}/segmentation_decoder".format(path), map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(w_dict, strict=True)
        w_dict = torch.load("{}/segmentation_head".format(path), map_location=lambda storage, loc: storage)
        self.head.load_state_dict(w_dict, strict=True)

    def forward(self, x, mask=None):
        y, skips = self.backbone(x)
        y = self.decoder(y, skips)
        y = self.head(y)
        y = F.softmax(y, dim=1)
        return y

