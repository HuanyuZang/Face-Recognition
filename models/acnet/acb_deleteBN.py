import torch.nn as nn
import torch.nn.init as init
from ACNet_master.custom_layers.crop_layer import CropLayer

class ACBlock_deleteBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                 use_affine=True, reduce_gamma=False, use_last_bn=False, gamma_init=None):
        super(ACBlock_deleteBN, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
            #self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (padding, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, padding)

            #不管是kernel_size=3,padding=1 or kernel_size=7,padding=3,都是这个
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            #基本不会是这个
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)

            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1), stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size), stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)

            #self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
            #self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            # 没用，define false
            if reduce_gamma:
                assert not use_last_bn
                self.init_gamma(1.0 / 3)

            # 没用，define false
            # if use_last_bn:
            #     assert not reduce_gamma
            #     self.last_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

            # 没用，define none
            if gamma_init is not None:
                assert not reduce_gamma
                self.init_gamma(gamma_init)

    #用不上，两个地方false和none
    def init_gamma(self, gamma_value):
        init.constant_(self.square_bn.weight, gamma_value)
        init.constant_(self.ver_bn.weight, gamma_value)
        init.constant_(self.hor_bn.weight, gamma_value)
        print('init gamma of square, ver and hor as ', gamma_value)

    #只有这一处，不知道干嘛用
    def single_init(self):
        init.constant_(self.square_bn.weight, 1.0)
        init.constant_(self.ver_bn.weight, 0.0)
        init.constant_(self.hor_bn.weight, 0.0)
        print('init gamma of square as 1, ver and hor as 0')

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            #square_outputs = self.square_bn(square_outputs)

            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            #vertical_outputs = self.ver_bn(vertical_outputs)

            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            #horizontal_outputs = self.hor_bn(horizontal_outputs)

            result = square_outputs + vertical_outputs + horizontal_outputs
            # if hasattr(self, 'last_bn'):
            #     return self.last_bn(result)
            return result
