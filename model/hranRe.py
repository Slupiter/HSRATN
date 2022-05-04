from model import common_hran
import torch.nn as nn

import torch.nn

import torch.nn.functional as F
from IPython import embed

import sys
sys.path.append('./')
sys.path.append('../')
from .recognizer.tps_spatial_transformer import TPSSpatialTransformer
from .recognizer.stn_head import STNHead

def make_model(args, parent=False):    return HRAN(args)

################S laplace channel attention
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = common_hran.BasicBlock_laplas(channel, channel // reduction, 3, 1, 3, 3)
        self.c2 = common_hran.BasicBlock_laplas(channel, channel // reduction, 3, 1, 5, 5)
        self.c3 = common_hran.BasicBlock_laplas(channel, channel // reduction, 3, 1, 7, 7)
        self.c4 = common_hran.BasicBlockSig_laplas((channel // reduction) * 3, channel, 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y

class HRAB(nn.Module):
    def __init__(self, conv=common_hran.default_conv, n_feats=64):
        super(HRAB, self).__init__()

        kernel_size_1 = 3

        reduction = 4

        #self.conv_du_1 = nn.Sequential(
         #   nn.AdaptiveAvgPool2d(1),
         #   conv(n_feats, n_feats // reduction, 1),
         #   nn.LeakyReLU(inplace=True),
         #   conv(n_feats // reduction, n_feats, 1),
         #   nn.Sigmoid()
        #)
        self.conv_du_1 = CALayer(n_feats)
        self.conv_3 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats, n_feats, kernel_size_1, dilation=2)

        self.conv_3_1 = conv(n_feats * 2, n_feats, kernel_size_1)
        self.conv_3_2_1 = conv(n_feats * 2, n_feats, kernel_size_1, dilation=2)

        self.LR = nn.LeakyReLU(inplace=True)

        self.conv_11 = conv(n_feats * 2, n_feats, 1)

    def forward(self, x):
        res_x = x

        a = self.conv_du_1(x)
        b1 = self.LR(self.conv_3(x))
        b2 = self.LR(self.conv_3_2(x)) + b1
        B = torch.cat([b1, b2], 1)

        b1 = self.conv_3_1(B)
        b2 = self.LR(self.conv_3_2_1(B)) + b1

        B = torch.cat([b1, b2], 1)

        B = self.conv_11(B)

        output = a * B

        output = output + res_x

        return output


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [HRAB(n_feats=n_feat) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class HRAN(nn.Module):
    def __init__(self, scale_factor=2,width=128, height=32,STN=False, mask=False, conv=common_hran.default_conv):
        super(HRAN, self).__init__()

        n_feats = 64
        self.n_blocks = 8
        n_resgroups = 4

        kernel_size = 3
        scale = scale_factor
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common_hran.MeanShift(255, rgb_mean, rgb_std)

        # 语义掩膜
        in_planes = 3
        if mask:
            in_planes = 4

        # define head module
        modules_head = [conv(in_planes, n_feats, kernel_size)]
        modules_head_2 = [conv(n_feats, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()

        # define body module
        modules_body = [ResidualGroup(conv, n_feats, kernel_size, self.n_blocks) for _ in range(n_resgroups)]

        modules_tail = [
            conv(n_feats, n_feats, kernel_size),
            common_hran.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, in_planes , kernel_size)]
        self.add_mean = common_hran.MeanShift(255, rgb_mean, rgb_std, 1)

        self.head_1 = nn.Sequential(*modules_head)
        self.head_2 = nn.Sequential(*modules_head_2)
        self.fusion = nn.Sequential(*[nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)])
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

        #########S
        self.tps_inputsize = [32, 64]
        # self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):
        ##################S
        if self.stn and self.training:
            # print('stn process')
            x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)

        #x = self.sub_mean(x)
        x = self.head_1(x)
        res = x

        x = self.head_2(x)

        res_x = x

        HRAB_out = []
        for i in range(4):
            x = self.body[i](x)
            HRAB_out.append(x)

        while len(HRAB_out) > 2:
            fusions = []
            for i in range(0, len(HRAB_out), 2):
                fusions.append(self.fusion(torch.cat((HRAB_out[i], HRAB_out[i + 1]), 1)))

            HRAB_out = fusions

        res = res + self.fusion(torch.cat(HRAB_out, 1))
        x = self.tail(res)
        #x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))