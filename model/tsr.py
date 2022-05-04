import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed

sys.path.append('./')
sys.path.append('../')
from .recognizer.tps_spatial_transformer import TPSSpatialTransformer
from .recognizer.stn_head import STNHead
#from . import AttentionalImageLoss


from torch import einsum


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class sr(nn.Module):
    def __init__(self, scale_factor=2, patch_size=4, width=128, height=32,mask=False,STN=False,rrb_nums=5,\
                 hidden_units=32,channels=64,dim=1024
                 ):
        super(sr, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))

        ######S lr image
        image_height, image_width = height//scale_factor, width//scale_factor
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        ## print('num_patches:%d'%num_patches)
        patch_dim = channels * patch_height * patch_width

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.task_embed = nn.Parameter(torch.zeros(1, num_patches, dim))

        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2*hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.rrb_nums = rrb_nums
        for i in range(rrb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2*hidden_units))

        setattr(self, 'block%d' % (rrb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2*hidden_units, 2*hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2*hidden_units)
                ))
        block_ = [UpsampleBLock(2*hidden_units, 2) for _ in range(upsample_block_num)]
        #################S c=4
        block_.append(nn.Conv2d(2*hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (rrb_nums + 3), nn.Sequential(*block_))
        #self.tps_inputsize = [height//scale_factor, width//scale_factor]
        self.tps_inputsize = [32, 64]
        tps_outputsize = [height//scale_factor, width//scale_factor]
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
        #####################S
        if self.stn and self.training:
            x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}
        for i in range(self.rrb_nums):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)],self.pos_embedding,self.task_embed)
        block[str(self.rrb_nums + 2)] = getattr(self, 'block%d' % (self.rrb_nums + 2))(block[str(self.rrb_nums + 1)])
        block[str(self.rrb_nums + 3)] = getattr(self, 'block%d' % (self.rrb_nums + 3)) \
            ((block['1'] + block[str(self.rrb_nums + 2)]))
        output = torch.tanh(block[str(self.rrb_nums + 3)])
        return output


class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.transformer1 = Transformer(channels, channels)
        # self.prelu = nn.PReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.transformer2 = Transformer(channels, channels)

    def forward(self, x,pos_embedding,task_embed):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.transformer1(residual,pos_embedding,task_embed)
        # residual = self.non_local(residual)

        return self.transformer2(x + residual,pos_embedding,task_embed)


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.PReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x

class PatchEmbed(nn.Module):
    """ Feature to Patch Embedding
        input : N C H W
        output: N num_patch P^2*C
        """
    def __init__(self,height=16,width=64,patch_size=4,channels=64,dim=1024):
        super(PatchEmbed, self).__init__()
        image_height, image_width = height,width
        #print('image_height, image_width:%d,%d'%(image_height, image_width))
        patch_height, patch_width = pair(patch_size)

        #assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        #num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
           Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        #self.dropout = nn.Dropout(emb_dropout)

    def forward(self,x,pos_embedding):
        x = self.to_patch_embedding(x)
        x += pos_embedding
        return x

class DePatchEmbed(nn.Module):
    """ Patch Embedding to Feature
        input : N num_patch P^2*C
        output: N C H W
        """
    def __init__(self, height=16,width=64,  patch_size=4, channels=64, dim=1024):
        super(DePatchEmbed, self).__init__()
        image_height, image_width = height, width
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.de_patch_embedding = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=image_height//patch_height,w=image_width// patch_width,p1=patch_height, p2=patch_width),
        )
    def forward(self, x):
        return self.de_patch_embedding(x)

class Transformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transformer, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        #self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

        self.patch_embedding = PatchEmbed(height=16,width=64,  patch_size=4, channels=64, dim=1024)
        self.encoder = Encoder(dim=1024, depth=1, heads=16, dim_head =64, mlp_dim=2048, dropout=0.1)
        self.depatch_embedding = DePatchEmbed(height=16,width=64,  patch_size=4, channels=64, dim=1024)

    def forward(self, x,pos_embedding,task_embed):


        x = self.conv1(x)
        x = self.patch_embedding(x,pos_embedding)
        x = self.encoder(x)
        x = self.depatch_embedding(x)
        return x


if __name__ == '__main__':
    img = torch.zeros(2, 3, 16, 64)
    model =sr()
    print(model(img))
    embed()
