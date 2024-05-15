import torch
import torch.nn as nn
import torch.nn.functional as F
from models.swin_transformer import SwinTransformer, PatchMerging, PatchEmbed, BasicLayer, to_2tuple
from einops import rearrange



class Upsample(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(
            dim, 2*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=2, p2=2, c=C//4)
        x = x.view(B, -1, C//4)
        x = self.norm(x)

        return x

class SwinDecoderHead(nn.Module):
    def __init__(self, channels, window_size, resolution, num_classes):
        super(SwinDecoderHead, self).__init__()
        self.outer_head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], num_classes, 1)
        )
        self.resolution = resolution
        self.scale0 = BasicLayer(dim=channels[0], input_resolution=to_2tuple(resolution[0]), depth=2, num_heads=1, window_size=window_size, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0)
        self.scale1 = BasicLayer(dim=channels[1], input_resolution=to_2tuple(resolution[1]), depth=2, num_heads=1, window_size=window_size, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0)
        self.scale2 = BasicLayer(dim=channels[2], input_resolution=to_2tuple(resolution[2]), depth=2, num_heads=1, window_size=window_size, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0)
        self.scale3 = BasicLayer(dim=channels[3], input_resolution=to_2tuple(resolution[3]), depth=2, num_heads=1,
                                 window_size=window_size, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0)
        self.upsample3 = nn.Linear(channels[3], channels[2])
        self.upsample2 = Upsample(input_resolution=to_2tuple(resolution[2]), dim=channels[2], dim_scale=2)
        self.upsample1 = Upsample(input_resolution=to_2tuple(resolution[1]), dim=channels[1], dim_scale=2)
        self.concat3 = nn.Linear(
            2*resolution[2]*resolution[2], resolution[2]*resolution[2])
        self.concat2 = nn.Linear(
            2*resolution[1]*resolution[1], resolution[1]*resolution[1])
        self.concat1 = nn.Linear(
            2*resolution[0]*resolution[0], resolution[0]*resolution[0])
    def forward(self, x):
        
        x0 = x[0].permute(0, 2, 3, 1)
        x0 = x0.view(x0.size(0), -1, x0.size(3))
        
        x1 = x[1].permute(0, 2, 3, 1)
        x1 = x1.view(x1.size(0), -1, x1.size(3))

        x2 = x[2].permute(0, 2, 3, 1)
        x2 = x2.view(x2.size(0), -1, x2.size(3))

        x3 = x[3].permute(0, 2, 3, 1)
        x3 = x3.view(x3.size(0), -1, x3.size(3))
        
        x3 = self.scale3(x3)
        x3 = self.upsample3(x3)

        x2 = torch.cat([x2, x3], 1)
        x2 = self.concat3(x2.permute(0, 2, 1)).permute(0, 2, 1)
        x2 = self.scale2(x2)
        x2 = self.upsample2(x2)

        x1 = torch.cat([x1, x2], 1)
        x1 = self.concat2(x1.permute(0, 2, 1)).permute(0, 2, 1) 
        x1 = self.scale1(x1)
        x1 = self.upsample1(x1)

        x0 = torch.cat([x0, x1], 1)
        x0 = self.concat1(x0.permute(0, 2, 1)).permute(0, 2, 1)
        x0 = self.scale0(x0).permute(0, 2, 1)
        x0 = x0.view(x0.size(0), -1, self.resolution[0], self.resolution[0])
        
        return self.outer_head(x0)