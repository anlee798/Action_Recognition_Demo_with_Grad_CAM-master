import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
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
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p t n (h d) -> b p t h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p t h n d -> b p t n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, depth, height, width)
    #permute
    x = x.permute(0,2,1,3,4,5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        if self.stride == 1:
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                #nn.ReLU(inplace=True),
                HardSwish(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                HardSwish(inplace=True),
                #nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv3d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm3d(inp),
                # pw-linear
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                #nn.ReLU(inplace=True),
                HardSwish(inplace=True),
            )
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                #nn.ReLU(inplace=True),
                HardSwish(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear``
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                #nn.ReLU(inplace=True),
                HardSwish(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1]//2), :, :, :]
            x2 = x[:, (x.shape[1]//2):, :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif self.stride == 2:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()
        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        _, _,t, h, w = x.shape #[1, 64, 2, 32, 32]
        x = rearrange(x, 'b d t (h ph) (w pw) -> b t (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b t (ph pw) (h w) d -> b d t (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x
# # python MobileViTVideo.py

class MobileViT(nn.Module):
    # dims = [64, 80, 96]
    # channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    # return MobileViT((224, 224), dims, channels, num_classes=1000, expansion=2)
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv = nn.Sequential(
            MV2Block(channels[0], channels[1], 1, expansion),
            MV2Block(channels[1], channels[2], 2, expansion),
            MV2Block(channels[2], channels[3], 1, expansion),
            MV2Block(channels[2], channels[3], 1, expansion),   # Repeat
            MV2Block(channels[3], channels[4], 2, expansion),
        )

        # self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        # self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        # self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        # self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        # self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        # self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        # self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        self.mvit = nn.Sequential(
            MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2))
        )
        self.mv2 = nn.Sequential(
            MV2Block(channels[5], channels[6], 2, expansion)
        )
        self.mvit2 = nn.Sequential(
            MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4))
        )
        self.mv3 = nn.Sequential(
            MV2Block(channels[7], channels[8], 2, expansion)
        )
        self.mvit3 = nn.Sequential(
            MobileViTBlock(dims[2], L[2], channels[9], kernel_size, (1,1), int(dims[2]*4))
        )
        # self.mvit = nn.ModuleList([])
        # self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        # self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        # self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, (1,1), int(dims[2]*4)))
       
        self.last_conv2 = conv_1x1_bn(channels[-2], channels[-1])

        #self.avgpool = nn.AvgPool3d(ih//32, 1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.mv2[0](x)

        # x = self.mv2[1](x)
        # x = self.mv2[2](x)
        # x = self.mv2[3](x)      # Repeat

        # x = self.mv2[4](x)
        x = self.mv(x)

        x= self.mvit(x) #x = self.mvit[0](x)

        x = self.mv2(x) #x = self.mv2[5](x)
        x= self.mvit2(x) # x = self.mvit[1](x)

        x = self.mv3(x) # x = self.mv2[6](x)
        x= self.mvit3(x) #x = self.mvit[2](x)
        x = self.last_conv2(x)

        #x = F.adaptive_avg_pool3d(x,x.data.size()[-3:])
        #x = F.avg_pool3d(x, x.data.size()[-3:])
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

def mobilevit_xxs2():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((224, 224), dims, channels, num_classes=1000, expansion=2)

def mobilevit_xxs(num_classes=101,sample_size=224):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((sample_size, sample_size), dims, channels, num_classes=num_classes, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(**kwargs):
    """
    Returns the model.
    """
    model = mobilevit_xxs(**kwargs)
    return model

def create_MobileNet(num_classes = 101,sample_size=224):
    return get_model(num_classes=num_classes,sample_size=sample_size)

if __name__ == '__main__':
    from nni.compression.pytorch.utils.counter import count_flops_params
    img = torch.randn(1, 3, 16, 224, 224)
    vit = mobilevit_xxs()
    outputs = vit(img)
    print(outputs.size())
    flops,params,results = count_flops_params(vit,img)
    print(vit)
    for name, module in vit._modules.items():
        print("name",name)

# python shufflevit.py