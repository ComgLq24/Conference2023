import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, input_nc=256, output_nc=3):
        super(Generator, self).__init__()

        self.l1 = nn.Sequential(nn.Linear(input_nc, 256 * 405 * 720))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, output_nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, 405, 720)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):

    ''' Discriminator Model '''

    def __init__(self,ngpu,ndf,nc):

        ''' initialising the variables '''

        super(Discriminator,self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.nc = nc

        '''
        Building the model - 

        We have 4 convolution layers for downsampling.

        Following the above, we have 4 LeakyReLU activation layers which according to the paper gives better results on the discriminator specially for
        higher-resolution images.

        The final layer has a sigmoid activation function that outputs the probabilty of an image being fake or real. 
        '''

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), #stride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), #stride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), #stride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), #stride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False), #stride=1, padding=0
            nn.Sigmoid()
        )
        
    ''' Function to forward the input into the model '''

    def forward(self,input):
        return self.main(input)

class DAM(nn.Module):
    
    def __init__(self, in_channels):
        self.conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        N, C, H, W = z.shape
        F = self.conv(z).view(N, C, H*W).permute(0, 2, 1) # shape N, H*w, C

        # spatial attention
        S = torch.bmm(F, F.permute(0, 2, 1))
        S = self.softamx(S)

        # channel attention
        R = torch.bmm(F.permute(0, 2, 1), F)
        R = self.softmax(R)

        S = torch.bmm(F, S)
        R = torch.bmm(R, S)

        return S+R

class SpatialAttentionMap(nn.Module):
    
    def __init__(self, in_channels):
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        N, C, H, W = z.shape
        F = z.view(N, C, H*W).permute(0, 2, 1) # shape N, H*w, C

        # spatial attention
        S = torch.bmm(F, F.permute(0, 2, 1))
        S = self.softamx(S)
        
        return S

class ChannelAttentionMap(nn.Module):
    def __init__(self, in_channels):
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        N, C, H, W = z.shape
        F = z.view(N, C, H*W).permute(0, 2, 1) # shape N, H*w, C

        # channel attention
        R = torch.bmm(F.permute(0, 2, 1), F)
        R = self.softmax(R)

        return R
