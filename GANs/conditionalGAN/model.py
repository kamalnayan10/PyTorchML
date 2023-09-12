import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self , channels_img , features_d , num_classes , img_size):
        super(Discriminator , self).__init__()

        self.img_size = img_size

        self.disc = nn.Sequential(
            # Input : N x channels_img x 64 x 64
            nn.Conv2d(
                in_channels=channels_img + 1,
                out_channels=features_d,
                kernel_size=4,
                stride=2,
                padding=1
            ),#32 x 32
            nn.LeakyReLU(0.2),

            self.block(features_d , features_d*2 , 4 , 2 , 1),#16 x 16

            self.block(features_d*2 , features_d*4 , 4 , 2 , 1),#8 x 8

            self.block(features_d*4 , features_d*8, 4 , 2 , 1),#4 x 4

            nn.Conv2d(features_d*8 , 1 , kernel_size = 4 , stride = 2 , padding = 0),#1 x 1
        )

        self.embed = nn.Embedding(num_classes , img_size*img_size)

    def block(self , in_channels , out_channels , kernel_size, stride , padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride= stride,
                padding = padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels , affine = True),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self , X , labels):
        embedding = self.embed(labels).view(labels.shape[0] , 1 , self.img_size , self.img_size)
        X = torch.cat([X , embedding] , dim = 1)
        return self.disc(X)
    

class Generator(nn.Module):
    def __init__(self , z_dim , channels_img , features_g , num_classes , img_size , embed_size):
        super(Generator , self).__init__()

        self.img_size = img_size

        self.net = nn.Sequential(
            #Input: N x z_dim x 1 x 1
            self.block(z_dim + embed_size , features_g*16, 4,1,0), #N x features_g*16 x 4 x 4

            self.block(features_g*16,features_g*8, 4,2,1), #8 x 8

            self.block(features_g*8,features_g*4, 4,2,1), #16 x 16

            self.block(features_g*4,features_g*2, 4,2,1), #32 x 32

            nn.ConvTranspose2d(
                features_g*2 , channels_img , kernel_size = 4 , stride = 2 , padding = 1
            ), #64 x 64
            nn.Tanh()

        )

        self.embed = nn.Embedding(num_classes , embed_size)

    def block(self , in_channels , out_channels , kernsel_size , stride , padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernsel_size,
                stride=stride,
                padding=padding,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self , X , labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        X = torch.cat([X , embedding] , dim = 1)
        return self.net(X)

    
def initialise_wt(model):
    for m in model.modules():
        if isinstance(m , (nn.Conv2d , nn.ConvTranspose2d , nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data , 0.0 , 0.02)

def test():
    N , in_channels , H , W = 8,3,64,64
    z_dim = 100
    X = torch.randn((N,in_channels , H , W))
    disc = Discriminator(in_channels , 8)
    initialise_wt(disc)
    assert disc(X).shape == (N,1,1,1)
    gen = Generator(z_dim ,in_channels, 8)
    z = torch.randn((N , z_dim , 1,1))
    assert gen(z).shape == (N , in_channels , H , W)