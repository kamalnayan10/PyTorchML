"""
Training the Discirminator and Generator from DCGAN paper
"""

import torch
from torch import nn
from torch import optim

import torchvision
from torchvision import datasets , transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Generator , Discriminator , initialise_wt

# DEVCE AGNOSTIC CODE
device = "cuda" if torch.cuda.is_available() else "cpu"

#HYPERPARAMETERS
LR = 2e-4
BATCH_SIZE = 128
IMG_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for i in range(CHANNELS_IMG)] , [0.5 for i in range(CHANNELS_IMG)]
        )
    ]
)

dataset = datasets.MNIST(root = "GANs/simpleGAN/dataset/" , train = True ,transform=transforms , download = True)
loader = DataLoader(dataset , batch_size=BATCH_SIZE , shuffle = True)

gen = Generator(Z_DIM , CHANNELS_IMG , FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG , FEATURES_DISC).to(device)

initialise_wt(gen)
initialise_wt(disc)

opt_gen = optim.Adam(params = gen.parameters() , lr = LR , betas = (0.5 , 0.999))
opt_disc = optim.Adam(params = disc.parameters(), lr = LR , betas = (0.5 , 0.999))

loss_fn = nn.BCELoss()

fixed_noise = torch.randn(32 , Z_DIM ,1 ,1).to(device)

writer_real = SummaryWriter("GANs/DCGAN/runs/real")
writer_fake = SummaryWriter("GANs/DCGAN/runs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(EPOCHS):
    for batch_idx , (img_real , _) in enumerate(loader):
        img_real = img_real.to(device)

        noise = torch.randn((BATCH_SIZE , Z_DIM , 1 ,1)).to(device)

        # TRAIN DISCRIMINATOR
        fake_img = gen(noise)

        disc_real = disc(img_real).reshape(-1)
        disc_fake = disc(fake_img.detach()).reshape(-1)

        loss_disc_real = loss_fn(disc_real , torch.ones_like(disc_real))
        loss_disc_fake = loss_fn(disc_fake , torch.zeros_like(disc_fake))
        
        loss_disc = (loss_disc_fake + loss_disc_real)/2

        disc.zero_grad()

        loss_disc.backward()

        opt_disc.step()

        # TRAIN GENERATOR
        output = disc(fake_img).reshape(-1)

        loss_gen = loss_fn(output , torch.ones_like(output))

        gen.zero_grad()

        loss_gen.backward()

        opt_gen.step()

        #Print Useful info like current epoch , Loss of gen and disc , tensorboard visualisation
        if batch_idx % 100 == 0:
            print(f"""
                  Epoch: {epoch}/{EPOCHS} | Batch: {batch_idx}/{len(loader)}
                  Loss Disc: {loss_disc:.3f} | Loss Gen: {loss_gen:.3f}
                """)

            with torch.inference_mode():
                fake = gen(fixed_noise)

                img_grid_fake = torchvision.utils.make_grid(fake[:32] , normalize = True)
                img_grid_real = torchvision.utils.make_grid(img_real[:32] , normalize = True)

                writer_fake.add_image("Fake Images" , img_grid_fake , global_step = step)
                writer_real.add_image("Real Images" , img_grid_real , global_step = step)

            step += 1