"""
Training the Discirminator and Generator for WGAN
"""

import torch
from torch import nn
from torch import optim

import torchvision
from torchvision import datasets , transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Generator , Discriminator , initialise_wt
from utils import gradient_penalty

# DEVCE AGNOSTIC CODE
device = "cuda" if torch.cuda.is_available() else "cpu"

#HYPERPARAMETERS
LR = 1e-4
BATCH_SIZE = 64
IMG_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITER = 5
LAMBDA_GP = 10

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
critic = Discriminator(CHANNELS_IMG , FEATURES_DISC).to(device)

initialise_wt(gen)
initialise_wt(critic)

opt_gen = optim.Adam(params = gen.parameters() , lr = LR , betas=(0, 0.9))
opt_critic = optim.Adam(params = critic.parameters(), lr = LR , betas=(0, 0.9))

fixed_noise = torch.randn(32 , Z_DIM ,1 ,1).to(device)

writer_real = SummaryWriter("GANs/WGAN/runs/real")
writer_fake = SummaryWriter("GANs/WGAN/runs/fake")

step = 0

gen.train()
critic.train()

for epoch in range(EPOCHS):
    for batch_idx , (img_real , _) in enumerate(loader):
        img_real = img_real.to(device)

        # TRAIN DISCRIMINATOR
        for _ in range(CRITIC_ITER):
            noise = torch.randn((BATCH_SIZE , Z_DIM , 1 ,1)).to(device)

            fake_img = gen(noise)

            critic_real = critic(img_real).reshape(-1)
            critic_fake = critic(fake_img).reshape(-1)

            gp = gradient_penalty(critic , img_real , fake_img , device)

            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp

            critic.zero_grad()

            loss_critic.backward(retain_graph=True)

            opt_critic.step()

        # TRAIN GENERATOR
        output = critic(fake_img).reshape(-1)

        loss_gen = -torch.mean(output)

        gen.zero_grad()

        loss_gen.backward()

        opt_gen.step()

        #Print Useful info like current epoch , Loss of gen and disc , tensorboard visualisation
        if batch_idx % 100 == 0:
            print(f"""
                  Epoch: {epoch}/{EPOCHS} | Batch: {batch_idx}/{len(loader)}
                  Loss Disc: {loss_critic:.3f} | Loss Gen: {loss_gen:.3f}
                """)

            with torch.inference_mode():
                fake = gen(fixed_noise)

                img_grid_fake = torchvision.utils.make_grid(fake[:32] , normalize = True)
                img_grid_real = torchvision.utils.make_grid(img_real[:32] , normalize = True)

                writer_fake.add_image("Fake Images" , img_grid_fake , global_step = step)
                writer_real.add_image("Real Images" , img_grid_real , global_step = step)

            step += 1