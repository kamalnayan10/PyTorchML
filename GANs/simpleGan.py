import torch
from torch import nn , optim
import torchvision
from torchvision import datasets , transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# DEFINING THE ADVERSARIAL NEURAL NETWORKS
# Discriminator: It distinguishes the fake data from the real data. It wants to maximise the loss function eqn(Supervised Learning)
class Discriminator(nn.Module):
    def __init__(self , img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim , 128),
            nn.LeakyReLU(0.1),

            nn.Linear(128 , 1),
            nn.Sigmoid()
        )

    def forward(self , X):
        return self.disc(X)

# Generator: It creates fake data to fool the discrtiminator into thinking its real data. It;s initialised from random noise which is tuned
# It wants to minimise the loss function eqn (Unsupervised Learning)
class Generator(nn.Module):
    def __init__(self , z_dim , img_dim):
        super().__init__()

        self.gen = nn.Sequential(
            nn.Linear(z_dim , 256),
            nn.LeakyReLU(0.1),

            nn.Linear(256 , img_dim),
            nn.Tanh()
        )

    def forward(self , X):
        return self.gen(X)


# DEVICE AGNOSTIC CODE
device = "cuda" if torch.cuda.is_available() else "cpu"

# HYPERPARAMETERS
lr = 3e-4
z_dim = 64 #Inintial noise passed to generator
image_dim = 28*28*1
batch_size = 32
epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim , image_dim).to(device)
fixed_noise = torch.randn((batch_size , z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,) , (0.5,))]
)


# Getting MNIST dataset and making DataLoader
dataset = datasets.MNIST(root = "dataset/" , transform=transforms , download = True)
loader = DataLoader(dataset , batch_size=batch_size , shuffle = True)
opt_disc = optim.Adam(params = disc.parameters() , lr = lr)
opt_gen = optim.Adam(params = gen.parameters() , lr = lr)

loss_fn = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(epochs):
    for batch_idx , (real_img , _) in enumerate(loader):
        real_img = real_img.view(-1 , 784).to(device)
        batch_size = real_img.shape[0]

        #Train Discriminator
        noise = torch.randn(batch_size , z_dim).to(device)

        fake = gen(noise) #generating an img from generator NN
        disc_real = disc(real_img).view(-1) #training discrimantor to identify real img
        disc_fake = disc(fake.detach()).view(-1) #training disc to identify fake img

        # Calculating Loss(Discriminator)
        lossDisc_real = loss_fn(disc_real , torch.ones_like(disc_real)) #loss of training disc to find real img
        lossDisc_fake = loss_fn(disc_fake , torch.zeros_like(disc_fake)) #loss of training disc to find fake img
        lossDisc = (lossDisc_real + lossDisc_fake) / 2 #avg loss of discriminator

        disc.zero_grad()

        lossDisc.backward()

        opt_disc.step()

        #Train Generator
        output = disc(fake).view(-1) #result of discriminator classifying fake img

        lossGen = loss_fn(output , torch.ones_like(output))#loss of training gen to create fake img

        gen.zero_grad()

        lossGen.backward()

        opt_gen.step()

        #Print Useful info like current epoch , Loss of gen and disc , tensorboard visualisation
        if batch_idx == 0:
            print(f"Epoch: {epoch}/{epochs} | Loss Disc: {lossDisc:.3f} | Loss Gen: {lossGen:.3f}")

            with torch.inference_mode():
                fake = gen(fixed_noise).reshape(-1 , 1 , 28,28)
                data = real_img.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake , normalize = True)
                img_grid_real = torchvision.utils.make_grid(data , normalize = True)

                writer_fake.add_image("MNIST Fake Images" , img_grid_fake , global_step = step)
                writer_real.add_image("MNIST Real Images" , img_grid_real , global_step = step)

                step += 1