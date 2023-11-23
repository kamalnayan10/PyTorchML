import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import UrbanSoundDataset
import torchaudio
from model import CNNnet

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 2
LR = 1e-3
ANNOTATION_FILE = "E:/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "E:/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_dataloader(train_data , batch_size):
    train_dataloader = DataLoader(train_data , batch_size=batch_size)

    return train_dataloader

def train_one_epoch(model , dataloader , loss_fn , optimiser , device):
    for input , target in dataloader:
        input , target = input.to(device) , target.to(device)
        prediction = model(input)
        loss = loss_fn(prediction , target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f'Loss: {loss.item()}')

def train(epochs , model , dataloader , loss_fn , optimiser , device):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model , dataloader , loss_fn , optimiser , device)
    print("Finished")

if __name__ == "__main__":
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length=512,
        n_mels = 64)

    usd = UrbanSoundDataset(ANNOTATION_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    
    train_dataloader = create_dataloader(usd , BATCH_SIZE)

    model = CNNnet().to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(params=model.parameters() , lr = LR)

    train(EPOCHS , model , train_dataloader , loss_fn , optimiser , device)

    torch.save(model , "CNNmodel.pth")