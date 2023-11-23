import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        
        self.device = device
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal , sr = torchaudio.load(audio_sample_path)
        
        signal = signal.to(self.device)

        signal = self._resample(signal , sr)
        signal = self._mix_down(signal)
        signal = self._cut(signal)
        signal = self._right_pad(signal)
        signal = self.transformation(signal)

        return signal , label
    
    def _cut(self , signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad(self , signal):
        length_signal = signal.shape[1]

        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal , last_dim_padding)
        return signal

    def _resample(self , signal , sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr , new_freq=self.target_sample_rate).to(self.device)
            signal = resampler(signal)

        return signal
    
    def _mix_down(self , signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal , dim = 0 , keepdim=True)

        return signal
    
    def _get_audio_sample_path(self , index):
        fold = f"fold{self.annotations.iloc[index , 5]}"
        file_name = self.annotations.iloc[index , 0]
        path = os.path.join(self.audio_dir , fold , file_name)

        return path
    
    def _get_audio_sample_label(self , index):
        label = self.annotations.iloc[index , 6]

        return label

if __name__ == "__main__":
    ANNOTATION_FILE = "E:/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "E:/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device , end = "\n")

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

    print(f"There are {len(usd)} samples in dataset")

    signal , label = usd[0]

    print(signal , label)