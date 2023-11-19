import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal , sr = torchaudio.load(audio_sample_path)
        signal = self._resample(signal , sr)
        signal = self._mix_down(signal)
        signal = self.transformation(signal)

        return signal , label
    
    def _resample(self , signal , sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr , new_freq=self.target_sample_rate)
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
    SAMPLE_RATE = 16000

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length=512,
        n_mel = 64)

    usd = UrbanSoundDataset(ANNOTATION_FILE , AUDIO_DIR , mel_spectrogram , SAMPLE_RATE)

    print(f"There are {len(usd)} samples in dataset")

    signal , label = usd[0]

    print(signal , label)