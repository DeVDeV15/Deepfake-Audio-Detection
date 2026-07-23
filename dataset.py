import os
import torch
import torchaudio
from torch.utils.data import Dataset

class SpoofDataset(Dataset):
    """
    Dataset loader for ASVspoof 2019 LA and standardized audio deepfake formats.
    Loads raw 16kHz mono audio, applying fixed-length padding or cropping (e.g., 4 seconds = 64,000 samples).
    Extracts utterance-level ground-truth labels (0.0 = bonafide / real, 1.0 = spoof / fake).
    """
    def __init__(self, audio_paths, labels, target_samples=64000, sr=16000):
        self.audio_paths = audio_paths
        self.labels = labels
        self.target_samples = target_samples
        self.sr = sr

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        waveform, original_sr = torchaudio.load(path)
        
        # Convert multi-channel / stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if sample rate does not match target (16 kHz)
        if original_sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.sr)
            waveform = resampler(waveform)
            
        waveform = waveform.squeeze(0)
        num_samples = waveform.shape[0]
        
        # Enforce exact length via padding or cropping
        if num_samples < self.target_samples:
            pad_amount = self.target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[:self.target_samples]
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return waveform, label


def parse_asvspoof_protocol(protocol_path, audio_dir):
    """
    Parses ASVspoof 2019 LA protocol text files.
    Returns lists of file paths and float labels (0.0 for bonafide, 1.0 for spoof).
    """
    audio_paths = []
    labels = []
    
    if not os.path.exists(protocol_path):
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            file_id = parts[1]
            label_str = parts[4]
            
            # Support both .flac and .wav extensions
            flac_path = os.path.join(audio_dir, f"{file_id}.flac")
            wav_path = os.path.join(audio_dir, f"{file_id}.wav")
            
            if os.path.exists(flac_path):
                file_path = flac_path
            elif os.path.exists(wav_path):
                file_path = wav_path
            else:
                continue
                
            if label_str == "bonafide":
                labels.append(0.0)
                audio_paths.append(file_path)
            elif label_str == "spoof":
                labels.append(1.0)
                audio_paths.append(file_path)
                
    return audio_paths, labels
