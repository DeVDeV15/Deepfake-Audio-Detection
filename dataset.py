import torch
import torchaudio
from torch.utils.data import Dataset

class SpoofDiarizationDataset(Dataset):
    """
    Slices raw waveforms into overlapping temporal frames for continuous diarization.
    Applies active artifact amplification to expose low-frequency synthetic anomalies.
    Handles short-file padding and label alignment defensively.
    """
    def __init__(self, audio_paths, labels=None, window_ms=400, hop_ms=20, sr=16000):
        """
        Args:
            audio_paths (list): List of strings containing paths to audio files (.flac or .wav).
            labels (list, optional): List of frame-level binary masks/arrays.
            window_ms (int): Sliding window size in milliseconds.
            hop_ms (int): Step size/hop length in milliseconds.
            sr (int): Target sampling rate for processing.
        """
        self.audio_paths = audio_paths
        self.labels = labels 
        self.sr = sr
        self.window_size = int((window_ms / 1000) * sr)
        self.hop_length = int((hop_ms / 1000) * sr)

    def active_artifact_amplification(self, waveform):
        """
        Inverts speech enhancement logic to isolate and amplify low-frequency 
        generative errors, preventing the network from utilizing noise shortcuts.
        """
        low_band = torchaudio.functional.lowpass_biquad(waveform, self.sr, cutoff_freq=3000)
        amplified_waveform = waveform + (1.5 * low_band) 
        return amplified_waveform

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.audio_paths[idx])
        
        # Ensure single channel (mono) tracking
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample on the fly if original file deviates from target rate
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
            
        # Pad ultra-short files to avoid unfold errors
        if waveform.shape[1] < self.window_size:
            pad_amount = self.window_size - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        # 1. Magnify structural deepfake defects
        waveform = self.active_artifact_amplification(waveform)
        
        # 2. Slice long waveform into sequential chunks [num_frames, window_size]
        frames = waveform.unfold(1, self.window_size, self.hop_length).squeeze(0)
        num_frames = frames.shape[0]
        
        # 3. Label Alignment Guard: Broadcasts the protocol label across all generated frames
        if self.labels is not None:
            label = self.labels[idx]
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
                
            if label.shape[0] < num_frames:
                padding_val = label[-1] if label.numel() > 0 else 0
                pad = torch.full((num_frames - label.shape[0],), padding_val, dtype=label.dtype)
                label = torch.cat([label, pad])
            else:
                label = label[:num_frames]
        else:
            label = torch.full((num_frames,), -1)
            
        return frames, label

def collate_diarization_batch(batch):
    """
    DYNAMIC COLLATION ENHANCEMENT:
    Pads variable frame counts within a batch to matching lengths on the fly.
    Pads features with 0 and masks with -1 so train.py drops them seamlessly.
    """
    batch_frames = [item[0] for item in batch]
    batch_labels = [item[1] for item in batch]
    
    # Identify the maximum frame length inside this specific batch block
    max_frames = max([f.shape[0] for f in batch_frames])
    window_size = batch_frames[0].shape[1]
    
    # Initialize padded target matrices hosted cleanly in CPU RAM
    padded_frames = torch.zeros(len(batch), max_frames, window_size)
    padded_labels = torch.full((len(batch), max_frames), -1.0) 
    
    for i, (f, l) in enumerate(zip(batch_frames, batch_labels)):
        num_f = f.shape[0]
        padded_frames[i, :num_f, :] = f
        padded_labels[i, :num_f] = l
        
    return padded_frames, padded_labels