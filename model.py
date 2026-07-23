import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2AASIST_Diarizer(nn.Module):
    """
    Hybrid architecture combining Wav2Vec 2.0 feature maps with an AASIST graph 
    attention bottleneck to perform continuous, zero-day partial deepfake localization.
    """
    def __init__(self, w2v2_path="facebook/wav2vec2-base", hidden_dim=256, dropout=0.3):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(w2v2_path)
        self.wav2vec2.gradient_checkpointing_enable()
        
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
            
        self.graph_attention_bottleneck = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.frame_classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        batch_size, num_frames, window_size = x.shape
        x_flattened = x.view(batch_size * num_frames, window_size)
        
        valid_frames_mask = (x_flattened.abs().sum(dim=-1) > 1e-5).long()
        attention_mask = valid_frames_mask.unsqueeze(-1).expand(-1, window_size)
        
        chunk_size = 128
        pooled_outputs = []
        
        for i in range(0, x_flattened.size(0), chunk_size):
            chunk_x = x_flattened[i : i + chunk_size]
            chunk_mask = attention_mask[i : i + chunk_size]
            
            w2v2_outputs = self.wav2vec2(
                chunk_x,
                attention_mask=chunk_mask
            ).last_hidden_state
            
            chunk_pooled = torch.mean(w2v2_outputs, dim=1)
            pooled_outputs.append(chunk_pooled)
            
        pooled_w2v2 = torch.cat(pooled_outputs, dim=0)
        sequence_features = pooled_w2v2.view(batch_size, num_frames, -1)
        
        graph_embeddings = self.graph_attention_bottleneck(sequence_features)
        frame_logits = self.frame_classifier(graph_embeddings).squeeze(-1)
        
        # AMP FIX: Return raw pre-sigmoid logits for numerically stable loss calculation
        return frame_logits