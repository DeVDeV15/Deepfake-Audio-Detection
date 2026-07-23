import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, WavLMModel

class BiDirectionalCrossAttention(nn.Module):
    """
    Bi-directional Cross-Attention module.
    Allows WavLM features to attend to Wav2Vec2 features and vice versa.
    """
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.attn_w2v_to_wavlm = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_wavlm_to_w2v = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, w2v_feats, wavlm_feats):
        # 1. Wav2Vec2 queries WavLM
        w2v_attended, _ = self.attn_w2v_to_wavlm(query=w2v_feats, key=wavlm_feats, value=wavlm_feats)
        
        # 2. WavLM queries Wav2Vec2
        wavlm_attended, _ = self.attn_wavlm_to_w2v(query=wavlm_feats, key=w2v_feats, value=w2v_feats)
        
        # 3. Fuse cross-attended feature maps
        fused = torch.cat([w2v_attended, wavlm_attended], dim=-1)  # [batch, seq_len, 1536]
        fused_proj = self.norm(self.fusion_proj(fused))            # [batch, seq_len, 768]
        return fused_proj


class GraphAttentionPooling(nn.Module):
    """AASIST-inspired Graph Attention Pooling."""
    def __init__(self, in_dim=768, hidden_dim=256):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attn_weights = self.attn_proj(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled, attn_weights


class DualSSL_AASIST(nn.Module):
    """
    Dual SSL Architecture (WavLM + Wav2Vec2) with Bi-Directional Cross-Attention
    and Graph Attention Pooling.
    """
    def __init__(self, w2v2_path="facebook/wav2vec2-base", wavlm_path="microsoft/wavlm-base-plus", hidden_dim=256, dropout=0.3):
        super().__init__()
        # Load both pretrained SSL frontends
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(w2v2_path)
        self.wavlm = WavLMModel.from_pretrained(wavlm_path)
        
        # Freeze initial CNN feature extractors
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.wavlm.feature_extractor.parameters():
            param.requires_grad = False
            
        self.wav2vec2.gradient_checkpointing_enable()
        self.wavlm.gradient_checkpointing_enable()
        
        # Cross-Attention Fusion
        self.cross_attention = BiDirectionalCrossAttention(embed_dim=768, num_heads=8)
        
        # AASIST Attentive Graph Pooler
        self.pooler = GraphAttentionPooling(in_dim=768, hidden_dim=hidden_dim)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Extract features from both SSL backbones
        w2v_out = self.wav2vec2(x).last_hidden_state      # [batch, seq_len, 768]
        wavlm_out = self.wavlm(x).last_hidden_state        # [batch, seq_len, 768]
        
        # Bi-directional Cross-Attention Fusion
        fused_features = self.cross_attention(w2v_out, wavlm_out) # [batch, seq_len, 768]
        
        # Graph Attention Pooling
        pooled, _ = self.pooler(fused_features)           # [batch, 768]
        
        # Output Logits
        logits = self.classifier(pooled).squeeze(-1)       # [batch]
        return logits
