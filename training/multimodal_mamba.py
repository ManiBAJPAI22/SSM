# training/multimodal_mamba.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MultimodalMambaModel(nn.Module):
    def __init__(self, d_model=384, audio_dim=80, vocab_size=50257):
        super().__init__()
        self.d_model = d_model
        
        # Audio encoder: Project mel spectrogram to d_model
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(audio_dim, d_model//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model//2, d_model, kernel_size=3, padding=1),
        )
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        
        # Mamba blocks for sequence modeling [2]
        self.mamba1 = Mamba(
            d_model=d_model,
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width  
            expand=2,    # Block expansion factor
        )
        
        self.mamba2 = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, audio_features, text_tokens):
        batch_size = audio_features.shape[0]
        
        # Process audio: [B, mel_bins, time] -> [B, time, d_model]
        audio_emb = self.audio_encoder(audio_features)  # [B, d_model, time]
        audio_emb = audio_emb.transpose(1, 2)  # [B, time, d_model]
        
        # Process text: [B, seq_len] -> [B, seq_len, d_model]  
        text_emb = self.text_embedding(text_tokens)
        
        # Concatenate audio and text embeddings
        combined = torch.cat([audio_emb, text_emb], dim=1)  # [B, time+seq_len, d_model]
        
        # Apply Mamba blocks
        x = self.mamba1(combined)
        x = self.mamba2(x)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits
