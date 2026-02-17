import torch
import torch.nn as nn
from torch import Tensor

class VisionTransformer256(nn.Module):
    """Vision Transformer custom pour 256×256"""
    
    def __init__(self, 
                 image_size=256,
                 patch_size=16,
                 num_classes=1,
                 dim=768,
                 depth=12,
                 heads=12,
                 mlp_dim=3072):
        super().__init__()
        
        num_patches = (image_size // patch_size) ** 2  # (256/16)² = 256
        patch_dim = 3 * patch_size * patch_size  # 3 * 256 = 768
        
        # 1. Patch Embedding
        self.patch_embed = nn.Linear(patch_dim, dim)  # 768 → 768
        
        # 2. Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 3. Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 5. Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
        self.patch_size = patch_size
        self.image_size = image_size
    
    def forward(self, x):  # x: [B, 3, 256, 256]
        B = x.shape[0]
        
        # 1. Divide into patches
        x = x.reshape(B, 3, self.image_size // self.patch_size, self.patch_size,
                      self.image_size // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, -1, 3 * self.patch_size * self.patch_size)  # [B, 256, 768]
        
        # 2. Patch embedding
        x = self.patch_embed(x)  # [B, 256, 768]
        
        # 3. Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 257, 768]
        
        # 4. Add positional embeddings
        x = x + self.pos_embedding  # [B, 257, 768]
        
        # 5. Transformer
        x = self.transformer(x)  # [B, 257, 768]
        
        # 6. Extract [CLS] token
        x = x[:, 0]  # [B, 768]
        
        # 7. Classification
        x = self.norm(x)
        x = self.head(x)  # [B, 1]
        
        return x