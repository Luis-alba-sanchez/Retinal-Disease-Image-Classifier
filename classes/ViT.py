import torch
import torch.nn as nn


class PatchEmbeddings(nn.Module):
    """Class to convert a 2D image into 1D learnable embedding tensor"""
    def __init__(self,
                 img_size: int = 64,
                 in_channels: int = 3,
                 patch_size: int = 4,
                 embedding_dim: int = 256):
        super(PatchEmbeddings, self).__init__()
        
        assert img_size % patch_size == 0, f"Image size should be a multiple of patch size. Image size {img_size} and patch size {patch_size}"
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Conv2d patching layer
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        
        # Creating class token
        self.class_token = nn.Parameter(data=torch.randn(1, 1, embedding_dim), requires_grad=True)
        
        # Creating position embeddinggs
        self.position_embeddings = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim), requires_grad=True)

        # Flatten layer
        # Flatten the height and width dimension into a single dimension
        # while preserving color channels and batch dimensions.
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, X: torch.Tensor):
        # Checking if inputs are correct
        batch_size = X.shape[0]
        
        # Perform the forward pass
        x_patched = self.patcher(X)
        x_flattened = self.flatten(x_patched)

        # Making sure that the output dimensions are in the right order
        x_flatten = x_flattened.permute(0, 2, 1)
        
        # Getting the class token
        class_token = self.class_token.expand(batch_size, -1, -1)
        
        # Prepending class token to patch embedding
        X = torch.cat((class_token, x_flatten), dim=1)
        
        # Adding position embeddings
        X = X + self.position_embeddings
        return X
    

class MultiHeadSelfAttentionBlock(nn.Module):
    """Creates a Multi-Headded Self-Attention block for Vision Transformer"""
    def __init__(self, embedding_dim: int = 256, num_heads: int = 8, attn_dropout: float = 0):
        super(MultiHeadSelfAttentionBlock, self).__init__()

        # Creating the normalization layer
        self.norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Creating the MHSA layer
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                          num_heads=num_heads,
                                          dropout=attn_dropout,
                                          batch_first=True)

    def forward(self, X: torch.Tensor):
        X = self.norm(X)
        X, _ = self.attn(query=X, key=X, value=X, need_weights=False)
        return X
    

class MLPLayer(nn.Module):
    """Creates a normalized multi layer perceptron block"""
    def __init__(self, embedding_dim:int = 256, mlp_size:int = 1024, dropout:float = 0.1):
        super(MLPLayer, self).__init__()

        # Lormalization layer
        self.norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Multi layer perceptron layer
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, X: torch.Tensor):
        X = self.norm(X)
        X = self.mlp(X)
        return X
    

class TransformerEncoderBlock(nn.Module):
    """Creates a ViT transformer encoder block"""
    def __init__(self,
                 embedding_dim:int=256,
                 num_heads:int=8,
                 mlp_size:int=1024,
                 mlp_dropout:float=0.1,
                 attn_dropout:float=0):
        super(TransformerEncoderBlock, self).__init__()

        # Create the MSA block
        self.msa = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim,
                                               num_heads=num_heads,
                                               attn_dropout=attn_dropout)

        # Create teh MLP block
        self.mlp = MLPLayer(embedding_dim=embedding_dim,
                            mlp_size=mlp_size,
                            dropout=mlp_dropout)

    def forward(self, X: torch.Tensor):
        # Create residual connection for MSA block
        X = self.msa(X) + X

        # Create residual connection for MLP block
        X = self.mlp(X) + X
        return X
    

class ViT(nn.Module):
    """Creates a vision transformer model"""
    def __init__(self,
                 img_size: int = 64,                # Training resolution from ViT paper
                 in_channels: int = 3,              # Number of color channels in the input image
                 patch_size: int = 4,               # Patch size
                 num_transformer_layers: int = 12,  # Layers from ViT paper for ViT-Base
                 embedding_dim: int = 256,          # Hidden size D from the paper for ViT-Base
                 mlp_size: int = 1024,              # MLP size from the paper for ViT-Base
                 num_heads: int = 8,                # Heads from the paper for ViT-Base
                 attn_dropout: float = 0,           # Dropout for attention projection
                 mlp_dropout: float = 0.1,          # Dropout for dense/MLP layers
                 embedding_dropout: float = 0.1,    # Dropout for patch and position embeddings
                 num_classes: int = 200):           # Default for ImageNet, can be customized
        super(ViT, self).__init__()
        
        # Check if image size is completely divisible by patch size
        assert img_size % patch_size == 0, f"Image size should be a multiple of patch size. Image size {img_size} and patch size {patch_size}"

        # Creating a dropout for patch embeddings
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Creating a patch embeddings layer
        self.patch_embed_layer = PatchEmbeddings(img_size=img_size,
                                                 in_channels=in_channels,
                                                 patch_size=patch_size,
                                                 embedding_dim=embedding_dim)

        # Creating a transformer encoder blocks
        self.tfm_enc_blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(embedding_dim=embedding_dim,
                                        num_heads=num_heads,
                                        mlp_size=mlp_size,
                                        mlp_dropout=mlp_dropout,
                                        attn_dropout=attn_dropout)
                for _ in range(num_transformer_layers)
            ]
        )

        # Creating the classifier head
        self.fcs = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, X: torch.Tensor):
        # Creating patch embeddings
        X = self.patch_embed_layer(X)

        # Passing it through embedding dropout
        X = self.embedding_dropout(X)

        # Passing it through transformer encoder blocks
        X = self.tfm_enc_blocks(X)

        # Passing the 0 indexed logits through classifier
        X = self.fcs(X[:, 0])
        return X