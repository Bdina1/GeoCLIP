import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size = 768):
        super().__init__()
        self.patch_sizes = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)
    


class ViTEncoder(nn.Module):
    def __init__(self, in_channels=3,  patch_size=16, emb_size=768, num_heads=12, num_layers=12):
        super().__init__()
        self.patch_emb = PatchEmbedding(in_channels, patch_size, emb_size)
        self.cntx_tokens = nn.Parameter(torch.randn(1,1,emb_size))  
        self.transformer = nn.Transformer(emb_size, num_heads, num_layers)
    def forward(self, x):
        x = self.patch_emb(x)
        cntx_tokens = self.cntx_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat([cntx_tokens, x], dim=1)
        x = self.transformer(x)
        return x
    
model = ViTEncoder(in_channels=9)
x = torch.rand(1,9,256,256)
embeddings = model(x)

print(embeddings.shape)

        