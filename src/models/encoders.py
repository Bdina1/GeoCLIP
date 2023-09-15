import torch
import torch.nn as nn
import timm

class SatelliteViT(nn.Module):
    def __init__(self, patch_size=16, out_dim=768, model_name="vit_base_patch16_224"):
        super(SatelliteViT, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.patch_embed.conv[0] = nn.Conv2d(3, self.model.patch_embed.proj.in_channels, kernel_size=patch_size, stride=patch_size)
        self.model.head = nn.Linear(self.model.head.in_features, out_dim)

    def forward(self, x):
        return self.model(x)

class GroundCoverViT(nn.Module):
    def __init__(self, patch_size=1, out_dim=768, model_name="vit_base_patch16_224"):
        super(GroundCoverViT, self).__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        self.model.patch_embed.conv[0] = nn.Conv2d(11, self.model.patch_embed.proj.in_channels, kernel_size=patch_size, stride=patch_size)
        self.model.head = nn.Linear(self.model.head.in_features, out_dim)

    def forward(self, x):
        return self.model(x)

class CLIPEncoder(nn.Module):
    def __init__(self):
        super(CLIPEncoder, self).__init__()
        self.satellite_encoder = SatelliteViT()
        self.ground_cover_encoder = GroundCoverViT()

    def forward(self, satellite_images, ground_cover_images):
        satellite_embedding = self.satellite_encoder(satellite_images)
        ground_cover_embedding = self.ground_cover_encoder(ground_cover_images)
        return satellite_embedding, ground_cover_embedding

# Example usage
clip_model = CLIPEncoder()
satellite_images = torch.randn(32, 3, 224, 224)  # Batch of 32
ground_cover_images = torch.randn(32, 11, 224, 224)  # Batch of 32

satellite_embedding, ground_cover_embedding = clip_model(satellite_images, ground_cover_images)


