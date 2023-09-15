from src.models.encoders import SatelliteViT, GroundCoverViT, CLIPEncoder
import torch
from torch import nn, F
from torch import nn, optim

from src.dataloader import SatelliteDataset, GccDataset, SuperDataset

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, satellite_embedding, ground_cover_embedding):
        # Normalize the embeddings
        satellite_embedding = F.normalize(satellite_embedding, p=2, dim=1)
        ground_cover_embedding = F.normalize(ground_cover_embedding, p=2, dim=1)
        
        # Compute the similarity matrix
        sim_matrix = torch.mm(satellite_embedding, ground_cover_embedding.t()) / self.temperature
        
        # Compute the contrastive loss
        loss = -torch.log(
            torch.exp(sim_matrix.diag()) /
            torch.sum(torch.exp(sim_matrix), dim=1)
        ).mean()
        
        return loss


# Initialize your models and loss function here
# For example: 
clip_model = CLIPEncoder()
criterion = ContrastiveLoss()

optimizer = optim.Adam(clip_model.parameters(), lr=0.001)

# Initialize your datasets and dataloaders here
satellite_dataset = SatelliteDataset(satellite_image_paths)
gcc_dataset = GccDataset(ground_cover_paths)
super_dataset = SuperDataset(satellite_dataset, gcc_dataset)

train_dataloader = DataLoader(super_dataset, batch_size=32, shuffle=True)

# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    for satellite_images, ground_cover_images in train_dataloader:
        optimizer.zero_grad()
        satellite_embedding, ground_cover_embedding = clip_model(satellite_images, ground_cover_images)
        loss = criterion(satellite_embedding, ground_cover_embedding)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")