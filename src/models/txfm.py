import torch
import torch.nn as nn 
import torch.nn.functional as F 
from typing import Tuple



def scaled_dot_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the scaled dot-product attention given query, key, and value tensors.
    MATH:
        M_qk = Q * K^T
        Z = (Q*K.T)/sqrt(d_k)
        W = softmax(Z)
        ouptut = W*V
    Args:
        q (torch.Tensor): Query tensor of shape `(batch_size, num_heads, seq_len, depth)`.
        k (torch.Tensor): Key tensor of shape `(batch_size, num_heads, seq_len, depth)`.
        v (torch.Tensor): Value tensor of shape `(batch_size, num_heads, seq_len, depth_v)`.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor of shape `(batch_size, num_heads, seq_len, depth_v)` and 
                                           the attention weights tensor of shape `(batch_size, num_heads, seq_len, seq_len)`.
    """
    # Calculate the product of q and the transpose of k.
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    
    # Scale the matmul_qk using the square root of the last dimension of q.
    d_k = q.size(-1)
    scaled_attention_logits = matmul_qk / (d_k ** 0.5)
    
    # Softmax to obtain attention weights
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    
    # Calculate the output tensor
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # Number of attention Heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0 # Check if model dimension is divisible by number of heads
        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(d_model, d_model) # Query mapping
        self.wk = nn.Linear(d_model, d_model) # Key mapping
        self.wv = nn.Linear(d_model, d_model) # Value mapping

        self.dense = nn.Linear(d_model, d_model) # Final output mapping 
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0,2,1,3)
    
    def forward(self, q, k, v):
        batch_size = q.size(0)
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wq(k), batch_size)
        v = self.split_heads(self.wq(v), batch_size)
        
        output, attention_weights = scaled_dot_product(q, k, v)
        output = output.permute(0,2,1,3).contiguous().view(batch_size, -1, self.d_model) # concat heads
        output = self.dense(output) # Final output linear layer
        return output

# Point-wise Feed-Forward Network
class PointWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)  # 512 -> 2048
        self.linear2 = nn.Linear(dff, d_model)  # 2048 -> 512

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))  # ReLU activation in between



# Single Transformer
class TransformerLayer(nn.Module):
    def __init__(self,  d_model, num_heads, dff):
        super(TransformerLayer, self).__init__()
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn_output = self.multi_head_attn(x, x, x) # Multi-head attn
        out1 = self.layernorm1(x+attn_output)  # Add & Norm 1
        ffn_output = self.ffn(out1) # Feed-Forward
        out2 = self.layernorm2(out1 + ffn_output) # Add & Norm 2
        return out2


# Full Transformer Model
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model, num_heads, dff) for _ in range(num_layers)])  # Multiple transformer layers

    def forward(self, x):
        for layer in self.layers:  # Loop through each layer
            x = layer(x)  # Apply the layer
        return x  # Return final output

# # Create an instance of the full Transformer model
# model = Transformer(num_layers=6, d_model=512, num_heads=8, dff=2048)
# test_tensor = torch.rand(1, 512, 512)

# print(model)
# # Let's say these are your latent variables
# latent1 = torch.rand((32, 10, 512))  # From encoder 1
# latent2 = torch.rand((32, 10, 728))  # From encoder 2

# # Option 1: Dimensionality Reduction for latent2
# reducer = nn.Linear(728, 512)
# latent2_reduced = reducer(latent2)
# output = model(latent2_reduced)
# print(f"Option 1: Dimensionality Reduction for latent2\nOutput Shape: {output.shape}")
# # Option 2: Dimensionality Expansion for latent1
# linear1 = nn.Linear(728, 512)
# latent1_expanded = linear1(latent2)
# output = model(latent1_expanded)
# print(f"Option 2: Dimensionality Expansion for latent1\nOutput Shape: {output.shape}")
# # Option 3: Concatenation + Linear layer
# concat_latent = torch.cat((latent1, latent2), dim=-1)  # concatenate along the last dimension
# linear3 = nn.Linear(512 + 728, 512)
# common_latent = linear3(concat_latent)
# output = model(common_latent)
# print(f"Option 3: Concatenation + Linear layer\nOutput Shape: {output.shape}")