import torch
import torch.nn as nn 
import torch.nn.functional as F 


def scaled_dot_product(q, k, v):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    d_k = q.size(-1)
    sclaed_attention_logits = matmul_qk / (d_k **0.5)
    attention_weights = F.softmax(sclaed_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights



