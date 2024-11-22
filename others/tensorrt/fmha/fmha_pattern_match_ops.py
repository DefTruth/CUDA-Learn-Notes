import math
import torch
from torch import nn


class QKVAttentionMatchTensorRTfusedMHA(nn.Module):
    """
    reference: https://github.com/NVIDIA/TensorRT/issues/3575  
    fused MHA Kernel in TensorRT 9.2+ need specific pattern for Attention: 
    Q: [B, S, H] -MatMul-> [B, S, H] -Reshape-> [B, S, N, h] -Transpose-> [B, N, S, h] -> MatMul -> [B, N, S, S] -> MatMul -> [B, N, S, h] -Transpose-> [B, S, N, h] -Reshape-> [B, S, H] -LayerNorm->...
    K: [B, S, H] -MatMul-> [B, S, H] -Reshape-> [B, S, N, h] -Transpose-> [B, N, h, S] ---^                           ^
    V: [B, S, H] -MatMul-> [B, S, H] -Reshape-> [B, S, N, h] -Transpose-> [B, N, S, h] --------------------------------
    """
    def __init__(self, n_heads: int = 1):
        super().__init__()
        self.n_heads = n_heads
    
    @torch.no_grad()
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """
        Apply QKV attention to match TRT MHA pattern
        :param Q: an [B, S, H] tensor of Qs
        :param K: an [B, S, H] tensor of Ks
        :param V: an [B, S, H] tensor of Vs
        :return: an output tensor with shape [B, S, H].
        """
        N = self.n_heads
        B, S, H = Q.shape
        h = H // N

        # Reshape
        Q = Q.view(B, S, N, h) # [B, S, H] -> [B, N, S, S]
        K = K.view(B, S, N, h) # [B, S, H] -> [B, N, S, S]
        V = V.view(B, S, N, h) # [B, S, H] -> [B, N, S, S]

        # Transpose  
        Q = Q.permute(0, 2, 1, 3) # [B, S, N, h] -> [B, N, S, h] 
        K = K.permute(0, 2, 3, 1) # [B, S, N, h] -> [B, N, h, S]
        V = V.permute(0, 2, 1, 3) # [B, S, N, h] -> [B, N, S, h]
        
        # --------------------- Attention Begin --------------------------------
        # Supported Patterns:
        # 0: MatMul -> [B, N, S, S] -> MatMul -> [B, N, S, h]
        # 1: MatMul -> [B, N, S, S] -> Softmax -> MatMul -> [B, N, S, h]
        # 2: MatMul -> [B, N, S, S] -> [Ops-DoNot-Change-Shape] -> MatMul -> [B, N, S, h]
        
        # Apply pattern: MatMul -> Softmax -> MatMul
        # Div: scaling
        scale = 1 / math.sqrt(h)
        # [0] MatMul: Q*K [B, N, S, S] = [B, N, S, h] * [B, N, h, S] 
        weight = torch.matmul(Q, K) * scale 
        # [1] Softmax: [B, N, S, S] 
        weight = torch.softmax(weight, dim=-1).type(weight.dtype) 
        # [2] MatMul: w*v [B, N, S, S]*[B, N, S, h]->[B, N, S, h] 
        out = torch.matmul(weight, V) 
        # ---------------------  Attention End  --------------------------------

        # Transpose [B, N, S, h] -> [B, S, N, h] -> Reshape -> [B, S, H]
        out = out.permute(0, 2, 1, 3).reshape(B, S, H)
        return out
