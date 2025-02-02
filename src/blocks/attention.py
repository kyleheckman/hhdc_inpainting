import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

class SelfAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 1,
            dropout: float = 0.0,
            bias: bool = True,
            causal: bool = False
    ):
        '''
        dim: int = embedding dimension
        num_heads: int = number of attention heads
        dropout: float = dropout probability for scaled_dot_product_attention
        bias: bool = use bias in Linear modules
        causal: bool = set scaled_dat_product is_causal parameter, defaults to False
        '''
        
        super(SelfAttentionBlock, self).__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.is_causal = causal

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        attn_mask: torch.Tensor = optional attention mask, must match the sequence length of the input tensor
        '''

        if mask and mask.shape[1] != x.shape[1]:
            raise Exception(f'Attention mask must match shape of input at dim (1). Got mask {mask.shape}, input {x.shape}')

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.k_proj(x)

        q = rearrange(q, 'b (L H) E -> b H L E', H=self.num_heads)
        k = rearrange(k, 'b (L H) E -> b H L E', H=self.num_heads)
        v = rearrange(v, 'b (L H) E -> b H L E', H=self.num_heads)

        attn_mask = mask
        if attn_mask:
            attn_mask = torch.matmul(attn_mask.unsqueeze(2), attn_mask.unsqueeze(1))
            attn_mask = attn_mask.to(torch.bool)
        
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal, attn_mask=attn_mask, dropout_p=self.dropout)

        attn_output = rearrange(attn_output, 'b H L E -> b (L H) E', H=self.num_heads)
        
        return self.out_proj(attn_output)

class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            q_heads: int = 1,
            kv_heads: int = 1,
            dropout: float = 0.0,
            bias: bool = True
    ):
        '''
        q_dim: int = embedding dimension of query
        kv_dim: int = embedding dimension of key,value
        q_heads: int = number of heads in query
        kn_heads: int = number of heads in key,value
        dropout: float = dropout probability for scaled_dot_product_attention
        biasL bool = use bias in Linear modules
        '''

        super(CrossAttentionBlock, self).__init__()

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
    
    def forward(self, x: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        attn_mask: torch.Tensor = optional attention mask, must match the sequence length of the input tensor
        '''

        q = self.q_proj(x)
        k = self.k_proj(kv)
        v = self.k_proj(kv)

        q = rearrange(q, 'b L (H E) -> b H L E', H=self.q_heads)
        k = rearrange(k, 'b L (H E) -> b H L E', H=self.kv_heads)
        v = rearrange(v, 'b L (H E) -> b H L E', H=self.kv_heads)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=mask, dropout_p=self.dropout)

        attn_output = rearrange(attn_output, 'b H L E -> b L (H E)', H=self.num_heads)
        
        return self.out_proj(attn_output)