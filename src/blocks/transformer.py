import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from .attention import SelfAttentionBlock, CrossAttentionBlock
from .feed_forward import FeedForward
from ..utils.positional_encoding import SinusoidalPositionalEmbedding

class BasicTransformer(nn.Module):
    def __init__(
            self,
            dim: int,
            max_seq_length: int = 1024,
            attn_num_heads: int = 1,
            attn_dropout: float = 0.0,
            bias: bool = True,
            norm: Optional[Tuple] = None,
            causal: bool = False
    ):
        '''
        dim: int = sequence length of input tensor
        attn_num_heads: int = number of heads for SelfAttentionBlock
        attn_dropout: float = dropout probability for SelfAttentionBlock
        bias: bool = use bias in modules
        norm: Tuple = normalized shape for LayerNorm, skip if None (not recommended)
        causal: bool = set scaled_dat_product is_causal parameter, defaults to False
        '''

        super(BasicTransformer, self).__init__()

        self.attn = SelfAttentionBlock(dim=dim, num_heads=attn_num_heads, dropout=attn_dropout, bias=bias, causal=causal)

        self.ff = FeedForward(dim=dim, bias=bias, norm=norm)

        self.pos_emb = SinusoidalPositionalEmbedding(max_seq_length=max_seq_length, embed_dim=dim)

        if norm:
            self.norm = nn.LayerNorm(normalized_shape=norm)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        x: input tensor, should contain [dim] length sequence of [L] length embeddings -> shape: (N,L,dim)
        attn_mask: torch.Tensor = optional attention mask, must match the sequence length of the input tensor
        '''
        
        embedding = self.pos_emb(x)

        # Add positional embedding to input
        h = x + embedding
        # Calculate attention
        attn_output = self.attn(h, mask=attn_mask)

        # Add attn_output and norm if norm
        h = h + attn_output
        if hasattr(self, 'norm'):
            h = self.norm(h)

        # Send to feed forward
        return self.ff(h)

class Transformer2d(nn.Module):
    '''
    Implementation of Vision Transformer described in
    Dosovitskiy et al. in An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    '''
    def __init__(
            self,
            block_size: Tuple,
            embed_dim: int,
            seq_dim: int,
            attn_num_heads: int = 1,
            attn_dropout: float = 0.0,
            bias: bool = True,
            norm: Optional[Tuple] = None,
            causal: bool = False,
            output_2d: bool = True
    ):
        '''
        blocks_size: Tuple = shape of patches input tensor is separated into
        embed_dim: int = dimension of embedding for each patch
        seq_dim: int = number of patches from input given block size
        attn_num_heads: int = number of attention heads
        attn_dropout: float = dropout probability for scaled_dot_product_attention
        bias: bool = use bias in modules
        norm: Tuple = normalized shaped for LayerNorm, skip if None (not recommended)
        causal: bool = set scaled_dot_product_attention is_causal parameter, defaults to False
        output_2d: bool = return value is given in original input shape (b c h w), else (b L E)
        '''

        super(Transformer2d, self).__init__()

        self.output_2d = output_2d
        self.h, self.w = block_size

        self.proj1 = nn.Linear(self.h*self.w, embed_dim, bias=bias)
        if output_2d:
            self.proj2 = nn.Linear(embed_dim, self.h*self.w, bias=bias)

        self.attn = SelfAttentionBlock(
            dim=embed_dim,
            num_heads=attn_num_heads,
            dropout=attn_dropout,
            bias=bias,
            causal=causal
        )

        self.pos_emb = SinusoidalPositionalEmbedding(max_seq_length=seq_dim, embed_dim=embed_dim)

        self.ff = FeedForward(dim=embed_dim, bias=bias, norm=norm)

        if norm:
            self.norm = nn.LayerNorm(normalized_shape=norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save initial shape H W
        H, W = x.shape[2:]
        H = H // self.h
        W = W // self.w
        
        # Separate x into patches and reshape in linear projections
        # Go from x (b c H W) -> (b L E) | E = h*w, L = c * (H/h) * (W/w) = seq_dim
        # Split x across rows and concat on channel dim
        p = torch.concat(torch.split(x, self.h, dim=2), dim=1)
        # Split p across columns and concat on channel dim
        p = torch.concat(torch.split(p, self.w, dim=3), dim=1)
        # Rearrange to (b L E)
        p = rearrange(p, 'b c h w -> b c (h w)')

        # Project p from h*w to embedding_dim
        p = self.proj1(p)

        embedding = self.pos_emb(p)

        # Add positional embedding
        p = p + embedding
        # Calculate attention
        attn_output = self.attn(p)

        p = p + attn_output
        if hasattr(self, 'norm'):
            p = self.norm(p)

        p = self.ff(p)

        # Reshape back to 2d if output_2d is set
        if self.output_2d:
            p = self.proj2(p)
            p = rearrange(p, 'b L (h w) -> b L h w', h=self.h, w=self.w)
            p = torch.concat(torch.split(p, p.shape[1]//W, dim=1), dim=3)
            p = torch.concat(torch.split(p, p.shape[1]//H, dim=1), dim=2)

        return p

class GeoEmbeddingTransformer2d(nn.Module):
    def __init__(
            self,
            block_size: Tuple,
            embed_dim: int,
            q_seg_dim: int,
            attn_num_heads: int = 1,
            attn_dropout: float = 0.0,
            bias: bool = True,
            norm: Optional[Tuple] = None,
            output_2d: bool = True
    ):
        '''
        blocks_size: Tuple = shape of patches 2D query tensor is separated into
        embed_dim: int = dimension of embedding for each patch, as well as key and value embeddings
        q_seq_dim: int = number of patches from input, given block size
        kv_seq_dim: int = length of geolocated key,value tensor
        attn_num_heads: int = number of attention heads
        attn_dropout: float = dropout probability for scaled_dot_product_attention
        bias: bool = use bias in modules
        norm: Tuple = normalized shape for LayerNorm, skip if None (not recommended)
        causal: bool = set scaled_dot_product_attention is_causal parameter, defaults to False
        output_2d: bool = return value is given in original input shape (b c h w), else (b L E)
        '''

        super(GeoEmbeddingTransformer2d, self).__init__()

        self.output_2d = output_2d
        self.h, self.w = block_size

        self.proj1 = nn.Linear(self.h*self.w, embed_dim, bias=bias)
        if output_2d:
            self.proj2 = nn.Linear(embed_dim, self.h*self.w, bias=bias)
        
        self.xattn = CrossAttentionBlock(
            dim=embed_dim,
            q_heads=attn_num_heads,
            kv_heads=1,
            dropout=attn_dropout,
            bias=bias
        )

        self.pos_emb = SinusoidalPositionalEmbedding(max_seq_length=q_seg_dim, embed_dim=embed_dim)

        self.ff = FeedForward(dim=embed_dim, bias=bias, norm=norm)

        if norm:
            self.norm = nn.LayerNorm(normalized_shape=norm)
    
    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Separate x into patches and reshape in linear projections
        # Go from x (b c H W) -> (b L E) | E = h*w, L = c * (H/h) * (W/w) = seq_dim
        # Split x across rows and concat on channel dim
        p = torch.concat(torch.split(q, self.h, dim=2), dim=1)
        # Split p across columns and concat on channel dim
        p = torch.concat(torch.split(p, self.w, dim=3), dim=1)
        # Rearrange to (b L E)
        p = rearrange(p, 'b c h w -> b c (h w)')

        # Project p from h*w to embedding_dim
        p = self.proj1(p)

        embedding = self.pos_emb(p)

        # Add positional embedding
        p = p + embedding

        # Calculate attention
        attn_output = self.xattn(x=p, kv=kv)

        p = p + attn_output
        if hasattr(self, 'norm'):
            p = self.norm(p)
        
        p = self.ff(p)

        # Reshape back to 2d if output_2d is set
        if self.output_2d:
            p = self.proj2(p)
            p = rearrange(p, 'b L (h w) -> b L h w', h=self.h, w=self.w)
            p = torch.concat((p[:,0::4,:,:], p[:,1::4,:,:], p[:,2::4,:,:], p[:,3::4,:,:]), dim=2)
            p = torch.concat((p[:,0::4,:,:], p[:,1::4,:,:], p[:,2::4,:,:], p[:,3::4,:,:]), dim=3)

        return p