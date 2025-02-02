import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(
            self,
            max_seq_length: int,
            embed_dim: int
    ):
        '''
        max_seq_length: int = maximum length of a sequence to provide positional embeddings
        embed dim: int = dimension of embedding
        '''

        super(SinusoidalPositionalEmbedding, self).__init__()

        pos = torch.arange(max_seq_length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10_000) / embed_dim))

        embed = torch.zeros(1,max_seq_length,embed_dim, requires_grad=False)
        embed[0,:,0::2] = torch.sin(pos * div)
        embed[0,:,1::2] = torch.cos(pos * div)

        self.embed = embed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_length, _ = x.shape
        embedding = self.embed[:,:seq_length].to(x.device)
        return embedding

class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(
            self,
            time_steps: int,
            max_dim: int
    ):
        super().__init__()

        pos = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, max_dim, 2).float() * -(math.log(10000.0) / max_dim))
        
        embeddings = torch.zeros(time_steps, max_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(pos * div)
        embeddings[:, 1::2] = torch.cos(pos * div)

        self.embeddings = embeddings
    
    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds[:, :, None, None]