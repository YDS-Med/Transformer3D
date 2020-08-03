import torch
from torch import nn
###import torch.nn.functional as F
import numpy as np

from schnetpack.nn import Dense
from schnetpack.nn.base import Aggregate

device = torch.device("cuda")

__all__ = ["MultiScaleAttention"]


class MultiScaleAttention(nn.Module):
    r"""Ego Attention block used in 3DT module.

    Args:
        n_in (int): number of input (i.e. atomic embedding) dimensions.
        
    """

    def __init__(
        self,
        n_atom_basis,
        n_scales,
        n_heads=8,   
        use_time_embedding=True,
    ):
        super(MultiScaleAttention, self).__init__()
        
        # dense layer as mh_attention
        assert (n_atom_basis%n_heads==0), "Mismatch Head Numbers."
        assert (n_atom_basis%2==0), "Must Be Even number of Atom Features."
        
        n_per_head = n_atom_basis//n_heads        
        self.n_per_head = n_per_head
        
        self.n_heads = n_heads
        self.n_scales = n_scales
        self.n_atom_basis = n_atom_basis        
        self.use_time_embedding = use_time_embedding        
        
        self.mh_q = Dense(n_atom_basis, n_atom_basis, bias=False, activation=None)        
        self.mh_k = Dense(n_atom_basis, n_atom_basis, bias=False, activation=None)
        self.mh_v = Dense(n_atom_basis, n_atom_basis, bias=False, activation=None)
        self.mh_o = Dense(n_atom_basis, n_atom_basis, bias=False, activation=None) 

#         self.self_attn = nn.MultiheadAttention(n_atom_basis,n_heads)

# #         ### For Time Embedding:
# #         self.even_mask = torch.cat( [torch.ones(n_atom_basis//2,1),torch.zeros(n_atom_basis//2,1)], dim=-1)
# #         self.even_mask = self.even_mask.reshape(1,1,1,n_atom_basis)
        
# #         self.period = torch.pow(10000, -2.* torch.range(1,n_atom_basis//2) /n_atom_basis).unsqueeze(-1)
# #         self.period = torch.cat( [self.period,self.period], dim=-1 )
# #         self.period = self.period.reshape(1,1,1,n_atom_basis)       
        
# #         tt = torch.range(1,n_scales).reshape(1,1,n_scales,1)
# #         tt = tt * self.period ### [1,1,n_scales,n_atom_basis]
# #         self.time_embedding = torch.sin( tt ) * self.even_mask + torch.cos( tt ) * (1.-self.even_mask)
                

    def forward( self, xs, e, t):
        """Compute convolution block.

        Args:
            e (torch.Tensor): Element Embedding.
                with (N_b, N_a, n_atom_basis) shape.
            xs (torch.Tensor): input representation/embedding of atomic environments
                with (N_b,N_a,ns,n_atom_basis) shape.
            t : time embedding.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_out) shape.

        """
        
        n_heads = self.n_heads
        n_per_head = self.n_per_head
        n_scales = self.n_scales
        
        if self.use_time_embedding:
            xt = xs + t
        else:
            xt = xs ### (N_b,N_a,n_scales,n_atom_basis)
       
        
        
        ######
        q = e ### [N_b,N_a,n_atom_basis]
        q_vec = self.mh_q(q) ### [N_b, N_a, n_atom_basis]
        q_vec = q_vec.unsqueeze(1) ### 4D tensor: [N_b, 1, N_a, n_atom_basis ]
        q_mh_vec = torch.cat( torch.split(q_vec, n_per_head, dim=-1), dim=1) ### 4D tensor: [N_b, nh, N_a, n_per_head ]
        q_mh_vec = q_mh_vec.unsqueeze(3).expand(-1,-1,-1,n_scales,-1) ### 5D tensor: [N_b, nh, N_a, ns, n_per_head ]
        
        ######
        
        k = xt
        k_vec = self.mh_k(k) ### 4D Tensor: [N_b, N_a, ns, n_atom_basis]
        k_vec = k_vec.unsqueeze(1) ### 5D Tensor: [N_b, 1, N_a, ns, n_atom_basis]
        k_mh_vec = torch.cat( torch.split(k_vec, n_per_head, dim=-1), dim=1) ### 5D tensor: [N_b, nh, N_a, N_n, n_per_head]
        
        ######
             
        v = xs ### (N_b,N_a,ns,n_atom_basis)
        v_vec = self.mh_v(v) ### 4D Tensor: [N_b, N_a, ns, n_atom_basis]
        v_vec = v_vec.unsqueeze(1) ### 5D Tensor: [N_b, 1, N_a, ns, n_atom_basis]
        v_mh_vec = torch.cat( torch.split(v_vec, n_per_head, dim=-1), dim=1) ### 5D tensor: [N_b, nh, N_a, ns, n_per_head]
        
        
        ### Attention:
        dot_product = torch.matmul( q_mh_vec.unsqueeze(-2), k_mh_vec.unsqueeze(-1) ).squeeze(-1) ### 5D: [N_b,nh,N_a,ns,1]
        logits = dot_product/ np.sqrt(self.n_per_head).astype(np.float32)
        att_score = nn.functional.softmax( logits, dim=-2 ) ### [N_b, N_h, N_a, N_n, 1]
        
        m_agg = torch.sum( v_mh_vec*att_score, dim=-2, keepdim=False) ### 4D: [N_b,nh,N_a,n_per_heads]
        
        y = torch.cat( torch.split(m_agg, 1, dim=1), dim=-1) ### 4D: [N_b, 1, N_a, n_atom_basis] 
        y = y.squeeze(1) ### 3D: [N_b, N_a, n_atom_basis] 
        
        # Universal Affine Transform:
        y = self.mh_o(y)

        
        return y

