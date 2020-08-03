import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from schnetpack.nn import Dense
from schnetpack.nn.base import Aggregate


__all__ = ["EgoAttention"]


class EgoAttention(nn.Module):
    r"""Ego Attention block used in 3DT module.

    Args:
        n_in (int): number of input (i.e. atomic embedding) dimensions.
        
    """

    def __init__(
        self,
        n_atom_basis,
        n_hidden,
        n_heads=8,        
        activation=None,
    ):
        super(EgoAttention, self).__init__()
        
        # dense layer as mh_attention
        assert (n_atom_basis%n_heads==0), "Mismatch Head Numbers."
        n_per_head = n_atom_basis//n_heads
        
        self.n_heads = n_heads
        self.n_per_head = n_per_head
        
        self.mh_q = Dense(n_atom_basis, n_atom_basis, bias=False, activation=None)        
        self.mh_k = Dense(n_atom_basis, n_atom_basis, bias=False, activation=None)
        self.mh_v = Dense(n_atom_basis, n_atom_basis, bias=False, activation=None)
        self.mh_o = Dense(n_atom_basis, n_atom_basis, bias=False, activation=None) 
        
        self.layer_norm_in = nn.LayerNorm([n_atom_basis]) ###(input.size()[-1])
              
        

    def forward(self, e, x, y, t, W, pairwise_mask, cutoff_mask=None):
        """Compute convolution block.

        Args:
            e (torch.Tensor): Element Embedding.
                with (N_b, N_a, n_atom_basis) shape.
            q (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_atom_basis) shape.
            k (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, N_n, n_atom_basis) shape.
            v (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, N_n, n_atom_basis) shape.
            pairwise_mask: (N_b, N_a, N_n)
            cutoff_mask : (N_b, N_a, N_n, 1)

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_out) shape.

        """
        
        n_heads = self.n_heads
        n_per_head = self.n_per_head
        ###print('Head , PerHead:', n_heads, n_per_head)
        
        k_size = y.size()
        ### k_size = [ N_b, N_a, 1+N_n, n_atom_basis ] = [64,9,8,128]
        ###print('k size: ', k_size)
        
        if cutoff_mask is None:
            f_ij = r_ij.unsqueeze(-1)
            cutoff_mask = torch.ones( size=[k_size[0], k_size[1], k_size[2], 1] ) ### [N_b,N_a,N_n,1]        
        cutoff_mask = cutoff_mask.unsqueeze(1) ### [N_b, 1(N_h), N_a, N_n, 1]
        
        ### ------------------###
        ### Note: q=x, k=y+W, v=y
        ### ------------------###
        ### According to Transformer:
                
        ### Query:
        ###print('x, W, t: ', x.size(),W.size(),t.size() )
        
        W_x = W[:,:,0,:]
        q = x * W_x + t
        
        q = self.layer_norm_in(q)
        
        q_vec = self.mh_q(q) ### [N_b,N_a,n_atom_basis]
        q_vec = q_vec.unsqueeze(1) ### 4D tensor: [N_b, 1, N_a, n_atom_basis ]
        q_mh_vec = torch.cat( torch.split(q_vec, n_per_head, dim=-1), dim=1) ### 4D tensor: [N_b, nh, N_a, n_per_head ] 
        q_mh_vec = q_mh_vec.unsqueeze(3) .expand(-1,-1,-1,k_size[2],-1) ### 5D tensor: [N_b, nh, N_a, N_n, n_per_head ]
                
        ### Key:
        k = y * W + t
        
        k = self.layer_norm_in(k)

        k_vec = self.mh_k(k) ### 4D Tensor: [N_b, N_a, N_n, n_atom_basis]
        k_vec = k_vec.unsqueeze(1) ### 5D Tensor: [N_b, 1, N_a, N_n, n_atom_basis]
        k_mh_vec = torch.cat( torch.split(k_vec, n_per_head, dim=-1), dim=1) ### 5D tensor: [N_b, nh, N_a, N_n, n_per_head]
        
        ### Value:
        ###v = y ### Do not contain Positional Information
        
        v = k ### As Transformer: contain Positional Information
        v_vec = self.mh_v(v) ### 4D Tensor: [N_b, N_a, N_n, n_atom_basis]
        v_vec = v_vec.unsqueeze(1) ### 5D Tensor: [N_b, 1, N_a, N_n, n_atom_basis]
        v_mh_vec = torch.cat( torch.split(v_vec, n_per_head, dim=-1), dim=1) ### 5D tensor: [N_b, nh, N_a, N_n, n_per_head]
        
        ### Dot-product Attention:
        dot_product = torch.matmul( q_mh_vec.unsqueeze(-2), k_mh_vec.unsqueeze(-1) ).squeeze(-1) ### 5D: [N_b,nh,N_a,N_n,1]
        logits = dot_product/ np.sqrt(self.n_per_head).astype(np.float32)        
        att_score = F.softmax( logits, dim=-2 ) ### [N_b, N_h, N_a, 1+N_n, 1]
        
        ### Masking:
        pairwise_mask = pairwise_mask.unsqueeze(1) ### [N_b,1, N_a,N_n] 
        att_score = att_score * pairwise_mask[..., None] ### [N_b, N_h, N_a, N_n, 1]        
        
#         sum_att_score = torch.sum(att_score, dim=-2, keepdim=True)
#         normed_score = att_score / sum_att_score ### 5D: [N_b, N_h, N_a, N_n, 1]        
#         ### Cut-off:
#         cut_att_score = normed_score * cutoff_mask ### [N_b, N_h, N_a, 1+N_n, 1]

        ####
        att_score = att_score * cutoff_mask
        sum_att_score = torch.sum(att_score, dim=-2, keepdim=True)
        cut_att_score = att_score / sum_att_score ### 5D: [N_b, N_h, N_a, N_n, 1]        
        
        
        # Perform message-aggregation:
        m_mh = cut_att_score * v_mh_vec  ### 5D: [N_b, N_h, N_a, N_n, n_per_head]  
        # Double masking, reassure:
        m_masked = m_mh * pairwise_mask[..., None] ### 5D: [N_b, N_h, N_a, N_n, n_per_head]         
        m_agg = torch.sum(m_masked, dim=-2, keepdim=False) ### 4D: [N_b, N_h, N_a, n_per_head] 
        
        y = torch.cat( torch.split(m_agg, 1, dim=1), dim=-1) ### 4D: [N_b, 1, N_a, n_atom_basis] 
        y = y.squeeze(1) ### 3D: [N_b, N_a, n_atom_basis] 
        
        # Universal Affine Transform:
        y = self.mh_o(y)                     
                     
        

        return y
