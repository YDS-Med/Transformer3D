import torch
from torch import nn
import numpy as np

from schnetpack.nn import Dense


__all__ = ["AdaptiveComputationTime"]


class AdaptiveComputationTime(nn.Module):
    r"""AdaptiveComputationTime used in 3DT module.

    Args:
        n_in (int): number of input (i.e. atomic embedding) dimensions.
        
    """

    def __init__(
        self,
        n_atom_basis,
        n_hidden,
        activation=None,
        dropout_rate=0,
        epsilon=0.01,
    ):
        super(AdaptiveComputationTime, self).__init__()
        
        ### # Regularization of Atomic Embedding:          
        self.ponder_net = nn.Sequential(
            nn.Dropout(dropout_rate),
            Dense(n_atom_basis, n_hidden, activation=activation),
            Dense(n_hidden, 1, activation=None),
        )
        
        ###self.affine_net = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)
        self.epsilon = epsilon
        self.sharpen_power = 5. ### to make the linear-interpolation sharper
        
        ###Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)
               

    def forward(self, xs, e ):
        """Compute convolution block.

        Args:
            xs List of (torch.Tensor): a list of hidden atom states; a list of (N_b,N_a,n_atom_basis)
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
        
        ###assert ( len(xs) == self.act_steps ), "Mismatch ACT Steps."
        
        x_size = xs[0].size() ### [N_b,N_a,n_atom_basis]
        
        accum_p = []   
        accum_x = 0 ### []
        x_expand_list = []
        
        act_steps = len(xs)
        p_max = 1. - self.epsilon
        
        ###for i_step in range(self.act_steps):
        for i_step in range(act_steps):
            if i_step < act_steps-1:
                x = xs[i_step] 
                x_expand_list.append( x.unsqueeze(-2) ) ### [N_b,N_a,1,n_atom_basis]
                
                ###hx = self.affine_net(x) ### [N_b,N_a,n_atom_basis]                
                ###hxe = hx + e                  
                ###hxe = x + self.affine_net( e )
                
                hxe = x + e
                
                h = torch.sigmoid( self.ponder_net(hxe) ) ### [N_b,N_a,1]
                
                if len(accum_p)>0:
                    accum_p_sum = torch.sum( torch.cat( accum_p, dim=-1 ), dim=-1, keepdim=True)
                else:
                    accum_p_sum = 0
                    
                accum_p_sum += h
                #####p = h - (accum_p_sum- 1.).clamp(min=0) ### [N_b,N_a,1]               
                res = (accum_p_sum - p_max)
                p = h - torch.relu( res ) ### [N_b,N_a,1]
                accum_p.append( p ) ### [N_b,N_a, 1 , 1]

                ###accum_x += (p / p_max) * x ### [N_b,N_a,n_atom_basis]  
                
            else:
                x = xs[i_step] ### [N_b,N_a,n_atom_basis] 
                x_expand_list.append( x.unsqueeze(-2) ) ### [N_b,N_a,1,n_atom_basis]
                
                if len(accum_p)>0:
                    accum_p_sum = torch.sum( torch.cat( accum_p, dim=-1 ), dim=-1, keepdim=True)
                else:
                    accum_p_sum = 0

                res = (p_max - accum_p_sum)
                p = torch.relu( res ) ### [N_b,N_a,1]   
                accum_p.append( p ) ### [N_b,N_a, 1]
                              
                ###accum_x += (p / p_max) * x ### [N_b,N_a,n_atom_basis] 
                
        p_tensor = torch.cat( accum_p, dim=-1).unsqueeze(-1) ### [N_b,N_a, ns ,1]
        sharpen_p = torch.pow( p_tensor, self.sharpen_power ) ### [N_b,N_a, ns ,1]
        sharpen_p = sharpen_p / torch.sum( sharpen_p, dim=-2, keepdim=True) ### [N_b,N_a, ns ,1]
        
        x_expand = torch.cat( x_expand_list, dim=-2 ) ### [N_b,N_a, ns ,n_atom_basis]
        x_out = torch.sum( x_expand * sharpen_p, dim=-2, keepdim=False ) ### [N_b,N_a,n_atom_basis]
        
#         accum_p_final = torch.sum( torch.cat( accum_p, dim=-1 ), dim=-1, keepdim=True)
#         p_history = torch.cat( tuple(accum_p), dim=-1 )
#         print('debug act: ', p_history.detach()[0,0,:] )
                
        ###x_out = accum_x
        
        return x_out
