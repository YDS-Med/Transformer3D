### This is a Universal TDT, i.e., parameter-tying for each pass and each scale

import torch
import torch.nn as nn
import numpy as np

from schnetpack.nn.base import Dense
from schnetpack import Properties
###from schnetpack.nn.cfconv import CFConv

### Added by Justin:
from schnetpack.nn.mpnn import MPNN
from schnetpack.nn.attention import EgoAttention
from schnetpack.nn.act import AdaptiveComputationTime
from schnetpack.nn.ms_attention import MultiScaleAttention

from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.acsf import LogNormalSmearing, GaussianSmearing
from schnetpack.nn.neighbors import AtomDistances
from schnetpack.nn.activations import swish,shifted_softplus


class Positional_Embedding(nn.Module):
    ### Added by Justin/Xing
    r"""Transition block for updating atomic embeddings.

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        n_hidden (int): number of hidden units in the FFMLP. Usually larger than n_atom_basis (recommend: 4*n_atom_basis).
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.

    """

    def __init__(
        self,
        n_atom_basis,
        n_hidden,
        n_gaussians,
        trainable_gaussians=False,

        cutoff=5.0,
        cutoff_network=CosineCutoff,
        distance_expansion=None,
        activation=None,
    ):
        super(Positional_Embedding, self).__init__()

        # layer for expanding interatomic distances in a basis
        if distance_expansion is None:
            ### Added by Justin:
            self.distance_expansion = LogNormalSmearing(
                np.log(0.1), np.log(cutoff), n_gaussians, trainable=trainable_gaussians
            )
        else:
            self.distance_expansion = distance_expansion


    def forward(self, positions, z, r_ij, v_ij, neighbors ):
        """Compute interaction output.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_atom_basis) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            z (torch.Tensor): block output with (N_b, N_a) shape.

        """
        ### Position Embedding:

        ### Generate fij:
        r_ii = torch.zeros( [ r_ij.size()[0],r_ij.size()[1], 1 ] ).cuda()
        r_ii += 0.01
        r_iij = torch.cat([r_ii,r_ij], dim=-1) ### [N_b,N_a,1+N_n]

        f_iij = self.distance_expansion(r_iij) ### Distance Embedding only


        return (r_iij, f_iij)


######################################################

class Transition(nn.Module):
    ### Added by Justin/Xing
    r"""Transition block for updating atomic embeddings.

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        n_hidden (int): number of hidden units in the FFMLP. Usually larger than n_atom_basis (recommend: 4*n_atom_basis).
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.

    """

    def __init__(
        self,
        n_atom_basis,
        n_hidden,
        activation=None,
        dropout_rate=0,
    ):
        super(Transition, self).__init__()

        # filter block used in interaction block
        ###self.layer_norm_in = nn.LayerNorm([n_atom_basis]) ###(input.size()[-1])

        self.layer_norm_in = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.LayerNorm([n_atom_basis]), ###(input.size()[-1])
        )

        self.transition_network = nn.Sequential(
            Dense(n_atom_basis, n_hidden, activation=activation),
            Dense(n_hidden, n_atom_basis, activation=None),
        )


    def forward(self, x, v, t, W):
        """Compute interaction output.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_atom_basis) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_atom_basis) shape.

        """
        ### Time-Position Embedding:
#         W_x = W[:,:,0,:]
#         x = x * W_x + t

        x = x + v
        x = self.layer_norm_in(x)

        x_t = self.transition_network(x)

        x = x + x_t
        ###x = self.layer_norm_out(x)

        return x

###########################################

class TDT_Interaction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems.

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        n_spatial_basis (int): number of input features of filter-generating networks.
        ###n_filters (int): number of Value vectors in Multi_Head_Attention. Default: = n_atom_basis.
        cutoff (float): cutoff radius.
        cutoff_network (nn.Module, optional): cutoff layer.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.

    """

    def __init__(
        self,
        n_atom_basis,
        n_gaussians,
        n_heads,
        n_hidden,

        cutoff,
        cutoff_network=CosineCutoff,
        gated_attention=True,
        activation=None,

        apply_transition_function=False,
    ):
        super(TDT_Interaction, self).__init__()

        # filter block used in interaction block
        self.filter_network = Dense(n_gaussians, n_atom_basis, bias=True, activation=None)
        ### Mark by Justin: Why not non-linear?

#         self.filter_network = nn.Sequential(
#             Dense(n_spatial_basis, n_hidden, activation=swish),
#             Dense(n_hidden, n_atom_basis),
#         )

        # cutoff layer used in interaction block
        self.cutoff_network = cutoff_network(cutoff)

        # Perform Ego-Attention:
        self.attention_network = EgoAttention(n_atom_basis, n_hidden=n_hidden, n_heads=n_heads, activation=activation )

        # For Message Passing (interaction block)
        self.mpnn = MPNN(
            self.filter_network, ### Filter_Network is for positional embedding
            self.attention_network,
            cutoff_network=self.cutoff_network,
            activation=activation,
        )

        # Transition function:
        self.apply_transition_function = apply_transition_function
        if apply_transition_function:
            self.transition = Transition(n_atom_basis=n_atom_basis,n_hidden=n_hidden, activation=activation, dropout_rate=dropout_rate )


    def forward(self, e, x, t, r_ij, neighbors, neighbor_mask, f_ij=None, angle_ij=None ):
        """Compute interaction output.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_atom_basis) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_atom_basis) shape.

        """
        # continuous-filter convolution interaction block followed by Dense layer
        v,W = self.mpnn(e, x, t, r_ij, neighbors, neighbor_mask, f_ij, angle_ij )

        if self.apply_transition_function:
            x = self.transition(x, v, t, W)
        else:
            x = x + v

        return x

########################################################

########################################################

class TDTNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems.

    Args:
        n_atom_basis (int, optional): number of features to describe atomic environments.
            This determines the size of each embedding vector; i.e. embeddings_dim.
        n_filters (int, optional): number of filters used in continuous-filter convolution
        n_interactions (int, optional): number of interaction blocks.
        cutoff (float, optional): cutoff radius.
        n_gaussians (int, optional): number of Gaussian functions used to expand
            atomic distances.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        return_intermediate (bool, optional): if True, `forward` method also returns
            intermediate atomic representations after each interaction block is applied.
        max_z (int, optional): maximum nuclear charge allowed in database. This
            determines the size of the dictionary of embedding; i.e. num_embeddings.
        cutoff_network (nn.Module, optional): cutoff layer.
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.
        distance_expansion (nn.Module, optional): layer for expanding interatomic
            distances in a basis.
        charged_systems (bool, optional):

    References:
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        ### Newtork Hyper-parameters:
        n_atom_basis=128,
        n_gaussians=32, ### 25
        n_heads=8,
        n_hidden=128,

        activation=swish,
        dropout_rate=0,

        ### Model Hyper-parameters:
        n_interactions=4,
        n_scales=1,
        cutoff=5.0,

        apply_transition_function=False, ### If true, Apply Transition function as in Transformer

        use_act=True, ### Adaptive Computation Time
        use_mcr=False, ### Multiple-Channel Rep.

        return_intermediate=False,
        max_z=100,
        cutoff_network=CosineCutoff,
        trainable_gaussians=False,

        distance_expansion=None,
        charged_systems=False,

        if_cuda=True,
    ):
        super(TDTNet, self).__init__()

        self.n_atom_basis = n_atom_basis
        # make a lookup table to store embeddings for each element (up to atomic
        # number max_z) each of which is a vector of size n_atom_basis
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # layer for computing interatomic distances
        self.distances = AtomDistances(return_directions=True)


        # layer for expanding interatomic distances in a basis
        self.positional_embedding = Positional_Embedding(
            n_atom_basis=n_atom_basis,
            n_hidden=n_hidden,
            n_gaussians=n_gaussians,
            trainable_gaussians=trainable_gaussians,

            activation=activation,
            cutoff=cutoff,
            cutoff_network=cutoff_network,
            distance_expansion=None,
        )



        # block for computing interaction

        self.interaction_blocks = nn.ModuleList(
                [
                    TDT_Interaction(
                        n_atom_basis=n_atom_basis,
                        n_gaussians=n_gaussians,
                        n_heads=n_heads,
                        n_hidden=n_hidden,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        activation=activation,
                        apply_transition_function=apply_transition_function,
                    )
                    for _ in range(n_scales)
                ]
        )

        ###
        ### For Time Embedding:
        ### Note by Justin: still some bugs here
        even_mask = torch.cat( [torch.ones(n_atom_basis//2,1),torch.zeros(n_atom_basis//2,1)], dim=-1)
        even_mask = even_mask.reshape(1,n_atom_basis)
        period = torch.pow(10000, -2.* torch.arange(n_atom_basis//2)/n_atom_basis).unsqueeze(-1)
        period = torch.cat( [period, period], dim=-1 )
        period = period.reshape(1,n_atom_basis)
        tt = torch.arange(n_interactions).reshape(n_interactions,1)
        tt = tt * period ### [n_interactions,n_atom_basis]
        self.time_embedding = torch.sin( tt ) * even_mask + torch.cos( tt ) * (1.-even_mask)
        if if_cuda:
            self.time_embedding = self.time_embedding.cuda()
        self.time_embedding_list = torch.split(self.time_embedding,1,dim=0) ### n_interactions*[1,n_atom_basis]
        ###print('debug: ', self.time_embedding)

        ### ACT:
        self.use_act = use_act
        if self.use_act and n_interactions>1 :
            self.act_blocks = nn.ModuleList(
                    [
                        AdaptiveComputationTime(
                            n_atom_basis=n_atom_basis,
                            n_hidden=n_hidden,
                            activation=activation,
                            dropout_rate=dropout_rate,
                        )
                        for _ in range(n_scales)
                    ]
            )


        ### MCR: Multiple-Channel Representation.
        self.use_mcr = use_mcr
        if self.use_mcr :
            assert (n_atom_basis%n_scales==0), "n_scales should divide-out n_atom_basis!"

            self.mcr_proj_blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            Dense(n_atom_basis, n_hidden, activation=activation),
                            Dense(n_atom_basis, n_atom_basis//n_scales, activation=None),
                        )

                        for _ in range(n_scales)
                    ]
            )


        #################
        # set attributes
        self.n_scales = n_scales
        self.use_act = use_act
        self.n_interactions = n_interactions

        self.return_intermediate = return_intermediate
        self.charged_systems = charged_systems
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, n_atom_basis))
            self.charge.data.normal_(0, 1.0 / n_atom_basis ** 0.5)

    def forward(self, inputs):
        """Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.

        """
        # get tensors from input dictionary
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]

        # get atom embeddings for the input atomic numbers
        e = self.embedding(atomic_numbers)
        x = e

        if False and self.charged_systems and Properties.charge in inputs.keys():
            n_atoms = torch.sum(atom_mask, dim=1, keepdim=True)
            charge = inputs[Properties.charge] / n_atoms  # B
            charge = charge[:, None] * self.charge  # B x F
            x = x + charge

        # compute interatomic distance of every atom to its neighbors
        r_ij, v_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask,
        ) ### r_ij: [N_b,N_a,N_n]


        ### Positional embedding:
        r_iij, f_iij = self.positional_embedding( positions, atomic_numbers, r_ij, v_ij, neighbors )


        # expand neighbor_mask
        ###neighbor_mask_ii = torch.gather(neighbor_mask, -1, shape_r_ii ) * 0 + 1 ### [N_b,N_a,1+N_n]
        neighbor_mask_ii = torch.ones( [ r_ij.size()[0],r_ij.size()[1], 1 ] ).cuda()  ### [N_b,N_a,1+N_n]
        neighbor_mask_iij = torch.cat([neighbor_mask_ii,neighbor_mask], dim=-1) ### [N_b,N_a,1+N_n]

        x0 = x

        # store intermediate representations
        if self.return_intermediate:
            xs = [x0]

        ###if self.use_msa:
        x_mcr_list = []

        # compute interaction block to update atomic embeddings

        for i_scale in range(self.n_scales):
            ###x = x0 ### reset

            x_act_list = [] ### or [x]

            interaction_function = self.interaction_blocks[i_scale]
            if self.use_act :
                act_function = self.act_blocks[i_scale]

            for i_interaction in range(self.n_interactions):
                t = self.time_embedding_list[i_interaction]

                x = interaction_function(e, x, t, r_iij, neighbors, neighbor_mask_iij, f_ij=f_iij )

                x_act_list.append(x)

                if self.use_act and i_interaction>0 :
                    x = act_function(x_act_list, e)
                    x_act_list[-1] = x

            ### After each Universal Transformer block:

            if self.use_mcr:
                mcr_proj = self.mcr_proj_blocks[i_scale]
                x_proj = mcr_proj( x )
                x_mcr_list.append( x_proj )


            if self.return_intermediate:
                xs.append(x)
        ##### End of Interaction Modules

        ##### Rendering the Final Atomic Embedding:
        if self.use_mcr and self.n_scales>1 :
            x_mcr_tensor = torch.cat(x_mcr_list, dim=-1) ### (N_b,N_a,n_atom_basis)
            x = x_mcr_tensor


        ##### End of Pair-wise Learning


        if self.return_intermediate:
            return x, xs
        return x
