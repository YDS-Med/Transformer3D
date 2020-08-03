import os
import numpy as np
import logging
from torch.optim import Adam
import schnetpack as spk
import schnetpack.atomistic.model
#from schnetpack.train import Trainer, CSVHook, ReduceLROnPlateauHook
from schnetpack.train import Trainer, CSVHook, ReduceLROnPlateauHook, EarlyStoppingHook
from schnetpack.train.metrics import MeanAbsoluteError
from schnetpack.train import build_mse_loss

from schnetpack.utils import spk_utils
spk_utils.set_random_seed(1)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


# basic settings
model_dir = "ethanol_model"  # directory that will be created for storing model
os.makedirs(model_dir)
properties = ["energy", "forces"]  # properties used for training

# data preparation
logging.info("get dataset")
###dataset = spk.datasets.MD17("data/ethanol.db", load_only=properties, molecule="ethanol")
dataset = spk.datasets.MD17("/home/jzhang/Documents/schnetpack.db/dataset/ethanol.db", load_only=properties, molecule="ethanol")
train, val, test = spk.train_test_split(
    data=dataset,
    num_train=1000,
    num_val=100,
    split_file=os.path.join(model_dir, "split.npz"),
)
train_loader = spk.AtomsLoader(train, batch_size=64)
val_loader = spk.AtomsLoader(val, batch_size=64)

# get statistics
atomrefs = dataset.get_atomref(properties)
per_atom = dict(energy=True, forces=False)
means, stddevs = train_loader.get_statistics(
    properties, single_atom_ref=atomrefs, divide_by_atoms=per_atom
)

# model build

logging.info("build model")
### ###bm1:
##representation = spk.SchNet(n_interactions=3,n_scales=1,n_filters=256,n_gaussians=32,cutoff=8.,coupled_interactions=True)
##representation = spk.SchNet(n_interactions=3,n_scales=2,n_filters=256,n_gaussians=32,cutoff=8.,coupled_interactions=True)

###debug:
##representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=1,cutoff=8.,dropout_rate=0.,apply_transition_function=True) # Note: multiplicative positional embedding as in SchNet, *** Best 1
##representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=1,cutoff=8.,dropout_rate=0.,apply_transition_function=False) # 
##representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=1,cutoff=8.,dropout_rate=0.,apply_transition_function=True) # Note: additive positional embedding
##representation = spk.TDTNet(n_heads=16,n_interactions=3,n_scales=1,cutoff=8.,dropout_rate=0.,apply_transition_function=True) # Note: additive positional embedding # *** Best 2
##representation = spk.TDTNet(n_heads=32,n_interactions=3,n_scales=1,cutoff=8.,dropout_rate=0.,apply_transition_function=True) # Note: additive positional embedding
##representation = spk.TDTNet(n_heads=8,n_interactions=6,n_scales=1,cutoff=8.,dropout_rate=0.,apply_transition_function=True) # Note: multiplicative positional embedding as in SchNet,
##representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=2,cutoff=8.,dropout_rate=0.,apply_transition_function=True) # Note: *** Best 3
##representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=2,cutoff=8.,dropout_rate=0.,apply_transition_function=True,use_msa=True) # Note: *** Best 4
##representation = spk.TDTNet(n_heads=8,n_interactions=4,n_scales=3,cutoff=8.,dropout_rate=0.,apply_transition_function=True,use_msa=True)
##representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=2,cutoff=8.,dropout_rate=0.,apply_transition_function=True,use_msa=True) # Note: No Time Embedding
#representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=2,cutoff=8.,dropout_rate=0.,apply_transition_function=True,use_msa=True) # Note: Value contains positional embedding *** Best 5
#representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=2,cutoff=8.,dropout_rate=0.,apply_transition_function=True,use_msa=True) # Note: Transition function contains time-position embedding

## SOTA:
##representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=2,cutoff=8.,dropout_rate=0.,apply_transition_function=False,use_msa=True) # Note: Transition function contains time-position embedding; *** SOTA
##representation = spk.TDTNet(n_heads=16,n_interactions=3,n_scales=2,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** Best 6
##representation = spk.TDTNet(n_heads=8,n_interactions=4,n_scales=2,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** Best 6
##representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=3,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** Best 7
##representation = spk.TDTNet(n_heads=8,n_interactions=4,n_scales=3,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** 
#representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=4,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** 
##representation = spk.TDTNet(n_heads=8,n_interactions=2,n_scales=6,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** SOTA 
##representation = spk.TDTNet(n_heads=16,n_interactions=3,n_scales=4,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** 


### Train:
###representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=3,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** SOTA
###representation = spk.TDTNet(n_heads=8,n_interactions=2,n_scales=6,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** SOTA
###representation = spk.TDTNet(n_heads=8,n_interactions=4,n_scales=3,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** SOTA
###representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=4,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** SOTA
###representation = spk.TDTNet(n_heads=8,n_interactions=2,n_scales=4,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** SOTA
###representation = spk.TDTNet(n_heads=8,n_interactions=4,n_scales=4,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** SOTA
###representation = spk.TDTNet(n_heads=8,n_interactions=12,n_scales=1,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** SOTA
###representation = spk.TDTNet(n_heads=8,n_interactions=6,n_scales=1,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** SOTA
###representation = spk.TDTNet(n_heads=8,n_interactions=2,n_scales=5,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** SOTA
representation = spk.TDTNet(n_heads=8,n_interactions=3,n_scales=5,cutoff=8.,apply_transition_function=False,use_msa=True) # Note: *** SOTA
###representation = spk.SchNet(n_interactions=3,n_scales=3,n_filters=256,n_gaussians=32,cutoff=8.,coupled_interactions=True)
###representation = spk.SchNet(n_interactions=1,n_scales=3,n_filters=256,n_gaussians=32,cutoff=8.,coupled_interactions=True)
###representation = spk.SchNet(n_interactions=1,n_scales=6,n_filters=256,n_gaussians=32,cutoff=8.,coupled_interactions=True)

###########################
output_modules = [
    spk.atomistic.Atomwise(
        n_in=representation.n_atom_basis,
        property="energy",
        derivative="forces",
        mean=means["energy"],
        stddev=stddevs["energy"],
        negative_dr=True,
    )
]
model = schnetpack.atomistic.model.AtomisticModel(representation, output_modules)

# build optimizer
optimizer = Adam(params=model.parameters(), lr=1e-4, )

# hooks
logging.info("build trainer")
metrics = [MeanAbsoluteError(p, p) for p in properties]
###hooks = [CSVHook(log_path=model_dir, metrics=metrics), ReduceLROnPlateauHook(optimizer)]
hooks = [CSVHook(log_path=model_dir, metrics=metrics) ]

# trainer
clip_norm=None

loss = build_mse_loss(properties, loss_tradeoff=[0.01, 0.99])
trainer = Trainer(
    model_dir,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
    clip_norm=clip_norm,
)

total_parms = sum(p.numel() for p in model.parameters() if p.requires_grad)
np.savetxt('./parms.txt', [total_parms], fmt='%d')

# run training
logging.info("training")
###trainer.train(device="cpu", n_epochs=1000)
#trainer.train(device="cuda", n_epochs=100)
trainer.train(device="cuda", n_epochs=300)

