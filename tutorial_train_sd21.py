from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ithaca365 import Ithaca365
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# make new config script
# python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_ini.ckpt

# Configs
resume_path = './models/control_sd21_3neighbor_instance.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = Ithaca365('/scratch/amodal_Ithaca365/')
dataloader = DataLoader(dataset, num_workers=1, batch_size=batch_size, shuffle=True)
# TODO: use wandb instead of image logger

logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)