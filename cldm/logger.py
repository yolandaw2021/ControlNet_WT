import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
import matplotlib.pyplot as plt


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.cmap = plt.colormaps['viridis']

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        result = []
        filename = "{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        for k in images:
            if k == "conditioning":
                continue
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1) # h,w,c
            if k == "control":
                neighbors = grid[:, :, 8:]
                neighbors = (neighbors + 1.0) / 2.0
                grid = grid[:, :, 0:8]
                grid = torch.argmax(grid, dim=2)
                grid = self.cmap(grid.numpy()/8.0)
                grid = grid[:, :, :3] # get rid of alpha channel

                # separate neighbors in to b,h,w,3 images, and concatenate them into an image (n*h, b*w, 3), store it somewhere
                n = int(neighbors.shape[2]/3)
                neighbor_images = neighbors.split(3, dim=2) # n*(h,w,3)
                neighbor_images = torch.concat(neighbor_images, dim=0) # (n*h, b*w, 3)
                neighbor_images = neighbor_images.numpy()
                neighbor_images = (neighbor_images * 255).astype(np.uint8)
                path = os.path.join(root, filename.replace(".png", "_neighbors.png"))
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(neighbor_images).save(path)
            else:
                grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            result.append(grid)
        result = np.concatenate(result, axis=0)
        Image.fromarray(result).save(path)
        

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            # TODO: change the folder name here
            self.log_img(pl_module, batch, batch_idx, split="3_neighbor_bordered")
