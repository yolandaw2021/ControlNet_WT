import random, json, sys
import os.path as osp
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import PIL.Image as pil
import pycocotools.mask as MaskUtil
from pycocotools.coco import COCO
from utils import read_img


class Dataset(data.Dataset):
    """Superclass for dataloaders"""

    def __init__(self,
                 json_path,
                 flip_freq=0.5,
                 is_train=False,
                 ):
        super(Dataset, self).__init__()

        self.coco = COCO(json_path)
        self.is_train = is_train
        self.flip_freq = flip_freq

        self.to_tensor = transforms.ToTensor()
        self.img_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        """
        
        do_flip = self.is_train and random.random() < self.flip_freq

        image_id = self.img_ids[index]
        img = self.coco.loadImgs(image_id)[0]
        height, width = img['height'], img['width']
        
        image = read_img(img['file_name'])
        if do_flip:
            image = image.transpose(pil.FLIP_LEFT_RIGHT)
        
        inputs = dict(image=self.to_tensor(image), index=index, target=dict())

        if self.is_train:   # only fill targets if training
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img['id'], iscrowd=None))

            if len(anns) == 0:
                masks = torch.zeros(0, height, width)
                boxes = torch.zeros(0,4).float()
                labels = torch.zeros(0).type(torch.long) 
            else:
                masks = torch.stack([torch.from_numpy(MaskUtil.decode(a['segmentation'])) for a in anns])
                boxes = torch.stack([torch.tensor(a['bbox']) for a in anns]).float()
                boxes[:,2] += boxes[:,0]
                boxes[:,3] += boxes[:,1]
                labels = torch.tensor([a['category_id'] for a in anns]).type(torch.long) 

                short_side = torch.min(torch.stack([boxes[:,2] - boxes[:,0], boxes[:,3] - boxes[:,1]]), 0).values
                valid = short_side > 0

                masks = masks[valid]
                boxes = boxes[valid]
                labels = labels[valid]

                if do_flip:
                    masks = torch.flip(masks, [-1]) # flip left/right on last dim
                    boxes[:, 0] = width - 1 - boxes[:, 0]
                    boxes[:, 2] = width - 1 - boxes[:, 2]
                    boxes = boxes[:, (2,1,0,3)]
                
            inputs['target'] = dict(masks=masks, boxes=boxes, labels=labels)

        return inputs

if __name__ == "__main__":
    dataset = Dataset(json_path='/home/ys732/share/datasets/amodal_Ithaca365/train.json', is_train=True)
    dataset.__getitem__(0)
