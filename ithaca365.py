import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from torch.utils.data import Dataset


class Ithaca365(Dataset):
    def __init__(self, data_dir, mode="train", image_size=128):
        # original size: (600, 960)
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size

        with open(os.path.join('./splits', 'split_weather.txt'), 'r') as fh:
            self.traversals = [l.strip().split('\t')[:2] for l in fh.readlines()[1:]]
        
        with open(os.path.join('./splits', f'locations_{self.mode}.txt'), 'r') as fh:
            self.locations = [l.strip() for l in fh.readlines()]

        self.num_traversal = len(self.traversals)
        self.num_location = len(self.locations)
        self.prompt = "real camera footage of road scenes, "

                

    def __len__(self):
        return self.num_traversal * self.num_location

    def __getitem__(self, index):
        t_idx = index // self.num_location
        l_idx = index % self.num_location 
        traversal_name, weather_cond = self.traversals[t_idx] # put weather name into condition prompt
        location_name = self.locations[l_idx]

        # target image: ground truth
        img_path = os.path.join(self.data_dir, traversal_name, 'images', location_name+'.jpg')
        target = cv2.imread(img_path) # h,w,c [0, 255]
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = (target.astype(np.float32) / 127.5) - 1.0 # (h,w,c), [-1, 1]
        target = target[:, 180: 780, :] # centercrop
        target = cv2.resize(target, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        # centercrop


        # source image: segementation mask
        mask_path = os.path.join(self.data_dir, traversal_name, 'annotations', location_name+'_mask.npy')
        mask = np.load(mask_path) # h,w
        mask = mask[:, 180: 780]
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        source = np.eye(7)[mask] # h,w,4

        # prompt: text conditioning
        prompt = self.prompt + weather_cond + " weather."

        # visualization
        cmap = plt.colormaps['viridis']
        check_mask = cmap(mask/7.0)
        check_mask = (check_mask * 255).astype(np.uint8)
        Image.fromarray(check_mask).save('./scratch/check_mask.png')
        check_img = ((target+1) / 2 * 255).astype(np.uint8)
        Image.fromarray(check_img).save('./scratch/check_img.png')
    
        # target : h,w,3; source: h,w,7; prompt: str
        return dict(jpg=target, txt=prompt, hint=source)



if __name__ == '__main__':
    ds = Ithaca365('/scratch/amodal_Ithaca365/')
    print(len(ds))
    first = ds[0]
    print(first['jpg'].shape, first['txt'], first['hint'].shape)