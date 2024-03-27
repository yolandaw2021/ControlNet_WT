import cv2
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import os

from torch.utils.data import Dataset

def get_image(img_path, image_size):
    image = cv2.imread(img_path) # h,w,c [0, 255]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image.astype(np.float32) / 127.5) - 1.0 # (h,w,c), [-1, 1]
    image = image[:, 180: 780, :] # centercrop
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return image

class Ithaca365(Dataset):
    def __init__(self, data_dir, mode="train", image_size=512):
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
        target = get_image(img_path, self.image_size)


        # source image: segementation mask
        mask_path = os.path.join(self.data_dir, traversal_name, 'annotations', location_name+'_mask.npy')
        mask = np.load(mask_path) # h,w
        mask = mask[:, 180: 780]
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        source = np.eye(7)[mask] # h,w,7

        # add a random past traversal to control image
        # rand_traversal = np.random.choice([t for t in range(self.num_traversal) if (self.traversals[t][1] == weather_cond)])
        rand_traversals = np.random.choice(range(self.num_traversal), size = 3)
        past_traversal_paths = [os.path.join(self.data_dir, self.traversals[t][0], 'images', location_name+'.jpg') for t in rand_traversals]
        control_images = [get_image(path, self.image_size) for path in past_traversal_paths] # h,w,3 * 3
        control_image = np.concatenate(control_images, axis=-1) # h,w,9

        # prompt: text conditioning
        prompt = self.prompt + weather_cond + " weather."

        # visualization
        # cmap = plt.colormaps['viridis']
        # check_mask = cmap(mask/7.0)
        # check_mask = (check_mask * 255).astype(np.uint8)
        # Image.fromarray(check_mask).save('./scratch/check_mask.png')
        # check_img = ((target+1) / 2 * 255).astype(np.uint8)
        # Image.fromarray(check_img).save('./scratch/check_img.png')
    
        # target : h,w,3; source: h,w,7; prompt: str
        return dict(jpg=target, txt=prompt, hint=np.concatenate([source, control_image], axis=-1))



if __name__ == '__main__':
    # ds = Ithaca365('/scratch/amodal_Ithaca365/')
    # print(len(ds))
    # first = ds[0]
    # print(first['jpg'].shape, first['txt'], first['hint'].shape) 
    folder_path = '/home/yw583/workspace/ControlNet_WT/image_log/3_neighbor'

    path = os.path.join(folder_path, '000000_e-000000_b-000000_neighbors.png')
    img = Image.open(path)
    img = np.array(img)
    img = img/255
    print(img.shape)
    img = img*2-0.5
    img = (img*255).astype(np.uint8)
    Image.fromarray(img).save('./scratch/check_img.png')



