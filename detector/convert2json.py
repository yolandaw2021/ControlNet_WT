import os, sys, json
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2
import torch
from torchvision.ops import masks_to_boxes
from tqdm import tqdm
import pycocotools.mask as MaskUtil

from options import DetectorOptions
from utils import readlines, get_img_dim


def gen_json(opt, data_type):
    print(f'Generating {data_type} json from data dir {opt.data_dir}...')
    fpath = getattr(opt, f'{data_type}_txt')
    assert osp.exists(fpath), f'{fpath} must exists (run python3 get_split.py --data_dir /home/ys732/share/datasets/amodal_Ithaca365)'
    instances = readlines(fpath)

    data = {
        'info' : {"description": f"Ithaca365 - {data_type}"},
        'licenses' : [None],
        'images' : list(),
        'annotations' : list(),
        'categories' : list(),
    }

    cat2id = dict()
    for c_i, cat in enumerate(opt.categories):
        cat_id = c_i+1
        data['categories'] += [{"supercategory": 'Ithaca Objects', "id": cat_id, "name": cat}]
        cat2id[cat] = cat_id
    
    traversals = [t for t in os.listdir(opt.data_dir) if '.' not in t ]
        
    for loc in tqdm(instances):
        for traversal in traversals:
            file_name = osp.join(opt.data_dir, traversal, 'images', loc+'.png')
            width, height = get_img_dim(file_name)
            image_id = len(data['images']) + 1

            data["images"].append({"id": image_id, "file_name": file_name, "height": height, "width": width})

            annos = json.loads(readlines(osp.join(opt.data_dir, traversal, 'annotations', loc+'.txt'))[0])
            annos.sort(key=lambda x: x['occlusion'], reverse=False)
            occupancy_mask = np.zeros((height, width))
            for anno in annos:
                obj_id = len(data["annotations"]) + 1
                category_id = cat2id[anno['class']]

                mask = np.zeros((height,width))
                contours = np.array(anno['data']).astype(np.int32)
                cv2.fillPoly(mask, pts = [contours], color=1)

                inst_mask = mask - mask*occupancy_mask      # obtain instance inmodal mask only
                occupancy_mask = occupancy_mask+inst_mask
                area = np.sum(inst_mask)
                if area == 0:
                    continue

                bbox = masks_to_boxes(torch.from_numpy(inst_mask)[None])[0]
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]

                enc_mask = dict(MaskUtil.encode(np.asfortranarray(inst_mask.astype(np.uint8))))
                enc_mask['counts'] = enc_mask['counts'].decode()

                occlusion = anno['occlusion']

                a = {
                    'image_id' : image_id,
                    'id' : obj_id,
                    'segmentation' : enc_mask,
                    'bbox' : bbox.tolist(),
                    'category_id' : category_id,
                    'area' : area,
                    'occlusion' : occlusion,
                    'iscrowd' : 0,
                }
                data["annotations"].append(a)

    save_path = osp.join(opt.data_dir, f'{data_type}.json')
    with open(save_path, 'w') as fh:
        json.dump(data, fh)
    print('Saved to', save_path)

''' python3 convert2json.py
'''
if __name__ == "__main__":

    # Load Options
    options = DetectorOptions()
    opt = options.parse()

    for data_type in ['train', 'val', 'test']:
        gen_json(opt, data_type)
