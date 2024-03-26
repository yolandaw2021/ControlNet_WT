import sys, json, os, time, random
import os.path as osp

from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pycocotools.mask as MaskUtil
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import imageio.v3 as iio

from utils import sec_to_hm_str, osp_join, read_img, _summarize, visualize_ann
from model import Model
from dataset import Dataset
from options import DetectorOptions

class Trainer:
    def __init__(self, options):
        self.opt = options
        # Set seed and run assertions
        random.seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)


        print('\n=============== Trainer Initialization ===============')

        self.device = torch.device("cuda:{}".format(self.opt.cuda_id) if torch.cuda.is_available() else "cpu")
        self.model = Model(num_classes=len(self.opt.categories)+1).to(self.device) 
        self.optim = optim.Adam(self.model.parameters(), self.opt.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optim, self.opt.scheduler_step_size, 0.5)

        self.log_path = os.path.join(self.opt.save_dir, self.opt.name)

        self.metrics = list()

        # set up loaders
        print('Using train json: ' + self.opt.train_json)
        self.train_dataset, self.train_loader = self.setup_loader(json_path=self.opt.train_json, is_train=True)
        print('Using val json: ' + self.opt.val_json)
        self.val_dataset, self.val_loader = self.setup_loader(json_path=self.opt.val_json, is_train=False)
        
        self.num_steps_per_epoch = len(self.train_loader)
        self.num_total_steps = self.num_steps_per_epoch * self.opt.epochs
        self.save_opts()
        print(f'Number of training items: {len(self.train_dataset)}')
        print('=============== Trainer Initialization ===============\n')

    # ========== Standard functions ========== #

    def train(self):
        """ Run the entire training pipeline 
        """

        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        for self.epoch in range(self.opt.epochs):
            print()
            self.run_epoch()

            if ((self.epoch + 1) % self.opt.save_freq == 0) or (self.epoch == self.opt.epochs - 1):
                self.save_model()

    def run_epoch(self):
        """ Run a single epoch of training and validation 
        """

        self.set_train()

        gpu_time, data_loading_time = 0, 0
        before_op_time = time.time()
        
        epoch_info = dict(epoch_idx=self.epoch, train_loss=0)
        for batch_idx, inputs in enumerate(self.train_loader):
            
            data_loading_time += (time.time() - before_op_time)
            before_op_time = time.time()

            # === Compute Starts === #
            outputs, losses = self.process_batch(inputs)
            losses['loss'].backward()

            self.optim.step()
            self.optim.zero_grad()
            # === Compute Ends === #

            compute_duration = time.time() - before_op_time
            gpu_time += compute_duration

            early_freq = self.opt.log_freq // 10; late_freq = self.opt.log_freq
            if (batch_idx % early_freq == 0 and self.step < late_freq) or self.step % late_freq == 0:
                self.log_time(batch_idx, compute_duration, losses['loss'].cpu().data, data_loading_time, gpu_time)
                gpu_time, data_loading_time = 0, 0
            del outputs

            epoch_info['train_loss'] = (batch_idx/(batch_idx+1)) * epoch_info['train_loss'] + (1/(batch_idx+1)) * losses['loss'].item()

            self.step += 1
            before_op_time = time.time()
        self.lr_scheduler.step()
        self.eval(epoch_info)
        

    def eval(self, epoch_info):
        
        pred_dict = self.get_pred_dict(self.val_dataset)

        val_pred = COCO()
        val_pred.dataset = pred_dict
        val_pred.createIndex()
        val_E = COCOeval(cocoGt=self.val_dataset.coco, cocoDt=val_pred, iouType='segm')
        val_E.evaluate()
        val_E.accumulate()

        epoch_info['ap'] = _summarize(val_E, ap=1, cat=None, iouThr=None, areaRng='all', maxDets=100)
        epoch_info['ap50'] = _summarize(val_E, ap=1, cat=None, iouThr=0.5, areaRng='all', maxDets=100)
        epoch_info['ap_l'] = _summarize(val_E, ap=1, cat=None, iouThr=None, areaRng='large', maxDets=100)
        epoch_info['ap_m'] = _summarize(val_E, ap=1, cat=None, iouThr=None, areaRng='medium', maxDets=100)
        epoch_info['ap_s'] = _summarize(val_E, ap=1, cat=None, iouThr=None, areaRng='small', maxDets=100)
        epoch_info['ar'] = _summarize(val_E, ap=0, cat=None, iouThr=None, areaRng='all', maxDets=100)
        epoch_info['ar_l'] = _summarize(val_E, ap=0, cat=None, iouThr=None, areaRng='large', maxDets=100)
        epoch_info['ar_m'] = _summarize(val_E, ap=0, cat=None, iouThr=None, areaRng='medium', maxDets=100)
        epoch_info['ar_s'] = _summarize(val_E, ap=0, cat=None, iouThr=None, areaRng='small', maxDets=100)

        imgIds = val_pred.getImgIds()
        for i in range(0, len(imgIds), 100):
            img = val_pred.loadImgs(imgIds[i])[0]
            I = iio.imread(img['file_name'])

            gt_anns = self.val_dataset.coco.loadAnns(self.val_dataset.coco.getAnnIds(imgIds=img['id'], iscrowd=None))
            pred_anns = val_pred.loadAnns(val_pred.getAnnIds(imgIds=img['id'], iscrowd=None))
            gt_vis = visualize_ann(I, gt_anns)
            pred_vis = visualize_ann(I, pred_anns)

            out_vis = np.hstack([pred_vis, gt_vis])
            iio.imwrite(osp_join(self.log_path, 'vis', f'ImgId{i}_Ep{self.epoch:02}.jpg'), out_vis)
        
        self.metrics.append(epoch_info)
        with open(osp_join(self.log_path, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f)

    def process_batch(self, inputs):
        """ Pass a minibatch through the network and generate images and losses 
        """
        image = [inputs['image'][0].to(self.device)]

        if self.is_train:
            masks = inputs['target']['masks'].to(self.device)       # (B, N, H, W)
            boxes = inputs['target']['boxes'].to(self.device)       # (B, N, 4)
            labels = inputs['target']['labels'].to(self.device)     # (B, N)
            target = [{'masks' : masks[0], 'boxes' : boxes[0], 'labels' : labels[0]}]
        else:
            target = None

        return self.model(image, target)
    
    def get_pred_dict(self, dataset):
        self.set_eval()
        self.model.set_score_thrd(0)

        data = {"images": dataset.coco.dataset['images'],
                "annotations": list(),
                "categories": dataset.coco.dataset['categories']}

        for img in tqdm(data['images']):
            image_id = img['id']
            image = read_img(img['file_name'])

            with torch.no_grad():
                out, _ = self.model([dataset.to_tensor(image).to(self.device)], None)
            masks = (out[0]['masks'][:,0].cpu().numpy() > 0.5).astype(np.uint8)  # (N, H, W)
            mask_enc = [dict(MaskUtil.encode(np.asfortranarray(m))) for m in masks] 
            boxes = out[0]['boxes'].cpu().numpy().astype(float)         # (N, 4)
            boxes[:,2] = boxes[:,2] - boxes[:,0]
            boxes[:,3] = boxes[:,3] - boxes[:,1]
            scores = out[0]['scores'].cpu().numpy().astype(float)       # (N)
            labels = out[0]['labels'].cpu().numpy()                     # (N)
            mask_area = masks.sum((1,2))

            num_anno = len(data["annotations"])
            anno = [
            {   'image_id' : image_id,
                'id' : num_anno + oi,
                'segmentation' : {'size': m['size'], 'counts': m['counts'].decode()},
                'bbox' : b,
                'score' : s,
                'category_id': int(l), 
                'area' : int(a),
            } for oi, (m, b, s, l, a) in enumerate(zip(mask_enc, boxes.tolist(), scores, labels, mask_area))]
            data["annotations"] += anno
        
        self.set_train()
        self.model.set_score_thrd(0.05)
        
        return data

    # ========== Helper functions ========== #

    def setup_loader(self, **kwargs):
        """ construct self.train_loader
        """
        dataset = Dataset(**kwargs)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True, drop_last=False, sampler=None)
        return dataset, loader

    def log_time(self, batch_idx, duration, loss, data_time, gpu_time):
        """ Print a logging statement to the terminal
        """

        samples_per_sec = 1 / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = 'epoch {:>3} | batch {:>6} | examples/s: {:5.1f}' + \
            ' | loss: {:.5f} | time elapsed: {} | time left: {} | CPU/GPU time: {:0.1f}s/{:0.1f}s'
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left),
                                  data_time, gpu_time))
    
    def save_opts(self,):
        """ Save options to disk so we know what we ran this experiment with 
        """
        with open(osp_join(self.log_path, 'opt.json'), 'w') as f:
            json.dump(self.opt.__dict__.copy(), f, indent=2)

    def save_model(self, save_name='weights'):
        """ Save model weights and opt to disk
        """
        model_path = osp_join(self.log_path, f'{save_name}_{self.epoch:02}.pth')
        self.model.save(model_path)

        best_ap = np.max([m['ap'] for m in self.metrics])
        if self.metrics[-1]['ap'] == best_ap:
            model_path = osp_join(self.log_path, f'{save_name}_best_{best_ap:.1f}.pth')
            self.model.save(model_path)
    
    def set_train(self):
        """ Convert all models to training mode
        """
        self.model.set_train()
        self.is_train = True

    def set_eval(self):
        """ Convert all models to testing/evaluation mode 
        """
        self.model.set_eval()
        self.is_train = False


if __name__ == "__main__":
    options = DetectorOptions()
    opt = options.parse()
    trainer = Trainer(opt)
    trainer.train()

