import os.path as osp
import torch
import torch.nn as nn
import torchvision


class Model(nn.Module):

    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        self.num_classes = num_classes  # should be 2 for class-agnostic 
        self.to_tensor = torchvision.transforms.ToTensor()

        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='COCO_V1')
        self.model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        self.model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, image, target=None):
        """ Forward pass of the model
        """

        out = self.model(image, target)

        outputs, losses = list(), dict()
        if target is not None:  # training
            losses['loss'] = sum([out[k] for k in out.keys()])
            losses.update(out)
        else:
            outputs = out
        return outputs, losses

    # ========== Helper functions ========== #

    def set_score_thrd(self, thrd):
        """ Reset score threshold for final predictions
        """
        self.model.roi_heads.score_thresh = thrd
    
    def save(self, model_path):
        """ Save model weights to disk
        """
        torch.save(self.model.state_dict(), model_path)
    
    def load(self, model_path, map_location='cpu'):
        print(f'Loading mrcnn weights from {model_path} ...')

        if not osp.exists(model_path):
            print(f'\t<<< FAILED :: Path {model_path} not found >>>')
            return

        try:
            ckpt = torch.load(model_path, map_location=map_location)
            self.model.load_state_dict(ckpt)
        except:
            print(f'\t<<< FAILED :: state_dict mismatched >>>')

    def set_train(self):
        self.model.train()
    
    def set_eval(self):
        self.model.eval()

