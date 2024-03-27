import os, cv2
import os.path as osp
from PIL import Image
import numpy as np
from einops import rearrange
import torch
import pycocotools.mask as MaskUtil

def readlines(filename):
    """ Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def write_to_file(data_list, fname, bool_newline=True):
    """ Write the given list of strings into the file
    """
    with open(fname, 'w') as fh:
        if bool_newline:
            fh.writelines([d+'\n' for d in data_list])
        else:
            fh.writelines(data_list)

def get_img_dim(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.size

def read_img(fname):
    with open(fname, 'rb') as f:
        with Image.open(f) as img:
            image = img.convert('RGB')
    return image

def osp_join(*tree):
    """ Returns os.path.join(*tree) but makes directories if missing
    """
    path = osp.join(*tree)
    dir_name = osp.dirname(path)
    if not osp.exists(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except:
            pass # catch sync racing errors
    
    return path

def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return "{:02d}h{:02d}m{:02d}s".format(t, m, s)

def _summarize(E, ap=1, cat=None, iouThr=None, areaRng='all', maxDets=100 ):
    # modified from  https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L427-L457
    p = E.params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.1f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap==1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = E.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,:,aind,mind]

        if cat is not None:
            kind = p.catIds.index(cat)
            s = s[:, :, kind:kind+1]
    else:
        # dimension of recall: [TxKxAxM]
        s = E.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,aind,mind]

        if cat is not None:
            kind = p.catIds.index(cat)
            s = s[:, kind:kind+1]

    if len(s[s>-1])==0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s>-1])
    mean_s *= 100
    print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s

def get_color_wheel(num_c):
    """ Return color wheel of shape (num_c, 3) where each color is R, G, B and in [0,1]
    """
    if num_c == 0:
        return list()
    hsv = np.ones((1, num_c, 3))
    hsv[...,0] = np.linspace(0,1,num_c, endpoint=False)
    cw = cv2.cvtColor((hsv*255).astype(np.uint8), cv2.COLOR_HSV2BGR)[0] # since cv2 store RGB in BGR order, HSV2BGR gives RGB order 
    return cw

def json2output(anns, img_dim):
    """ Convert annotations from coco json form to torchvision maskrcnn output format
    """
    if len(anns) == 0:
        return {'masks' : torch.zeros(0, *img_dim), 'boxes' : torch.zeros(0, 4)}

    out = dict()
    masks = [a['segmentation'] for a in anns]
    out['masks'] = torch.stack([torch.from_numpy(MaskUtil.decode(m)) for m in masks])
    
    boxes = torch.tensor([a['bbox'] for a in anns])
    boxes[:,2] += boxes[:,0]; boxes[:,3] += boxes[:,1]
    out['boxes'] = boxes
    
    if 'category_id' in anns[0]:
        out['labels'] = torch.tensor([a['category_id'] for a in anns])
    if 'score' in anns[0]:
        out['scores'] = torch.tensor([a['score'] for a in anns]) 
    return out

def visualize_ann(img, anns, m_conf_thrd=0.5, m_vis_thrd=0.5, m_alpha=0.3, inst_id=None, cwheel_density=250, bool_words=True, box_thickness=None):
    """ Visualize the anns onto the given img
    :param img -            input image, can be either np.ndarray (H, W, 3) or torch.Tensor (3, H, W), RGB-ordered
    :param anns             annotations to visualize
    :param m_conf_thrd -    confidence threshold used to visualize confident detections
    :param b_conf_thrd -    confidence threshold used to visualize somewhat confident detections
    :param m_vis_thrd -     threshold used to binary threshold mask prediction ranging from 0 to 1
    :param m_alpha -        alpha used when overlaying the object mask
    :param inst_id -        object instance id to ensure consistent same color for same instance
    :param np_bgr -         input np img is in bgr format
    """
    # Processing Input
    is_np = type(img) == np.ndarray 
    is_tensor = torch.is_tensor(img)
    if is_np:
        assert len(img.shape) == 3 and img.shape[2] == 3
    elif is_tensor:
        dev = img.device
        assert img.size(0) == 3
        img = (rearrange(img, 'c h w -> h w c').detach().cpu().numpy() * 255).astype(np.uint8)
    else:
        raise Exception('Input img type not recognized:', type(img))
    
    # Check inputs

    # Handling Different Inputs
    if type(anns) == list:  # a list of annotations are passed in -> need to convert from coco format to mrcnn output format
        anns = json2output(anns, img.shape[:2])
    
    bool_boxes = 'boxes' in anns
    bool_scores = 'scores' in anns
    bool_labels = 'labels' in anns

    if len(anns['masks'].shape) == 4:
        anns['masks'] = anns['masks'][:,0]

    # Resolve Dimension inconsistency
    img_height, img_width = img.shape[:2]
    mask_height, mask_width = anns['masks'].shape[-2:]
    if mask_height != img_height or mask_width != img_width:
        anns['masks'] = F.interpolate(anns['masks'][:,None], (img_height, img_width), mode='nearest')[:,0]
        if bool_boxes:
            anns['boxes'][:, 0] *= img_width / mask_width
            anns['boxes'][:, 2] *= img_width / mask_width
            anns['boxes'][:, 1] *= img_height / mask_height
            anns['boxes'][:, 3] *= img_height / mask_height
    
    
    masks = anns['masks'].detach().cpu().numpy()            # (N, H, W)
    boxes = anns['boxes'].detach().cpu().numpy() if bool_boxes else None    # (N, 4)
    scores = anns['scores'].detach().cpu().numpy() if bool_scores else None # (N)
    labels = anns['labels'].detach().cpu().numpy() if bool_labels else None # (N)
    
    N = masks.shape[0]
    num_m2v = np.where(scores > m_conf_thrd)[0].size if bool_scores else N  # num of masks to visualize

    color_collection = get_color_wheel(cwheel_density)      # (num_m2v, 3)
    np.random.seed(0) 
    np.random.shuffle(color_collection)

    if inst_id is None:
        inst_id = list(range(num_m2v))
    else:
        assert len(inst_id) == N, 'length must match'
    
    # If not specified, compute thickness given image dimensions
    box_thickness = max(int(0.004 * img.shape[0]), 1) if box_thickness is None else box_thickness
    mask_thickness = max(box_thickness // 2, 1)
    text_scale = 0.1 * box_thickness
    alpha = m_alpha**0.5
    
    def put_box(x, box, color, th):
        start_point = int(box[0]), int(box[1]); end_point = int(box[2]), int(box[3])
        return cv2.rectangle(x, start_point, end_point, color, th)  
    
    def put_text(x, label, score, left_top, font_scale, font_th=1, font=cv2.FONT_HERSHEY_SIMPLEX):
        if label is None:
            text = f'{int(score*100)}%'
        elif score is None:
            text = f'[{label}]'
        else:
            text = f'[{label}] {int(score*100)}%'
            
        left, top = int(left_top[0]), int(left_top[1])
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_th)
        text_w, text_h = text_size
        x = put_box(x, (left, top-text_h, left+text_w, top+1), (0, 0, 0), -1)
        return cv2.putText(x, text, (left, top), font, font_scale, (255, 255, 255), font_th, cv2.LINE_AA)

    # Draw out_vis
    out_vis = img.copy()
    # contour + mask
    for obj_id in list(range(0, num_m2v))[::-1]:
        mask = (masks[obj_id] >= m_vis_thrd).astype(np.uint8)
        color = color_collection[inst_id[obj_id]]
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out_vis = cv2.drawContours(out_vis, contours, -1, color.tolist(), mask_thickness)
        out_vis[mask > 0] = ((1-alpha) * out_vis[mask > 0] + alpha * color).astype(np.uint8)
    # box
    if bool_boxes: 
        for obj_id in list(range(0, num_m2v))[::-1]:
            box = boxes[obj_id]; color = color_collection[inst_id[obj_id]]
            out_vis = put_box(out_vis, box, color.tolist(), box_thickness)

    # apply alpha on contour+box
    out_vis = cv2.addWeighted(out_vis, alpha, img, 1 - alpha, 0, out_vis) 

    if bool_boxes and (bool_scores or bool_labels) and bool_words: # draw text
        for obj_id in list(range(0, num_m2v))[::-1]:
            box = boxes[obj_id]
            score = scores[obj_id] if bool_scores else None
            label = labels[obj_id] if bool_labels else None
            out_vis = put_text(out_vis, label, score, box, font_scale=text_scale)
    
    if is_tensor:
        out_vis = rearrange(torch.from_numpy(out_vis / 255.), 'h w c -> c h w').to(dev)
    
    return out_vis