import json, time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from options import DetectorOptions
from train import Trainer
from utils import _summarize


''' python3 eval.py --load_ckpt save/example/weights_00.pth --test_json /home/ys732/share/datasets/amodal_Ithaca365/val.json
'''
if __name__ == "__main__":
    options = DetectorOptions()
    opt = options.parse()
    trainer = Trainer(opt)

    trainer.model.load(opt.load_ckpt)

    print('Using test json: ' + opt.test_json)
    test_dataset, test_loader = trainer.setup_loader(json_path=opt.test_json, is_train=False)

    test_pred = COCO()
    test_pred.dataset = trainer.get_pred_dict(test_dataset)
    test_pred.createIndex()
    val_E = COCOeval(cocoGt=test_dataset.coco, cocoDt=test_pred, iouType='segm')
    val_E.evaluate()
    val_E.accumulate()

    record = dict()

    for cat in ['all'] + test_dataset.coco.dataset['categories']:

        if cat == 'all':
            cat_name = 'all'
            cat_id = None
        else:
            cat_name = cat['name'] 
            cat_id = cat['id']

        print('Category:', cat_name)
        r = dict()
        r['ap'] = _summarize(val_E, ap=1, cat=cat_id, iouThr=None, areaRng='all', maxDets=100)
        r['ap50'] = _summarize(val_E, ap=1, cat=cat_id, iouThr=0.5, areaRng='all', maxDets=100)
        r['ap_l'] = _summarize(val_E, ap=1, cat=cat_id, iouThr=None, areaRng='large', maxDets=100)
        r['ap_m'] = _summarize(val_E, ap=1, cat=cat_id, iouThr=None, areaRng='medium', maxDets=100)
        r['ap_s'] = _summarize(val_E, ap=1, cat=cat_id, iouThr=None, areaRng='small', maxDets=100)
        r['ar'] = _summarize(val_E, ap=0, cat=cat_id, iouThr=None, areaRng='all', maxDets=100)
        r['ar_l'] = _summarize(val_E, ap=0, cat=cat_id, iouThr=None, areaRng='large', maxDets=100)
        r['ar_m'] = _summarize(val_E, ap=0, cat=cat_id, iouThr=None, areaRng='medium', maxDets=100)
        r['ar_s'] = _summarize(val_E, ap=0, cat=cat_id, iouThr=None, areaRng='small', maxDets=100)

        record[cat_name] = r
        print()

    with open('eval_record.json', 'w') as fh:
        json.dump(record, fh, indent=2)
