## Simple Detector Training/Eval Code

To run training code:
```
python3 train.py -n <EXPERIMENT_NAME> --cuda_id <CUDA_ID> --seed <RANDOM_SEED> --train_json <JSON_FOR_TRAIN> --val_json <JSON_FOR_VAL> --epochs <NUM_EP> 
```

Example:
```
python3 train.py --train_json /home/ys732/share/datasets/amodal_Ithaca365/train.json --val_json /home/ys732/share/datasets/amodal_Ithaca365/val.json
```

To run evaluation code:
```
python3 eval.py --load_ckpt <MODEL_CKPT> --test_json <JSON_FOR_TEST>
```

Example:
```
python3 eval.py --load_ckpt save/example/weights_00.pth --test_json /home/ys732/share/datasets/amodal_Ithaca365/val.json
```