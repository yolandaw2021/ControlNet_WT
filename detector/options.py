import os
import argparse

class DetectorOptions:
    def __init__(self):
        self.p = argparse.ArgumentParser(description="Detector options")

        # Experiment options
        self.p.add_argument("--name", "-n",
							type=str,
							help="the name of the experiment",
                            default="example")
        self.p.add_argument("--seed",
							type=int,
							help="random seed",
                            default=0)
        self.p.add_argument("--cuda_id",
							type=int,
							help="cuda id",
                            default=0)

        self.p.add_argument("--train_json",
							type=str,
							help="the path of the json containing training instances",
                            default='/home/ys732/share/datasets/amodal_Ithaca365/train.json')
        self.p.add_argument("--val_json",
							type=str,
							help="the path of the json containing validation instances",
                            default='/home/ys732/share/datasets/amodal_Ithaca365/val.json')
        self.p.add_argument("--test_json",
							type=str,
							help="the path of the json containing testing instances",
                            default="")
        
        self.p.add_argument("--save_dir", 
							type=str,
							help="save dir",
                            default="./save/")
        
        self.p.add_argument("--data_dir", 
							type=str,
							help="data dir",
                            default="/home/ys732/share/datasets/amodal_Ithaca365")
        self.p.add_argument("--categories",
							type=list,
							help="categories",
                            default=['Bus', 'Car', 'Cyclist', 'Motorcyclist', 'Pedestrian', 'Truck'])
        self.p.add_argument("--train_txt",
							type=str,
							help="the path of the txt containing training instances",
                            default="/home/ys732/share/datasets/amodal_Ithaca365/train.txt")
        self.p.add_argument("--val_txt",
							type=str,
							help="the path of the txt containing validation instances",
                            default="/home/ys732/share/datasets/amodal_Ithaca365/val.txt")
        self.p.add_argument("--test_txt",
							type=str,
							help="the path of the txt containing testing instances",
                            default="/home/ys732/share/datasets/amodal_Ithaca365/test.txt")
		
        # Training options
        self.p.add_argument("--epochs",
							type=int,
							help="number of epochs",
                            default=20)
        self.p.add_argument("--num_workers",
							type=int,
							help="number of dataloader workers",
                            default=2)
        self.p.add_argument("--learning_rate",
							type=float,
							help="learning rate",
							default=1e-4)
        self.p.add_argument("--scheduler_step_size",
							type=int,
							help="step size of the scheduler",
							default=10)
        self.p.add_argument("--log_freq",
							type=int,
							help="number of iterations in between print lines",
                            default=1000)
        self.p.add_argument("--save_freq",
							type=int,
							help="number of epochs in between model saves",
                            default=1)
    
        # Inference options
        self.p.add_argument("--load_ckpt", "-l",
							type=str,
							help="path to model to load",
                            default=None)
    
    def parse(self, **kwargs):
        self.options = self.p.parse_args(**kwargs)
        	
        return self.options