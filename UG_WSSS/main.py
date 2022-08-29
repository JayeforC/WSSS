import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from rich.progress import Progress
from rich.live import Live 
import argparse
import numpy as np
import random
import time 
import datetime
from sklearn.metrics import auc, roc_auc_score
import json
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import accuracy, AverageMeter

from utils.update_config import get_config
from create_dataset.build_dataloader import build_loader
from build_backbone.UG_CAM import Build_UG_CAM
from utils.create_optimizer import build_optimizer
from utils.create_scheduler import build_scheduler
# from a06_utils import load_checkpoint, load_pretrained, save_checkpoint, auto_resume_helper

from utils.create_logger import create_logger
logger = create_logger("./")


def parse_option():
    parser = argparse.ArgumentParser(description="Name")

    parser.add_argument('--cfg', type=str, default="UG_WSSS/config/repeat.yaml",required=False, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')

    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    parser.add_argument("--type_dir",type=list,default=["train","val"],help="train dataset and val dataset")
    parser.add_argument("--device_ids",type=list,default=[0],help="gpu used")
    parser.add_argument("--base_lr",type=float,default=0.001,help="base learning rate")
    parser.add_argument("--weight_decay_rate",type=float,default=0.04,help="weight decay rate")

    parser.add_argument("--train_min_lr",type=float,default=5e-6,help="Cosin_Train_decay_Lr")
    parser.add_argument("--train_warmup_lr",type=float,default=5e-7,help="warm up learning rate")
    parser.add_argument("--train_epoches",type=int,default=90,help="Train Epoches")
    parser.add_argument("--train_warmup_epoches",type=int,default=20,help="Train warm up epoches")

    parser.add_argument("--num_classes",type=int,default=2,help="number of classes in the dataset")
    parser.add_argument("--embed_dim",type=int,default=512,help='input token dimension')
    parser.add_argument("--depths",type=list,default=[3,6,3],help="structure of Transformer")
    parser.add_argument("--num_heads",type=int,default=[4,8,16],help='number of heads')
    parser.add_argument("--pooling_size",type=list,default=[6,6,6],help="pooling size in Fusion blocks")
    parser.add_argument("--window_size",type=int,default=6,help='size of windows')
    parser.add_argument("--mlp_ratio",type=float,default=4.,help="Expanding ratio in MLP layer")

    parser.add_argument("--is_print",type=bool,default=True,help="print training information")
    parser.add_argument("--print_gap",type=int,default=100,help="every print_gap then print")

    parser.add_argument("--datasetCropSize",type=int,default=224,help="crop size")
    parser.add_argument("--datasetIgnoreIndex",type=int,default=255,help="ignored index")
    parser.add_argument("--CamScales",type=list,default=[1, 0.5, 1.5],help="scaled size")
    parser.add_argument("--CamRadius",type=int,default=8,help="cam radius")

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config
def main(config):

    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config,num=None,transform=None)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = Build_UG_CAM(config=config)
    # logger.info(str(model)

    model.cuda()
    model_without_ddp = model
    optimizer = build_optimizer(config,model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    criterion = torch.nn.CrossEntropyLoss()

    for data in data_loader_train:
        samples, targets = data["image"], data["cls_label"]
        samples = samples.float()
        if samples is not None:
            samples = samples.cuda()
            targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs=samples,cls_labels=targets.unsqueeze(1),n_iter=1)


if __name__ == "__main__":
    args,config = parse_option()
    print(config)
    main(config=config)
    # seed = config.SEED
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # cudnn.benchmark = True

    # config.defrost()
    # config.TRAIN.BASE_LR = config.TRAIN.BASE_LR
    # config.TRAIN.WARMUP_LR = config.TRAIN.WARMUP_LR
    # config.TRAIN.MIN_LR = config.TRAIN.MIN_LR
    # config.freeze()
    
    # os.makedirs(config.OUTPUT, exist_ok=True)
    # logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # path = os.path.join(config.OUTPUT, "config.json")
    # with open(path, "w") as f:
    #     f.write(config.dump())
    # logger.info(f"Full config saved to {path}")
    # logger.info(config.dump())
    # logger.info(json.dumps(vars(args)))
    # main(config)