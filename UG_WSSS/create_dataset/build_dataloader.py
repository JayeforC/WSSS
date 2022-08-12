from cmath import sin
import os 
import cv2
import torch
import random
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


class BaseDataSets_ACDC(Dataset):
    def __init__(self,root_dir='/home/jaye/Documents/MedicalDatasets/ACDC/processed_training/',
                    split='train',num=None,transform=None,single_model=True,organ_cls = 3):
        self.root_dir = root_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.single_model = single_model

        if single_model:
            self.organ_cls = organ_cls

        if self.split == "train":
            with open(self.root_dir + '/lists_ACDC/train.txt','r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n','') 
                                for item in self.sample_list]

        elif self.split == "val":
            with open(self.root_dir + '/lists_ACDC/test.txt','r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n','') 
                                for item in self.sample_list]

        if num is not None and self.split == 'train':
            self.sample_list = self.sample_list[:num]

        print(f"[DATA INFO] Total Samples:{len(self.sample_list)}")
        if single_model:
            print(f"[DATA INFO] Single Organ Model is on. Only Single Organ Label {organ_cls} is Returning")
        

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.sample_list == "train":
            data_path = os.path.join(self.root_dir,"train_npz", case )
            data = np.load(data_path)
        else: 
            data_path = os.path.join(self.root_dir,"train_npz", case )
            data = np.load(data_path)

        image, label = data['image'], data['label']
        if self.single_model:
            seg_label = (label == self.organ_cls).astype(np.uint8)
            cls_label = seg_label.max()
            sample = {'image':image,'cls_label':cls_label,'seg_label':seg_label}
        else:
            sample = {'image':image,'seg_label':seg_label}
        if self.transform is not None:
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class BaseDataSets_ProX2(Dataset):
    def __init__(self,root_dir='/home/jaye/Documents/MedicalDatasets/ProstateX2/processed_training/',
                split='train',num=None,transform=None,):
        self.root_dir = root_dir
        self.sample_list = []
        self,split = split
        self.transform = transform
        if self.spli == "train":
            with open(self.root_dir + 'lists_ProstateX2/train.txt','r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n','') 
                                for item in self.sample_list]

class BaseDataset_CHAOS(Dataset):
    def __init__(self,root_dir='./CHAOS/train') -> None:
        super().__init__()

def build_loader(config,num,transform):

    if config.DATASET == "ACDC":
        train_ds = BaseDataSets_ACDC(split='train',num=num,transform=transform)
        val_ds   = BaseDataSets_ACDC(split='val',num=num,transform=transform)

        train_loader = torch.utils.data.Dataloader(train_ds,batch_size=config.DATA.BATCH_SZIE,shuffle=True,
                                                    num_workers=config.DATA.NUM_WORKERS,pin_memory=True,drop_last=False)
        val_loader   = torch.utils.data.Dataloader(val_ds,batch_size=config.DATA.BATCH_SZIE,shuffle=False,
                                                    num_workers=config.DATA.NUM_WORKERS,pin_memory=True,drop_last=False)
        return train_ds, val_ds, train_loader, val_loader
    if config.DATASET == "CHAOS":
        pass
    if config.DATASET == "ProMRI":
        pass

if __name__ == "__main__":
    train_ds = BaseDataSets_ACDC()
    train_loader = DataLoader(train_ds,batch_size=1,shuffle=True)
    for i,data in enumerate(train_loader):
        if i == 0:
            image, cls_label, seg_label = data['image'], data['cls_label'],data['seg_label']

            plt.imshow(image[0] + seg_label[0],'gray')
            plt.axis('off')
            plt.show()
            print(cls_label)
        else:
            break

        