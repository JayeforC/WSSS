'''
Author: Jaye		
Date: 2022-07-11 22:52:06
LastEditors: JayeforC 2967752698@qq.com
Description: 
'''
from logging import root
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.model_selection import train_test_split



class Dataset_load(torch.utils.data.Dataset):
    def __init__(self,root_dir='/data/ziqing/Jaye_Files/Dataset/transformer',data_type='train'):
        self.root_dir = root_dir
        self.data_type = data_type

        sample_label = []
        with open("./data_all_path.txt","r") as f:
            sample_lsts = f.readlines()
        sample_lst = [item.replace("\n","")for item in sample_lsts]
        for item in sample_lst:
            if "/0/" in item:
                sample_label.append(0)
            elif "/1/" in item:
                sample_label.append(1)
            elif "/2/" in item:
                sample_label.append(1)
                
        self.sample_lst = sample_lst
        self.label = sample_label
        
        train_data_path, test_data_path = train_test_split(sample_lst,test_size=0.25,random_state=2022)
        train_label_path, test_label_path = train_test_split(sample_label,test_size=0.25,random_state=2022)

        if data_type == "train":
            self.data_path = train_data_path
            self.label_path = train_label_path
            for item in train_data_path:
                with open("./train_path.txt",'a') as f:
                    f.write(f"train_path{item}\n")
            print(len(self.data_path))
        else:
            self.data_path = test_data_path
            self.label_path = test_label_path
            for item in test_data_path:
                with open("./test_path.txt",'a') as f:
                    f.write(f"test_path{item}\n")
            print(len(self.data_path))

        print("[Data Lodaded]")

    def __len__(self):
        return(len(self.data_path))

    def __getitem__(self,idx):

        case_path = self.data_path[idx]
        data = np.load(case_path,allow_pickle=True)
        label = self.label_path[idx]

        return np.expand_dims(data,axis=0),label

def build_loader_():
    train_ds = Dataset_load(data_type='train')
    val_ds = Dataset_load(data_type='test')

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=12,
                                               shuffle=True, num_workers=4, drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=12,
                                                 shuffle=False, num_workers=4, drop_last=False, pin_memory=True,
                                                 )
    
    return train_ds,val_ds,train_loader,val_loader


import matplotlib.pyplot as plt
if __name__ == "__main__":
    np.random.seed(1)
    train_ds,val_ds,trainloader,valloader = build_loader_()
    n = 0 

    