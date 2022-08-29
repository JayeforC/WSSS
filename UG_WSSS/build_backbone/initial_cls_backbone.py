import torch 
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
from attention_cam.mix_transformer import MixVisionTransformer
from attention_cam.segformer_head import SegFormerHead

import matplotlib.pyplot as plt

class Wetr(nn.Module):
    def __init__(self,num_classes=2, embedding_dim=256, stride=None, pretrained=None, pooling=None,):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_stride = [4,8,16,32]
        self.stride = stride 

        self.encoder = MixVisionTransformer(stride=[4, 2, 2, 1])
        self.decoder = SegFormerHead(in_channels=[64, 128, 256, 512],feature_strides = [4, 8, 16, 32])
        self.classifier = classifier = nn.Conv2d(self.encoder.embed_dims[3],num_classes,kernel_size=1,bias=False)
        
        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")
    
    def forward(self,x,cam_only=False):
        _x, _attns = self.encoder(x)
        _x1,_x2,_x3,_x4 = _x
        seg = self.decoder(_x)

        attn_cat = torch.cat(_attns[-2:],dim=1) ## [1,16,196,196]
        attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)

        attn_pred = self.attn_proj(attn_cat) # MLP [B,1,N,N]
        attn_pred = torch.sigmoid(attn_pred)[:,0,...] #[1,N,N] 类似于squeeze
        
        
        if cam_only:
            cam_s4 = F.conv2d(_x4,self.classifier.weight,)
            # vis = cam_s4[0,0,:,:].detach().cpu()
            # plt.imshow(vis)
            # plt.show()

            return cam_s4, attn_pred
        
        pooling = F.adaptive_avg_pool2d
        cls_x4 = pooling(_x4,(1,1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1,self.num_classes)
        return cls_x4, _x, seg, _attns, attn_pred

def get_classification_loss(predict_cls,cls_label):
    cls_loss = F.multilabel_margin_loss(predict_cls,cls_label)
    return cls_label