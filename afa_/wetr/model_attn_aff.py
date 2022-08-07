import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
from . import mix_transformer
import numpy as np



class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None,):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/'+backbone+'.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict,)

        if pooling=="gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling=="gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        #self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)

        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes-1, kernel_size=1, bias=False)


    def get_param_groups(self):

        param_groups = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;
        
        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups


    def forward(self, x, cam_only=False, seg_detach=True,):

        """
        Args:
            if cam is only: after training to generate cam
            return : [cam_x4, attn_pred]
                cam_x4 : feature map from the last layer multiplied by the classifier weights to generate CAM
                        shape: [B,num_classes,14,14]
                attn_pred: fusing the last two att maps along the num_head dimension 
                        shape:[B,N,N] -> [B,196,196]
            
            if cam is not only: training part
            return : [cls_x4, seg, _attns, attn_pred]
            cls_x4 : final classification scores [B,num_classes]
            seg    : fusing all feature maps from each blocks, different from the cam_x4
                     shape [B,num_classes,14,14]
            _attns : list that contains all attns from each blocks [12,B,num_heads,N,N]
            attn_pred: affinity prediction 
        """

        _x, _attns = self.encoder(x) ## all patch tokens and attn maps in 4 block outputs
        _x1, _x2, _x3, _x4 = _x

        seg = self.decoder(_x) # fuse all _xi to one x, fuse all features from different stages [B,num_classes,H,W]
        #seg = self.decoder(_x4)

        attn_cat = torch.cat(_attns[-2:], dim=1)#.detach() # abstract attn maps from the last two layers 
        # and concatenate along dim = 1, [B,2*num_heads,N,N]
        attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2) # A = S + S.T
        attn_pred = self.attn_proj(attn_cat) # MLP [B,1,N,N]
        attn_pred = torch.sigmoid(attn_pred)[:,0,...]

        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach() # feature maps multiplied by classifier weights
            return cam_s4, attn_pred

        #_x4 = self.dropout(_x4.clone()
        cls_x4 = self.pooling(_x4,(1,1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes-1) ###[1,num_classes] class scores
 
        #attns = [attn[:,0,...] for attn in _attns]
        #attns.append(attn_pred)
        return cls_x4, seg, _attns, attn_pred
    

if __name__=="__main__":

    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    wetr = WeTr('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    wetr._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    wetr(dummy_input)