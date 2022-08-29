from UGTR.model.transformer import Transformer
import torch
from afa_.utils.evaluate import pseudo_scores
from attention_cam.mix_transformer import MixVisionTransformer
from attention_cam.segformer_head import SegFormerHead
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from utils.losses import DenseEnergyLoss, get_aff_loss, get_energy_loss
from attention_cam.camutils import (cam_to_label, cams_to_affinity_label, ignore_img_box,
                            multi_scale_cam, multi_scale_cam_with_aff_mat,
                            propagte_aff_cam_with_bkg, refine_cams_with_bkg_v2,
                            refine_cams_with_cls_label)


class Wetr(nn.Module):
    def __init__(self,num_classes=20, embedding_dim=256, stride=None, pretrained=None, pooling=None,):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_stride = [4,8,16,32]
        self.stride = stride 

        self.encoder = MixVisionTransformer(stride=[4, 2, 2, 1])
        self.decoder = SegFormerHead(in_channels=[64, 128, 256, 512],feature_strides = [4, 8, 16, 32])
        self.classifier = classifier = nn.Conv2d(self.encoder.embed_dims[3],20,kernel_size=1,bias=False)
        
    
    def forward(self,x,cam_only=False):
        _x, _attns = self.encoder(x)
        _x1,_x2,_x3,_x4 = _x
        seg = self.decoder(_x)

        attn_cat = torch.cat(_attns[-2:],dim=1) ## [1,16,196,196]
        attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
        attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(attn_proj.weight, a=np.sqrt(5), mode="fan_out")
        attn_pred = attn_proj(attn_cat) # MLP [B,1,N,N]
        attn_pred = torch.sigmoid(attn_pred)[:,0,...] #[1,N,N] 类似于squeeze
        
        
        if cam_only:
            cam_s4 = F.conv2d(_x4,self.classifier.weight,)
            return cam_s4, attn_pred
        

        pooling = F.adaptive_avg_pool2d
        cls_x4 = pooling(_x4,(1,1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1,self.num_classes)
        return cls_x4, _x, seg, _attns, attn_pred

def cam_to_label(cam, cls_label, img_box=None, ignore_mid=False, cfg=None):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value<=0.9] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value<=0.55] = 255
        _pseudo_label[cam_value<=0.35] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * 255

    # for idx, coord in enumerate(img_box):
    #     pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return valid_cam, pseudo_label

def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w 
    #_hw = (h + max(dilations)) * (w + max(dilations)) 
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask

def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5

class get_uncertainty(nn.Module):
    def __init__(self,input_dim,hidden_dim,BatchNorm=nn.BatchNorm2d,dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(nn.Conv2d(input_dim,hidden_dim,kernel_size=1,bias=False),BatchNorm(self.hidden_dim),nn.ReLU(inplace=True),nn.Dropout2d(p=dropout))
        self.conv = nn.Conv2d(hidden_dim,hidden_dim,kernel_size=1)
        self.mean_conv = nn.Conv2d(self.input_dim, 1, kernel_size=1, bias=False)
        self.std_conv  = nn.Conv2d(self.input_dim, 1, kernel_size=1, bias=False)

        ##kernel weight define
        kernel = torch.ones((7,7))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel,requires_grad=False)

    def reparameterize(self,mean,var,iter=1):
        sample_z = []
        for _ in range(iter):
            std = var.mul(0.5).exp_()  # type: Variable
            eps = std.data.new(std.size()).normal_()
            sample_z.append(eps.mul(std).add_(mean))
        sample_z = torch.cat(sample_z, dim=1)
        return sample_z
    
    def forward(self,x):
        """
        Input: x [B,C,H,W]
        """
        x = self.input_proj(x)
        residual = self.conv(x)
        mean = self.mean_conv(x)
        std  = self.std_conv(x)
        prob_x = self.reparameterize(mean,std,1)
        prob_out = self.reparameterize(mean,std,50)
        prob_out = torch.sigmoid(prob_out)

        #uncertainty
        uncertainty_map = prob_out.var(dim=1,keepdim=True).detach()
        if self.training:
            uncertainty_map=F.conv2d(uncertainty_map,self.weight,padding=3,groups=1)
            uncertainty_map=F.conv2d(uncertainty_map,self.weight,padding=3,groups=1)
            uncertainty_map=F.conv2d(uncertainty_map,self.weight,padding=3,groups=1)
        uncertainty_map = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min())
        residual *= (1-uncertainty_map)

        #random mask 
        if self.training:
            rand_mask = uncertainty_map < torch.Tensor(np.random.random(uncertainty_map.size())).to(uncertainty_map.device)
            residual *= rand_mask.to(torch.float32)
      
        return residual,prob_x

"""
##0. Import Data
"""
img_path = "/home/jaye/Documents/DeepLearning/08_ConvTrans/cat.jpg"     # 输入图片的路径
save_path = './cat_cam.jpg' 
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.2,0.2,0.2])])

img = Image.open(img_path).convert("RGB")
plt.imshow(img)
plt.show
img = transform(img)
inputs = img.unsqueeze(0)

"""
##1. Initial CAM and pseudo Generation 
"""

wetr = Wetr()
inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
cls,_x, segs, attns, attn_pred = wetr(inputs)
# attn_pred and aff_mat are the same thing 
# refine 
cams, aff_mat = multi_scale_cam_with_aff_mat(wetr, inputs=inputs, scales=cfg.cam.scales)
valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True, cfg=cfg)

"""
##2. Affinity part 
"""
refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=cams, cls_labels=cls_labels, cfg=cfg, img_box=img_box)
# affinity label
aff_label = cams_to_affinity_label(refined_pseudo_label, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
##affinity Loss
aff_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

"""
##3. Uncertainty Part
"""
uncertainty_model = get_uncertainty()
# use refined psudo lable to generate uncertainty map
uncertainty_masked, prob_x = uncertainty_model(pseudo_label)
#calculate the uncertainty map part loss
BCE_criterior = nn.CrossEntropyLoss(ignore_index=255)
KL_divergence = nn.KLDivLoss(size_average=False,reduce=False)
uncertainty_loss = BCE_criterior(prob_x,refined_pseudo_label) 

"""
##4. final segmentation pseudo label part 

In this part, the affinity map, pseudo label and uncertainty map should be
integrated as a whole to generate the final segmentation pseudo label

Consider two different place to integerate the uncertainty map 
"""

# across attention part to integrate 
# random walk part
"""place one: integrate uncertanity with valid_cam_resized"""
valid_cam_resized = F.interpolate(valid_cam, size=(infer_size, infer_size), mode='bilinear', align_corners=False)
"""
integrate uncertanity with valid_cam_resized by across attention mechanism 
"""
valid_cam_resized = Transformer(across_attention, valid_cam_resized,uncertainty_masked)

aff_cam_l = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels, bkg_score=cfg.cam.low_thre)
aff_cam_l = F.interpolate(aff_cam_l, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
aff_cam_h = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels, bkg_score=cfg.cam.high_thre)
aff_cam_h = F.interpolate(aff_cam_h, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

bkg_cls = bkg_cls.to(cams.device)
_cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

# final segmentation pseudo label generation
refined_aff_cam_l = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_l, labels=_cls_labels, img_box=img_box)
refined_aff_label_l = refined_aff_cam_l.argmax(dim=1)
refined_aff_cam_h = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_h, labels=_cls_labels, img_box=img_box)
refined_aff_label_h = refined_aff_cam_h.argmax(dim=1)

aff_cam = aff_cam_l[:,1:]
refined_aff_cam = refined_aff_cam_l[:,1:,]
refined_aff_label = refined_aff_label_h.clone()
refined_aff_label[refined_aff_label_h == 0] = cfg.dataset.ignore_index
refined_aff_label[(refined_aff_label_h + refined_aff_label_l) == 0] = 0
refined_aff_label = ignore_img_box(refined_aff_label, img_box=img_box, ignore_index=cfg.dataset.ignore_index)

"""place two: integrate uncertanity with refined aff label"""
refined_aff_label = Transformer(across_attention, refined_aff_label,uncertainty_masked)
#seg loss
seg_loss = get_seg_loss(segs, refined_aff_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)

"""
##5. Classification loss
"""
cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)

"""###6. Training Strategy"""

loss = 1.0 * cls_loss + 0.1 * seg_loss + 0.1 * aff_loss + 0.1 * uncertainty_loss


features_x4 = _x[3]

_cam, _aff_mat = wetr(inputs_cat, cam_only=True)
print(_aff_mat.size())
cams, aff_mat = multi_scale_cam_with_aff_mat(wetr, inputs=inputs, scales=[1.,0.5,1.5])
valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=torch.tensor([[2]]), img_box=True, ignore_mid=True, cfg=None)

mask_size = int(320 // 16)
infer_size = int((320 * 1.5) // 16)
attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=8)
attn_mask_infer = get_mask_by_radius(h=infer_size, w=infer_size, radius=8)
valid_cam_resized = F.interpolate(valid_cam, size=(infer_size, infer_size), mode='bilinear', align_corners=False)

aff_cam_l = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=torch.tensor([2]), bkg_score=0.35)
aff_cam_l = F.interpolate(aff_cam_l, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
aff_cam_h = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=torch.tensor([2]), bkg_score=0.55)
aff_cam_h = F.interpolate(aff_cam_h, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
print(aff_cam_h.size())
