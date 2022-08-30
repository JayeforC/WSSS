from unittest.mock import NonCallableMock
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .initial_cls_backbone import Wetr
from .uncertainty_guidance import get_uncertainty
from attention_cam.PAR import PAR
from attention_cam.imutils import denormalize_img2
from attention_cam.camutils import (cam_to_label, cams_to_affinity_label, ignore_img_box,
                            multi_scale_cam, multi_scale_cam_with_aff_mat,
                            propagte_aff_cam_with_bkg, refine_cams_with_bkg_v2,
                            refine_cams_with_cls_label)

import matplotlib.pyplot as plt 

class Build_UG_CAM(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.cfg = config
    
    def get_mask_by_radius(self,h=20, w=20, radius=8):
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
            
    def forward(self,inputs,cls_labels,n_iter):
        """
        ##1. Initial CAM and pseudo Generation 
        """
        wetr = Wetr()
        wetr.cuda()
        cls,_x, segs, attns, attn_pred = wetr(inputs)
        # attn_pred and aff_mat are the same thing 
        # refine 
        cams, aff_mat = multi_scale_cam_with_aff_mat(wetr, inputs=inputs, scales=self.cfg.CAM.SCALES)
        valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=cls_labels, img_box=None, ignore_mid=True, cfg=self.cfg)
        """
        ##2. Affinity part 
        """
        par = PAR(num_iter=15, dilations=[1,2,4,8,12,24])
        par.cuda()
        # inputs_denorm = denormalize_img2(inputs.clone())
        inputs_denorm = inputs.clone()
        # dataset.crop_size == 320 cfg.cam.scales==[0.5 1. 1.5] cfg.radius == 8 
        mask_size = int(self.cfg.DATA.CROP_SIZE // 16)
        infer_size = int((self.cfg.DATA.CROP_SIZE * max(self.cfg.CAM.SCALES)) // 16)
        attn_mask = self.get_mask_by_radius(h=mask_size, w=mask_size, radius=self.cfg.CAM.RADIUS)
        attn_mask_infer = self.get_mask_by_radius(h=infer_size, w=infer_size, radius=self.cfg.CAM.RADIUS)
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=cams, cls_labels=cls_labels, cfg=self.cfg, img_box=None)
        # plt.imshow(pseudo_label[1].detach().cpu())
        # plt.imshow()
        # affinity label
        aff_label = cams_to_affinity_label(refined_pseudo_label, mask=attn_mask, ignore_index=self.cfg.DATA.IGNORE_INDEX)
        # plt.imshow(aff_label[0].detach().cpu())
        # plt.imshow()

        ##affinity Loss
        # aff_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)
        """
        ##3. Uncertainty Part
        """
        uncertainty_model = get_uncertainty(input_dim=1,hidden_dim=512)
        uncertainty_model.cuda()
        # use refined psudo lable to generate uncertainty map
        uncertainty_masked, prob_x = uncertainty_model(pseudo_label) 
        ##uncertainty Loss
        # uncertainty_loss = get_uncertainty_loss(estimate_map=prob_x,pseudo_label=refined_pseudo_label)
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
        # valid_cam_resized = Transformer(across_attention, valid_cam_resized,uncertainty_masked)

        aff_cam_l = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels, bkg_score=self.cfg.CAM.LOW_THRE)
        aff_cam_l = F.interpolate(aff_cam_l, size=pseudo_label.shape[2:], mode='bilinear', align_corners=False)
        aff_cam_h = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels, bkg_score=self.cfg.CAM.HIGH_THRE)
        aff_cam_h = F.interpolate(aff_cam_h, size=pseudo_label.shape[2:], mode='bilinear', align_corners=False)

        bkg_cls = torch.ones(size=(self.cfg.TRAIN.SAMPLES_PER_GPU, 1)).cuda()
        _cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

        # final segmentation pseudo label generation
        refined_aff_cam_l = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_l, labels=_cls_labels, img_box=None)
        refined_aff_label_l = refined_aff_cam_l.argmax(dim=1)
        refined_aff_cam_h = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_h, labels=_cls_labels, img_box=None)
        refined_aff_label_h = refined_aff_cam_h.argmax(dim=1)

        aff_cam = aff_cam_l[:,1:]
        refined_aff_cam = refined_aff_cam_l[:,1:,]
        refined_aff_label = refined_aff_label_h.clone()
        refined_aff_label[refined_aff_label_h == 0] = self.cfg.DATA.IGNORE_INDEX
        refined_aff_label[(refined_aff_label_h + refined_aff_label_l) == 0] = 0
        refined_aff_label = ignore_img_box(refined_aff_label, img_box=None, ignore_index=self.cfg.DATA.IGNORE_INDEX)

        """place two: integrate uncertanity with refined aff label"""
        refined_aff_label = Transformer(across_attention, refined_aff_label,uncertainty_masked)
        # plt.imshow(refined_aff_label[0].detach().cpu())
        # plt.imshow(uncertainty_masked[0,0,:,:].detach().cpu())
        # plt.show()
        # #seg loss
        # seg_loss = self.get_seg_loss(segs, refined_aff_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)

        loss_items = {"cls_loss":(cls, cls_labels),
                      "seg_loss":(segs, refined_aff_label),
                      "aff_loss":(attn_pred, aff_label),
                      "uncertainty_loss":(prob_x,refined_pseudo_label)}
        return loss_items


if __name__ == "__main__":
    model = Build_UG_CAM(config=None)
    print(model.modules)