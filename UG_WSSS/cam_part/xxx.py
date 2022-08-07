import math
import torch
import numpy as np
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch import Tensor
from matplotlib import cm
from torchvision.transforms.functional import to_pil_image

img_path = "/home/jaye/Documents/DeepLearning/08_ConvTrans/cat.jpg"     # 输入图片的路径
save_path = './cat_cam.jpg'    # 类激活图保存路径
preprocess = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


net = models.resnet18(pretrained=True)  # 导入模型
feature_map = []     # 建立列表容器，用于盛放输出特征图

def forward_hook(module, inp, outp):     # 定义hook
    feature_map.append(outp)    # 把输出装入字典feature_map

net.layer4.register_forward_hook(forward_hook)    # 对net.layer4这一层注册前向传播
orign_img = Image.open(img_path).convert('RGB')    # 打开图片并转换为RGB模型
img = preprocess(orign_img)     # 图片预处理
img = torch.unsqueeze(img, 0)     # 增加batch维度 [1, 3, 224, 224]

with torch.no_grad():
    out = net(img)     # 前向传播
    print(feature_map[0].size())

cls = torch.argmax(out).item()    # 获取预测类别编码
weights = net._modules.get('fc').weight.data[cls,:]    # 获取类别对应的权重
print(net._modules.get('fc').weight.data.size())
print(*weights.shape)
print(weights.view(*weights.shape, 1, 1).size())
cam = (weights.view(*weights.shape, 1, 1) * feature_map[0].squeeze(0)).sum(0)
def _normalize(cams: Tensor) -> Tensor:
        """CAM normalization"""
        cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))

        return cams
def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
    """Overlay a colormapped mask on a background image

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)  # control the colour style   
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img

cam = _normalize(F.relu(cam, inplace=True)).cpu()
mask = to_pil_image(cam.detach().numpy(), mode='F')
result = overlay_mask(orign_img, mask) 
result.show()
result.save(save_path)