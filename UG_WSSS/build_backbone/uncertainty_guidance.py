import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class get_uncertainty(nn.Module):
    def __init__(self,input_dim,hidden_dim,BatchNorm=nn.BatchNorm2d,dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(nn.Conv2d(input_dim,hidden_dim,kernel_size=1,bias=False),BatchNorm(self.hidden_dim),nn.ReLU(inplace=True),nn.Dropout2d(p=dropout))
        self.conv = nn.Conv2d(hidden_dim,hidden_dim,kernel_size=1)
        self.mean_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=False)
        self.std_conv  = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=False)

        ##kernel weight define
        kernel = torch.ones((7,7))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel,requires_grad=False)

    def reparameterize(self,mean,var,iter=1):
        sample_z = []
        for _ in range(iter):
            std = var.mul(0.5).exp_()  #variable 
            eps = std.data.new(std.size()).normal_()
            sample_z.append(eps.mul(std).add_(mean))
        sample_z = torch.cat(sample_z, dim=1)
        return sample_z
    
    def forward(self,x):
        """
        Input: x [B,C,H,W]
        return: Certainty map and prob_x(estimate map)
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

