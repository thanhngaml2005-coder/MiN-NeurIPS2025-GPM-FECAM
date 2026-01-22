import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
from torch.amp import autocast

class BaseIncNet(nn.Module):
    def __init__(self, args: dict):
        super(BaseIncNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.feature_dim = self.backbone.out_dim
        self.fc = None

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    @staticmethod
    def generate_fc(in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        hyper_features = self.backbone(x)
        logits = self.fc(hyper_features)['logits']
        return {
            'features': hyper_features,
            'logits': logits
        }


class RandomBuffer(torch.nn.Linear):
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        
        # Dùng float32 cho chính xác khi tính RLS
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)
        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight.dtype)
        return F.relu(X @ self.W)


class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        # Random Buffer
        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        # Analytic Learning Params
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) 

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R) 

        # Normal FC (Dùng để train Noise bằng SGD)
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    def update_fc(self, nb_classes):
        """Cập nhật Normal FC cho training"""
        self.cur_task += 1
        self.known_class += nb_classes
        
        if self.cur_task > 0:
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            
        if self.normal_fc is not None:
            old_nb_output = self.normal_fc.out_features
            with torch.no_grad():
                new_fc.weight[:old_nb_output] = self.normal_fc.weight.data
                if new_fc.bias is not None and self.normal_fc.bias is not None:
                     new_fc.bias[:old_nb_output] = self.normal_fc.bias.data
            
            del self.normal_fc
            self.normal_fc = new_fc
        else:
            # Init zero cho task đầu
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None:
                nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    # =========================================================================
    # [BRIDGE TO VIT_MIN & BILORA]
    # =========================================================================
    
    def update_noise(self):
        """Kích hoạt Task mới cho BiLORA"""
        for i in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[i].update_noise()

    def after_task_magmax_merge(self):
        """Kích hoạt Merge cho BiLORA"""
        print(f"--> [IncNet] Task {self.cur_task}: Triggering Parameter-wise MagMax Merging...")
        for i in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[i].after_task_training()

    def unfreeze_noise(self):
        """Chỉ mở khóa BiLORA"""
        for i in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[i].unfreeze_noise()

    def init_unfreeze(self):
        """Mở khóa Task 0 (BiLORA + Norm)"""
        for i in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[i].unfreeze_noise()
            # Unfreeze Norm layers để ổn định hơn với SGD
            for p in self.backbone.blocks[i].norm1.parameters(): p.requires_grad = True
            for p in self.backbone.blocks[i].norm2.parameters(): p.requires_grad = True
                
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

    # =========================================================================
    # [ANALYTIC & FORWARD]
    # =========================================================================

    def forward_fc(self, features):
        features = features.to(self.weight) 
        return features @ self.weight

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """RLS Update"""
        with autocast('cuda', enabled=False):
            X = self.backbone(X).float() 
            X = self.buffer(X)            
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.weight.shape[1]:
                increment_size = self.weight.shape[1] - num_targets
                tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
                Y = torch.cat((Y, tail), dim=1)

            I = torch.eye(X.shape[0]).to(X)
            term = I + X @ self.R @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            K = torch.inverse(term + jitter)
            
            self.R -= self.R @ X.T @ K @ X @ self.R
            self.weight += self.R @ X.T @ (Y - X @ self.weight)

    def forward(self, x, new_forward: bool = False):
        """Inference (Analytic)"""
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {'logits': logits}

    def forward_normal_fc(self, x, new_forward: bool = False):
        """Training (SGD)"""
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}