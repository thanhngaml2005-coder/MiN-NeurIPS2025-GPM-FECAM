import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from torch.nn import functional as F
from torch.cuda.amp import autocast 

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
        
        # Dùng float32
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
        self.task_prototypes = []

        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        factory_kwargs = {"device": self.device, "dtype": torch.float32}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)

        self.Pinoise_list = nn.ModuleList()
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        self.fc2 = nn.ModuleList()

    # [FIX] Thêm property này để tránh lỗi attribute, 
    # nhưng logic bên dưới tôi đã chuyển sang dùng shape trực tiếp cho chắc ăn.
    @property
    def out_features(self) -> int:
        return self.weight.shape[1]

    def set_grad_checkpointing(self, enable=True):
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)
        elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'set_grad_checkpointing'):
             self.backbone.model.set_grad_checkpointing(enable)
        elif hasattr(self.backbone, 'grad_checkpointing'): 
            self.backbone.grad_checkpointing = enable

    def forward_fc(self, features):
        features = features.to(self.weight.dtype)
        return features @ self.weight

    @property
    def in_features(self) -> int:
        return self.weight.shape[0]

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        if self.cur_task > 0:
            fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
        
        if self.normal_fc is None:
            self.normal_fc = fc
        else:
            nn.init.constant_(fc.weight, 0.)
            del self.normal_fc
            self.normal_fc = fc

    # =========================================================================
    # HÀM FIT CHO ẢNH (QUA BACKBONE)
    # =========================================================================
    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        with autocast(enabled=False): 
            X = self.backbone(X)
            X = X.float() 
            X = self.buffer(X) 

            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            num_targets = Y.shape[1]
            # [FIX] Dùng self.weight.shape[1] thay vì self.out_features để tránh lỗi
            current_out_features = self.weight.shape[1]
            
            if num_targets > current_out_features:
                increment_size = num_targets - current_out_features
                tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < current_out_features:
                increment_size = current_out_features - num_targets
                tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
                Y = torch.cat((Y, tail), dim=1)

            I = torch.eye(X.shape[0]).to(X)
            term = I + X @ self.R @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            
            try:
                K = torch.inverse(term + jitter)
            except:
                K = torch.pinverse(term + jitter)
            
            self.R -= self.R @ X.T @ K @ X @ self.R
            self.weight += self.R @ X.T @ (Y - X @ self.weight)

    # =========================================================================
    # HÀM FIT CHO FEATURES (TỪ CFS) - KHÔNG QUA BACKBONE
    # =========================================================================
    @torch.no_grad()
    def fit_features(self, features: torch.Tensor, Y: torch.Tensor) -> None:
        with autocast(enabled=False): 
            # Features đã có sẵn, chỉ cần ép kiểu
            X = features.float() 
            X = self.buffer(X)

            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            num_targets = Y.shape[1]
            # [FIX] Dùng trực tiếp shape để tránh AttributeError
            current_out_features = self.weight.shape[1]

            if num_targets > current_out_features:
                increment_size = num_targets - current_out_features
                tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < current_out_features:
                increment_size = current_out_features - num_targets
                tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
                Y = torch.cat((Y, tail), dim=1)

            I = torch.eye(X.shape[0]).to(X)
            term = I + X @ self.R @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            
            try:
                K = torch.inverse(term + jitter)
            except:
                K = torch.pinverse(term + jitter)
            
            self.R -= self.R @ X.T @ K @ X @ self.R
            self.weight += self.R @ X.T @ (Y - X @ self.weight)

    def forward(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {
            'logits': logits
        }
    
    def update_task_prototype(self, mean, std=None):
        if std is not None:
            self.task_prototypes[-1] = (mean.detach().cpu(), std.detach().cpu())
        else:
            self.task_prototypes[-1] = mean.detach().cpu()

    def extend_task_prototype(self, mean, std=None):
        if std is not None:
            self.task_prototypes.append((mean.detach().cpu(), std.detach().cpu()))
        else:
            self.task_prototypes.append(mean.detach().cpu())

    def extract_feature(self, x):
        hyper_features = self.backbone(x)
        return hyper_features

    def forward_normal_fc(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        logits = self.normal_fc(hyper_features)['logits']
        return {
            "logits": logits
        }

    def update_noise(self, new_mean=None, new_std=None):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()
        
        if new_mean is not None:
            # print("[IncNet] Recalculating Noise Weights...")
            full_prototypes_device = []
            for p in self.task_prototypes:
                if isinstance(p, tuple):
                    full_prototypes_device.append((p[0].to(self.device), p[1].to(self.device)))
                else:
                    full_prototypes_device.append((p.to(self.device), torch.ones_like(p).to(self.device)))
            
            for j in range(self.backbone.layer_num):
                self.backbone.noise_maker[j].init_weight_noise(full_prototypes_device)

    # [FIX] Thêm hàm dummy này để Min.py không bị lỗi nếu gọi
    def after_task_magmax_merge(self):
        pass

    def unfreeze_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].unfreeze_noise()

    def init_unfreeze(self):
        for j in range(self.backbone.layer_num):
            for param in self.backbone.noise_maker[j].parameters():
                param.requires_grad = True
            for p in self.backbone.blocks[j].norm1.parameters():
                p.requires_grad = True
            for p in self.backbone.blocks[j].norm2.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True