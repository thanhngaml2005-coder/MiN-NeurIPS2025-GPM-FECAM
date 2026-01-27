import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from torch.nn import functional as F
import scipy.stats as stats
import timm
import random
# Thêm đoạn này vào đầu file utils/inc_net.py
try:
    from torch.amp import autocast
except ImportError:
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
        # Sửa dtype từ torch.double -> torch.float32
        factory_kwargs = {"device": device, "dtype": torch.float32} 
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)
        self.reset_parameters()

    # @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
        return F.relu(X @ self.W)


# -------------------------------------------------
# MAIN NETWORK
# -------------------------------------------------
class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        
        # RLS Params
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        # Random Buffer
        self.buffer = RandomBuffer(in_features=self.feature_dim, 
                                   buffer_size=self.buffer_size, 
                                   device=self.device)

        # RLS Weights (Classifier)
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) 

        # Covariance Matrix R (Inverse)
        R = torch.eye(self.buffer_size, **factory_kwargs) / self.gamma
        self.register_buffer("R", R) 

        # Normal FC (Cho việc học SGD - PiNoise training)
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        
        # Tạo Normal FC mới để train task hiện tại
        if self.cur_task > 0:
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False).float()
        else:
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True).float()
            
        # Copy trọng số cũ (nếu muốn warm-start normal FC, dù RLS mới là chính)
        if self.normal_fc is not None:
            old_nb = self.normal_fc.out_features
            with torch.no_grad():
                new_fc.weight[:old_nb] = self.normal_fc.weight.data
                nn.init.constant_(new_fc.weight[old_nb:], 0.)
            del self.normal_fc
            self.normal_fc = new_fc
        else:
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None: nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    def update_noise(self):
        # Trigger update mask trong PiNoise
        for j in range(self.backbone.layer_num):
            if hasattr(m, 'update_noise'):
                # PiNoise cần nhận tham số task_id
                m.update_noise(task_id=self.cur_task)
    def after_task_magmax_merge(self):
        print(f"--> [IncNet] Task {self.cur_task}: Triggering MagMax Merging...")
        for m in self.backbone.noise_maker:
            # Hàm này trong PiNoise mới đã bao gồm logic _magmax_update
            m.after_task_training()

    def unfreeze_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].unfreeze_noise()

    def init_unfreeze(self):
        # Mở khóa các lớp cần thiết ban đầu
        for j in range(self.backbone.layer_num):
            # Unfreeze Noise
            self.backbone.noise_maker[j].unfreeze_noise()
            
            # Unfreeze LayerNorms trong từng Block ViT
            for p in self.backbone.blocks[j].norm1.parameters():
                p.requires_grad = True
            for p in self.backbone.blocks[j].norm2.parameters():
                p.requires_grad = True
                
        # Unfreeze LayerNorm cuối cùng
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
    def forward_fc(self, features):
        features = features.to(self.weight.dtype)
        # Classifier chính thức (RLS Weight)
        return features @ self.weight

    # -------------------------------------------------
    # ✅ FIX 2: FIT FUNCTION VỚI EVAL MODE & MEMORY SAFE
    # -------------------------------------------------
    @torch.no_grad()

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Thuật toán Recursive Least Squares (RLS).
        Cập nhật self.weight và self.R trực tiếp bằng công thức toán học.
        """
        # [QUAN TRỌNG] Tắt Autocast để tính toán chính xác cao (FP32)
        with autocast('cuda', enabled=False):
            # 1. Feature Extraction & Expansion
            X = self.backbone(X).float() # ViT Features
            X = self.buffer(X)           # Random Expansion -> float32
            
            # Đưa về cùng device
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            # 2. Mở rộng chiều của classifier nếu có lớp mới
            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.weight.shape[1]:
                increment_size = self.weight.shape[1] - num_targets
                tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
                Y = torch.cat((Y, tail), dim=1)

            # 3. Công thức cập nhật RLS
            I = torch.eye(X.shape[0]).to(X)
            term = I + X @ self.R @ X.T
            
            # Thêm jitter để tránh lỗi ma trận suy biến (Singular Matrix)
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            
            # Nghịch đảo ma trận
            K = torch.inverse(term + jitter)
            
            # Cập nhật R (Covariance Matrix)
            self.R -= self.R @ X.T @ K @ X @ self.R
            
            # Cập nhật Trọng số Classifier
            self.weight += self.R @ X.T @ (Y - X @ self.weight)

    def forward(self, x, new_forward: bool = False):
        # Hàm forward tổng quát cho Inference
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        hyper_features = hyper_features.to(self.weight.dtype)
        
        # Qua buffer rồi nhân với RLS Weights
        logits = self.forward_fc(self.buffer(hyper_features))
        
        return {'logits': logits}

    def forward_normal_fc(self, x, new_forward: bool = False):
        # Hàm forward dành riêng cho lúc Training PiNoise (dùng SGD)
        hyper_features = self.backbone(x)
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        # Dùng Normal FC (có bias, đang học)
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}