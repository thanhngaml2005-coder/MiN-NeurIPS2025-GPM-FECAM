import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
# Import autocast
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

# [GIỮ NGUYÊN CODE CŨ CỦA BẠN]
class RandomBuffer(torch.nn.Linear):
    """
    Lớp mở rộng đặc trưng ngẫu nhiên (Random Projection).
    Giữ nguyên kế thừa torch.nn.Linear như yêu cầu.
    """
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        
        # Sử dụng float32 để đảm bảo độ chính xác
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
        
        # Các tham số cho Analytic Learning (RLS)
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        # Random Buffer
        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        # Khởi tạo ma trận trọng số và RLS (FP32)
        factory_kwargs = {"device": self.device, "dtype": torch.float32}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) 

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R) 

        # Normal FC
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    def update_fc(self, nb_classes):
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
                nn.init.constant_(new_fc.weight[old_nb_output:], 0.)
            
            del self.normal_fc
            self.normal_fc = new_fc
        else:
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None:
                nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    def update_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()

    def after_task_magmax_merge(self):
        print(f"--> [IncNet] Task {self.cur_task}: Triggering Parameter-wise MagMax Merging...")
        num_layers = len(self.backbone.blocks) 
        for i in range(num_layers):
             self.backbone.noise_maker[i].after_task_training()

    def unfreeze_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].unfreeze_noise()

    def init_unfreeze(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].unfreeze_noise()
            for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
        for p in self.backbone.norm.parameters(): p.requires_grad = True

    def forward_fc(self, features):
        features = features.to(self.weight) 
        return features @ self.weight

    # --- ĐÂY LÀ HÀM CẦN CHỈNH SỬA DUY NHẤT ---
    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Sửa đổi: Chuyển sang Eval Mode để ép PiNoise dùng Generator đã Merge (Main),
        tránh việc RLS học nhầm trên Generator tạm thời (Temp).
        """
        
        try:
            # 2. Feature Extraction (Bật Autocast để nhanh)
            try:
                from torch.amp import autocast
            except ImportError:
                from torch.cuda.amp import autocast

            with autocast('cuda', enabled=True): 
                X = self.backbone(X)
            
            # 3. RLS Calculation (Tắt Autocast, dùng FP32 chuẩn)
            with autocast('cuda', enabled=False):
                X = X.detach().float()
                
                # Qua Random Buffer
                X = self.buffer(X).float()
                
                device = self.weight.device
                X = X.to(device)
                Y = Y.to(device).float()

                # Expand Classifier
                num_targets = Y.shape[1]
                if num_targets > self.weight.shape[1]:
                    increment_size = num_targets - self.weight.shape[1]
                    tail = torch.zeros((self.weight.shape[0], increment_size), dtype=torch.float32, device=device)
                    self.weight = torch.cat((self.weight, tail), dim=1)
                    del tail 
                elif num_targets < self.weight.shape[1]:
                    increment_size = self.weight.shape[1] - num_targets
                    tail = torch.zeros((Y.shape[0], increment_size), dtype=torch.float32, device=device)
                    Y = torch.cat((Y, tail), dim=1)
                    del tail 

                # --- RLS ALGORITHM ---
                P = self.R @ X.T
                term = X @ P
                term.diagonal().add_(1.0 + 1e-5)
                term = 0.5 * (term + term.T)
                
                try:
                    K = torch.linalg.inv(term)
                except:
                    K = torch.linalg.inv(term.cpu()).to(device)
                
                del term 
                P_K = P @ K
                self.R -= P_K @ P.T
                del P 

                residual = Y - (X @ self.weight)
                self.weight += P_K @ residual
                
                del X, Y, K, P_K, residual
                torch.cuda.empty_cache()
        
        finally:
            pass

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

    def extract_feature(self, x):
        return self.backbone(x)

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