import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear

# Xử lý tương thích Import autocast
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
        return SimpleLinear(in_dim, out_dim)

    def forward(self, x):
        hyper_features = self.backbone(x)
        logits = self.fc(hyper_features)['logits']
        return {'features': hyper_features, 'logits': logits}


class RandomBuffer(torch.nn.Linear):
    """
    Lớp Random Projection giúp RLS hoạt động tốt hơn trong không gian cao chiều.
    """
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        
        # [SAFEGUARD] Luôn dùng float32 cho Buffer
        factory_kwargs = {"device": device, "dtype": torch.float32} 
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming Uniform chuẩn
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight.dtype)
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

        # RLS Weights (Classifier) - Luôn Float32
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) 

        # Covariance Matrix R (Inverse)
        R = torch.eye(self.buffer_size, **factory_kwargs) / self.gamma
        self.register_buffer("R", R) 

        # Normal FC (Dùng để train Noise bằng SGD)
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        
        # [FIX] Luôn bật bias=True để giữ kiến thức cũ tốt hơn
        new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=True).float()
            
        if self.normal_fc is not None:
            old_nb = self.normal_fc.out_features
            with torch.no_grad():
                # Copy Weight
                new_fc.weight[:old_nb] = self.normal_fc.weight.data
                # Copy Bias
                if self.normal_fc.bias is not None:
                    new_fc.bias[:old_nb] = self.normal_fc.bias.data
                
                # Init phần mới về 0 (Zero Init)
                nn.init.constant_(new_fc.weight[old_nb:], 0.)
                nn.init.constant_(new_fc.bias[old_nb:], 0.)
            
            del self.normal_fc
            self.normal_fc = new_fc
        else:
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None:
                nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    def update_noise(self):
        # [FIX QUAN TRỌNG] Truyền self.cur_task xuống PiNoise
        # Để đảm bảo PiNoise không tự tăng ID lung tung
        for m in self.backbone.noise_maker:
            # Bạn cần sửa thêm 1 chút ở class PiNoise để nhận tham số này
            # Nếu lười sửa PiNoise, thì giữ nguyên dòng này, nhưng hãy cẩn thận đừng gọi update_noise 2 lần.
            # Tốt nhất là thêm method set_task_id cho PiNoise
            if hasattr(m, 'set_task_id'):
                m.set_task_id(self.cur_task)
            else:
                m.update_noise() 

    def after_task_magmax_merge(self):
        print(f"--> [IncNet] Task {self.cur_task}: Saving PiNoise History...")
        for m in self.backbone.noise_maker:
            m.after_task_training()

    def unfreeze_noise(self):
        for m in self.backbone.noise_maker:
            m.unfreeze_noise()

    def init_unfreeze(self):
        self.unfreeze_noise()
        # Unfreeze LayerNorms trong Backbone
        for name, param in self.backbone.named_parameters():
            if "norm" in name:
                param.requires_grad = True

    def forward_fc(self, features):
        features = features.to(self.weight.dtype)
        return features @ self.weight

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        RLS Fit Function
        """
        # [SAFEGUARD] Ép Backbone về Eval mode để PiNoise hoạt động ổn định (cộng dồn noise tất cả tasks)
        self.backbone.eval()

        with autocast('cuda', enabled=False):
            # 1. Feature Extraction (Float32)
            features = self.backbone(X).float()
            
            # 2. Random Expansion
            X_embedded = self.buffer(features)
            X_embedded = X_embedded.to(self.weight.device)
            Y = Y.to(self.weight.device).float()

            # 3. Expand RLS Weight if needed
            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
                self.weight = torch.cat((self.weight, tail), dim=1)
            
            # 4. RLS Update
            I = torch.eye(X_embedded.shape[0]).to(X_embedded)
            term = I + X_embedded @ self.R @ X_embedded.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            
            K = torch.inverse(term + jitter)
            
            self.R -= self.R @ X_embedded.T @ K @ X_embedded @ self.R
            self.weight += self.R @ X_embedded.T @ (Y - X_embedded @ self.weight)

    def forward(self, x, new_forward: bool = False):
        # Inference Mode
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {'logits': logits}

    def forward_normal_fc(self, x, new_forward: bool = False):
        # Training Mode
        hyper_features = self.backbone(x)
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}