import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
# Import autocast để tắt nó trong quá trình tính toán ma trận chính xác cao
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
    """
    Lớp mở rộng đặc trưng ngẫu nhiên (Random Projection).
    Đóng vai trò như Kernel Approximation.
    """
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        
        # [QUAN TRỌNG] Sử dụng float32 để đảm bảo độ chính xác khi tính RLS
        # FP16 (Half) không đủ chính xác cho ma trận hiệp phương sai
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Ép kiểu input X về cùng kiểu với weight (float32)
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

        # Khởi tạo ma trận trọng số và ma trận hiệp phương sai cho RLS
        # Dùng float32 để tránh lỗi singular matrix khi tính nghịch đảo
        factory_kwargs = {"device": self.device, "dtype": torch.float32}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) # Trọng số của Analytic Classifier (Test time)

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R) # Ma trận hiệp phương sai đảo (Inverse Covariance Matrix)

        # Normal FC: Dùng để train Gradient Descent cho Noise Generator (BiLORA)
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    def update_fc(self, nb_classes):
        """
        Cập nhật lớp Normal FC (cho việc training Noise).
        Lớp Analytic FC (self.weight) sẽ tự động mở rộng trong hàm fit().
        """
        self.cur_task += 1
        self.known_class += nb_classes
        
        # Tạo mới Normal FC cho task hiện tại
        if self.cur_task > 0:
            # Task sau: Không dùng Bias để tránh bias vào lớp mới quá nhiều
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            # Task đầu: Có bias
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            
        if self.normal_fc is not None:
            # Sequential Init: Copy trọng số cũ sang layer mới mở rộng
            old_nb_output = self.normal_fc.out_features
            with torch.no_grad():
                # Copy phần cũ
                new_fc.weight[:old_nb_output] = self.normal_fc.weight.data
                # Init phần mới về 0 (để bắt đầu học từ neutral)
                nn.init.constant_(new_fc.weight[old_nb_output:], 0.)
            
            del self.normal_fc
            self.normal_fc = new_fc
        else:
            # Task đầu tiên
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None:
                nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    # =========================================================================
    # [BiLORA & MAGMAX CONTROL SECTION]
    # =========================================================================
    
    def update_noise(self):
        """
        [STEP 1 - BiLORA]: Gọi khi bắt đầu Task mới.
        Hàm này kích hoạt BiLORA_Linear.new_task() trong vit_min.py.
        Tác dụng: Sinh Mask ngẫu nhiên mới và tạo Parameter mới để train.
        """
        # Duyệt qua các blocks của ViT (hoặc module chứa noise_maker)
        # Giả sử noise_maker được gắn vào self.backbone.noise_maker (như trong vit_min.py)
        if hasattr(self.backbone, 'noise_maker'):
            for j in range(len(self.backbone.noise_maker)):
                self.backbone.noise_maker[j].update_noise()

    def after_task_magmax_merge(self):
        """
        [STEP 2 - MagMax]: Gọi sau khi kết thúc Task.
        Hàm này kích hoạt BiLORA_Linear.perform_magmax_merge() trong vit_min.py.
        Tác dụng: Gộp tham số vừa train vào bộ nhớ tần số toàn cục dựa trên độ lớn.
        """
        print(f"--> [IncNet] Task {self.cur_task}: Triggering Frequency-Domain MagMax Merging...")
        if hasattr(self.backbone, 'noise_maker'):
            for j in range(len(self.backbone.noise_maker)):
                self.backbone.noise_maker[j].after_task_training()

    def unfreeze_noise(self):
        """Chỉ mở khóa gradient cho các module Noise (cho các task > 0)"""
        # Logic freeze/unfreeze chi tiết đã được xử lý bên trong BiLORA_Linear
        # Hàm này chỉ mang tính chất kích hoạt update state nếu cần
        self.update_noise() 

    def init_unfreeze(self):
        """
        Mở khóa gradient cho Task 0.
        Bao gồm Noise modules và các lớp Normalization của Backbone để ổn định hơn.
        """
        self.update_noise() # Sinh mask cho task 0
        
        # Unfreeze LayerNorms (Thường giúp hội tụ tốt hơn ở Task đầu)
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
            
        # Nếu block có norm riêng
        for blk in self.backbone.blocks:
            if hasattr(blk, 'norm1'):
                for p in blk.norm1.parameters(): p.requires_grad = True
            if hasattr(blk, 'norm2'):
                for p in blk.norm2.parameters(): p.requires_grad = True

    # =========================================================================
    # [ANALYTIC LEARNING (RLS) SECTION]
    # =========================================================================

    def forward_fc(self, features):
        """Forward qua Analytic Classifier (Test Time)"""
        # Đảm bảo features cùng kiểu với trọng số RLS (float32)
        features = features.to(self.weight) 
        return features @ self.weight

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Thuật toán Recursive Least Squares (RLS).
        Cập nhật self.weight và self.R trực tiếp bằng công thức toán học.
        """
        # [QUAN TRỌNG] Tắt Autocast để tính toán chính xác cao (FP32)
        # Nếu dùng FP16, ma trận hiệp phương sai sẽ rất dễ bị suy biến (singular)
        with autocast(enabled=False):
            # 1. Feature Extraction & Expansion
            X = self.backbone(X).float() # ViT Features
            X = self.buffer(X)           # Random Expansion -> float32
            
            # Đưa về cùng device
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            # 2. Mở rộng chiều của classifier nếu có lớp mới (Class Incremental)
            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.weight.shape[1]:
                increment_size = self.weight.shape[1] - num_targets
                tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
                Y = torch.cat((Y, tail), dim=1)

            # 3. Công thức cập nhật RLS (Woodbury Matrix Identity application)
            I = torch.eye(X.shape[0]).to(X)
            term = I + X @ self.R @ X.T
            
            # Thêm jitter (nhiễu cực nhỏ) trên đường chéo để đảm bảo khả nghịch
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            
            # Nghịch đảo ma trận
            K = torch.inverse(term + jitter)
            
            # Cập nhật R (Inverse Covariance Matrix)
            self.R -= self.R @ X.T @ K @ X @ self.R
            
            # Cập nhật Trọng số Classifier
            # W_new = W_old + K * (Y - X * W_old)
            self.weight += self.R @ X.T @ (Y - X @ self.weight)

    # =========================================================================
    # [FORWARD PASSES]
    # =========================================================================

    def forward(self, x, new_forward: bool = False):
        """
        Dùng cho Inference/Testing.
        Chạy qua backbone (đã merge noise) -> Buffer -> Analytic Classifier.
        """
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        # Ép kiểu về float32 cho Buffer và Classifier
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        
        return {
            'logits': logits
        }

    def extract_feature(self, x):
        """Chỉ trích xuất đặc trưng từ Backbone"""
        return self.backbone(x)

    def forward_normal_fc(self, x, new_forward: bool = False):
        """
        Dùng cho Training (Gradient Descent).
        Chạy qua backbone -> Buffer -> Normal FC (trainable).
        Gradient sẽ lan truyền ngược về Backbone -> PiNoise -> BiLORA Params.
        """
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = self.buffer(hyper_features)
        
        # Ép kiểu để khớp với Normal FC (có thể là FP16 nếu autocast bật bên ngoài)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        
        logits = self.normal_fc(hyper_features)['logits']
        
        return {
            "logits": logits
        }