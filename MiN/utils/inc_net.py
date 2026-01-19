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

# [FIXED] Chuyển sang kế thừa nn.Module để tránh lỗi bias của nn.Linear
class RandomBuffer(nn.Module):
    """
    Lớp mở rộng đặc trưng ngẫu nhiên (Random Projection).
    """
    def __init__(self, in_features: int, buffer_size: int, device):
        super().__init__()
        self.in_features = in_features
        self.out_features = buffer_size
        
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        # Tạo ma trận ngẫu nhiên nhưng lưu dưới dạng buffer (không train)
        # Shape: [In, Out] để nhân X @ W
        self.register_buffer("weight", torch.empty((in_features, buffer_size), **factory_kwargs))
        
        # Explicitly set bias to None
        self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        # Tự khởi tạo Kaiming Uniform (giống Linear)
        # Lưu ý: weight shape là [In, Out], kaiming_uniform_ mặc định tính fan_in theo dim 1.
        # Ở đây ta muốn fan_in là In_features (dim 0).
        # Tuy nhiên với Random Projection, phân phối chuẩn là đủ tốt.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Ép kiểu input X về cùng kiểu với weight (float32)
        X = X.to(self.weight.dtype)
        return F.relu(X @ self.weight)


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
        self.register_buffer("weight", weight) # Trọng số của Analytic Classifier

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R) # Ma trận hiệp phương sai đảo (Inverse Covariance Matrix)

        # Normal FC: Dùng để train Gradient Descent cho Noise Generator
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        
        # [NEW] Lưu trữ CFS samples cho mỗi task (để re-fit)
        self.task_cfs_samples = []  # List of [20, feature_dim] tensors

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
            # Sequential Init: Copy trọng số cũ
            old_nb_output = self.normal_fc.out_features
            with torch.no_grad():
                # Copy phần cũ
                new_fc.weight[:old_nb_output] = self.normal_fc.weight.data
                # Init phần mới về 0
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
    # [PROTOTYPE MANAGEMENT]
    # =========================================================================
    def extend_task_prototype(self, prototype):
        # Hàm này có thể dùng để lưu prototype tính bằng mean nếu cần
        pass

    def update_task_prototype(self, prototype):
        pass

    # =========================================================================
    # [MAGMAX & NOISE CONTROL SECTION]
    # =========================================================================
    
    def update_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()

    def after_task_magmax_merge(self):
        print(f"--> [IncNet] Task {self.cur_task}: Triggering Parameter-wise MagMax Merging...")
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].after_task_training()

    def unfreeze_noise(self):
        """Chỉ mở khóa gradient cho các module Noise (cho các task > 0)"""
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].unfreeze_noise()

    def init_unfreeze(self):
        """
        Mở khóa gradient cho Task 0.
        Bao gồm Noise modules và các lớp Normalization của Backbone để ổn định hơn.
        """
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

    # =========================================================================
    # [ANALYTIC LEARNING (RLS) SECTION]
    # =========================================================================

    def forward_fc(self, features):
        """Forward qua Analytic Classifier"""
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
        with autocast('cuda', enabled=False):
            
            # --- [FIX QUAN TRỌNG] Kiểm tra đầu vào là Ảnh hay Feature ---
            if X.ndim == 4: 
                # Nếu là Ảnh [B, C, H, W] -> Chạy qua Backbone để lấy Feature
                X = self.backbone(X).float() 
            else:
                # Nếu đã là Feature [B, Dim] -> Chỉ cần ép kiểu
                X = X.float()

            # 1. Feature Expansion (Random Buffer)
            X = self.buffer(X)           
            
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
            term = torch.eye(X.shape[0], device=X.device) + X @ self.R @ X.T
            
            # Thêm jitter để tránh lỗi ma trận suy biến
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            
            # Nghịch đảo ma trận
            try:
                K = torch.inverse(term + jitter)
            except RuntimeError:
                K = torch.linalg.pinv(term + jitter)
            
            # Cập nhật R và Weight
            self.R -= self.R @ X.T @ K @ X @ self.R
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