import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
# Import autocast cho Mixed Precision
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

# -------------------------------------------------
# ✅ FIX 1: RandomBuffer kế thừa nn.Module (Sạch hơn)
# -------------------------------------------------
class RandomBuffer(nn.Module):
    def __init__(self, in_features: int, buffer_size: int, device):
        super(RandomBuffer, self).__init__()
        self.in_features = in_features
        self.out_features = buffer_size
        
        # Tạo ma trận chiếu cố định (Không học)
        factory_kwargs = {"device": device, "dtype": torch.float32}
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        
        # Kaiming Init cho projection tốt
        nn.init.kaiming_normal_(self.W, nonlinearity='linear')
        
        # Đăng ký là buffer để nó được lưu vào state_dict nhưng không có gradient
        self.register_buffer("proj_weight", self.W)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.proj_weight.dtype)
        # Random Projection + ReLU activation
        return F.relu(X @ self.proj_weight)

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
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            
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
        # Mở khóa các lớp cần thiết ban đầu
        self.unfreeze_noise()
        for p in self.backbone.norm.parameters(): p.requires_grad = True
        # Mở khóa LayerNorm của backbone (tùy chọn)
        for j in range(self.backbone.layer_num):
             for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
             for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True

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
        Huấn luyện Analytic Classifier (RLS).
        Bắt buộc chuyển sang EVAL mode để PiNoise dùng Merged Weights.
        """
        # Lưu trạng thái cũ
        old_training_state = self.training
        self.eval() 
        
        try:
            # 1. Feature Extraction (Dùng Autocast cho nhanh)
            with autocast('cuda', enabled=True): 
                # backbone() sẽ gọi PiNoise.forward()
                # Ở eval mode, PiNoise sẽ kích hoạt logic Merged + Deterministic
                X_feat = self.backbone(X)
            
            # 2. RLS Calculation (Dùng FP32 chuẩn xác)
            with autocast('cuda', enabled=False):
                X_feat = X_feat.detach().float()
                
                # Qua Random Buffer
                X_feat = self.buffer(X_feat).float()
                
                device = self.weight.device
                X_feat = X_feat.to(device)
                Y = Y.to(device).float()

                # Tự động mở rộng Classifier (Weight matrix)
                num_targets = Y.shape[1]
                current_outputs = self.weight.shape[1]
                
                if num_targets > current_outputs:
                    diff = num_targets - current_outputs
                    tail = torch.zeros((self.weight.shape[0], diff), dtype=torch.float32, device=device)
                    self.weight = torch.cat((self.weight, tail), dim=1)
                elif num_targets < self.out_features:
                    increment_size = self.out_features - num_targets
                    tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
                    Y = torch.cat((Y, tail), dim=1)
                # --- RLS CORE ---
                # P = R * X^T
                P = self.R @ X_feat.T
                
                # Term = X * P = X * R * X^T
                term = X_feat @ P
                # Regularization (Dampening factor)
                term.diagonal().add_(1.0) 
                # Symmetrization (giữ tính đối xứng)
                term = 0.5 * (term + term.T)
                
                # Invert (K = term^-1)
                try:
                    K = torch.linalg.inv(term)
                except RuntimeError:
                    # ✅ Fallback CPU nếu OOM
                    print("⚠️ GPU OOM during RLS inversion, switching to CPU...")
                    K = torch.linalg.inv(term.cpu()).to(device)
                
                del term # Giải phóng bộ nhớ ngay
                
                # Update R
                # R_new = R - P * K * P^T
                P_K = P @ K # [Buffer, Batch] * [Batch, Batch]
                self.R -= P_K @ P.T
                
                del P # Giải phóng
                
                # Update Weights
                # W_new = W_old + P * K * (Y - X * W_old)
                # residual = Y - Prediction
                residual = Y - (X_feat @ self.weight)
                self.weight += P_K @ residual
                
                # Dọn dẹp cuối cùng
                del X_feat, Y, K, P_K, residual
                torch.cuda.empty_cache()
        
        finally:
            # ✅ QUAN TRỌNG: Trả lại trạng thái cũ
            self.train(old_training_state)

    def forward(self, x, new_forward: bool = False):
        # Hàm forward tổng quát cho Inference
        hyper_features = self.backbone(x)
        hyper_features = hyper_features.to(self.weight.dtype)
        
        # Qua buffer rồi nhân với RLS Weights
        logits = self.forward_fc(self.buffer(hyper_features))
        
        return {'logits': logits}

    def forward_normal_fc(self, x, new_forward: bool = False):
        # Hàm forward dành riêng cho lúc Training PiNoise (dùng SGD)
        hyper_features = self.backbone(x)
        hyper_features = self.buffer(hyper_features)
        
        # Dùng Normal FC (có bias, đang học)
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}