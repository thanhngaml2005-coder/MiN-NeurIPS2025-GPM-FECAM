import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from backbones.pretrained_backbone import get_pretrained_backbone 
from backbones.linears import SimpleLinear
from torch.nn import functional as F
import gc

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

class BaseIncNet(nn.Module):
    def __init__(self, args: dict):
        super(BaseIncNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        
        # [QUAN TRỌNG]: Tắt gradient backbone ngay lập tức để tiết kiệm VRAM khởi tạo
        for param in self.backbone.parameters():
            param.requires_grad = False
            
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

import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone 
from backbones.linears import SimpleLinear
import gc

# -----------------------------------------------------------
# 1. RandomBuffer: Float32 & GPU
# (Giữ trên GPU vì cần cho Forward Pass nhanh)
# -----------------------------------------------------------
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone 
from backbones.linears import SimpleLinear
import gc

# -----------------------------------------------------------------------------
# 1. RandomBuffer: Float32 & GPU (Nhẹ nhất có thể)
# -----------------------------------------------------------------------------
class RandomBuffer(torch.nn.Linear):
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.in_features = in_features
        self.out_features = buffer_size
        
        # [OPTIMIZATION 1]: Ép cứng Float32 ngay từ đầu
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        # Tạo ma trận rỗng trước
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        
        # In-place Init (Không tạo bản copy)
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        
        # Register buffer để lưu vào state_dict nhưng không tính gradient
        self.register_buffer("weight", self.W)
        self.weight.requires_grad = False

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight.dtype)
        return F.relu(X @ self.W)

# -----------------------------------------------------------------------------
# 2. MiNbaseNet: Hybrid Init (R nằm ở CPU)
# -----------------------------------------------------------------------------
class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        
        # [OPTIMIZATION 2]: Dọn sạch VRAM trước khi bắt đầu
        gc.collect()
        torch.cuda.empty_cache()
        
        self.args = args
        self.device = args['device']
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size'] # Khuyên dùng 8192
        
        print(f"📉 [Init] Starting Initialization... Target Buffer Size: {self.buffer_size}")

        # --- BƯỚC 1: Load Backbone & Đóng băng ngay lập tức ---
        self.backbone = get_pretrained_backbone(args)
        
        # [OPTIMIZATION 3]: Tắt Gradient NGAY LẬP TỨC
        # Nếu không tắt ngay, PyTorch có thể cấp phát bộ nhớ dự phòng cho Gradients
        print("❄️  [Init] Freezing Backbone Gradients...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.feature_dim = self.backbone.out_dim 
        
        # Dọn dẹp rác sinh ra khi load backbone
        torch.cuda.empty_cache()

        # --- BƯỚC 2: Init Random Buffer (GPU) ---
        print("🎲 [Init] Creating Random Buffer on GPU...")
        self.buffer = RandomBuffer(in_features=self.feature_dim, 
                                   buffer_size=self.buffer_size, 
                                   device=self.device)
        
        # --- BƯỚC 3: Init RLS Matrix (CPU ONLY) ---
        # Đây là bước quan trọng nhất để cứu VRAM lúc khởi tạo
        print("💾 [Init] Allocating Covariance Matrix R on CPU RAM...")
        
        # Tạo trực tiếp trên CPU (Không bao giờ chạm vào GPU)
        self.R_cpu = torch.eye(self.buffer_size, dtype=torch.float32, device='cpu')
        self.R_cpu.div_(self.gamma) # In-place division
        
        # Lưu ý: Không register_buffer cho R_cpu để tránh nó bị đẩy lên GPU khi model.to(device)
        
        # --- BƯỚC 4: Init Classifier Weight (GPU - Size 0) ---
        print("⚖️  [Init] Creating Empty Classifier on GPU...")
        # Khởi tạo kích thước 0. Nó sẽ tự mở rộng khi train. Tốn 0 VRAM lúc này.
        self.register_buffer("weight", torch.zeros((self.buffer_size, 0), device=self.device, dtype=torch.float32))

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        
        print("✅ [Init] Model Initialized Successfully.")
        print(f"   - GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        
        # Tạo FC mới cho PiNoise training (SGD)
        if self.cur_task > 0:
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False).float()
        else:
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True).float()
            
        if self.normal_fc is not None:
            old_nb = self.normal_fc.out_features
            with torch.no_grad():
                new_fc.weight[:old_nb] = self.normal_fc.weight.data
                nn.init.constant_(new_fc.weight[old_nb:], 0.)
            del self.normal_fc
        else:
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None: nn.init.constant_(new_fc.bias, 0.)
            
        # Đẩy FC mới lên GPU
        self.normal_fc = new_fc.to(self.device)

    def update_noise(self):
        if hasattr(self.backbone, 'noise_maker'):
            for j in range(len(self.backbone.noise_maker)):
                self.backbone.noise_maker[j].expand_new_task(self.cur_task)

    def after_task_magmax_merge(self):
        if hasattr(self.backbone, 'noise_maker'):
            for j in range(len(self.backbone.noise_maker)):
                 self.backbone.noise_maker[j].after_task_training()

    def unfreeze_noise(self):
        if hasattr(self.backbone, 'noise_maker'):
            for j in range(len(self.backbone.noise_maker)):
                for param in self.backbone.noise_maker[j].parameters():
                    param.requires_grad = True

    def init_unfreeze(self):
        self.unfreeze_noise()
        if hasattr(self.backbone, 'blocks'):
            for block in self.backbone.blocks:
                if hasattr(block, 'norm1'): 
                    for p in block.norm1.parameters(): p.requires_grad = True
                if hasattr(block, 'norm2'):
                    for p in block.norm2.parameters(): p.requires_grad = True
        if hasattr(self.backbone, 'norm'):
            for p in self.backbone.norm.parameters(): p.requires_grad = True

    def forward_fc(self, features):
        return features @ self.weight

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Hybrid Fit:
        1. Feature Extraction -> GPU
        2. RLS Math -> CPU (Dùng self.R_cpu)
        """
        old_training_state = self.training
        self.eval() 
        
        try:
            # --- 1. GPU: Feature Extraction ---
            from torch.amp import autocast
            with autocast('cuda', enabled=True): 
                X_feat = self.backbone(X)
            
            # Detach, Float32, Project
            X_feat = X_feat.detach().float()
            X_proj = self.buffer(X_feat) 
            del X_feat 
            
            # --- 2. TRANSFER: GPU -> CPU ---
            # Chỉ chuyển Feature đã project (nhỏ hơn nhiều so với ảnh gốc)
            X_final_cpu = X_proj.cpu()
            Y_cpu = Y.cpu().float()
            del X_proj # Xóa ngay trên GPU
            
            # Lấy weight hiện tại về CPU để update
            weight_cpu = self.weight.detach().cpu()

            # Expand Weight trên CPU
            num_targets = Y_cpu.shape[1]
            if num_targets > weight_cpu.shape[1]:
                tail = torch.zeros((weight_cpu.shape[0], num_targets - weight_cpu.shape[1]))
                weight_cpu = torch.cat((weight_cpu, tail), dim=1)
            elif num_targets < weight_cpu.shape[1]:
                tail = torch.zeros((Y_cpu.shape[0], weight_cpu.shape[1] - num_targets))
                Y_cpu = torch.cat((Y_cpu, tail), dim=1)

            # --- 3. CPU: RLS Calculation ---
            # Dùng self.R_cpu (đã nằm sẵn trên RAM)
            P = self.R_cpu @ X_final_cpu.T
            
            term = X_final_cpu @ P
            term.diagonal().add_(1.0) 
            term = 0.5 * (term + term.T)
            
            # Nghịch đảo trên CPU (An toàn)
            K = torch.linalg.inv(term)
            del term
            
            P_K = P @ K 
            self.R_cpu -= P_K @ P.T
            del P
            
            residual = Y_cpu - (X_final_cpu @ weight_cpu)
            weight_cpu += P_K @ residual
            
            # --- 4. TRANSFER: CPU -> GPU ---
            # Đẩy weight kết quả về lại GPU
            self.weight = weight_cpu.to(self.device)
            
            del X_final_cpu, Y_cpu, K, P_K, residual, weight_cpu
            gc.collect()
            torch.cuda.empty_cache()

        finally:
            self.train(old_training_state)

    def forward(self, x, new_forward: bool = False):
        hyper_features = self.backbone(x)
        hyper_features = hyper_features.float()
        proj_features = self.buffer(hyper_features)
        logits = self.forward_fc(proj_features)
        return {'logits': logits}

    def forward_normal_fc(self, x, new_forward: bool = False):
        hyper_features = self.backbone(x)
        hyper_features = hyper_features.float()
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}