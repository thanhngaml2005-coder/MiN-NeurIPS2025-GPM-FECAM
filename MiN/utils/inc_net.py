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
import torch.nn.init as init
import gc

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

# =============================================================================
# 1. CLASS PiNoiseBiLoRA (Logic Task ID Tường minh)
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.init as init
import math
import copy
import gc

class PiNoiseBiLoRA(nn.Module):
    def __init__(self, in_dim, sparsity_ratio=0.10, hidden_dim=256):
        super(PiNoiseBiLoRA, self).__init__()
        self.in_dim = in_dim
        self.freq_dim = in_dim // 2 + 1
        
        # 1. Cấu hình kích thước (Đảm bảo k >= 1)
        self.k = max(1, int(self.freq_dim * sparsity_ratio))
        
        # Input cho MLP là (Real + Imag) của k tần số => k * 2
        self.mlp_in_dim = self.k * 2 
        self.hidden_dim = hidden_dim if hidden_dim else self.mlp_in_dim * 2
        
        # 2. Định nghĩa Noise Generator (Shared MLP)
        self.mu_net = nn.Sequential(
            nn.Linear(self.mlp_in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.mlp_in_dim)
        )
        
        self.sigma_net = nn.Sequential(
            nn.Linear(self.mlp_in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.mlp_in_dim)
        )

        # 3. Quản lý Task
        self.task_indices = []       
        self.current_task_id = -1 
        
        # [MEMORY OPTIMIZATION]: Running MagMax Storage (Chỉ lưu 1 bản gộp trên CPU)
        self.merged_mu_state = None    
        self.merged_sigma_state = None 
        
        # Buffer để track device
        self.register_buffer('dummy_buffer', torch.zeros(1))

    def reset_parameters(self):
        """Khởi tạo tối ưu: Zero cho Task 0, In-place Perturbation cho Task > 0"""
        if self.current_task_id <= 0:
            # --- TASK 0: Zero Init ---
            print(f"🚀 [PiNoise] Task {self.current_task_id}: Zero Initialization.")
            for name, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    is_last = ("mu_net" in name and str(len(self.mu_net)-1) in name) or \
                              ("sigma_net" in name and str(len(self.sigma_net)-1) in name)
                    
                    if is_last:
                        init.constant_(m.weight, 0)
                        if m.bias is not None: init.constant_(m.bias, 0)
                    else:
                        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                        if m.bias is not None: init.constant_(m.bias, 0)
        else:
            # --- TASK > 0: Warm-start ---
            print(f"🔄 [PiNoise] Task {self.current_task_id}: Warm-starting (In-place).")
            with torch.no_grad():
                for param in self.parameters():
                    # [FIX OOM]: Cộng trực tiếp, không tạo biến trung gian lớn
                    # Dùng randn (float32) * 0.001
                    param.add_(torch.randn_like(param), alpha=0.001)

    def _get_spectral_mask(self, task_id):
        start_freq = 1 
        available = torch.arange(start_freq, self.freq_dim)
        max_supported_tasks = 20 
        indices = available[task_id % max_supported_tasks :: max_supported_tasks]
        
        if len(indices) >= self.k:
            indices = indices[:self.k]
        else:
            needed = self.k - len(indices)
            repeat_factor = math.ceil(needed / len(indices))
            padding = indices.repeat(repeat_factor)[:needed]
            indices = torch.cat([indices, padding])
        return indices.long()

    def expand_new_task(self, target_task_id):
        """
        [LOGIC FIX]: Nhận target_task_id từ Controller.
        Ngăn chặn việc gọi lặp dẫn đến sai lệch index hoặc warm-start nhầm.
        """
        # Idempotency check: Nếu đã ở task này rồi thì thôi
        if target_task_id <= self.current_task_id:
            return

        self.current_task_id = target_task_id
        device = self.dummy_buffer.device
        
        new_indices = self._get_spectral_mask(self.current_task_id).to(device)
        self.task_indices.append(new_indices)
        
        # Reset parameters cho task mới (hoặc Zero hoặc Warm-start tùy ID)
        self.reset_parameters()

    def after_task_training(self):
        """
        [MEMORY OPTIMIZATION]: Running MagMax Merge.
        Gộp trọng số hiện tại vào bản merged trên CPU ngay lập tức.
        """
        # 1. Lấy snapshot hiện tại (đưa về CPU để giải phóng VRAM)
        current_mu = {k: v.detach().cpu().clone() for k, v in self.mu_net.state_dict().items()}
        current_sigma = {k: v.detach().cpu().clone() for k, v in self.sigma_net.state_dict().items()}
        
        # 2. Gộp vào bộ nhớ tích lũy
        self.merged_mu_state = self._update_running_magmax(self.merged_mu_state, current_mu)
        self.merged_sigma_state = self._update_running_magmax(self.merged_sigma_state, current_sigma)
        
        # 3. Load ngược lại vào model để chuẩn bị cho task sau
        # (Lúc này model trên GPU chứa tri thức tổng hợp)
        self._load_state_to_module(self.mu_net, self.merged_mu_state)
        self._load_state_to_module(self.sigma_net, self.merged_sigma_state)
        
        # Dọn dẹp biến tạm
        del current_mu, current_sigma
        gc.collect()

    def _update_running_magmax(self, merged_state, current_state):
        """
        So sánh và cập nhật trọng số lớn nhất (MagMax) trên CPU.
        """
        if merged_state is None:
            return copy.deepcopy(current_state)
        
        new_merged = {}
        for key in merged_state.keys():
            w_merged = merged_state[key]
            w_new = current_state[key]
            
            # Chọn giá trị có biên độ lớn hơn
            mask = torch.abs(w_new) > torch.abs(w_merged)
            w_updated = torch.where(mask, w_new, w_merged)
            new_merged[key] = w_updated
            
        return new_merged

    def _load_state_to_module(self, module, state_dict):
        """Helper load state từ CPU -> GPU Module"""
        if state_dict is None: return
        device_state = {k: v.to(module.dummy_buffer.device) for k, v in state_dict.items()}
        module.load_state_dict(device_state)

    def forward(self, x):
        if len(self.task_indices) == 0:
            return torch.zeros_like(x)

        device = x.device
        x_freq = torch.fft.rfft(x, dim=-1)
        total_freq_noise = torch.zeros_like(x_freq, dtype=torch.complex64)

        if self.training:
            curr_indices = self.task_indices[self.current_task_id].to(device)
            x_selected = x_freq[..., curr_indices]
            x_mlp_in = torch.cat([x_selected.real, x_selected.imag], dim=-1)
            
            mu = self.mu_net(x_mlp_in)
            sigma = self.sigma_net(x_mlp_in)
            
            # Reparameterization
            z = mu + torch.randn_like(mu) * sigma
            z_complex = torch.complex(z[..., :self.k], z[..., self.k:])
            
            total_freq_noise.index_add_(-1, curr_indices, z_complex)

        else:
            # Eval: Deterministic
            for indices in self.task_indices:
                indices = indices.to(device)
                x_selected = x_freq[..., indices]
                x_mlp_in = torch.cat([x_selected.real, x_selected.imag], dim=-1)
                
                # Chỉ lấy Mu
                mu_out = self.mu_net(x_mlp_in)
                z_complex = torch.complex(mu_out[..., :self.k], mu_out[..., self.k:])
                
                current_vals = total_freq_noise[..., indices]
                mask_better = z_complex.abs() > current_vals.abs()
                updated_vals = torch.where(mask_better, z_complex, current_vals)
                
                total_freq_noise.index_copy_(-1, indices, updated_vals)

        noise_spatial = torch.fft.irfft(total_freq_noise, n=self.in_dim, dim=-1)
        return noise_spatial
    
    def unfreeze_noise(self):
        for param in self.parameters(): param.requires_grad = True

    def freeze_noise(self):
        for param in self.parameters(): param.requires_grad = False
# 2. MiNbaseNet (Controller truyền ID)
# =============================================================================
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
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        
        # [FIX OOM 2]: Dùng Float32 thay vì Double
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)
        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Ép kiểu float32
        X = X.to(self.weight.dtype) 
        return F.relu(X @ self.W)

class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        gc.collect(); torch.cuda.empty_cache()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 
        
        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)
        
        # RLS dùng Float32
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) 
        R = torch.eye(self.buffer_size, **factory_kwargs) / self.gamma
        self.register_buffer("R", R) 
        
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    # ... (Các hàm update_fc, update_noise, unfreeze... giữ nguyên) ...

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        RLS Fit: Float32 Only.
        Không convert sang Double, không tạo bản sao thừa.
        """
        old_training_state = self.training
        self.eval() 
        try:
            with autocast('cuda', enabled=True): 
                X_feat = self.backbone(X)
            
            with autocast('cuda', enabled=False):
                # [FIX OOM 2]: Chỉ dùng Float32
                X_feat = X_feat.detach().float() # Detach và ép float32
                
                # Chiếu qua buffer (Float32 @ Float32 -> Nhẹ nhàng)
                X_proj = self.buffer(X_feat)
                
                # Giải phóng feature gốc ngay
                del X_feat
                
                device = self.weight.device
                X_final = X_proj.to(device)
                Y = Y.to(device).float()

                # Expand weights (Giữ nguyên logic)
                num_targets = Y.shape[1]
                if num_targets > self.weight.shape[1]:
                    increment_size = num_targets - self.weight.shape[1]
                    tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
                    self.weight = torch.cat((self.weight, tail), dim=1)
                elif num_targets < self.weight.shape[1]:
                    increment_size = self.weight.shape[1] - num_targets
                    tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
                    Y = torch.cat((Y, tail), dim=1)

                # RLS Math (Vẫn dùng Float32)
                # Tính toán từng bước để dễ xóa biến tạm
                P = self.R @ X_final.T
                
                term = X_final @ P
                term.diagonal().add_(1.0) 
                term = 0.5 * (term + term.T)
                
                try:
                    K = torch.linalg.inv(term)
                except RuntimeError:
                    print("⚠️ Switch CPU inv...")
                    K = torch.linalg.inv(term.cpu()).to(device)
                del term 
                
                P_K = P @ K 
                self.R -= P_K @ P.T
                del P 
                
                residual = Y - (X_final @ self.weight)
                self.weight += P_K @ residual
                
                del X_final, Y, K, P_K, residual
                torch.cuda.empty_cache()
        finally:
            self.train(old_training_state)

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
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
            self.normal_fc = new_fc
        else:
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None: nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc
            

    def update_noise(self):
        if hasattr(self.backbone, 'noise_maker'):
            print(f"--> [IncNet] Syncing BiLoRA Noise for Task {self.cur_task}")
            for j in range(len(self.backbone.noise_maker)):
                # [FIX]: Truyền cur_task vào để PiNoise biết chính xác đang ở đâu
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
        features = features.to(self.weight.dtype)
        return features @ self.weight

    
    def forward(self, x, new_forward: bool = False):
        hyper_features = self.backbone(x)
        hyper_features = hyper_features.double() 
        proj_features = self.buffer(hyper_features)
        proj_features = proj_features.float()
        logits = self.forward_fc(proj_features)
        return {'logits': logits}

    def forward_normal_fc(self, x, new_forward: bool = False):
        hyper_features = self.backbone(x)
        hyper_features = hyper_features.double()
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}