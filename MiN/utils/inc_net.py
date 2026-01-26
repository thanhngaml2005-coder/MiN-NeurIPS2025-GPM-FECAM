import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
# Giả sử bạn để file backbone ở đường dẫn này
from backbones.pretrained_backbone import get_pretrained_backbone 
from backbones.linears import SimpleLinear
from torch.nn import functional as F
import torch.nn.init as init

# Xử lý Autocast cho các phiên bản Pytorch khác nhau
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

# =============================================================================
# 1. CLASS PiNoiseBiLoRA (Tích hợp BiLoRA + MIN)
# =============================================================================
class PiNoiseBiLoRA(nn.Module):
    def __init__(self, in_dim, sparsity_ratio=0.15, hidden_dim=None):
        """
        Implementation của BiLoRA (Frequency Domain) cho cơ chế sinh Noise của MIN.
        """
        super(PiNoiseBiLoRA, self).__init__()
        self.in_dim = in_dim
        # RFFT output shape: (N // 2) + 1
        self.freq_dim = in_dim // 2 + 1
        
        # 1. Cấu hình kích thước (Sparsity)
        self.k = int(self.freq_dim * sparsity_ratio) 
        self.mlp_in_dim = self.k * 2 
        self.hidden_dim = hidden_dim if hidden_dim else self.mlp_in_dim * 2
        
        # 2. Định nghĩa Noise Generator (Shared MLP)
        # Thay thế cho Down/Up projection của MIN gốc
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

        # 3. Quản lý Task & History
        self.task_indices = []       
        self.current_task_id = -1 
        self.history_mu = []         
        self.history_sigma = []
        
        # Buffer tạm
        self.register_buffer('dummy_buffer', torch.zeros(1))

    def reset_parameters(self):
        """Khởi tạo: Random nhỏ cho Task 0, Warm-start + Perturbation cho Task > 0"""
        if self.current_task_id <= 0:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=0.002) 
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
        else:
            # Warm-start: Giữ weight đã merge, thêm nhiễu nhẹ
            # print(f"   -> [PiNoise] Perturbing weights for adaptation...")
            with torch.no_grad():
                for param in self.parameters():
                    noise = torch.randn_like(param) * 0.01 
                    param.add_(noise)

    def _get_spectral_mask(self, task_id):
        """Tạo mask tần số trực giao (Interleaved Sampling)"""
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

    def expand_new_task(self):
        """Gọi khi bắt đầu Task mới"""
        self.current_task_id += 1
        device = self.dummy_buffer.device
        
        # Tạo mask mới
        new_indices = self._get_spectral_mask(self.current_task_id).to(device)
        self.task_indices.append(new_indices)
        
        # Reset weights (Warm-start)
        self.reset_parameters()

    def after_task_training(self):
        """Gọi sau khi kết thúc Task để Merge weights"""
        mu_state = {k: v.detach().cpu().clone() for k, v in self.mu_net.state_dict().items()}
        sigma_state = {k: v.detach().cpu().clone() for k, v in self.sigma_net.state_dict().items()}
        self.history_mu.append(mu_state)
        self.history_sigma.append(sigma_state)
        
        self._perform_parameter_magmax(self.mu_net, self.history_mu)
        self._perform_parameter_magmax(self.sigma_net, self.history_sigma)

    def _perform_parameter_magmax(self, module, history_list):
        if not history_list: return
        base_state = history_list[0]
        final_state = {}
        with torch.no_grad():
            for key in base_state.keys():
                all_versions = torch.stack([h[key] for h in history_list], dim=0)
                magnitudes = torch.abs(all_versions)
                max_indices = torch.argmax(magnitudes, dim=0, keepdim=True)
                best_values = torch.gather(all_versions, 0, max_indices).squeeze(0)
                final_state[key] = best_values
        module.load_state_dict(final_state)

    def forward(self, x):
        """Trả về NOISE (Không cộng x ở đây để dễ tích hợp với Residual gốc)"""
        if len(self.task_indices) == 0:
            return torch.zeros_like(x)

        device = x.device
        x_freq = torch.fft.rfft(x, dim=-1)
        total_freq_noise = torch.zeros_like(x_freq, dtype=torch.complex64)

        if self.training:
            # --- Training: Chỉ dùng Mask của Task hiện tại + Stochastic ---
            curr_indices = self.task_indices[self.current_task_id].to(device)
            x_selected = x_freq[..., curr_indices]
            x_mlp_in = torch.cat([x_selected.real, x_selected.imag], dim=-1)
            
            mu = self.mu_net(x_mlp_in)
            sigma = self.sigma_net(x_mlp_in)
            
            epsilon = torch.randn_like(mu)
            z = mu + epsilon * sigma
            
            z_complex = torch.complex(z[..., :self.k], z[..., self.k:])
            total_freq_noise.index_add_(-1, curr_indices, z_complex)

        else:
            # --- Eval: Deterministic & Competition (Thay cho Noise Mixture) ---
            for indices in self.task_indices:
                indices = indices.to(device)
                x_selected = x_freq[..., indices]
                x_mlp_in = torch.cat([x_selected.real, x_selected.imag], dim=-1)
                
                mu_out = self.mu_net(x_mlp_in) # Chỉ dùng Muy
                z_complex = torch.complex(mu_out[..., :self.k], mu_out[..., self.k:])
                
                current_vals = total_freq_noise[..., indices]
                # MagMax Activation: Chọn noise mạnh nhất tại mỗi tần số
                mask_better = z_complex.abs() > current_vals.abs()
                updated_vals = torch.where(mask_better, z_complex, current_vals)
                
                total_freq_noise.index_copy_(-1, indices, updated_vals)

        # IFFT về lại miền không gian
        noise_spatial = torch.fft.irfft(total_freq_noise, n=self.in_dim, dim=-1)
        return noise_spatial


# =============================================================================
# 2. BASE NETWORKS
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
        return {
            'features': hyper_features,
            'logits': logits
        }
# inc_net.py

class RandomBuffer(torch.nn.Linear):
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        
        # --- FIX: Đổi torch.double thành torch.float32 ---
        factory_kwargs = {"device": device, "dtype": torch.float32} 
        
        # Khởi tạo buffer với float32
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)
        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
        return F.relu(X @ self.W)


# =============================================================================
# 3. MAIN NETWORK (MiNbaseNet - Controller)
# =============================================================================
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

        # Random Buffer (Projector)
        self.buffer = RandomBuffer(in_features=self.feature_dim, 
                                   buffer_size=self.buffer_size, 
                                   device=self.device)

        # RLS Weights (Analytic Classifier)
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
        
        # 1. Cập nhật Normal FC (cho SGD training của PiNoise)
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
            
        # 2. TRIGGER UPDATE NOISE (Quan trọng cho BiLoRA)
        # Tự động tạo Mask mới và reset/perturb weights cho task mới
        self.update_noise()

    def update_noise(self):
        """Duyệt qua tất cả các lớp PiNoise trong backbone và expand task"""
        # Lưu ý: backbone phải có thuộc tính 'noise_maker' (list các PiNoiseBiLoRA)
        if hasattr(self.backbone, 'noise_maker'):
            print(f"--> [IncNet] Expanding BiLoRA Noise for Task {self.cur_task}")
            for j in range(len(self.backbone.noise_maker)):
                self.backbone.noise_maker[j].expand_new_task()
        else:
            print("⚠️ Warning: Backbone does not have 'noise_maker'. PiNoise logic skipped.")

    def after_task_magmax_merge(self):
        """Gọi hàm này sau khi train xong 1 task để merge weights"""
        print(f"--> [IncNet] Task {self.cur_task}: Triggering Parameter-wise MagMax Merging...")
        if hasattr(self.backbone, 'noise_maker'):
            for j in range(len(self.backbone.noise_maker)):
                 self.backbone.noise_maker[j].after_task_training()

    def unfreeze_noise(self):
        # Mở khóa gradient cho noise
        if hasattr(self.backbone, 'noise_maker'):
            for j in range(len(self.backbone.noise_maker)):
                for param in self.backbone.noise_maker[j].parameters():
                    param.requires_grad = True

    def init_unfreeze(self):
        # Mở khóa các lớp cần thiết ban đầu
        self.unfreeze_noise()
        
        # Mở khóa LayerNorms (Thường dùng trong ViT Continual Learning)
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
        # Classifier chính thức (RLS Weight)
        return features @ self.weight

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Huấn luyện Analytic Classifier (RLS).
        Bắt buộc chuyển sang EVAL mode để PiNoise dùng Merged Weights (Deterministic).
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
                if num_targets > self.weight.shape[1]:
                    increment_size = num_targets - self.weight.shape[1]
                    tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
                    self.weight = torch.cat((self.weight, tail), dim=1)
                elif num_targets < self.weight.shape[1]:
                    increment_size = self.weight.shape[1] - num_targets
                    tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
                    Y = torch.cat((Y, tail), dim=1)

                # --- RLS CORE ALGORITHM ---
                # P = R * X^T
                P = self.R @ X_feat.T
                
                # Term = X * P = X * R * X^T
                term = X_feat @ P
                # Regularization (Dampening factor)
                term.diagonal().add_(1.0) 
                # Symmetrization
                term = 0.5 * (term + term.T)
                
                # Invert (K = term^-1)
                try:
                    K = torch.linalg.inv(term)
                except RuntimeError:
                    print("⚠️ GPU OOM during RLS inversion, switching to CPU...")
                    K = torch.linalg.inv(term.cpu()).to(device)
                
                del term 
                
                # Update R: R_new = R - P * K * P^T
                P_K = P @ K 
                self.R -= P_K @ P.T
                del P 
                
                # Update Weights: W_new = W_old + P * K * (Y - X * W_old)
                residual = Y - (X_feat @ self.weight)
                self.weight += P_K @ residual
                
                del X_feat, Y, K, P_K, residual
                torch.cuda.empty_cache()
        
        finally:
            # Trả lại trạng thái cũ
            self.train(old_training_state)

    def forward(self, x, new_forward: bool = False):
        # Hàm forward tổng quát cho Inference
        # Nếu backbone hỗ trợ new_forward (cho task mới nhất), truyền cờ này vào
        # Nhưng với cơ chế PiNoise tự động, thường chỉ cần gọi backbone(x)
        hyper_features = self.backbone(x)
        hyper_features = hyper_features.to(self.weight.dtype)
        
        # Qua buffer rồi nhân với RLS Weights
        logits = self.forward_fc(self.buffer(hyper_features))
        
        return {'logits': logits}

    def forward_normal_fc(self, x, new_forward: bool = False):
        # Hàm forward dành riêng cho lúc Training PiNoise (dùng SGD)
        # Lúc này backbone đang ở mode Train -> PiNoise sinh Stochastic Noise
        hyper_features = self.backbone(x)
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        
        # Dùng Normal FC (có bias, đang học)
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}