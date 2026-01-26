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
class PiNoiseBiLoRA(nn.Module):
    def __init__(self, in_dim, sparsity_ratio=0.10, hidden_dim=256):
        super(PiNoiseBiLoRA, self).__init__()
        self.in_dim = in_dim
        self.freq_dim = in_dim // 2 + 1
        
        self.k = max(1, int(self.freq_dim * sparsity_ratio))
        self.mlp_in_dim = self.k * 2 
        self.hidden_dim = hidden_dim
        
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

        self.task_indices = []       
        self.current_task_id = -1 
        self.history_mu = []         
        self.history_sigma = []
        
        self.register_buffer('dummy_buffer', torch.zeros(1))

    def reset_parameters(self):
        """Khởi tạo dựa trên Task ID hiện tại"""
        if self.current_task_id <= 0:
            # Task 0: Zero Init
            print(f"🚀 [PiNoise] Task {self.current_task_id}: Zero Initialization (Clean Start).")
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
            # Task > 0: Warm-start (In-place)
            print(f"🔄 [PiNoise] Task {self.current_task_id}: Warm-starting with in-place perturbation.")
            with torch.no_grad():
                for param in self.parameters():
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
        [FIX]: Nhận target_task_id từ Controller thay vì tự cộng += 1.
        Điều này ngăn chặn việc gọi lặp dẫn đến sai lệch index.
        """
        # Nếu đã ở task này rồi thì không làm gì cả (Idempotent)
        if target_task_id <= self.current_task_id:
            return

        self.current_task_id = target_task_id
        
        device = self.dummy_buffer.device
        new_indices = self._get_spectral_mask(self.current_task_id).to(device)
        self.task_indices.append(new_indices)
        
        self.reset_parameters()

    def after_task_training(self):
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
                max_indices = torch.argmax(torch.abs(all_versions), dim=0, keepdim=True)
                best_values = torch.gather(all_versions, 0, max_indices).squeeze(0)
                final_state[key] = best_values.to(module.dummy_buffer.device)
        module.load_state_dict(final_state)

    def forward(self, x):
        if len(self.task_indices) == 0: return torch.zeros_like(x)
        device = x.device
        x_freq = torch.fft.rfft(x, dim=-1)
        total_freq_noise = torch.zeros_like(x_freq, dtype=torch.complex64)

        if self.training:
            curr_indices = self.task_indices[self.current_task_id].to(device)
            x_selected = x_freq[..., curr_indices]
            x_mlp_in = torch.cat([x_selected.real, x_selected.imag], dim=-1)
            
            mu = self.mu_net(x_mlp_in)
            sigma = self.sigma_net(x_mlp_in)
            z = mu + torch.randn_like(mu) * sigma
            z_complex = torch.complex(z[..., :self.k], z[..., self.k:])
            total_freq_noise.index_add_(-1, curr_indices, z_complex)
        else:
            for indices in self.task_indices:
                indices = indices.to(device)
                x_selected = x_freq[..., indices]
                x_mlp_in = torch.cat([x_selected.real, x_selected.imag], dim=-1)
                mu_out = self.mu_net(x_mlp_in)
                z_complex = torch.complex(mu_out[..., :self.k], mu_out[..., self.k:])
                
                current_vals = total_freq_noise[..., indices]
                mask_better = z_complex.abs() > current_vals.abs()
                updated_vals = torch.where(mask_better, z_complex, current_vals)
                total_freq_noise.index_copy_(-1, indices, updated_vals)

        return torch.fft.irfft(total_freq_noise, n=self.in_dim, dim=-1)

# =============================================================================
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
        factory_kwargs = {"device": device, "dtype": torch.double}
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)
        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight) 
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
        
        # RLS (Float32)
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) 
        R = torch.eye(self.buffer_size, **factory_kwargs) / self.gamma
        self.register_buffer("R", R) 
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

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
            
        # Tự động gọi update_noise đúng với cur_task hiện tại
        self.update_noise()

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

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """RLS Fit Optimized"""
        old_training_state = self.training
        self.eval() 
        try:
            with autocast('cuda', enabled=True): 
                X_feat = self.backbone(X)
            
            with autocast('cuda', enabled=False):
                # Double for Buffer
                X_double = X_feat.detach().double()
                del X_feat 
                X_proj = self.buffer(X_double) 
                del X_double
                
                # Float for RLS
                X_final = X_proj.float() 
                del X_proj
                
                device = self.weight.device
                X_final = X_final.to(device)
                Y = Y.to(device).float()

                num_targets = Y.shape[1]
                if num_targets > self.weight.shape[1]:
                    increment_size = num_targets - self.weight.shape[1]
                    tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
                    self.weight = torch.cat((self.weight, tail), dim=1)
                elif num_targets < self.weight.shape[1]:
                    increment_size = self.weight.shape[1] - num_targets
                    tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
                    Y = torch.cat((Y, tail), dim=1)

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