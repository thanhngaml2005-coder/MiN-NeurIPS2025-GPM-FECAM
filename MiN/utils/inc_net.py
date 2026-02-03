import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear

# Xử lý tương thích phiên bản PyTorch cho Autocast
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
        return {'features': hyper_features, 'logits': logits}


class RandomBuffer(nn.Module):
    def __init__(self, in_features: int, buffer_size: int, device):
        super(RandomBuffer, self).__init__()
        self.in_features = in_features
        self.out_features = buffer_size
        
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)
        
        self.use_relu = True 
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Dùng Normal Init để đúng bản chất Random Projection (JL Lemma)
        nn.init.normal_(self.W, mean=0.0, std=1.0 / math.sqrt(self.in_features))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight.dtype)
        out = X @ self.W
        return F.relu(out) if self.use_relu else out


class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        # Random Buffer
        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        # Analytic Parameters
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) 
        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R) 

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        
        # --- FeCAM Storage ---
        self.class_means = []      
        self.class_vars = []  
        self.use_fecam = True      

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
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].after_task_training()

    def unfreeze_noise(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_incremental()

    def init_unfreeze(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_task_0()
            if hasattr(self.backbone.blocks[j], 'norm1'):
                for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            if hasattr(self.backbone.blocks[j], 'norm2'):
                for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            for p in self.backbone.norm.parameters(): p.requires_grad = True

    # =========================================================================
    # [ANALYTIC LEARNING]
    # =========================================================================

    def forward_fc(self, features):
        features = features.to(self.weight.dtype) 
        return features @ self.weight

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.buffer.use_relu = True 
        
        with autocast('cuda', enabled=False):
            X = self.backbone(X).float() 
            X = self.buffer(X) 
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device)
                new_weight = torch.cat((self.weight, tail), dim=1)
                self.weight = new_weight 
                
            elif num_targets < self.weight.shape[1]:
                increment_size = self.weight.shape[1] - num_targets
                tail = torch.zeros((Y.shape[0], increment_size), device=Y.device)
                Y = torch.cat((Y, tail), dim=1)

            term = torch.eye(X.shape[0], device=X.device) + X @ self.R @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            try:
                K = torch.linalg.solve(term + jitter, X @ self.R).T
            except:
                K = self.R @ X.T @ torch.inverse(term + jitter)

            self.R -= K @ X @ self.R
            self.weight += K @ (Y - X @ self.weight)
            
            self.buffer.use_relu = True
            del term, jitter, K, X, Y 

    # =========================================================================
    # [FeCAM: DIAGONAL + LINEAR BUFFER + ROBUST STATS]
    # =========================================================================
    
    def _robust_transform(self, x, beta=0.5):
        # Power Transform: Thêm epsilon để tránh gradient nổ tại 0
        x = torch.sign(x) * torch.pow(torch.abs(x) + 1e-6, beta)
        
        # LayerNorm: Chuẩn PyTorch (ổn định hơn tự code)
        x = F.layer_norm(x, x.shape[-1:])
        return x

    def build_fecam_stats(self, train_loader):
        self.eval()
        print(f"--> [FeCAM] Building Diagonal Stats (Linear Buffer D={self.buffer_size})...")
        
        self.buffer.use_relu = False
        running_stats = {} 
        
        with torch.no_grad(), autocast('cuda', enabled=False):
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device)
                
                feats = self.backbone(inputs)
                if torch.isnan(feats).any(): feats = torch.nan_to_num(feats)
                
                feats = self.buffer(feats) 
                feats = self._robust_transform(feats)
                
                unique_labels = torch.unique(targets)
                for label in unique_labels:
                    label_item = label.item()
                    mask = (targets == label)
                    class_feats = feats[mask]
                    
                    if label_item not in running_stats:
                        D = class_feats.shape[1]
                        running_stats[label_item] = {
                            'sum_x': torch.zeros(D, device=self.device, dtype=torch.float32),
                            'sum_sq_x': torch.zeros(D, device=self.device, dtype=torch.float32),
                            'n': 0
                        }
                    
                    # Accumulate Sum(X)
                    running_stats[label_item]['sum_x'] += class_feats.sum(dim=0)
                    
                    # [OPTIMIZATION] Dùng phép nhân thay vì pow(2) để tối ưu VRAM/Speed
                    running_stats[label_item]['sum_sq_x'] += torch.sum(class_feats * class_feats, dim=0)
                    
                    running_stats[label_item]['n'] += class_feats.shape[0]

        sorted_labels = sorted(running_stats.keys())
        if self.cur_task == 0:
            self.class_means = []
            self.class_vars = []
        
        for label in sorted_labels:
            stats = running_stats[label]
            n = stats['n']
            sum_x = stats['sum_x']
            sum_sq_x = stats['sum_sq_x']
            
            mean = sum_x / n
            
            if n > 1:
                var = (sum_sq_x / n) - (mean ** 2)
                var = var * (n / (n - 1)) 
            else:
                var = torch.ones_like(mean)
            
            var = torch.clamp(var, min=1e-4)
            mean_var = torch.mean(var)
            alpha = 0.1
            var = (1 - alpha) * var + alpha * mean_var
            
            self.class_means.append(mean)
            self.class_vars.append(var)
            del running_stats[label]
            
        self.buffer.use_relu = True
        print(f"--> [FeCAM] Stats Built. Total classes: {len(self.class_means)}")
        del running_stats
        torch.cuda.empty_cache()

    def predict_fecam_internal(self, feats):
        with autocast('cuda', enabled=False):
            self.buffer.use_relu = False
            
            feats = feats.float()
            feats = self.buffer(feats) 
            feats = self._robust_transform(feats)
            
            dists = []
            
            for c in range(len(self.class_means)):
                mean = self.class_means[c].float()
                var = self.class_vars[c].float()
                
                diff_sq = (feats - mean.unsqueeze(0)) ** 2
                
                # Thêm epsilon khi chia để an toàn tuyệt đối
                dist = torch.sum(diff_sq / (var + 1e-6), dim=1)
                dists.append(dist)
            
            self.buffer.use_relu = True
            return -torch.stack(dists, dim=1)

    # =========================================================================
    # [FORWARD PASSES]
    # =========================================================================

    def forward(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        if self.training or not self.use_fecam or len(self.class_means) == 0:
            hyper_features_fp32 = hyper_features.to(self.weight.dtype)
            features_buffer = self.buffer(hyper_features_fp32)
            logits = self.forward_fc(features_buffer)
        else:
            logits = self.predict_fecam_internal(hyper_features)
            
        return {'logits': logits}

    def extract_feature(self, x):
        return self.backbone(x)

    def forward_normal_fc(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = self.buffer(hyper_features.to(self.buffer.weight.dtype))
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}

    def collect_projections(self, mode='threshold', val=0.95):
        print(f"--> [IncNet] Collecting Projections (Mode: {mode}, Val: {val})...")
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].compute_projection_matrix(mode=mode, val=val)

    def apply_gpm_to_grads(self, scale=1.0):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].apply_gradient_projection(scale=scale)
