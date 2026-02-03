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
        return {
            'features': hyper_features,
            'logits': logits
        }


class RandomBuffer(torch.nn.Linear):
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
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
        self.class_covs = []       
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
        with autocast('cuda', enabled=False):
            X = self.backbone(X).float() 
            X = self.buffer(X) 
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device)
                self.weight = torch.cat((self.weight, tail), dim=1)
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
            del term, jitter, K, X, Y 

    # =========================================================================
    # [FeCAM INTEGRATION - ROBUST FP32 MODE]
    # =========================================================================
    
    def _tukeys_transform(self, x, beta=0.5):
        # [FIX NAN]: ReLU chặn số âm, Clamp chặn số vô cực
        x = F.relu(x)
        x = torch.clamp(x, min=0.0, max=1e5) 
        return torch.pow(x, beta)

    def _shrink_cov(self, cov):
        diag = torch.diagonal(cov)
        diag_mean = torch.mean(diag)
        sum_all = torch.sum(cov)
        sum_diag = torch.sum(diag)
        
        n = cov.shape[0]
        off_diag_mean = (sum_all - sum_diag) / (n * n - n) if n > 1 else 0.0
        
        alpha1, alpha2 = 0.01, 0.01 
        
        cov.add_(alpha2 * off_diag_mean)
        torch.diagonal(cov).add_(alpha1 * diag_mean - alpha2 * off_diag_mean)
        return cov

    def build_fecam_stats(self, train_loader):
        self.eval()
        print(f"--> [FeCAM] Building Statistics (Backbone D={self.backbone.out_dim})...")
        
        running_stats = {} 
        
        # [QUAN TRỌNG] Tắt Autocast khi build stats
        with torch.no_grad(), autocast(enabled=False):
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device).float() # Ép kiểu input về float32
                targets = targets.to(self.device)
                
                feats = self.backbone(inputs) 
                
                # Sanitize Features
                if torch.isnan(feats).any():
                    feats = torch.nan_to_num(feats)
                
                feats = self._tukeys_transform(feats)
                
                unique_labels = torch.unique(targets)
                for label in unique_labels:
                    label_item = label.item()
                    mask = (targets == label)
                    class_feats = feats[mask]
                    
                    if label_item not in running_stats:
                        D = class_feats.shape[1]
                        running_stats[label_item] = {
                            'sum_x': torch.zeros(D, device=self.device, dtype=torch.float32),
                            'sum_xxT': torch.zeros((D, D), device=self.device, dtype=torch.float32),
                            'n': 0
                        }
                    
                    running_stats[label_item]['sum_x'] += class_feats.sum(dim=0)
                    running_stats[label_item]['sum_xxT'].addmm_(class_feats.T, class_feats)
                    running_stats[label_item]['n'] += class_feats.shape[0]

        sorted_labels = sorted(running_stats.keys())
        if self.cur_task == 0:
            self.class_means = []
            self.class_covs = []
        
        for label in sorted_labels:
            stats = running_stats[label]
            n = stats['n']
            sum_x = stats['sum_x']
            sum_xxT = stats['sum_xxT']
            
            mean = sum_x / n
            if n > 1:
                term2 = torch.outer(sum_x, sum_x) / n
                cov = (sum_xxT - term2) / (n - 1)
            else:
                cov = torch.eye(mean.shape[0], device=self.device) * 1e-6
            
            if torch.isnan(cov).any() or torch.isinf(cov).any():
                print(f"[Warning] Covariance matrix for class {label} corrupted. Using Identity.")
                cov = torch.eye(mean.shape[0], device=self.device)

            cov = self._shrink_cov(cov)
            
            self.class_means.append(mean)
            self.class_covs.append(cov)
            
            del running_stats[label]
            
        print(f"--> [FeCAM] Stats Built. Total classes: {len(self.class_means)}")
        del running_stats
        torch.cuda.empty_cache()

    # [FIX] Tắt Autocast trong hàm inference này
    @torch.cuda.amp.autocast(enabled=False)
    def predict_fecam_internal(self, feats):
        """
        Robust Inference with FP32 enforcement
        """
        # Đảm bảo đầu vào là Float32
        feats = feats.float()
        
        feats = self._tukeys_transform(feats)
        dists = []
        JITTER = 1e-5 
        
        for c in range(len(self.class_means)):
            mean = self.class_means[c].float()
            cov = self.class_covs[c].float()
            
            diff = feats - mean.unsqueeze(0)
            
            try:
                cov_stable = cov + torch.eye(cov.shape[0], device=cov.device) * JITTER
                term = torch.linalg.solve(cov_stable, diff.T).T 
            except RuntimeError:
                try:
                    cov_cpu = cov.detach().cpu() + torch.eye(cov.shape[0]) * JITTER
                    inv_cov = torch.linalg.pinv(cov_cpu)
                    term = (diff.detach().cpu() @ inv_cov).to(self.device)
                except:
                    term = diff

            dist = torch.sum(diff * term, dim=1)
            dists.append(dist)
            
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
            # Inference: Ép kiểu FP32 trước khi vào FeCAM
            logits = self.predict_fecam_internal(hyper_features.float())
            
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
