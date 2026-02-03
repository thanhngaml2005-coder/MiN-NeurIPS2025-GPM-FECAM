import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
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
    # [FeCAM INTEGRATION - BACKBONE ONLY - ROBUST]
    # =========================================================================
    
    def _tukeys_transform(self, x, beta=0.5):
        # [FIX NAN]: ReLU để tránh căn bậc 2 số âm
        x = F.relu(x)
        return torch.pow(x, beta)

    def _shrink_cov(self, cov):
        """In-place shrinkage to save memory"""
        diag = torch.diagonal(cov)
        diag_mean = torch.mean(diag)
        sum_all = torch.sum(cov)
        sum_diag = torch.sum(diag)
        
        n = cov.shape[0]
        off_diag_mean = (sum_all - sum_diag) / (n * n - n) if n > 1 else 0.0
        
        alpha1, alpha2 = 0.01, 0.01 
        
        # In-place modifications
        cov.add_(alpha2 * off_diag_mean)
        torch.diagonal(cov).add_(alpha1 * diag_mean - alpha2 * off_diag_mean)
        return cov

    def build_fecam_stats(self, train_loader):
        """
        Build FeCAM stats on Backbone Features (D=768).
        Runs on GPU safely (2MB per class).
        """
        self.eval()
        print(f"--> [FeCAM] Building Statistics (Backbone D={self.backbone.out_dim})...")
        
        running_stats = {} 
        
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # [QUAN TRỌNG]: Lấy features từ Backbone (768 chiều), KHÔNG qua Buffer
                feats = self.backbone(inputs) 
                
                # Biến đổi Tukey (có ReLU)
