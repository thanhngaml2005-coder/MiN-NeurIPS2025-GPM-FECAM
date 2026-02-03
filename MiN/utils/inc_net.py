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

# =============================================================================
# [MODULE] DPCR Estimator: Bộ sửa lỗi trôi đặc trưng (Feature Drift Corrector)
# Hoạt động trên không gian gốc (768 dim) để tránh OOM
# =============================================================================
class DPCREstimator(nn.Module):
    def __init__(self, feature_dim, device):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        
        # Lưu trữ Covariance và Prototype (Mean) ở không gian gốc (768 dim)
        self.saved_covs = {} 
        self.saved_protos = {} 
        self.projectors = {} # Lưu SVD Projector (CIP)

    def get_projector_svd(self, raw_matrix):
        """Tính SVD để lấy ma trận chiếu CIP"""
        try:
            U, S, V = torch.svd(raw_matrix + 1e-4 * torch.eye(raw_matrix.shape[0], device=self.device))
        except:
            return torch.eye(raw_matrix.shape[0], device=self.device)
            
        non_zeros = torch.where(S > 1e-5)[0]
        if len(non_zeros) == 0:
            return torch.eye(raw_matrix.shape[0], device=self.device)
            
        left_vecs = U[:, non_zeros]
        projector = left_vecs @ left_vecs.T
        return projector

    def update_stats(self, model, loader):
        """
        [FIXED] Tự động quét và lưu stats cho TẤT CẢ class trong loader.
        Đã thêm bước Centering Features để tính Covariance đúng.
        """
        model.eval()
        print(f"--> [DPCR] Scanning & Saving Stats (Auto-detect classes)...")
        
        temp_features = {}
        
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                feats = model.extract_feature(inputs) # [B, 768]
                
                unique_labels = torch.unique(targets)
                for label in unique_labels:
                    label_item = label.item()
                    mask = (targets == label)
                    
                    if label_item not in temp_features:
                        temp_features[label_item] = []
                    
                    temp_features[label_item].append(feats[mask].cpu())

        count = 0
        for c, feat_list in temp_features.items():
            if len(feat_list) == 0: continue
            
            features = torch.cat(feat_list, dim=0).to(self.device).float()
            
            # Tính Mean
            mean = torch.mean(features, dim=0)
            
            # [FIX CRITICAL] Centering Features: Trừ Mean trước khi nhân ma trận
            # Covariance = (X - Mu)^T @ (X - Mu)
            features_centered = features - mean.unsqueeze(0)
            cov = features_centered.T @ features_centered
            
            self.saved_protos[c] = mean
            self.saved_covs[c] = cov
            self.projectors[c] = self.get_projector_svd(cov)
            count += 1
            
        print(f"--> [DPCR] Stats Saved. Total classes found: {count}")
        del temp_features

    def correct_drift(self, old_model, new_model, current_loader, known_classes):
        """Tính toán ma trận Drift và sửa lại Stats của Class CŨ"""
        print(f"--> [DPCR] Calculating Drift Matrix & Correcting {known_classes} old classes...")
        old_model.eval()
        new_model.eval()
        
        cov_old = torch.zeros(self.feature_dim, self.feature_dim).to(self.device)
        cross_corr = torch.zeros(self.feature_dim, self.feature_dim).to(self.device)
        
        sample_count = 0
        MAX_SAMPLES = 2000 
        
        with torch.no_grad():
            for _, inputs, _ in current_loader:
                inputs = inputs.to(self.device)
                
                feat_old = old_model.extract_feature(inputs).float() 
                feat_new = new_model.extract_feature(inputs).float() 
                
                cov_old += feat_old.T @ feat_old
                cross_corr += feat_old.T @ feat_new
                
                sample_count += inputs.shape[0]
                if sample_count > MAX_SAMPLES: break
        
        epsilon = 1e-4 * torch.eye(self.feature_dim, device=self.device)
        try:
            P_tssp = torch.linalg.solve(cov_old + epsilon, cross_corr)
        except:
            P_tssp = torch.inverse(cov_old + epsilon) @ cross_corr
        
        corrected_count = 0
        for c in range(known_classes):
            if c not in self.saved_covs: continue
            
            W_c = P_tssp @ self.projectors[c]
            
            self.saved_covs[c] = W_c.T @ self.saved_covs[c] @ W_c
            self.saved_protos[c] = self.saved_protos[c] @ W_c
            self.projectors[c] = self.get_projector_svd(self.saved_covs[c])
            corrected_count += 1
            
        print(f"--> [DPCR] Correction Done for {corrected_count} classes.")


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
        
        # --- DPCR Module ---
        self.dpcr = DPCREstimator(self.feature_dim, self.device)

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

    # =========================================================================
    # [FeCAM + DPCR INTEGRATION]
    # =========================================================================

    def extract_feature(self, x):
        return self.backbone(x)

    def _robust_transform(self, x, beta=0.5):
        # Power + LayerNorm chuẩn hóa
        x = torch.sign(x) * torch.pow(torch.abs(x) + 1e-6, beta)
        x = F.layer_norm(x, x.shape[-1:])
        return x

    def update_fecam_stats_with_dpcr(self):
        """
        Đồng bộ dữ liệu: Chuyển stats đã sửa từ DPCR (768) sang FeCAM (16k).
        """
        print("--> [FeCAM] Syncing Stats from DPCR corrected memory...")
        self.class_means = []
        self.class_vars = []
        
        sorted_classes = sorted(self.dpcr.saved_protos.keys())
        
        self.buffer.use_relu = False # Tắt ReLU để lấy linear projection
        
        with torch.no_grad():
            for c in sorted_classes:
                mean_768 = self.dpcr.saved_protos[c].float()
                
                # Chiếu Mean qua Buffer (16k dim)
                mean_16k = self.buffer(mean_768.unsqueeze(0)).squeeze(0)
                
                # Robust Transform
                mean_16k = self._robust_transform(mean_16k.unsqueeze(0)).squeeze(0)
                
                # FeCAM Hybrid (Nearest Mean): Set Var = 1
                var_16k = torch.ones_like(mean_16k)
                
                self.class_means.append(mean_16k)
                self.class_vars.append(var_16k)
                
        self.buffer.use_relu = True
        print(f"--> [FeCAM] Sync Done. Classes: {len(self.class_means)}")

    def predict_fecam_internal(self, inputs):
        """FeCAM Inference"""
        self.buffer.use_relu = False
        with torch.no_grad(), autocast('cuda', enabled=False):
            if inputs.ndim == 4:
                feats = self.backbone(inputs)
            else:
                feats = inputs
            
            feats = feats.float()
            feats = self.buffer(feats)
            feats = self._robust_transform(feats)
            
            dists = []
            for c in range(len(self.class_means)):
                mean = self.class_means[c]
                var = self.class_vars[c]
                
                diff_sq = (feats - mean.unsqueeze(0)) ** 2
                
                # [FIX SCALE] Chia cho sqrt(D) để giảm magnitude của khoảng cách
                D = feats.shape[1]
                dist = torch.sum(diff_sq / (var + 1e-6), dim=1) / math.sqrt(D)
                
                dists.append(dist)
                
        self.buffer.use_relu = True
        # Trả về số âm: Distance nhỏ -> Logits lớn
        return -torch.stack(dists, dim=1)

    # --- Các hàm khác ---
    def forward(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        logits = self.forward_fc(self.buffer(hyper_features.to(self.weight.dtype)))
        return {'logits': logits}

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
