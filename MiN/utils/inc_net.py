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
# [MODULE] DPCR Estimator: SỬA LỖI UNCENTERED & DRIFT
# =============================================================================
class DPCREstimator(nn.Module):
    def __init__(self, feature_dim, device):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        
        self.saved_covs = {} 
        self.saved_protos = {} 
        self.projectors = {} 

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
        Lưu Stats 768-dim Centered (Raw features)
        """
        model.eval()
        print(f"--> [DPCR] Scanning Stats (768-dim Centered)...")
        temp_features = {}
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                feats = model.extract_feature(inputs) # 768
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
            
            # Mean
            mean = torch.mean(features, dim=0)
            
            # Centered Covariance (Full Matrix)
            features_centered = features - mean.unsqueeze(0)
            cov = features_centered.T @ features_centered / (features.shape[0] - 1 + 1e-6)
            
            self.saved_protos[c] = mean
            self.saved_covs[c] = cov
            self.projectors[c] = self.get_projector_svd(cov)
            count += 1
        print(f"--> [DPCR] Stats Saved. Classes: {count}")
        del temp_features

    def correct_drift(self, old_model, new_model, current_loader, known_classes):
        """Tính P và sửa Stats cũ ngay trên 768-dim"""
        print(f"--> [DPCR] Calculating Drift Matrix P (768-dim)...")
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
                
                # Centering cục bộ
                feat_old = feat_old - feat_old.mean(dim=0, keepdim=True)
                feat_new = feat_new - feat_new.mean(dim=0, keepdim=True)
                
                cov_old += feat_old.T @ feat_old
                cross_corr += feat_old.T @ feat_new
                
                sample_count += inputs.shape[0]
                if sample_count > MAX_SAMPLES: break
        
        RG_TSSP = 0.01 
        epsilon = RG_TSSP * torch.eye(self.feature_dim, device=self.device)
        
        try:
            P_tssp = torch.linalg.solve(cov_old + epsilon, cross_corr)
        except:
            P_tssp = torch.inverse(cov_old + epsilon) @ cross_corr
        
        print(f"--> [DPCR] Drift Calculated. Correcting Stats...")
        
        # Cập nhật trực tiếp Stats cũ
        for c in range(known_classes):
            if c not in self.saved_covs: continue
            
            old_mean = self.saved_protos[c]
            old_cov = self.saved_covs[c]
            
            # Công thức biến đổi Mean và Covariance
            new_mean = old_mean @ P_tssp
            new_cov = P_tssp.T @ old_cov @ P_tssp
            
            self.saved_protos[c] = new_mean
            self.saved_covs[c] = new_cov
            self.projectors[c] = self.get_projector_svd(new_cov)

# ... (BaseIncNet, RandomBuffer giữ nguyên) ...
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

        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) 
        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R) 

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        
        # FeCAM Storage (Lưu 768-dim Stats)
        self.class_means = []
        self.class_covs_inv = []
        
        self.dpcr = DPCREstimator(self.feature_dim, self.device)

    # ... (Các hàm update_fc, update_noise, Analytic fit giữ nguyên) ...
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
    # [FeCAM IMPLEMENTATION] 768-dim + Tukey + Correlation Norm
    # =========================================================================

    def extract_feature(self, x):
        return self.backbone(x)

    def _tukey_transform(self, x, lambda_val=0.5):
        """Tukey's Ladder of Powers Transformation"""
        if lambda_val != 0:
            return torch.sign(x) * torch.pow(torch.abs(x) + 1e-6, lambda_val)
        else:
            return torch.log(torch.abs(x) + 1e-6)

    def compute_fecam_stats_direct(self, loader):
        """
        Dùng cho Task 0: Tính Stats TRỰC TIẾP từ dữ liệu (Chính xác cao).
        Apply Tukey -> Tính Mean/Cov -> Shrink -> Norm -> Invert.
        """
        self.eval()
        self.class_means = []
        self.class_covs_inv = []
        
        # 1. Gom tất cả features
        print("--> [FeCAM] Initializing Stats directly from Data (Task 0)...")
        temp_features = {}
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                feats = self.extract_feature(inputs).float() # Raw 768
                
                # Apply Tukey Transform NGAY LẬP TỨC
                feats = self._tukey_transform(feats, lambda_val=0.5)
                
                unique_labels = torch.unique(targets)
                for label in unique_labels:
                    label_item = label.item()
                    mask = (targets == label)
                    if label_item not in temp_features:
                        temp_features[label_item] = []
                    temp_features[label_item].append(feats[mask].cpu())
        
        # 2. Tính Stats
        sorted_classes = sorted(temp_features.keys())
        for c in sorted_classes:
            # Gom lại thành Tensor: [N, 768]
            feats_c = torch.cat(temp_features[c], dim=0).to(self.device)
            
            # Mean
            mean = torch.mean(feats_c, dim=0)
            
            # Covariance (Centered)
            feats_centered = feats_c - mean.unsqueeze(0)
            cov = feats_centered.T @ feats_centered / (feats_c.shape[0] - 1 + 1e-6)
            
            # Shrinkage (lam=0.5 chuẩn bài báo)
            lam = 0.5 
            identity = torch.eye(cov.shape[0], device=self.device)
            cov_shrunk = (1 - lam) * cov + lam * identity
            
            # Norm
            diag_std = torch.sqrt(torch.diag(cov_shrunk))
            outer_std = torch.outer(diag_std, diag_std)
            cov_norm = cov_shrunk / (outer_std + 1e-6)
            
            # Inverse
            try:
                cov_inv = torch.inverse(cov_norm + 1e-4 * identity)
            except:
                cov_inv = identity
            
            self.class_means.append(mean)
            self.class_covs_inv.append(cov_inv)
        
        print(f"--> [FeCAM] Initialized {len(self.class_means)} classes directly.")

    def update_fecam_stats_with_dpcr(self):
        """
        Dùng cho Task > 0: Generative Replay Stats.
        Pipeline: DPCR Stats -> Sample -> Tukey -> Mean/Cov -> Shrink -> Norm -> Invert.
        """
        print("--> [FeCAM] Syncing Stats with Generative Tukey Replay...")
        self.class_means = []
        self.class_covs_inv = []
        
        sorted_classes = sorted(self.dpcr.saved_protos.keys())
        NUM_SAMPLES = 2000
        
        with torch.no_grad():
            for c in sorted_classes:
                # 1. Get Raw Stats (from DPCR - drift corrected)
                raw_mean = self.dpcr.saved_protos[c].float()
                raw_cov = self.dpcr.saved_covs[c].float()
                
                # 2. Sample (Generative Replay in Raw Space)
                jitter = 1e-5 * torch.eye(raw_cov.shape[0], device=self.device)
                try:
                    dist = torch.distributions.MultivariateNormal(raw_mean, covariance_matrix=raw_cov + jitter)
                    samples = dist.sample((NUM_SAMPLES,))
                except:
                    samples = raw_mean.unsqueeze(0) + torch.randn(NUM_SAMPLES, self.feature_dim, device=self.device) * 0.1
                
                # 3. Apply Tukey Transform
                samples_trans = self._tukey_transform(samples, lambda_val=0.5)
                
                # 4. Compute New Stats in Transformed Space
                mean = torch.mean(samples_trans, dim=0)
                samples_centered = samples_trans - mean.unsqueeze(0)
                cov = samples_centered.T @ samples_centered / (NUM_SAMPLES - 1)
                
                # 5. Shrinkage & Norm
                lam = 0.5 
                identity = torch.eye(cov.shape[0], device=self.device)
                cov_shrunk = (1 - lam) * cov + lam * identity
                
                diag_std = torch.sqrt(torch.diag(cov_shrunk))
                outer_std = torch.outer(diag_std, diag_std)
                cov_norm = cov_shrunk / (outer_std + 1e-6)
                
                try:
                    cov_inv = torch.inverse(cov_norm + 1e-4 * identity)
                except:
                    cov_inv = identity
                
                self.class_means.append(mean)
                self.class_covs_inv.append(cov_inv)
                
        print(f"--> [FeCAM] Sync Done. Stats calculated on Tukey Space.")

    def predict_fecam_internal(self, inputs):
        """
        Inference FeCAM trên không gian 768-dim + Tukey.
        """
        with torch.no_grad(), autocast('cuda', enabled=False):
            if inputs.ndim == 4:
                feats = self.backbone(inputs)
            else:
                feats = inputs
            
            feats = feats.float() # [B, 768]
            
            # 1. Tukey's Transformation (CRITICAL)
            feats = self._tukey_transform(feats, lambda_val=0.5)
            
            dists = []
            for c in range(len(self.class_means)):
                mean = self.class_means[c]
                cov_inv = self.class_covs_inv[c]
                
                diff = feats - mean.unsqueeze(0)
                
                # Mahalanobis Distance: (x-u)^T * Sigma^-1 * (x-u)
                temp = torch.matmul(diff, cov_inv) 
                dist = torch.sum(temp * diff, dim=1)
                
                dists.append(dist)
                
        # Trả về negative distance
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
