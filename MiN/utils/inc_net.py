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

# =============================================================================
# [MODULE] DPCR Estimator: CHỈ GIỮ LOGIC SỬA MEAN (PROTOTYPES)
# =============================================================================
class DPCREstimator(nn.Module):
    def __init__(self, feature_dim, device):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        
        # Chỉ cần lưu Mean (Prototypes)
        self.saved_protos = {} 
        # Vẫn cần lưu Cov tạm thời để tính Drift Matrix P, nhưng không dùng để predict
        self.saved_covs = {} 

    def update_stats(self, model, loader):
        """Quét và lưu Stats cơ bản"""
        model.eval()
        print(f"--> [DPCR] Scanning Stats...")
        temp_features = {}
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                feats = model.extract_feature(inputs)
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
            
            # Vẫn tính Covariance để phục vụ thuật toán DPCR (tính P)
            features_centered = features - mean.unsqueeze(0)
            cov = features_centered.T @ features_centered / (features.shape[0] - 1 + 1e-6)
            
            self.saved_protos[c] = mean
            self.saved_covs[c] = cov
            count += 1
        print(f"--> [DPCR] Stats Saved. Classes: {count}")

    def correct_drift(self, old_model, new_model, current_loader, known_classes):
        """Tính P và Sửa lỗi Mean cũ"""
        print(f"--> [DPCR] Calculating Drift Matrix P...")
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
                
                # Normalize features (Quan trọng để tính P ổn định)
                feat_old = F.normalize(feat_old, p=2, dim=1)
                feat_new = F.normalize(feat_new, p=2, dim=1)
                
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
        
        print(f"--> [DPCR] Drift Calculated. Correcting Prototypes...")
        
        # CHỈ CẦN SỬA MEAN
        for c in range(known_classes):
            if c not in self.saved_protos: continue
            
            old_mean = self.saved_protos[c]
            
            # Công thức biến đổi Mean: u_new = u_old * P
            new_mean = old_mean @ P_tssp
            
            # Lưu lại Mean mới (Mean cũ bị đè)
            self.saved_protos[c] = new_mean
            
            # (Không cần sửa Covariance vì ta không dùng FeCAM nữa)

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
        
        # DPCR Module
        self.dpcr = DPCREstimator(self.feature_dim, self.device)

    # ... (Các hàm update_fc, update_noise, Analytic fit giữ nguyên không đổi) ...
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
    # [THAY THẾ FeCAM BẰNG SIMPLE NCM (EUCLIDEAN/COSINE)]
    # =========================================================================

    def extract_feature(self, x):
        return self.backbone(x)

    # Không cần update_fecam_stats_with_dpcr nữa
    # DPCR đã tự lưu Mean vào self.dpcr.saved_protos rồi

    def predict_ncm_simple(self, inputs):
        """
        Dùng khoảng cách Cosine đến các Mean đã được DPCR sửa lỗi.
        Nhanh, Gọn, Ổn định.
        """
        with torch.no_grad(), autocast('cuda', enabled=False):
            if inputs.ndim == 4:
                feats = self.backbone(inputs)
            else:
                feats = inputs
            
            feats = feats.float() # [B, 768]
            # Normalize query features (cho Cosine Distance)
            feats = F.normalize(feats, p=2, dim=1)
            
            # Gom Mean của các class lại
            sorted_classes = sorted(self.dpcr.saved_protos.keys())
            if len(sorted_classes) == 0:
                # Fallback nếu chưa có class nào
                return torch.zeros(feats.shape[0], 0, device=self.device)

            prototypes = []
            for c in sorted_classes:
                # Lấy mean từ DPCR
                proto = self.dpcr.saved_protos[c].float()
                # Normalize prototypes (cho Cosine Distance)
                proto = F.normalize(proto, p=2, dim=0)
                prototypes.append(proto)
            
            # [C_total, 768]
            prototypes = torch.stack(prototypes, dim=0)
            
            # Tính Cosine Similarity: X @ W.T
            # [B, 768] @ [768, C] -> [B, C]
            logits = torch.matmul(feats, prototypes.T)
            
            # Logits này chính là Cosine Similarity (-1 đến 1)
            # Có thể nhân với một Temperature nếu cần (ví dụ * 10) để Softmax nhọn hơn
            return logits * 10.0

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
