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
    """
    Lớp mở rộng đặc trưng ngẫu nhiên (Random Projection).
    """
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        
        # [QUAN TRỌNG] Sử dụng float32 để đảm bảo độ chính xác khi tính RLS
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Ép kiểu input X về cùng kiểu với weight (float32)
        X = X.to(self.weight.dtype)
        return F.relu(X @ self.W)


class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        
        # Các tham số cho Analytic Learning (RLS)
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        # Random Buffer
        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        # Khởi tạo ma trận trọng số và ma trận hiệp phương sai cho RLS
        # Dùng float32 để tránh lỗi singular matrix khi tính nghịch đảo
        factory_kwargs = {"device": self.device, "dtype": torch.float32}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) # Trọng số của Analytic Classifier

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R) # Ma trận hiệp phương sai đảo (Inverse Covariance Matrix)

        # Normal FC: Dùng để train Gradient Descent cho Noise Generator
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        
        # --- FeCAM Storage ---
        self.class_means = []      # List chứa prototypes (Mu)
        self.class_covs = []       # List chứa covariance matrices (Sigma)
        self.use_fecam = True      # Flag bật tắt FeCAM

    def update_fc(self, nb_classes):
        """
        Cập nhật lớp Normal FC (cho việc training Noise).
        Lớp Analytic FC (self.weight) sẽ tự động mở rộng trong hàm fit().
        """
        self.cur_task += 1
        self.known_class += nb_classes
        
        # Tạo mới Normal FC cho task hiện tại
        if self.cur_task > 0:
            # Task sau: Không dùng Bias để tránh bias vào lớp mới quá nhiều
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            # Task đầu: Có bias
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            
        if self.normal_fc is not None:
            # Sequential Init: Copy trọng số cũ
            old_nb_output = self.normal_fc.out_features
            with torch.no_grad():
                # Copy phần cũ
                new_fc.weight[:old_nb_output] = self.normal_fc.weight.data
                # Init phần mới về 0
                nn.init.constant_(new_fc.weight[old_nb_output:], 0.)
            
            del self.normal_fc
            self.normal_fc = new_fc
        else:
            # Task đầu tiên
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None:
                nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    # =========================================================================
    # [MAGMAX & NOISE CONTROL SECTION]
    # =========================================================================
    
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
            
            # Giữ LayerNorm trainable ở Task 0 để ổn định base
            if hasattr(self.backbone.blocks[j], 'norm1'):
                for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            if hasattr(self.backbone.blocks[j], 'norm2'):
                for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
                
        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            for p in self.backbone.norm.parameters(): p.requires_grad = True
    
    # =========================================================================
    # [ANALYTIC LEARNING (RLS) SECTION]
    # =========================================================================

    def forward_fc(self, features):
        """Forward qua Analytic Classifier"""
        # Đảm bảo features cùng kiểu với trọng số RLS (float32)
        features = features.to(self.weight.dtype) 
        return features @ self.weight

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Phiên bản RLS tối ưu bộ nhớ (Memory-Efficient RLS)
        """
        # Tắt Autocast để tính toán chính xác FP32 (tránh lỗi Singular Matrix)
        with autocast('cuda', enabled=False):
            # 1. Feature Extraction & Expansion
            X = self.backbone(X).float() 
            X = self.buffer(X) 
            
            # Đảm bảo cùng device
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            # 2. Mở rộng chiều của classifier nếu có lớp mới
            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.weight.shape[1]:
                increment_size = self.weight.shape[1] - num_targets
                tail = torch.zeros((Y.shape[0], increment_size), device=Y.device)
                Y = torch.cat((Y, tail), dim=1)

            # 3. RLS Update (Tối ưu OOM)
            term = torch.eye(X.shape[0], device=X.device) + X @ self.R @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            
            try:
                K = torch.linalg.solve(term + jitter, X @ self.R)
                K = K.T 
            except:
                K = self.R @ X.T @ torch.inverse(term + jitter)

            self.R -= K @ X @ self.R
            self.weight += K @ (Y - X @ self.weight)
            
            del term, jitter, K, X, Y 

    # =========================================================================
    # [FeCAM INTEGRATION - OPTIMIZED FOR BACKBONE FEATURES]
    # =========================================================================
    
    def _tukeys_transform(self, x, beta=0.5):
        return torch.pow(x, beta)

    def _shrink_cov(self, cov):
        """
        Co ngót ma trận hiệp phương sai (In-place & No extra allocation).
        """
        diag = torch.diagonal(cov)
        diag_mean = torch.mean(diag)
        sum_all = torch.sum(cov)
        sum_diag = torch.sum(diag)
        sum_off_diag = sum_all - sum_diag
        n = cov.shape[0]
        num_off_diag = n * n - n
        
        if num_off_diag > 0:
            off_diag_mean = sum_off_diag / num_off_diag
        else:
            off_diag_mean = 0.0
            
        alpha1 = 0.01; alpha2 = 0.01 
        
        # In-place add
        cov.add_(alpha2 * off_diag_mean)
        torch.diagonal(cov).add_(alpha1 * diag_mean - alpha2 * off_diag_mean)
        return cov

    def build_fecam_stats(self, train_loader):
        """
        Tính FeCAM trên Backbone Features (D=768).
        Nhanh, Chính xác và Siêu nhẹ (No OOM).
        """
        self.eval()
        print(f"--> [FeCAM] Building Statistics (Backbone Features D={self.backbone.out_dim})...")
        
        running_stats = {} 
        
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # [QUAN TRỌNG]: Chỉ dùng backbone, KHÔNG qua buffer
                # Feats shape: [Batch, 768] -> Ma trận Cov chỉ 2.3MB
                feats = self.backbone(inputs) 
                
                feats = self._tukeys_transform(feats)
                
                unique_labels = torch.unique(targets)
                for label in unique_labels:
                    label_item = label.item()
                    mask = (targets == label)
                    class_feats = feats[mask]
                    
                    if label_item not in running_stats:
                        D = class_feats.shape[1]
                        running_stats[label_item] = {
                            'sum_x': torch.zeros(D, device=self.device),
                            'sum_xxT': torch.zeros((D, D), device=self.device),
                            'n': 0
                        }
                    
                    # Accumulate on GPU (Safe because D=768 is small)
                    running_stats[label_item]['sum_x'] += class_feats.sum(dim=0)
                    running_stats[label_item]['sum_xxT'].addmm_(class_feats.T, class_feats)
                    running_stats[label_item]['n'] += class_feats.shape[0]

        # Tổng hợp kết quả
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
            
            cov = self._shrink_cov(cov)
            
            self.class_means.append(mean)
            self.class_covs.append(cov)
            
            del running_stats[label]
            
        print(f"--> [FeCAM] Stats Built. Total classes: {len(self.class_means)}")
        del running_stats
        torch.cuda.empty_cache()

    def predict_fecam_internal(self, feats):
        """Hàm tính khoảng cách nhận features đầu vào (đã trích xuất từ backbone)"""
        feats = self._tukeys_transform(feats)
        dists = []
        for c in range(len(self.class_means)):
            mean = self.class_means[c]
            cov = self.class_covs[c]
            
            diff = feats - mean.unsqueeze(0)
            
            # Solve linear system: cov * x = diff.T
            # Với D=768, solve cực nhanh
            try:
                term = torch.linalg.solve(cov, diff.T).T 
            except:
                term = diff @ torch.linalg.pinv(cov)
                
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
        
        # [LOGIC MỚI]: Tách luồng xử lý
        if self.training or not self.use_fecam or len(self.class_means) == 0:
            # Training: Dùng Buffer -> Analytic Classifier (Proxy Task)
            hyper_features_fp32 = hyper_features.to(self.weight.dtype)
            features_buffer = self.buffer(hyper_features_fp32)
            logits = self.forward_fc(features_buffer)
        else:
            # Inference: Dùng trực tiếp Backbone Features -> FeCAM
            # Không đi qua self.buffer nữa
            logits = self.predict_fecam_internal(hyper_features)
            
        return {'logits': logits}

    def extract_feature(self, x):
        """Chỉ trích xuất đặc trưng từ Backbone"""
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
