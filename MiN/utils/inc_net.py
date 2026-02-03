import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear

# Xử lý tương thích phiên bản PyTorch cho Autocast
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

# =============================================================================
# [NEW MODULE] DPCR Estimator - Bộ sửa lỗi trôi đặc trưng
# =============================================================================
class DPCREstimator(nn.Module):
    def __init__(self, feature_dim, device):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        
        # Lưu trữ Covariance và Prototype (Mean) ở không gian gốc (768 dim)
        # Key: class_id, Value: Tensor
        self.saved_covs = {} 
        self.saved_protos = {} 
        self.projectors = {} # Lưu SVD Projector của từng class

    def get_projector_svd(self, raw_matrix):
        """Tính SVD để lấy ma trận chiếu CIP"""
        # raw_matrix: [768, 768]
        try:
            U, S, V = torch.svd(raw_matrix + 1e-4 * torch.eye(raw_matrix.shape[0], device=self.device))
        except:
            # Fallback nếu SVD lỗi
            return torch.eye(raw_matrix.shape[0], device=self.device)
            
        # Lấy các chiều có giá trị singular > 0 (hoặc top k)
        # Trong bài báo họ lấy full rank
        non_zeros = torch.where(S > 1e-5)[0]
        if len(non_zeros) == 0:
            return torch.eye(raw_matrix.shape[0], device=self.device)
            
        left_vecs = U[:, non_zeros]
        # P = U * U^T
        projector = left_vecs @ left_vecs.T
        return projector

    def update_stats(self, model, loader, class_list):
        """
        Lưu trữ Covariance và Mean của Task vừa học xong (khi chưa bị drift)
        """
        model.eval()
        print(f"--> [DPCR] Saving Stats for classes {class_list}...")
        
        with torch.no_grad():
            for c in class_list:
                features = []
                for _, inputs, targets in loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    mask = targets == c
                    if mask.sum() == 0: continue
                    
                    # Lấy feature ở tầng Backbone (768), KHÔNG qua Buffer
                    feat = model.extract_feature(inputs[mask]) 
                    features.append(feat)
                
                if len(features) > 0:
                    features = torch.cat(features, dim=0).float() # [N, 768]
                    
                    # Tính Mean
                    mean = torch.mean(features, dim=0)
                    # Tính Uncentered Covariance (X^T * X) theo bài báo
                    cov = features.T @ features
                    
                    self.saved_protos[c] = mean
                    self.saved_covs[c] = cov
                    self.projectors[c] = self.get_projector_svd(cov)

    def correct_drift(self, old_model, new_model, current_loader, known_classes):
        """
        Tính toán ma trận Drift và sửa lại Stats cũ
        """
        print(f"--> [DPCR] Calculating Drift Matrix & Correcting {known_classes} old classes...")
        old_model.eval()
        new_model.eval()
        
        # 1. Tính TSSP (Task-wise Shift)
        # Gom dữ liệu task mới để học sự thay đổi từ Old -> New Model
        cov_old = torch.zeros(self.feature_dim, self.feature_dim).to(self.device)
        cross_corr = torch.zeros(self.feature_dim, self.feature_dim).to(self.device)
        
        with torch.no_grad():
            for _, inputs, _ in current_loader:
                inputs = inputs.to(self.device)
                
                feat_old = old_model.extract_feature(inputs).float() # f_t-1(x)
                feat_new = new_model.extract_feature(inputs).float() # f_t(x)
                
                cov_old += feat_old.T @ feat_old
                cross_corr += feat_old.T @ feat_new
        
        # Giải phương trình: P = (X_old^T X_old + epsilon)^-1 @ (X_old^T X_new)
        # P: [768, 768] - Ma trận biến đổi toàn cục
        epsilon = 1e-4 * torch.eye(self.feature_dim, device=self.device)
        P_tssp = torch.linalg.solve(cov_old + epsilon, cross_corr)
        
        # 2. Áp dụng CIP và Update Stats cũ
        # Duyệt qua các class cũ
        for c in range(known_classes):
            if c not in self.saved_covs: continue
            
            # Kết hợp TSSP và CIP (Projector riêng của class)
            # W_c = P_tssp @ Projector_c
            W_c = P_tssp @ self.projectors[c]
            
            # Update Covariance: Cov_new = W^T @ Cov_old @ W
            self.saved_covs[c] = W_c.T @ self.saved_covs[c] @ W_c
            
            # Update Mean: Mean_new = Mean_old @ W
            # Lưu ý: vector nhân ma trận -> (1, D) @ (D, D)
            self.saved_protos[c] = self.saved_protos[c] @ W_c
            
            # Update lại Projector cho vòng sau
            self.projectors[c] = self.get_projector_svd(self.saved_covs[c])
            
        print("--> [DPCR] Correction Done.")

# =============================================================================
# [MODIFIED] RandomBuffer & MiNbaseNet
# =============================================================================

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
        
        # --- FeCAM & DPCR ---
        self.class_means = []
        self.class_vars = []
        # Khởi tạo bộ sửa lỗi DPCR
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

    # --- DPCR Helper ---
    def extract_feature(self, x):
        return self.backbone(x)

    def _robust_transform(self, x, beta=0.5):
        x = torch.sign(x) * torch.pow(torch.abs(x) + 1e-6, beta)
        x = F.layer_norm(x, x.shape[-1:])
        return x

    def update_fecam_stats_with_dpcr(self):
        """
        Hàm quan trọng nhất: Chuyển đổi từ DPCR Stats (768 dim) sang FeCAM Stats (16k dim)
        """
        print("--> [Sync] Updating FeCAM Stats from DPCR corrected memory...")
        self.class_means = []
        self.class_vars = []
        
        # Duyệt qua tất cả các class đã lưu trong DPCR (đã được correct)
        sorted_classes = sorted(self.dpcr.saved_protos.keys())
        
        self.buffer.use_relu = False # Tắt ReLU để lấy linear projection
        
        with torch.no_grad():
            for c in sorted_classes:
                # 1. Lấy Mean/Cov gốc (768)
                mean_768 = self.dpcr.saved_protos[c].float()
                # cov_768 = self.dpcr.saved_covs[c].float() # (Ít dùng để tính var chéo vì nặng)
                
                # 2. Chiếu Mean qua Buffer (16k)
                # mean_16k = Buffer(mean_768)
                mean_16k = self.buffer(mean_768.unsqueeze(0)).squeeze(0)
                
                # 3. Ước lượng Variance ở 16k
                # Cách chuẩn: Var[Ax] = A Var[x] A^T. Nhưng A (buffer) quá lớn.
                # Cách "đường tắt" (Heuristic): 
                # Chúng ta giả định Buffer bảo toàn phân phối tương đối.
                # Tuy nhiên, để đơn giản và hiệu quả, ta dùng Variance trung bình toàn cục 
                # hoặc tính Var từ mean_16k (nếu có mẫu).
                # Vì DPCR chỉ lưu Covariance ma trận, việc chiếu Covariance 768x768 -> 16k Diagonal 
                # là phép tính rất nặng: diag(W @ Cov @ W^T).
                
                # [THỦ THUẬT]: Tính Variance trực tiếp từ công thức Var(Projected)
                # Var_i = Sum_j(W_ij^2 * Cov_jj) + ... (Phức tạp)
                
                # [GIẢI PHÁP TỐT NHẤT CHO HYBRID]:
                # Dùng Mean đã sửa (Mean chuẩn quan trọng hơn Var).
                # Dùng Variance mặc định hoặc Variance cũ (nếu có).
                # Ở đây tôi set Variance đồng nhất = 1 để FeCAM hoạt động như NCM (Nearest Mean) 
                # nhưng trên không gian đã được DPCR sửa lỗi. Điều này an toàn hơn tính Var sai.
                var_16k = torch.ones_like(mean_16k)
                
                # Robust transform cho Mean
                mean_16k = self._robust_transform(mean_16k.unsqueeze(0)).squeeze(0)
                
                self.class_means.append(mean_16k)
                self.class_vars.append(var_16k)
                
        self.buffer.use_relu = True
        print(f"--> [Sync] Done. Total classes ready for FeCAM: {len(self.class_means)}")

    def predict_fecam_internal(self, inputs):
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
                # var = self.class_vars[c] # Var = 1 nên bỏ qua chia
                
                # Euclidean Distance tới Mean đã được DPCR sửa
                diff_sq = (feats - mean.unsqueeze(0)) ** 2
                dist = torch.sum(diff_sq, dim=1)
                dists.append(dist)
                
        self.buffer.use_relu = True
        return -torch.stack(dists, dim=1)

    # ... (Các hàm forward, collect_projections... giữ nguyên) ...
    def update_fc(self, nb_classes): # Copy lại hàm cũ
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
