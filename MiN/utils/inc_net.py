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
            # [FIX LỖI TẠI ĐÂY]: Đổi fc thành new_fc
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
        """
        Gọi khi bắt đầu Task mới.
        Kích hoạt chế độ Sequential Initialization trong PiNoise.
        """
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()

    def after_task_magmax_merge(self):
        """
        Gọi sau khi kết thúc Task.
        Kích hoạt việc LƯU (Save) và TRỘN (Merge) tham số theo MagMax.
        """
        print(f"--> [IncNet] Task {self.cur_task}: Triggering Parameter-wise MagMax Merging...")
        for j in range(self.backbone.layer_num):
            # Hàm này nằm trong PiNoise
            self.backbone.noise_maker[j].after_task_training()

    def unfreeze_noise(self):
        """Chỉ mở khóa gradient cho các module Noise (cho các task > 0)"""
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].unfreeze_noise()

    def init_unfreeze(self):
        """
        Mở khóa gradient cho Task 0.
        Bao gồm Noise modules và các lớp Normalization của Backbone để ổn định hơn.
        """
        for j in range(self.backbone.layer_num):
            # Unfreeze Noise
            self.backbone.noise_maker[j].unfreeze_noise()
            
            # Unfreeze LayerNorms trong từng Block ViT
            for p in self.backbone.blocks[j].norm1.parameters():
                p.requires_grad = True
            for p in self.backbone.blocks[j].norm2.parameters():
                p.requires_grad = True
                
        # Unfreeze LayerNorm cuối cùng
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

    # =========================================================================
    # [ANALYTIC LEARNING (RLS) SECTION]
    # =========================================================================

    def forward_fc(self, features):
        """Forward qua Analytic Classifier"""
        # Đảm bảo features cùng kiểu với trọng số RLS (float32)
        features = features.to(self.weight) 
        return features @ self.weight
    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Huấn luyện Analytical Classifier sử dụng Recursive Least Squares (RLS).
        Phiên bản ổn định cao (Robust Version) sử dụng Cholesky Solve.
        """
        # [QUAN TRỌNG] Tắt Autocast: RLS cần độ chính xác FP32 để không bị lỗi ma trận suy biến
        with torch.cuda.amp.autocast(enabled=False):
            # 1. Feature Extraction & Expansion
            # Chuyển về float32 ngay lập tức
            X = self.backbone(X).float() 
            if hasattr(self, 'buffer'):
                X = self.buffer(X) # Random Expansion (nếu có)
            
            # Đưa về cùng device với trọng số RLS
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            # 2. Mở rộng chiều của Classifier (Dynamic Expansion)
            # Nếu số lớp trong Y lớn hơn số lớp hiện tại của weight, ta mở rộng ma trận
            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.weight.shape[1]:
                # Trường hợp hiếm: Y ít lớp hơn (ví dụ chỉ batch của task cũ), pad Y thêm số 0
                increment_size = self.weight.shape[1] - num_targets
                tail = torch.zeros((Y.shape[0], increment_size), device=Y.device)
                Y = torch.cat((Y, tail), dim=1)

            # 3. Tính toán RLS Update (Woodbury Matrix Identity)
            # Công thức gốc: P_new = P - P X^T (I + X P X^T)^-1 X P
            # Đặt term = (I + X P X^T)
            
            I = torch.eye(X.shape[0], device=X.device)
            term = I + X @ self.R @ X.T
            
            # [STABILITY FIX 1] Thêm Jitter để đảm bảo ma trận không bị suy biến (Singular)
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            term_stable = term + jitter
            
            # [STABILITY FIX 2] Dùng Cholesky Solve thay vì Inverse trực tiếp
            # Tính term_inv = (term_stable)^-1
            try:
                # Cholesky nhanh và ổn định cho ma trận đối xứng dương (Symmetric Positive Definite)
                L = torch.linalg.cholesky(term_stable)
                term_inv = torch.cholesky_solve(I, L)
            except RuntimeError:
                # Fallback: Nếu Cholesky fail (rất hiếm), dùng linalg.solve thường (chậm hơn chút nhưng an toàn)
                term_inv = torch.linalg.solve(term_stable, I)
            
            # Tính Kalman Gain: K = P * X^T * term_inv
            # self.R chính là ma trận P (Inverse Covariance Matrix)
            K = self.R @ X.T @ term_inv
            
            # 4. Cập nhật Ma trận hiệp phương sai (R)
            # R_new = R - K * X * R
            self.R = self.R - K @ X @ self.R
            
            # 5. Cập nhật Trọng số Classifier (Weight)
            # W_new = W + K * (Y - X * W)
            # (Y - X @ self.weight) là Prediction Error
            self.weight = self.weight + K @ (Y - X @ self.weight)
    # =========================================================================
    # [FORWARD PASSES]
    # =========================================================================

    def forward(self, x, new_forward: bool = False):
        """
        Dùng cho Inference/Testing.
        Chạy qua backbone (đã merge noise) -> Buffer -> Analytic Classifier.
        """
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        # Ép kiểu về float32 cho Buffer và Classifier
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        
        return {
            'logits': logits
        }

    def extract_feature(self, x):
        """Chỉ trích xuất đặc trưng từ Backbone"""
        return self.backbone(x)

    def forward_normal_fc(self, x, new_forward: bool = False):
        """
        Dùng cho Training (Gradient Descent).
        Chạy qua backbone -> Buffer -> Normal FC (trainable).
        """
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = self.buffer(hyper_features)
        
        # Ép kiểu để khớp với Normal FC (có thể là FP16 nếu autocast bật bên ngoài)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        
        logits = self.normal_fc(hyper_features)['logits']
        
        return {
            "logits": logits
        }
    def update_GPM(self, threshold=0.99):
        """
        [FIX ISSUE 4] Robust GPM Update with Orthonormalization
        """
        print(f"--> [GPM] Updating Basis (Threshold={threshold})...")
        
        for module in self.backbone.noise_maker:
            R = module.cur_matrix
            if R.sum() == 0: continue

            # 1. SVD trên covariance matrix hiện tại
            # R là ma trận đối xứng (X^T X), dùng eigh sẽ nhanh và ổn định hơn svd thường
            try:
                # eigh trả về eigenvalues tăng dần, ta cần đảo ngược lại
                S, U = torch.linalg.eigh(R) 
                S = S.flip(0)
                U = U.flip(1)
            except:
                # Fallback nếu lỗi
                U, S, V = torch.linalg.svd(R)

            # 2. Chọn k vector
            s_sq = S.abs() # S là eigenvalues của R (chính là s^2 của X)
            s_sum = torch.sum(s_sq)
            s_cum = torch.cumsum(s_sq, dim=0) / s_sum
            k = torch.searchsorted(s_cum, threshold).item() + 1
            
            # Basis đề xuất từ task mới
            new_basis = U[:, :k] # [Hidden, k]

            # 3. [CRITICAL FIX] Orthonormalize against OLD Basis
            # Nếu đã có basis cũ, phải đảm bảo basis mới vuông góc với nó
            if module.basis.shape[1] > 0:
                old_basis = module.basis # [Hidden, M]
                
                # Chiếu new_basis lên không gian cũ: P = B_old @ B_old^T
                # proj = old_basis @ (old_basis.T @ new_basis)
                # new_basis_ortho = new_basis - proj
                # Cách viết tối ưu bộ nhớ:
                projection = torch.matmul(old_basis.T, new_basis)
                new_basis_residual = new_basis - torch.matmul(old_basis, projection)
                
                # 4. Normalize lại (Gram-Schmidt)
                # Để đảm bảo tính trực chuẩn (Orthonormality)
                norms = torch.norm(new_basis_residual, dim=0)
                
                # Chỉ giữ lại các vector có norm đủ lớn (chưa nằm trong không gian cũ)
                # Epsilon nhỏ để tránh chia cho 0
                valid_indices = norms > 1e-5 
                
                if valid_indices.sum() > 0:
                    new_basis_clean = new_basis_residual[:, valid_indices]
                    new_basis_clean = new_basis_clean / norms[valid_indices] # Normalize
                    
                    # Cập nhật: Ghép vào
                    module.basis = torch.cat((module.basis, new_basis_clean), dim=1)
            else:
                # Task đầu tiên: Chỉ cần Normalize là đủ (SVD U đã trực giao sẵn)
                # Nhưng U của SVD đã là trực chuẩn rồi, không cần làm gì thêm
                module.basis = new_basis

            # Reset buffers
            module.cur_matrix.zero_()
            module.n_cur_matrix.zero_()