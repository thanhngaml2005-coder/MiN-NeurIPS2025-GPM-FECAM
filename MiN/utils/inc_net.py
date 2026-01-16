import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear

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
        
        # Analytic Classifier Init
        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight)
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        if self.cur_task > 0:
            fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
        
        if self.normal_fc is not None:
            del self.normal_fc
        self.normal_fc = fc

    # --- [MAGMAX SECTION START] ---
    def update_noise(self):
        """Gọi khi bắt đầu task mới: Chuẩn bị Noise Generator (Sequential Init)"""
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()

    def after_task_magmax_merge(self):
        """Gọi khi kết thúc task: Lưu Task Vector và Merge"""
        print(f"Task {self.cur_task}: Performing MagMax Merging on Parameters...")
        for j in range(self.backbone.layer_num):
            # Hàm này nằm trong PiNoise bạn vừa sửa
            self.backbone.noise_maker[j].after_task_training()

    def init_unfreeze(self):
        """Unfreeze noise layers cho task đầu tiên"""
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].unfreeze_noise()
            # Unfreeze LayerNorms của Backbone để ổn định
            for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
        for p in self.backbone.norm.parameters(): p.requires_grad = True

    def unfreeze_noise(self):
        """Chỉ unfreeze noise modules (cho các task sau)"""
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].unfreeze_noise()
    # --- [MAGMAX SECTION END] ---

    # Các hàm hỗ trợ khác giữ nguyên
    def forward_fc(self, features):
        features = features.to(self.weight)
        return features @ self.weight

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        # Tắt autocast cho đoạn tính toán ma trận chính xác cao
        # (Giữ nguyên logic RLS của bạn)
        X = self.backbone(X).float()
        X = self.buffer(X)
        X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

        num_targets = Y.shape[1]
        if num_targets > self.weight.shape[1]:
            increment_size = num_targets - self.weight.shape[1]
            tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
            self.weight = torch.cat((self.weight, tail), dim=1)
        
        I = torch.eye(X.shape[0]).to(X)
        term = I + X @ self.R @ X.T
        jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
        K = torch.inverse(term + jitter)
        self.R -= self.R @ X.T @ K @ X @ self.R
        self.weight += self.R @ X.T @ (Y - X @ self.weight)

    def forward(self, x, new_forward: bool = False):
        # Forward inference (Test)
        hyper_features = self.backbone(x)
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {'logits': logits}

    def forward_normal_fc(self, x, new_forward: bool = False):
        # Forward training (với Normal FC)
        hyper_features = self.backbone(x)
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}
