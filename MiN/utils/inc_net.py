import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ===============================
# PiNoise (Bilora-style)
# ===============================
class PiNoise(nn.Module):
    def __init__(self, dim, k, freq_range):
        super().__init__()
        self.dim = dim
        self.k = k
        self.freq_range = freq_range  # (start, end)

        # low-rank noise
        self.mu = nn.Linear(k, dim, bias=False)
        self.w_up = nn.Parameter(torch.zeros(dim, dim))
        nn.init.kaiming_uniform_(self.w_up, a=math.sqrt(5))

        self.active = True

    def forward(self, x, noise):
        if not self.active:
            return torch.zeros_like(x)

        noise = noise[:, self.freq_range[0]:self.freq_range[1]]
        noise = self.mu(noise)
        return noise @ self.w_up

    def freeze(self):
        self.active = False
        for p in self.parameters():
            p.requires_grad = False


# ===============================
# Backbone + Noise
# ===============================
class BackboneWithNoise(nn.Module):
    def __init__(self, backbone, k, freq_ranges):
        super().__init__()
        self.backbone = backbone
        self.noise_maker = nn.ModuleList()

        dim = backbone.embed_dim

        for fr in freq_ranges:
            self.noise_maker.append(PiNoise(dim, k, fr))

    def forward(self, x, noise, task_id, new_forward=True):
        feat = self.backbone.forward_features(x)

        if new_forward:
            feat = feat + self.noise_maker[task_id](feat, noise)

        return feat

    def freeze_old_tasks(self, cur_task):
        for i in range(cur_task):
            self.noise_maker[i].freeze()


# ===============================
# MiN Base Net
# ===============================
class MiNbaseNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.device = args["device"]
        self.k = args["k"]
        self.num_tasks =10

        self.backbone = args["backbone"]  # timm ViT đã build sẵn

        # chia miền tần số
        step = self.k
        self.freq_ranges = [
            (i * step, (i + 1) * step)
            for i in range(self.num_tasks)
        ]

        self.backbone = BackboneWithNoise(
            self.backbone,
            k=self.k,
            freq_ranges=self.freq_ranges
        )

        self.normal_fc = nn.Linear(self.backbone.backbone.embed_dim, 0, bias=False)

        self.cur_task = 0

    # ===============================
    # FC
    # ===============================
    def update_fc(self, num_new_classes):
        in_dim = self.normal_fc.in_features
        out_dim = self.normal_fc.out_features

        new_fc = nn.Linear(in_dim, out_dim + num_new_classes, bias=False).to(self.device)
        if out_dim > 0:
            new_fc.weight.data[:out_dim] = self.normal_fc.weight.data

        self.normal_fc = new_fc

    # ===============================
    # Noise
    # ===============================
    def update_noise(self):
        self.backbone.freeze_old_tasks(self.cur_task)

    def after_task_magmax_merge(self):
        # MagMax: giữ noise mạnh nhất theo norm
        with torch.no_grad():
            for i in range(self.cur_task + 1):
                w = self.backbone.noise_maker[i].w_up
                w.copy_(w / (torch.norm(w) + 1e-6))

        self.cur_task += 1

    # ===============================
    # Forward
    # ===============================
    def forward(self, x, noise=None, new_forward=True):
        feat = self.backbone(x, noise, self.cur_task, new_forward)
        logits = self.normal_fc(feat)
        return {"logits": logits}

    def forward_normal_fc(self, x, new_forward=True):
        feat = self.backbone.backbone.forward_features(x)
        logits = self.normal_fc(feat)
        return {"logits": logits}
