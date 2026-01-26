import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.inc_net import BackboneWithPiNoise


class MinNet(nn.Module):
    """
    MiN main network

    - backbone: ViT / CNN (timm)
    - PiNoise handled in BackboneWithNoise
    - logits = normal_fc(feat + noise)
    """

    def __init__(self, args):
        super().__init__()

        self.device = args["device"]
        self.backbone_name = args["backbone_name"]
        self.backbone = args["backbone"]          # timm model (built outside)
        self.k = args["k"]                         # freq per task
        self.hidden_dim = args.get("hidden_dim", 128)

        self.num_tasks = args["num_tasks"]
        self.cur_task = 0

        embed_dim = self.backbone.embed_dim

        # ===============================
        # Backbone + PiNoise
        # ===============================
        self.backbone = BackboneWithPiNoise(
            backbone=self.backbone,
            in_dim=embed_dim,
            k=self.k,
            hidden_dim=self.hidden_dim,
            device=self.device
        )

        # ===============================
        # Classifier (grows over tasks)
        # ===============================
        self.normal_fc = nn.Linear(embed_dim, 0, bias=False).to(self.device)

    # =====================================================
    # FC logic (Giữ nguyên MiN gốc)
    # =====================================================
    def update_fc(self, num_new_classes):
        """
        Expand classifier for new task
        """
        in_dim = self.normal_fc.in_features
        out_dim = self.normal_fc.out_features

        new_fc = nn.Linear(
            in_dim,
            out_dim + num_new_classes,
            bias=False
        ).to(self.device)

        if out_dim > 0:
            new_fc.weight.data[:out_dim].copy_(
                self.normal_fc.weight.data
            )

        self.normal_fc = new_fc

    # =====================================================
    # Noise logic
    # =====================================================
    def init_train(self, data_manager=None):
        """
        Called at beginning of each task
        """
        # freeze old PiNoise
        self.update_noise()

        # unfreeze current task PiNoise
        self.backbone.noise_maker[self.cur_task].unfreeze_noise()

    def update_noise(self):
        """
        Freeze all old task PiNoise
        """
        for j in range(self.cur_task):
            self.backbone.noise_maker[j].update_noise()

    def after_task(self):
        """
        Called after finishing training a task
        """
        self.cur_task += 1

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, x, noise=None, new_forward=True):
        """
        Standard MiN forward
        """
        feat = self.backbone(
            x,
            task_id=self.cur_task,
            noise=noise,
            new_forward=new_forward
        )

        logits = self.normal_fc(feat)
        return {"logits": logits}

    def forward_without_noise(self, x):
        """
        Used for eval / refit_fc
        """
        feat = self.backbone.backbone.forward_features(x)
        logits = self.normal_fc(feat)
        return {"logits": logits}
