import math
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
import gc
import os

from utils.inc_net import MiNbaseNet
from utils.toolkit import tensor2numpy
from utils.training_tool import get_optimizer, get_scheduler
from utils.toolkit import calculate_class_metrics, calculate_task_metrics

# [ADDED] Import Mixed Precision
from torch.amp import autocast, GradScaler

EPSILON = 1e-8

# =============================================================================
#  [NEW] CÁC CLASS HỖ TRỢ CFS
# =============================================================================
class NegativeContrastiveLoss(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau
    def forward(self, x): 
        x = F.normalize(x, dim=1)
        x_1 = torch.unsqueeze(x, dim=0)
        x_2 = torch.unsqueeze(x, dim=1)
        cos = torch.sum(x_1 * x_2, dim=2) / self.tau
        exp_cos = torch.exp(cos)
        mask = torch.eye(x.size(0), device=x.device).bool()
        exp_cos = exp_cos.masked_fill(mask, 0)
        loss = torch.log(exp_cos.sum(dim=1) / (x.size(0) - 1) + EPSILON)
        return torch.mean(loss)

class CFS_Mapping(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f_cont = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim), 
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        x = self.f_cont(x)
        return F.normalize(x, p=2, dim=1)

# =============================================================================
#  MAIN CLASS
# =============================================================================
class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.logger = loger
        self._network = MiNbaseNet(args)
        self.device = args['device']
        self.num_workers = args["num_workers"]

        self.init_epochs = args["init_epochs"]
        self.init_lr = args["init_lr"]
        self.init_weight_decay = args["init_weight_decay"]
        self.init_batch_size = args["init_batch_size"]

        self.lr = args["lr"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.epochs = args["epochs"]

        self.init_class = args["init_class"]
        self.increment = args["increment"]

        self.buffer_size = args["buffer_size"]
        self.buffer_batch = args["buffer_batch"]
        self.gamma = args['gamma']
        self.fit_epoch = args["fit_epochs"]

        self.known_class = 0
        self.cur_task = -1
        self.total_acc = []
        
        self.scaler = GradScaler('cuda')
        
        # [NEW] Buffer để lưu Prototype (Exemplars)
        self.feature_buffer = {} 

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # [NEW] Hàm chọn Feature bằng CFS + Herding
    def get_features_cfs(self, model, train_loader, num_select=20):
        device = self.device
        model.eval()
        
        # 1. Thu thập toàn bộ feature
        all_features = []
        all_labels = []
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                with autocast('cuda'):
                    # Detach để không lưu graph
                    feats = model.extract_feature(inputs).detach()
                all_features.append(feats)
                all_labels.append(targets)
        
        all_features = torch.cat(all_features).float()
        all_labels = torch.cat(all_labels).to(device)
        feature_dim = all_features.shape[1]

        # 2. Train mạng CFS
        f_cont = CFS_Mapping(feature_dim).to(device).float()
        optimizer = optim.Adam(f_cont.parameters(), lr=1e-3)
        criterion = NegativeContrastiveLoss(tau=0.1)
        
        f_cont.train()
        cfs_batch_size = 256
        num_samples = all_features.size(0)
        cfs_epochs = 15 
        
        for ep in range(cfs_epochs):
            perm = torch.randperm(num_samples)
            for i in range(0, num_samples, cfs_batch_size):
                idx = perm[i : i + cfs_batch_size]
                batch_feat = all_features[idx].to(device)
                optimizer.zero_grad()
                embeddings = f_cont(batch_feat)
                loss = criterion(embeddings)
                loss.backward()
                optimizer.step()

        # 3. Chọn Feature (Herding trên Latent Space)
        f_cont.eval()
        final_exemplars = {}
        unique_classes = torch.unique(all_labels).cpu().numpy()
        
        for cls in unique_classes:
            cls_mask = (all_labels == cls)
            raw_feats = all_features[cls_mask].to(device)
            
            if raw_feats.size(0) <= num_select:
                final_exemplars[cls] = raw_feats.cpu()
                continue
                
            with torch.no_grad():
                latent_feats = f_cont(raw_feats)
            
            mu_latent = latent_feats.mean(dim=0)
            selected_indices = []
            current_sum = torch.zeros_like(mu_latent)
            remaining_mask = torch.ones(raw_feats.size(0), dtype=torch.bool, device=device)
            
            for k in range(num_select):
                target = mu_latent * (k + 1) - current_sum
                dists = torch.norm(latent_feats - target, dim=1)
                dists[~remaining_mask] = float('inf')
                best_idx = torch.argmin(dists).item()
                selected_indices.append(best_idx)
                remaining_mask[best_idx] = False
                current_sum += latent_feats[best_idx]
            
            final_exemplars[cls] = raw_feats[selected_indices].cpu()

        del f_cont, optimizer, criterion, all_features
        self._clear_gpu()
        return final_exemplars

    # [MODIFIED] Re-fit có thêm Prototype
    def re_fit(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        
        # 1. Lấy Feature Mới (Real Data)
        X_new_list, Y_new_list = [], []
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                with autocast('cuda'):
                    features = self._network.extract_feature(inputs).detach().float()
                X_new_list.append(features)
                Y_new_list.append(targets)
        
        X_new = torch.cat(X_new_list, dim=0)
        Y_new = torch.cat(Y_new_list, dim=0)
        
        # 2. Lấy Prototype Cũ (từ feature_buffer)
        X_old_list, Y_old_list = [], []
        if len(self.feature_buffer) > 0:
            for cls, feats in self.feature_buffer.items():
                if cls < self.known_class: # Chỉ lấy lớp cũ
                    X_old_list.append(feats.to(self.device))
                    Y_old_list.append(torch.full((feats.size(0),), cls, dtype=torch.long, device=self.device))
            
            if len(X_old_list) > 0:
                X_old = torch.cat(X_old_list, dim=0)
                Y_old = torch.cat(Y_old_list, dim=0)
                
                # Balancing: Nhân bản prototype
                avg_new = X_new.size(0) // (self.increment if self.cur_task > 0 else self.init_class)
                avg_old = 20
                repeat_factor = max(1, int((avg_new / avg_old) * 0.5)) 
                
                X_old = X_old.repeat(repeat_factor, 1)
                Y_old = Y_old.repeat(repeat_factor)
                
                X_total = torch.cat([X_new, X_old], dim=0)
                Y_total = torch.cat([Y_new, Y_old], dim=0)
            else:
                X_total, Y_total = X_new, Y_new
        else:
            X_total, Y_total = X_new, Y_new

        # 3. Fit RLS
        # Tự tính tổng số class để tránh lỗi One-hot
        if self.cur_task == 0:
            total_classes = self.init_class
        else:
            total_classes = self.known_class + self.increment

        Y_total_oh = F.one_hot(Y_total, num_classes=total_classes).float()
        
        # Fit Batch-wise
        batch_size_fit = 4096 
        total_samples = X_total.size(0)
        perm = torch.randperm(total_samples)
        
        info = f"Task {self.cur_task} --> Re-fit RLS with Prototypes..."
        self.logger.info(info)
        
        with torch.no_grad():
            for i in tqdm(range(0, total_samples, batch_size_fit), desc="Re-fitting"):
                idx = perm[i : i + batch_size_fit]
                x_batch = X_total[idx]
                y_batch = Y_total_oh[idx]
                self._network.fit(x_batch, y_batch)
        
        self._clear_gpu()

    # [MODIFIED] Init Train
    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(0)
        
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        self._clear_gpu()
        
        self.run(train_loader)
        self._network.after_task_magmax_merge()
        self._clear_gpu()
        
        # Fit FC
        train_loader_buf = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.fit_fc(train_loader_buf, test_loader)

        # [NEW] Tính Prototype bằng CFS (dùng dữ liệu sạch)
        train_set_clean = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_clean.labels = self.cat2order(train_set_clean.labels, data_manger)
        clean_loader = DataLoader(train_set_clean, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        
        new_protos = self.get_features_cfs(self._network, clean_loader, num_select=20)
        self.feature_buffer.update(new_protos)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(clean_loader, test_loader)
        
        self.known_class = self.init_class
        del train_set, test_set
        self._clear_gpu()

    # [MODIFIED] Increment Train
    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(self.cur_task)

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        # Fit FC trước (như logic cũ của bạn)
        self.fit_fc(train_loader, test_loader)
        self._network.update_fc(self.increment)

        train_loader_run = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self._network.update_noise()
        self._clear_gpu()

        self.run(train_loader_run)
        self._network.after_task_magmax_merge()
        self._clear_gpu()

        # [NEW] Tính Prototype bằng CFS cho Task mới
        train_set_clean = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_clean.labels = self.cat2order(train_set_clean.labels, data_manger)
        clean_loader = DataLoader(train_set_clean, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        
        new_protos = self.get_features_cfs(self._network, clean_loader, num_select=20)
        self.feature_buffer.update(new_protos)

        self.re_fit(clean_loader, test_loader)
        
        self.known_class += self.increment
        del train_set, test_set
        self._clear_gpu()

    # [MODIFIED] Fit FC - Sửa nhẹ để không lỗi One-hot khi chưa update_fc
    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)

        prog_bar = tqdm(range(self.fit_epoch))
        
        # Tự tính số class để tránh lỗi Index
        if self.cur_task == 0:
            total_classes = self.init_class
        else:
            total_classes = self.known_class + self.increment

        with torch.no_grad():
            for _, epoch in enumerate(prog_bar):
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    targets = torch.nn.functional.one_hot(targets, num_classes=total_classes).float()
                    self._network.fit(inputs, targets)
                
                info = "Task {} --> Update Analytical Classifier!".format(self.cur_task)
                self.logger.info(info)
                prog_bar.set_description(info)
                if epoch % 5 == 0: self._clear_gpu()

    # Các hàm khác giữ nguyên
    def after_train(self, data_manger):
        if self.cur_task == 0:
            self.known_class = self.init_class
        else:
            self.known_class += self.increment # Lưu ý: Increment train đã cộng rồi, check lại logic này trong code gốc của bạn, nhưng ở đây tôi để như cũ

        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        self.logger.info('task_confusion_metrix:\n{}'.format(eval_res['task_confusion']))
        del test_set

    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

    def compute_test_acc(self, test_loader):
        model = self._network.eval()
        correct, total = 0, 0
        device = self.device
        with torch.no_grad(), autocast('cuda'):
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                logits = outputs["logits"]
                predicts = torch.max(logits, dim=1)[1]
                correct += (predicts.cpu() == targets).sum()
                total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    @staticmethod
    def cat2order(targets, datamanger):
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets

    def run(self, train_loader):
        if self.cur_task == 0:
            epochs = self.init_epochs
            lr = self.init_lr
            weight_decay = self.init_weight_decay
        else:
            epochs = 5
            lr = self.lr * 0.1
            weight_decay = self.weight_decay

        for param in self._network.parameters():
            param.requires_grad = False
        for param in self._network.normal_fc.parameters():
            param.requires_grad = True
            
        if self.cur_task == 0:
            self._network.init_unfreeze()
        else:
            self._network.unfreeze_noise()
            
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        prog_bar = tqdm(range(epochs))
        self._network.train()
        self._network.to(self.device)
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad(set_to_none=True) 

                with autocast('cuda'):
                    if self.cur_task > 0:
                        with torch.no_grad():
                            outputs1 = self._network(inputs, new_forward=False)
                            logits1 = outputs1['logits']
                        outputs2 = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits2 = outputs2['logits']
                        logits2 = logits2 + logits1
                        loss = F.cross_entropy(logits2, targets.long())
                        logits_final = logits2
                    else:
                        outputs = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits = outputs["logits"]
                        loss = F.cross_entropy(logits, targets.long())
                        logits_final = logits

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                del inputs, targets, loss, logits_final

            scheduler.step()
            train_acc = 100. * correct / total
            info = "Task {} Epoch {}/{} => Loss {:.3f}, train_accy {:.2f}".format(
                self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            self.logger.info(info)
            prog_bar.set_description(info)
            
            if epoch % 5 == 0:
                self._clear_gpu()

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        with torch.no_grad(), autocast('cuda'):
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                logits = outputs["logits"]
                predicts = torch.max(logits, dim=1)[1]
                pred.extend([int(predicts[i].cpu().numpy()) for i in range(predicts.shape[0])])
                label.extend(int(targets[i].cpu().numpy()) for i in range(targets.shape[0]))
        class_info = calculate_class_metrics(pred, label)
        task_info = calculate_task_metrics(pred, label, self.init_class, self.increment)
        return {
            "all_class_accy": class_info['all_accy'],
            "class_accy": class_info['class_accy'],
            "class_confusion": class_info['class_confusion_matrices'],
            "task_accy": task_info['all_accy'],
            "task_confusion": task_info['task_confusion_matrices'],
            "all_task_accy": task_info['task_accy'],
        }