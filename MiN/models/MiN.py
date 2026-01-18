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
from utils.toolkit import tensor2numpy, count_parameters, calculate_class_metrics, calculate_task_metrics
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler

from torch.amp import autocast, GradScaler 

EPSILON = 1e-8

# --- CFS Helper Classes ---
class NegativeContrastiveLoss(torch.nn.Module):
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

class CFS_Mapping(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f_cont = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.LayerNorm(dim), 
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim)
        )
    def forward(self, x):
        x = self.f_cont(x)
        return F.normalize(x, p=2, dim=1)

# --- Main Class ---
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
        self.fit_epoch = args["fit_epochs"]

        self.known_class = 0
        self.cur_task = -1
        self.total_acc = []
        
        self.scaler = GradScaler('cuda')

    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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
        
        # [HYBRID PROTOTYPE] - Lưu CFS samples vào network
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        self._clear_gpu()

        self.run(train_loader)
        
        # Refinement
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.update_task_prototype(prototype)
        self._clear_gpu()
        
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self.fit_fc(train_loader, test_loader)

        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader, test_loader)
        
        self.known_class = self.init_class

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
        
        self.fit_fc(train_loader, test_loader)
        self._clear_gpu()

        self._network.update_fc(self.increment)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self._network.update_noise()
        
        # [HYBRID PROTOTYPE] - Lưu CFS samples vào network
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        self._clear_gpu()
        
        self.run(train_loader)
        
        # Refinement
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.update_task_prototype(prototype)
        self._clear_gpu()
        
        del train_set, train_loader

        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader, test_loader)
        
        self.known_class += self.increment

    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        
        if hasattr(self._network, 'set_grad_checkpointing'):
            self._network.set_grad_checkpointing(True)

        prog_bar = tqdm(range(self.fit_epoch))
        for _, epoch in enumerate(prog_bar):
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = torch.nn.functional.one_hot(targets)
                
                self._network.fit(inputs, targets)
            
            info = "Task {} --> Update Analytical Classifier!".format(self.cur_task)
            self.logger.info(info)
            prog_bar.set_description(info)
            if epoch % 5 == 0: gc.collect()

    # =========================================================================
    # [NEW] RE-FIT VỚI CFS SAMPLES
    # =========================================================================
    def re_fit(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        
        # [NEW] PHASE 1: Fit CFS Representative Samples trước
        if hasattr(self._network, 'task_cfs_samples') and self.cur_task < len(self._network.task_cfs_samples):
            cfs_samples = self._network.task_cfs_samples[self.cur_task]  # [20, feature_dim]
            
            # Tính toán base class ID và số class trong task
            if self.cur_task == 0:
                num_classes_in_task = self.init_class
                base_class_id = 0
            else:
                num_classes_in_task = self.increment
                base_class_id = self.init_class + (self.cur_task - 1) * self.increment
            
            # Phân bổ 20 CFS samples cho các class (chia đều)
            samples_per_class = 20 // num_classes_in_task
            cfs_labels = []
            for c in range(num_classes_in_task):
                class_id = base_class_id + c
                cfs_labels.extend([class_id] * samples_per_class)
            
            # Bù phần dư
            remainder = 20 - len(cfs_labels)
            if remainder > 0:
                cfs_labels.extend([base_class_id] * remainder)
            
            # Chuyển sang one-hot
            cfs_labels = torch.tensor(cfs_labels, device=self.device, dtype=torch.long)
            cfs_targets = torch.nn.functional.one_hot(cfs_labels, num_classes=self.known_class)
            
            # Fit CFS samples (KHÔNG dùng autocast vì RLS cần FP32)
            self._network.fit(cfs_samples, cfs_targets)
            
            self.logger.info(f"Task {self.cur_task} --> Fitted {len(cfs_samples)} CFS representative samples first!")
            print(f"[Re-fit] Fitted {len(cfs_samples)} CFS samples for Task {self.cur_task}")
        
        # [ORIGINAL] PHASE 2: Fit real data như cũ
        prog_bar = tqdm(train_loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = torch.nn.functional.one_hot(targets)
            self._network.fit(inputs, targets)
            
            info = "Task {} --> Reupdate Analytical Classifier with Real Data!".format(self.cur_task)
            self.logger.info(info)
            prog_bar.set_description(info)

    def run(self, train_loader):
        if self.cur_task == 0:
            epochs, lr, weight_decay = self.init_epochs, self.init_lr, self.init_weight_decay
        else:
            epochs, lr, weight_decay = self.epochs, self.lr, self.weight_decay

        for param in self._network.parameters(): param.requires_grad = False
        for param in self._network.normal_fc.parameters(): param.requires_grad = True
            
        if self.cur_task == 0: self._network.init_unfreeze()
        else: self._network.unfreeze_noise()
            
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        self._clear_gpu()
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        if hasattr(self._network, 'set_grad_checkpointing'):
            self._network.set_grad_checkpointing(True)

        prog_bar = tqdm(range(epochs))
        self._network.train()
        self._network.to(self.device)

        for _, epoch in enumerate(prog_bar):
            losses, correct, total = 0.0, 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast('cuda'):
                    if self.cur_task > 0:
                        with torch.no_grad():
                            outputs1 = self._network(inputs, new_forward=False)
                        outputs2 = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits_final = outputs2['logits'] + outputs1['logits']
                        loss = F.cross_entropy(logits_final, targets.long())
                    else:
                        outputs = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits_final = outputs["logits"]
                        loss = F.cross_entropy(logits_final, targets.long())

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                del inputs, targets, logits_final, loss 
            
            scheduler.step()
            train_acc = 100. * correct / total
            info = "Task {} Epoch {}/{} => Loss {:.3f}, Acc {:.2f}".format(self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            self.logger.info(info)
            prog_bar.set_description(info)
            if epoch % 5 == 0: gc.collect()
        self._clear_gpu()

    # =========================================================================
    #  [UPDATED] HYBRID PROTOTYPE + LƯU CFS SAMPLES
    # =========================================================================
    def get_task_prototype(self, model, train_loader):
        device = self.device
        model.eval()
        model.to(device)
        
        # 1. Thu thập Feature
        features_cpu = []
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                with autocast('cuda'):
                    feature = model.extract_feature(inputs)
                features_cpu.append(feature.detach().cpu())
        
        all_features = torch.cat(features_cpu, dim=0)
        
        # --- PART A: SIMPLE MEAN ---
        raw_simple_mean = torch.mean(all_features, dim=0).to(device)
        norm_simple_mean = F.normalize(raw_simple_mean, dim=0)
        
        # --- PART B: CFS MEAN ---
        feature_dim = all_features.shape[1]
        MAX_CFS_SAMPLES = 500
        
        if all_features.size(0) > MAX_CFS_SAMPLES:
            indices = torch.randperm(all_features.size(0))[:MAX_CFS_SAMPLES]
            train_feats = all_features[indices].to(device)
        else:
            train_feats = all_features.to(device)

        train_feats = train_feats.float()

        f_cont = CFS_Mapping(feature_dim).to(device)
        f_cont.float() 
        
        optimizer = torch.optim.Adam(f_cont.parameters(), lr=1e-3)
        criterion = NegativeContrastiveLoss(tau=0.1)
        
        f_cont.train()
        for _ in range(30): 
            optimizer.zero_grad(set_to_none=True)
            embeddings = f_cont(train_feats)
            loss = criterion(embeddings)
            loss.backward()
            optimizer.step()
            
        f_cont.eval()
        all_selected_feats = None
        samples_needed = 20
        sup_batch = 100
        
        # Greedy Selection
        with torch.no_grad():
            for step in range(samples_needed):
                eps = torch.randn([sup_batch, feature_dim], device=device)
                std_temp = torch.std(train_feats, dim=0) + EPSILON
                if step == 0: eps[0] = 0 
                candidate_feats = eps * std_temp + raw_simple_mean 
                
                if all_selected_feats is None:
                    all_selected_feats = candidate_feats[:1] 
                else:
                    cont_cand = f_cont(candidate_feats)
                    cont_selected = f_cont(all_selected_feats)
                    sim_matrix = torch.matmul(cont_cand, cont_selected.t())
                    avg_sim = torch.mean(sim_matrix, dim=1)
                    slt_ids = torch.argsort(avg_sim)[:1]
                    all_selected_feats = torch.cat([all_selected_feats, candidate_feats[slt_ids]], dim=0)

        # [NEW] LƯU 20 CFS SAMPLES VÀO NETWORK
        # Chuyển về feature space gốc (chưa qua buffer) để re-fit có thể dùng
        model._network.task_cfs_samples.append(all_selected_feats.detach().clone())
        self.logger.info(f"Task {self.cur_task} --> Saved {all_selected_feats.size(0)} CFS samples for re-fit")

        # Batch-wise Calculation
        cfs_mean_numerator = torch.zeros(feature_dim, device=device)
        total_weight_sum = 0.0
        batch_calc = 1024 
        
        with torch.no_grad():
            z_anchors = f_cont(all_selected_feats)
            
            for i in range(0, all_features.size(0), batch_calc):
                batch_real = all_features[i:i+batch_calc].to(device).float()
                z_real = f_cont(batch_real)
                
                sim_matrix = torch.matmul(z_real, z_anchors.t())
                max_sim, _ = torch.max(sim_matrix, dim=1)
                
                max_sim = torch.clamp(max_sim / 0.1, max=50) 
                raw_weights = torch.exp(max_sim)
                
                weighted_sum = torch.sum(batch_real * raw_weights.unsqueeze(1), dim=0)
                cfs_mean_numerator += weighted_sum
                total_weight_sum += torch.sum(raw_weights)
                
                del batch_real, z_real

        cfs_mean = cfs_mean_numerator / (total_weight_sum + EPSILON)
        norm_cfs_mean = F.normalize(cfs_mean, dim=0)

        # --- PART C: HYBRID FUSION ---
        final_prototype = 0.5 * norm_simple_mean + 0.5 * norm_cfs_mean
        
        del f_cont, optimizer, criterion, all_features, train_feats
        self._clear_gpu()
        
        return F.normalize(final_prototype, dim=0)

    def after_train(self, data_manger):
        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        print('total acc: {}'.format(self.total_acc))
        del test_set

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
            "task_confusion": task_info['task_confusion_matrices']
        }