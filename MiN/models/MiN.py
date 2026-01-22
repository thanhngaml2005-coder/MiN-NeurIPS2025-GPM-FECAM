import math
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
import gc
import os

from utils.inc_net import MiNbaseNet
from torch.utils.data import WeightedRandomSampler
from utils.toolkit import tensor2numpy, calculate_class_metrics, calculate_task_metrics
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler
# [FIXED] Chỉ import autocast, bỏ GradScaler vì không dùng cho số phức
from torch.amp import autocast

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
        self.fit_epoch = args["fit_epochs"]
        self.buffer_batch = args["buffer_batch"]

        self.known_class = 0
        self.cur_task = -1
        self.total_acc = []
        
        # [FIXED] Bỏ self.scaler = GradScaler('cuda') vì gây lỗi với BiLORA

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def after_train(self, data_manger):
        if self.cur_task == 0:
            self.known_class = self.init_class
        else:
            self.known_class += self.increment

        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
                                 num_workers=self.num_workers)
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        self.logger.info('task_confusion_metrix:\n{}'.format(eval_res['task_confusion']))
        print('total acc: {}'.format(self.total_acc))
        print('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        del test_set

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        with torch.no_grad():
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
    
    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)
    
    @staticmethod
    def cat2order(targets, datamanger):
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets

    # =========================================================================
    # TASK 0
    # =========================================================================
    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(0)
        
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        
        # Loader
        train_loader_noise = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        train_loader_analytic = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False
                
        self._network.update_fc(self.init_class)
        self._network.update_noise() 
        self._network.to(self.device)
        self._clear_gpu()

        # STEP 1: Analytic Warm-up
        self.logger.info(">>> Step 1: Analytic Warm-up...")
        self.fit_fc(train_loader_analytic)

        # STEP 2: Train BiLORA Noise
        self.logger.info(">>> Step 2: Training BiLORA Noise...")
        self.run(train_loader_noise)

        # STEP 3: Merge & Refit
        self.logger.info(">>> Step 3: MagMax Merge & Refit...")
        self._network.after_task_magmax_merge()
        
        del train_set
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_refit = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        
        self.re_fit(train_loader_refit)
        
        self._clear_gpu()

    # =========================================================================
    # TASK > 0
    # =========================================================================
    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(self.cur_task)

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)

        train_loader_noise = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        train_loader_analytic = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        
        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False
                
        self._network.update_fc(self.increment) 
        self._network.update_noise()
        self._network.to(self.device)
        self._clear_gpu()

        # STEP 1: Analytic Update
        self.logger.info(">>> Step 1: Analytic Update...")
        self.fit_fc(train_loader_analytic)

        # STEP 2: Train Noise
        self.logger.info(">>> Step 2: Training BiLORA Noise...")
        self.run(train_loader_noise)

        # STEP 3: Merge & Refit
        self.logger.info(">>> Step 3: MagMax Merge & Refit...")
        self._network.after_task_magmax_merge()
        
        del train_set
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_refit = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        
        self.re_fit(train_loader_refit)
        
        self._clear_gpu()

    # =========================================================================
    # FIT FC (Step 1)
    # =========================================================================
    def fit_fc(self, train_loader):
        self._network.eval()
        for param in self._network.parameters(): 
            param.requires_grad = False
            
        prog_bar = tqdm(range(self.fit_epoch), desc="Analytic Fitting")
        for _ in prog_bar:
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                num_classes = self._network.normal_fc.out_features
                targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
                
                with autocast('cuda', enabled=False):
                    self._network.fit(inputs, targets_onehot)
            self._clear_gpu()

    # =========================================================================
    # RE-FIT (Step 3)
    # =========================================================================
    def re_fit(self, train_loader):
        self._network.eval()
        self._network.to(self.device)
        
        prog_bar = tqdm(train_loader, desc="Analytic Re-fitting")
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            num_classes = self._network.normal_fc.out_features
            targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
            
            with autocast('cuda', enabled=False):
                self._network.fit(inputs, targets_onehot)
                
        self._clear_gpu()

    # =========================================================================
    # TRAIN LOOP (SGD) - NO SCALER
    # =========================================================================
    def run(self, train_loader):
        if self.cur_task == 0:
            epochs = self.init_epochs
            lr = self.init_lr
            weight_decay = self.init_weight_decay
        else:
            epochs = self.epochs
            lr = self.lr
            weight_decay = self.weight_decay

        # Freeze All
        for param in self._network.parameters():
            param.requires_grad = False
            
        # Unfreeze BiLORA
        if self.cur_task == 0:
            self._network.init_unfreeze()
        else:
            self._network.unfreeze_noise()
            
        params = list(filter(lambda p: p.requires_grad, self._network.parameters()))
        if not params: return

        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        self._network.train()
        self._network.to(self.device)
        
        prog_bar = tqdm(range(epochs), desc=f"Training Noise T{self.cur_task}")
        for epoch in prog_bar:
            losses = 0.0
            correct = 0
            total = 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)

                # Giữ Autocast cho backbone (FP16), BiLORA tự chạy FP32
                with autocast('cuda'):
                    outputs = self._network.forward_normal_fc(inputs)
                    logits = outputs['logits']
                    loss = F.cross_entropy(logits, targets.long())

                # [FIXED] Backward trực tiếp, không dùng scaler
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                
                losses += loss.item()

                # Tính Accuracy
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += len(targets)

            scheduler.step()
            
            train_acc = 100. * correct / total if total > 0 else 0.0
            info = f"Ep {epoch+1}/{epochs} | Loss: {losses/len(train_loader):.3f} | Acc: {train_acc:.2f}%"
            prog_bar.set_description(info)
            self.logger.info(info)
            
            if epoch % 5 == 0:
                self._clear_gpu()