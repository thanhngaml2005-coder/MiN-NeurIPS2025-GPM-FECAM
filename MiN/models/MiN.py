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
# Mixed Precision
from torch.amp import autocast, GradScaler

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
        
        # Scaler cho Mixed Precision
        self.scaler = GradScaler('cuda')

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
    # [NEW] HÀM TÍNH PROTOTYPE (Class-wise Mean)
    # Tính trung bình feature của từng class để khởi tạo Classifier
    # =========================================================================
    def get_class_prototypes(self, train_loader):
        model = self._network
        model.eval()
        
        # Dùng Dictionary để cộng dồn feature trên CPU (Tránh OOM)
        features_sum = {}
        features_count = {}
        
        self.logger.info(">>> Calculating Class Prototypes (on CPU)...")
        
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(tqdm(train_loader, desc="Extracting Features")):
                inputs = inputs.to(self.device)
                
                # Lấy feature từ Backbone (qua buffer nếu cần)
                with autocast('cuda'):
                    # Gọi trực tiếp backbone để lấy raw features
                    features = model.backbone(inputs)
                    # Nếu model có buffer, có thể cần qua buffer: features = model.buffer(features)
                    # Nhưng thường prototype lấy raw feature từ backbone
                
                # Chuyển về CPU để tính toán
                features = features.detach().cpu()
                targets = targets.cpu()
                
                for feat, label in zip(features, targets):
                    lbl = int(label.item())
                    if lbl not in features_sum:
                        features_sum[lbl] = feat
                        features_count[lbl] = 1
                    else:
                        features_sum[lbl] += feat
                        features_count[lbl] += 1
        
        # Tính trung bình và tạo Tensor Weight
        num_classes = self._network.normal_fc.out_features
        feature_dim = self._network.feature_dim # Hoặc self._network.backbone.out_dim
        
        # Init weight bằng 0
        prototypes = torch.zeros(num_classes, feature_dim)
        
        for lbl in features_sum:
            if lbl < num_classes:
                # Mean = Sum / Count
                prototypes[lbl] = features_sum[lbl] / features_count[lbl]
        
        return prototypes.to(self.device)

    # =========================================================================
    # TASK 0: PROTOTYPE INITIALIZATION
    # =========================================================================
    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(0)
        
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        
        train_loader_noise = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        # Loader tuần tự để refit
        train_loader_final = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False
                
        # 1. Khởi tạo Classifier
        self._network.update_fc(self.init_class)
        
        # 2. Kích hoạt BiLORA
        self.logger.info(">>> Activating BiLORA for Task 0...")
        self._network.update_noise() 
        self._network.to(self.device)
        self._clear_gpu()

        # ---------------------------------------------------------------------
        # [PROTOTYPE INITIALIZATION]
        # Thay vì RLS, ta dùng Mean Feature của từng class gán vào Weight
        # Đây chính là lý do MiN gốc đạt 70% Acc ngay Epoch 0
        # ---------------------------------------------------------------------
        prototypes = self.get_class_prototypes(train_loader_noise)
        
        # Gán Prototype vào Classifier Weight
        with torch.no_grad():
            self._network.normal_fc.weight.data.copy_(prototypes)
            self.logger.info(">>> Classifier initialized with Class Prototypes!")
        
        # Dọn dẹp RAM
        self._clear_gpu()
        # ---------------------------------------------------------------------

        # STEP 1: Train SGD (BiLORA)
        # Lúc này Classifier đã biết mặt mũi dữ liệu (nhờ Prototype) nên Acc sẽ cao
        self.logger.info(">>> Step 1: Training BiLORA Noise (Joint Training)...")
        self.run(train_loader_noise)

        # STEP 2: Merge
        self.logger.info(">>> Step 2: MagMax Merge...")
        self._network.after_task_magmax_merge()
        
        # STEP 3: Re-fit
        self.logger.info(">>> Step 3: Analytic Re-fit...")
        
        del train_set
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_refit = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        
        self.re_fit(train_loader_refit)
        self._clear_gpu()

    # =========================================================================
    # TASK > 0: ENSEMBLE LOGIC
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

        # STEP 1: Analytic Update (Chuẩn bị Teacher cho Ensemble)
        self.logger.info(">>> Step 1: Analytic Update (Preparing Teacher)...")
        self.fit_fc(train_loader_analytic)
        
        # [QUAN TRỌNG] KHÔNG SYNC!
        # Vì ta dùng phép cộng (Ensemble), nên SGD phải học độc lập với RLS.

        # STEP 2: Train Noise (ENSEMBLE MODE)
        self.logger.info(">>> Step 2: Training BiLORA Noise (Ensemble Mode)...")
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
    # FIT FC (RLS)
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
                
                # RLS cần chính xác cao, không dùng autocast
                with autocast('cuda', enabled=False):
                    self._network.fit(inputs, targets_onehot)
            self._clear_gpu()

    # =========================================================================
    # RE-FIT
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
    # TRAIN LOOP (SGD) - ENSEMBLE LOGIC
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

        # Freeze Backbone & RLS
        for param in self._network.parameters():
            param.requires_grad = False
        
        # Unfreeze SGD Classifier
        for param in self._network.normal_fc.parameters():
            param.requires_grad = True
            
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

                with autocast('cuda'):
                    # ----------------------------------------------------
                    # LOGIC ENSEMBLE CHO TASK > 0
                    # ----------------------------------------------------
                    if self.cur_task > 0:
                        # 1. Analytic Teacher (RLS) - Freeze
                        with torch.no_grad():
                            outputs_analytic = self._network(inputs, new_forward=False)
                            logits_analytic = outputs_analytic['logits']
                        
                        # 2. SGD Student (Normal FC) - Trainable
                        outputs_sgd = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits_sgd = outputs_sgd['logits']
                        
                        # 3. Cộng gộp
                        logits = logits_analytic + logits_sgd
                    else:
                        # Task 0: Chỉ chạy SGD (Đã được init tốt nhờ Prototype ở trên)
                        outputs = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits = outputs['logits']

                    loss = F.cross_entropy(logits, targets.long())

                # Backward với Scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += len(targets)

            scheduler.step()
            
            train_acc = 100. * correct / total if total > 0 else 0.0
            info = f"Ep {epoch+1}/{epochs} | Loss: {losses/len(train_loader):.3f} | Acc: {train_acc:.2f}%"
            prog_bar.set_description(info)
            self.logger.info(info)
            print(f"Ep {epoch+1}/{epochs} | Loss: {losses/len(train_loader):.3f} | Acc: {train_acc:.2f}%")
            
            if epoch % 5 == 0:
                self._clear_gpu()