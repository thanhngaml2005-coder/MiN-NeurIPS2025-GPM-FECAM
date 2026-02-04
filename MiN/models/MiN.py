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
from utils.toolkit import tensor2numpy, calculate_class_metrics, calculate_task_metrics
from utils.training_tool import get_optimizer, get_scheduler

# Import Mixed Precision
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

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
        
        # Snapshot Model cũ để dùng cho DPCR
        self._old_network = None

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

    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

    # =========================================================================
    # [EVALUATION] Softmax Ensemble (Analytic + DPCR-NCM)
    # =========================================================================

    def compute_test_acc(self, test_loader):
        model = self._network.eval()
        correct, total = 0, 0
        
        # Hệ số hòa trộn giữa Analytic Classifier và NCM (đã sửa Drift)
        LAMBDA = 0.6 
        
        with torch.no_grad(), autocast('cuda'):
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                
                # 1. Analytic Prediction (16k-dim)
                outputs = model(inputs)
                logits_anal = outputs["logits"]
                
                # 2. DPCR-NCM Prediction (768-dim)
                # Chỉ chạy nếu đã có prototypes (từ Task 0 trở đi)
                if len(model.dpcr.saved_protos) > 0:
                    logits_ncm = model.predict_ncm_simple(inputs)
                    
                    # Softmax Ensemble
                    prob_anal = F.softmax(logits_anal, dim=1)
                    prob_ncm = F.softmax(logits_ncm, dim=1)
                    
                    final_prob = prob_anal + LAMBDA * prob_ncm
                    predicts = torch.max(final_prob, dim=1)[1]
                else:
                    predicts = torch.max(logits_anal, dim=1)[1]
                
                correct += (predicts.cpu() == targets).sum()
                total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        LAMBDA = 0.6
        
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                
                outputs = model(inputs)
                logits_anal = outputs["logits"]
                
                if len(model.dpcr.saved_protos) > 0:
                    logits_ncm = model.predict_ncm_simple(inputs)
                    
                    prob_anal = F.softmax(logits_anal, dim=1)
                    prob_ncm = F.softmax(logits_ncm, dim=1)
                    
                    final_prob = prob_anal + LAMBDA * prob_ncm
                    predicts = torch.max(final_prob, dim=1)[1]
                else:
                    predicts = torch.max(logits_anal, dim=1)[1]
                
                pred.extend(predicts.cpu().numpy().tolist())
                label.extend(targets.cpu().numpy().tolist())
                
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

    # =========================================================================
    # [TRAINING LOGIC]
    # =========================================================================

    @staticmethod
    def cat2order(targets, datamanger):
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets

    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(0)
        self.logger.info("task_list: {}".format(train_list_name))
        self.logger.info("task_order: {}".format(train_list))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        
        self._clear_gpu()
        self.run(train_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        
        self._clear_gpu()
        train_loader_buf = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader_buf = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self.fit_fc(train_loader_buf, test_loader_buf)

        train_set_noaug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_noaug.labels = self.cat2order(train_set_noaug.labels, data_manger)
        train_loader_noaug = DataLoader(train_set_noaug, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        self.re_fit(train_loader_noaug, test_loader_buf)
        
        # [DPCR INITIALIZATION]
        # Lưu Prototypes cho Task 0 để làm mốc so sánh drift cho Task 1
        print("--> [DPCR] Initializing stats/prototypes for Task 0...")
        self._network.dpcr.update_stats(self._network, train_loader_noaug)
        
        # Snapshot Model
        self._old_network = copy.deepcopy(self._network)
        self._old_network.eval()
        for p in self._old_network.parameters(): p.requires_grad = False
        
        del train_set, test_set, train_set_noaug
        self._clear_gpu()

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info("task_list: {}".format(train_list_name))
        self.logger.info("task_order: {}".format(train_list))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        self.fit_fc(train_loader, test_loader)
        self._network.update_fc(self.increment)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self._network.update_noise()
        
        if self._old_network is None:
            self._old_network = copy.deepcopy(self._network)
        
        self._clear_gpu()
        self.run(train_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        
        self._clear_gpu()
        del train_set

        train_set_noaug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_noaug.labels = self.cat2order(train_set_noaug.labels, data_manger)
        train_loader_noaug = DataLoader(train_set_noaug, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        self.re_fit(train_loader_noaug, test_loader)
        
        # [DPCR DRIFT CORRECTION LOGIC]
        
        # 1. Tính Stats của Task mới (Mean/Cov mới)
        self._network.dpcr.update_stats(self._network, train_loader_noaug)
        
        # 2. Tính Drift & Sửa Prototypes cũ
        # Đây là bước quan trọng nhất: Prototypes cũ sẽ được xoay/dịch chuyển theo P
        self._network.dpcr.correct_drift(
            old_model=self._old_network, 
            new_model=self._network, 
            current_loader=train_loader_noaug, 
            known_classes=self.known_class - self.increment
        )
        
        # Lưu ý: Không cần gọi update_fecam_stats nữa vì ta đã bỏ FeCAM.
        # Hàm predict_ncm_simple sẽ tự lấy Prototypes từ self._network.dpcr.saved_protos
        
        # 3. Snapshot Model
        self._old_network = copy.deepcopy(self._network)
        self._old_network.eval()
        for p in self._old_network.parameters(): p.requires_grad = False
        
        del train_set_noaug, test_set
        self._clear_gpu()

    # --- Các hàm hỗ trợ GIỮ NGUYÊN ---
    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        prog_bar = tqdm(range(self.fit_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.to(self.device)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = torch.nn.functional.one_hot(targets)
                self._network.fit(inputs, targets)
            info = f"Task {self.cur_task} --> Update Analytical Classifier!"
            self.logger.info(info)
            prog_bar.set_description(info)
            self._clear_gpu()

    def re_fit(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        prog_bar = tqdm(train_loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = torch.nn.functional.one_hot(targets)
            self._network.fit(inputs, targets)
            info = f"Task {self.cur_task} --> Reupdate Analytical Classifier!"
            self.logger.info(info)
            prog_bar.set_description(info)
        self._clear_gpu()

    def compute_adaptive_scale(self, current_loader):
        curr_proto = self.get_task_prototype(self._network, current_loader)
        if not hasattr(self, 'old_prototypes'): self.old_prototypes = []
        if not self.old_prototypes:
            self.old_prototypes.append(curr_proto)
            return 0.95 
        max_sim = 0.0
        curr_norm = F.normalize(curr_proto.unsqueeze(0), p=2, dim=1)
        for old_p in self.old_prototypes:
            old_norm = F.normalize(old_p.unsqueeze(0), p=2, dim=1)
            sim = torch.mm(curr_norm, old_norm.t()).item()
            if sim > max_sim: max_sim = sim
        self.old_prototypes.append(curr_proto)
        scale = 0.5 + 0.5 * (1.0 - max_sim)
        scale = max(0.65, min(scale, 0.95))
        self.logger.info(f"--> [ADAPTIVE] Similarity: {max_sim:.4f} => Scale: {scale:.4f}")
        return scale

    def run(self, train_loader):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay
        current_scale = 0.85 
        if self.cur_task > 0: current_scale = self.compute_adaptive_scale(train_loader)

        for param in self._network.parameters(): param.requires_grad = False
        for param in self._network.normal_fc.parameters(): param.requires_grad = True
        
        if self.cur_task == 0: self._network.init_unfreeze()
        else: self._network.unfreeze_noise()
            
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        prog_bar = tqdm(range(epochs))
        self._network.train()
        self._network.to(self.device)
        WARMUP_EPOCHS = 2

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True) 
                with autocast('cuda'):
                    if self.cur_task > 0:
                        with torch.no_grad():
                            logits1 = self._network(inputs, new_forward=False)['logits']
                        logits2 = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                        logits_final = logits2 + logits1
                    else:
                        logits_final = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                    loss = F.cross_entropy(logits_final, targets.long())

                self.scaler.scale(loss).backward()
                if self.cur_task > 0 and epoch >= WARMUP_EPOCHS:
                    self.scaler.unscale_(optimizer)
                    self._network.apply_gpm_to_grads(scale=0.85)
                
                self.scaler.step(optimizer)
                self.scaler.update()
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                del inputs, targets, loss, logits_final

            scheduler.step()
            train_acc = 100. * correct / total
            info = f"Task {self.cur_task} | Ep {epoch + 1}/{epochs} | Loss {losses / len(train_loader):.3f} | Acc {train_acc:.2f} | Scale {current_scale:.2f}"
            self.logger.info(info)
            prog_bar.set_description(info)
            if epoch % 5 == 0: self._clear_gpu()

    def get_task_prototype(self, model, train_loader):
        model = model.eval()
        model.to(self.device)
        features = []
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                with autocast('cuda'):
                    feature = model.extract_feature(inputs)
                features.append(feature.detach().cpu())
        all_features = torch.cat(features, dim=0)
        prototype = torch.mean(all_features, dim=0).to(self.device)
        self._clear_gpu()
        return prototype
