import math
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
import gc  # [ADDED] Để dọn rác bộ nhớ
import os

from utils.inc_net import MiNbaseNet
from torch.utils.data import WeightedRandomSampler
from utils.toolkit import tensor2numpy, count_parameters
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler
from utils.toolkit import calculate_class_metrics, calculate_task_metrics

# [ADDED] Import Mixed Precision
from torch.amp import autocast, GradScaler

EPSILON = 1e-8

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
        self.class_acc = []
        self.task_acc = []
        
        # [ADDED] Scaler cho Mixed Precision
        self.scaler = GradScaler('cuda')
    

    def _clear_gpu(self):
        # [ADDED] Hàm dọn dẹp GPU
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

    def compute_test_acc(self, test_loader):
        model = self._network.eval()
        correct, total = 0, 0
        device = self.device
        # [MODIFIED] Thêm no_grad và autocast để test nhanh và nhẹ hơn
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
        train_list, test_list, train_list_name = data_manger.get_task_list(0)
        self.logger.info("task_list: {}".format(train_list_name))
        self.logger.info("task_order: {}".format(train_list))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
                                 num_workers=self.num_workers)

        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        
        # [FIX OOM] Dọn GPU trước và sau khi tính proto
        self._clear_gpu()
        
        
        self.run(train_loader)
        self._network.after_task_magmax_merge()
        
        self._clear_gpu()
       
        
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)
        self.fit_fc(train_loader, test_loader)

        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        # Gọi kiểu mới (Nhanh, không OOM)
        # PHẢI THÊM trước khi gọi re_fit:
        cfs_feats, cfs_lbls = self.select_diverse_features_cfs(
            self._network, 
            train_loader, 
            n_samples_per_class=self.buffer_size // self.known_class
        )

        # Sau đó mới gọi:
        self.re_fit((cfs_feats, cfs_lbls), test_loader)
        # [ADDED] Clear memory
        del train_set, test_set
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

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)

        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.fit_fc(train_loader, test_loader)

        self._network.update_fc(self.increment)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers)
        self._network.update_noise()
        
        self._clear_gpu()

        
        self.run(train_loader)
        self._network.after_task_magmax_merge()
        self._clear_gpu()


        del train_set

        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                    num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                    num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        # Gọi kiểu mới (Nhanh, không OOM)
        # PHẢI THÊM trước khi gọi re_fit:
        cfs_feats, cfs_lbls = self.select_diverse_features_cfs(
            self._network, 
            train_loader, 
            n_samples_per_class=self.buffer_size // self.known_class
        )

        # Sau đó mới gọi:
        self.re_fit((cfs_feats, cfs_lbls), test_loader)
        del train_set, test_set
        self._clear_gpu()

    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        
        # [FIX] Bật Gradient Checkpointing (nếu có)
        if hasattr(self._network.backbone, 'set_grad_checkpointing'):
            self._network.backbone.set_grad_checkpointing(True)

        # [CHUẨN HÓA] Lấy số class trực tiếp từ mạng -> Không bao giờ sai!
        real_num_classes = self._network.normal_fc.out_features

        prog_bar = tqdm(range(self.fit_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.to(self.device)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # [FIX QUAN TRỌNG] Phải ép num_classes cố định
                # Nếu không, one_hot sẽ bị co giãn theo max(targets) của từng batch -> LỖI MA TRẬN
                targets_onehot = F.one_hot(targets, num_classes=real_num_classes)
                
                # Fit Analytical (Dùng FP32 cho chính xác)
                self._network.fit(inputs, targets_onehot)
            
            info = "Task {} --> Update Analytical Classifier!".format(self.cur_task)
            self.logger.info(info)
            prog_bar.set_description(info)
            
            self._clear_gpu()

    def re_fit(self, train_data, test_loader=None):
        self._network.eval()
        self._network.to(self.device)
        
        # [CHUẨN HÓA] Lấy số class chuẩn từ mạng
        real_num_classes = self._network.normal_fc.out_features

        # 1. Nếu dùng CFS (Features, Targets)
        if isinstance(train_data, tuple) or isinstance(train_data, list):
            features, targets = train_data
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # One-hot chuẩn
            targets_onehot = F.one_hot(targets.long(), num_classes=real_num_classes)
            
            # Gọi hàm fit_features (Bypass backbone)
            self._network.fit_features(features, targets_onehot)
            
            self.logger.info(f"Task {self.cur_task} --> Re-fit done using {len(targets)} CFS features.")

        # 2. Nếu dùng DataLoader (Ảnh)
        else:
            prog_bar = tqdm(train_data)
            for i, (_, inputs, targets) in enumerate(prog_bar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # One-hot chuẩn
                targets_onehot = F.one_hot(targets, num_classes=real_num_classes)
                
                # Gọi hàm fit thường (Qua backbone)
                self._network.fit(inputs, targets_onehot)

                info = "Task {} --> Reupdate Analytical Classifier!".format(self.cur_task)
                prog_bar.set_description(info)
        
        self._clear_gpu()
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
                
                # [ADDED] set_to_none=True tiết kiệm RAM hơn
                optimizer.zero_grad(set_to_none=True) 

                # [ADDED] Autocast để giảm 50% VRAM khi train
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

                # [ADDED] Backward với Scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()

                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
                # [ADDED] Xóa biến tạm
                del inputs, targets, loss, logits_final

            scheduler.step()
            train_acc = 100. * correct / total

            info = "Task {} --> Learning Beneficial Noise!: Epoch {}/{} => Loss {:.3f}, train_accy {:.2f}".format(
                self.cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                train_acc,
            )
            self.logger.info(info)
            prog_bar.set_description(info)
            
            # [ADDED] Clear cache sau mỗi epoch
            if epoch % 5 == 0:
                self._clear_gpu()

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        # [ADDED] no_grad
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
            "class_accy": class_info['class_accy'],
            "class_confusion": class_info['class_confusion_matrices'],
            "task_accy": task_info['all_accy'],
            "task_confusion": task_info['task_confusion_matrices'],
            "all_task_accy": task_info['task_accy'],
        }

    def get_task_prototype(self, model, train_loader):
        model = model.eval()
        model.to(self.device)
        features = []
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                
                # Dùng autocast khi extract feature để nhanh hơn
                with autocast('cuda'):
                    feature = model.extract_feature(inputs)
                
                # .detach().cpu() là chìa khóa để tránh OOM
                features.append(feature.detach().cpu())
        
        # 2. Concat trên CPU (RAM thường lớn hơn VRAM)
        all_features = torch.cat(features, dim=0)
        prototype = torch.mean(all_features, dim=0).to(self.device)
        
        self._clear_gpu()
        return prototype
    def select_diverse_features_cfs(self, model, train_loader, n_samples_per_class=20):
        """
        Chọn feature đa dạng dùng CFS (Contrastive Feature Selection).
        """
        model.eval()
        device = self.device
        
        # 1. Thu thập toàn bộ feature (Đưa về CPU để tránh OOM)
        all_features = []
        all_labels = []
        
        with torch.no_grad(), autocast('cuda'):
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                features = model.extract_feature(inputs)
                all_features.append(features.detach().cpu())
                all_labels.append(targets.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        feature_dim = all_features.shape[1]
        unique_classes = torch.unique(all_labels)
        
        selected_features = []
        selected_targets = []
        
        print(f"Selecting {n_samples_per_class} diverse features/class via CFS...")
        
        # 2. Train nhẹ một mạng CFS để phân biệt feature
        # (Train trên tập hợp ngẫu nhiên nhỏ để học phân phối chung)
        # Lấy tối đa 1000 mẫu ngẫu nhiên để train CFS cho nhanh
        perm = torch.randperm(all_features.size(0))[:1000]
        train_feats = all_features[perm].to(device).float() # CFS cần float32
        
        cfs_net = CFS_Mapping(feature_dim).to(device).float()
        optimizer = torch.optim.Adam(cfs_net.parameters(), lr=1e-3)
        criterion = NegativeContrastiveLoss(tau=0.1)
        
        cfs_net.train()
        # Train nhanh 20 epoch
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            embeds = cfs_net(train_feats)
            loss = criterion(embeds)
            loss.backward()
            optimizer.step()
            
        cfs_net.eval()
        
        # 3. Dùng CFS đã train để chọn mẫu cho TỪNG CLASS
        for c in unique_classes:
            indices = (all_labels == c)
            class_feats = all_features[indices].to(device).float()
            
            # Nếu ít mẫu hơn số cần chọn thì lấy hết
            if class_feats.size(0) <= n_samples_per_class:
                selected_features.append(class_feats.cpu())
                selected_targets.append(torch.full((class_feats.size(0),), c))
                continue

            # Greedy Selection dựa trên CFS embedding
            with torch.no_grad():
                # Map sang không gian CFS
                class_embeds = cfs_net(class_feats) # [N, D]
                
                # Chọn mẫu đầu tiên: Mẫu gần Mean nhất (đại diện nhất)
                class_mean = torch.mean(class_embeds, dim=0, keepdim=True)
                sim_to_mean = (class_embeds @ class_mean.T).squeeze()
                best_idx = torch.argmax(sim_to_mean)
                
                chosen_indices = [best_idx.item()]
                chosen_embeds = class_embeds[best_idx].unsqueeze(0)
                
                # Chọn các mẫu tiếp theo: Khác biệt nhất với tập đã chọn
                for _ in range(n_samples_per_class - 1):
                    # Tính sim với các mẫu đã chọn
                    # [N_remaining, N_chosen]
                    sim_matrix = class_embeds @ chosen_embeds.T
                    
                    # Tìm mẫu mà độ tương đồng TRUNG BÌNH với tập đã chọn là NHỎ NHẤT (Đa dạng nhất)
                    # Hoặc MAX Similarity nhỏ nhất (xa mẫu gần nhất)
                    max_sim_values, _ = torch.max(sim_matrix, dim=1)
                    
                    # Mask những thằng đã chọn
                    max_sim_values[chosen_indices] = 999.0
                    
                    # Chọn thằng có max_sim nhỏ nhất (xa lạ nhất)
                    next_idx = torch.argmin(max_sim_values).item()
                    
                    chosen_indices.append(next_idx)
                    chosen_embeds = torch.cat([chosen_embeds, class_embeds[next_idx].unsqueeze(0)], dim=0)
            
            # Lưu lại feature GỐC (chưa qua CFS) của các index đã chọn
            selected_features.append(class_feats[chosen_indices].cpu())
            selected_targets.append(torch.full((len(chosen_indices),), c))
            
        # Dọn dẹp
        del cfs_net, train_feats, optimizer, criterion
        self._clear_gpu()
        
        return torch.cat(selected_features, dim=0), torch.cat(selected_targets, dim=0)
class NegativeContrastiveLoss(torch.nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau
    
    def forward(self, x): 
        # x: [Batch, Dim]
        x = F.normalize(x, dim=1)
        
        # Tính Cosine Similarity Matrix
        # [Batch, 1, Dim] * [1, Batch, Dim] -> [Batch, Batch]
        # Dùng mm thay vì unsqueeze để tiết kiệm bộ nhớ: x @ x.T
        cos = x @ x.T / self.tau
        
        exp_cos = torch.exp(cos)
        
        # Mask bỏ đường chéo (chính nó)
        mask = torch.eye(x.size(0), device=x.device).bool()
        exp_cos = exp_cos.masked_fill(mask, 0)
        
        # Loss: Muốn các feature khác nhau càng xa nhau càng tốt
        # log(sum(exp)) -> Minimize sự tương đồng
        loss = torch.log(exp_cos.sum(dim=1) / (x.size(0) - 1) + 1e-8)
        return torch.mean(loss)

class CFS_Mapping(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Mạng nhỏ để map feature sang không gian contrastive
        self.f_cont = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.LayerNorm(dim), 
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        x = self.f_cont(x)
        return F.normalize(x, p=2, dim=1)