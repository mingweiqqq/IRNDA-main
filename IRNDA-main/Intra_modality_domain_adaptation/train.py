import random
import time
import warnings
import argparse
import shutil

import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F

import utils
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
import datapreprocessing
import torchvision.transforms as transforms
from tllib.vision.transforms import MultipleApply
import numpy as np
import copy
import wandb
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, balanced_accuracy_score, precision_score, \
    f1_score
# from tllib.vision.transforms import ResizeImage
from tllib.self_training.dst import ImageClassifier, WorstCaseEstimationLoss
from augmentation import RandAugment, FIX_MATCH_AUGMENTATION_POOL

# 记录训练和验证的性能数据
# wandb.init(project="EXP", name='Test')
# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 添加这行导入
from PIL import Image


# 数据增强
# data_augmentation
class StainColorJitter(object):
    M = torch.tensor([
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78]], dtype=torch.float32)
    Minv = torch.inverse(M)
    eps = 1e-6

    def __init__(self, sigma=0.05, p=0.5):
        self.sigma = sigma
        self.p = p  # 执行变换的概率
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        # 将 PIL 图像转换为张量
        if isinstance(img, Image.Image):
            img = self.to_tensor(img)

        # 确保张量在 [0, 1] 范围内
        if img.max() > 1:
            img = img / 255.0

        assert img.shape[0] == 3, "输入张量应为3通道图像"

        # 以概率 p 决定是否执行颜色变换
        if random.random() < self.p:
            # 颜色变换逻辑
            S = -torch.log(255 * img.permute(1, 2, 0) + self.eps).matmul(self.Minv)
            alpha = 1 + (torch.rand(3, dtype=torch.float32) - 0.5) * 2 * self.sigma
            beta = (torch.rand(3, dtype=torch.float32) - 0.5) * 2 * self.sigma
            Sp = S * alpha + beta
            Pp = torch.exp(-Sp.matmul(self.M)) - self.eps
            Pp = Pp.permute(2, 0, 1) / 255
            Pp = torch.clip(Pp, 0.0, 1.0)
            return Pp
        else:
            # 不执行变换，直接返回原图像
            return img

    def __repr__(self):
        return f'{self.__class__.__name__}(sigma={self.sigma}, p={self.p})'


# KL散度计算
def kl_div_with_logit(q_logit, p_logit):
    ### return a matrix without mean over samples.
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = (q * logq).sum(dim=1)
    qlogp = (q * logp).sum(dim=1)

    return qlogq - qlogp


# EMA模型参数更新（保持模型稳定）
class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())

        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.data.copy_(param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)


def create_model(model, ema=False):
    if ema:
        # 创建独立的EMA模型，参数为原始模型的副本
        ema_model = copy.deepcopy(model)
        # 确保EMA模型参数不需要梯度
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    else:
        return model


# 平衡采样（对源域数据和目标域数据进行平衡采样）
def rebalance_loader(dataset, args, model=None, use_true_target=False, device=None, target_dataset=None, save_path = None):
    """
    Simple function to rebalance a dataset loader.
    """
    if not use_true_target:
        assert model is not None, "If use_true_target is False, model must be provided"

    if use_true_target:
        all_labels = []
        for sub_dataset in dataset.datasets:
            all_labels.extend(sub_dataset.targets)
        target = all_labels
        target_dataset = dataset

    else:
        dataset_copy = copy.deepcopy(dataset)
        dataloader = DataLoader(
            dataset_copy,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        target = infer_predictions(model, dataloader, device)
        model.train()

    # import pdb; pdb.set_trace()
    class_counts = np.bincount(target)
    print(class_counts)
    max_count = max(class_counts)
    class_counts_proxy = class_counts + 1e-8
    class_weights = max_count / class_counts_proxy
    class_weights[class_counts == 0] = 0
    weights = class_weights[target]
    sampler = WeightedRandomSampler(weights, len(weights))
    loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        drop_last=True
    )
    return loader


def save_metrics(metrics_dict, save_path="metrics.csv"):
    """保存指标到 CSV 文件，如果文件不存在会创建新文件，否则追加"""
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        df = pd.concat([df, pd.DataFrame([metrics_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([metrics_dict])
    df.to_csv(save_path, index=False)

from sklearn.metrics import recall_score, accuracy_score  # 导入accuracy_score
def infer_predictions(model, loader, device, epoch=None, save_path="new_our-pseudo-labels-metrics.csv"):
    """
    推理函数，计算 GM, bACC, Precision, F1，并保存到 CSV。
    epoch: 可以传入当前 epoch 或步骤，便于画曲线
    """
    model.eval()
    y_pred = []
    y_labels = []
    losses = AverageMeter('Loss', ':.4e')
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            # ---- 计算损失 ----
            loss = F.cross_entropy(outputs, predictions)
            losses.update(loss.item(), images.size(0))
            y_pred.extend(predictions.cpu().numpy())
            y_labels.extend(labels.cpu().numpy())

        # 计算指标
        cm = confusion_matrix(y_labels, y_pred)
        # 计算总体准确率
        acc = accuracy_score(y_labels, y_pred)
        per_class_acc = np.diag(cm) / np.sum(cm, axis=1)
        gm = np.prod(per_class_acc) ** (1.0 / len(per_class_acc))
        bACC = balanced_accuracy_score(y_labels, y_pred)
        precision = precision_score(y_labels, y_pred)
        recall = recall_score(y_labels, y_pred)
        f1 = f1_score(y_labels, y_pred)
        avg_loss = losses.avg

        # 打印时包含准确率
        print(f"Acc: {acc:.4f} | GM: {gm:.4f} | bACC: {bACC:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Loss: {avg_loss:.4f}")

        # 保存指标到 CSV（包含准确率）
        metrics = {
            'epoch': epoch if epoch is not None else 0,
            'Acc': acc,
            'GM': gm,
            'bACC': bACC,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Loss': avg_loss
        }
        save_metrics(metrics, save_path)
    return y_pred

from typing import Optional, List, Dict, Tuple, Callable
from tllib.modules.grl import WarmStartGradientReverseLayer


class GeneralModule(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: nn.Module,
                 head: nn.Module, adv_head: nn.Module, grl: Optional[WarmStartGradientReverseLayer] = None,
                 finetune: Optional[bool] = True):
        super(GeneralModule, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.bottleneck = bottleneck
        self.head = head
        self.adv_head = adv_head
        self.finetune = finetune
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
                                                       auto_step=False) if grl is None else grl

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        features = self.backbone(x)
        features = self.bottleneck(features)
        outputs = self.head(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.adv_head(features_adv)
        if self.training:
            return outputs, outputs_adv, features
        else:
            return outputs

    def step(self):
        """
        Gradually increase :math:`\lambda` in GRL layer.
        """
        self.grl_layer.step()

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """
        Return a parameters list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer.
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else base_lr},
            {"params": self.bottleneck.parameters(), "lr": base_lr},
            {"params": self.head.parameters(), "lr": base_lr},
            {"params": self.adv_head.parameters(), "lr": base_lr}
        ]
        return params


class ImageClassifier(GeneralModule):
    r"""Classifier for MDD.

    Classifier for MDD has one backbone, one bottleneck, while two classifier heads.
    The first classifier head is used for final predictions.
    The adversarial classifier head is only used when calculating MarginDisparityDiscrepancy.


    Args:
        backbone (torch.nn.Module): Any backbone to extract 1-d features from data
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024
        width (int, optional): Feature dimension of the classifier head. Default: 1024
        grl (nn.Module): Gradient reverse layer. Will use default parameters if None. Default: None.
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main classifier
        - outputs_adv: logits outputs by the adversarial classifier

    Shape:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, C)`, where C means the number of classes.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, classifier is an ImageClassifier
            >>> outputs, outputs_adv = classifier(x)
            >>> classifier.step()

    """

    def __init__(self, backbone: nn.Module, num_classes: int,
                 bottleneck_dim: Optional[int] = 1024, width: Optional[int] = 1024,
                 grl: Optional[WarmStartGradientReverseLayer] = None, finetune=True, pool_layer=None):
        grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
                                                  auto_step=False) if grl is None else grl

        if pool_layer is None:
            pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        bottleneck = nn.Sequential(
            pool_layer,
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        bottleneck[1].weight.data.normal_(0, 0.005)
        bottleneck[1].bias.data.fill_(0.1)

        # The classifier head used for final predictions.
        head = nn.Sequential(
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_classes)
        )
        # The adversarial classifier head
        adv_head = nn.Sequential(
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_classes)
        )
        for dep in range(2):
            head[dep * 3].weight.data.normal_(0, 0.01)
            head[dep * 3].bias.data.fill_(0.0)
            adv_head[dep * 3].weight.data.normal_(0, 0.01)
            adv_head[dep * 3].bias.data.fill_(0.0)
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck,
                                              head, adv_head, grl_layer, finetune)


def main(args: argparse.Namespace):
    log_address = os.path.join(args.log, args.domain)
    logger = CompleteLogger(log_address, args.phase)
    save_path_1 = os.path.join(log_address, "metrics.csv")
    save_path_2 = os.path.join(log_address, "pseudo-label-metrics.csv")


    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    # Data loading code
    # train_source_transform
    train_source_transform_weak = transforms.Compose([
        transforms.Resize(256),
        datapreprocessing.AdvancedHairAugmentation(hairs=5, hairs_folder='../data/melanoma_hairs', p=0.5),  # 头发
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=args.ratio),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        StainColorJitter(sigma=0.05, p=0.5),  # 颜色抖动
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_target_transform_strong = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        RandAugment(2, FIX_MATCH_AUGMENTATION_POOL),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # val_transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # weak_transform
    weak_transform = transforms.Compose([
        transforms.Resize(256),
        datapreprocessing.AdvancedHairAugmentation(hairs=5, hairs_folder='../data/melanoma_hairs', p=0.5),  # 头发
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        StainColorJitter(sigma=0.05, p=0.5),  # 颜色抖动
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # strong_transform
    strong_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        RandAugment(2, FIX_MATCH_AUGMENTATION_POOL),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_source_transform = train_source_transform_weak
    train_target_transform = MultipleApply([weak_transform, strong_transform])

    print("train_transform_weak: ", train_source_transform_weak)
    print("train_target_transform: ", train_target_transform)
    print("val_transform: ", val_transform)

    args.class_names = [0, 1]
    args.num_classes = 2


    labeled_train_dataset_1 = datapreprocessing.MyDataset(annotations_file='../data/metadata/H_Train.csv',
                                                          img_dir='../data/H_Train',
                                                          transform=train_source_transform)
    labeled_train_dataset_2 = datapreprocessing.MyDataset(annotations_file='../data/metadata/H_Val.csv',
                                                          img_dir='../data/H_Val',
                                                          transform=train_source_transform)
    train_source_dataset = torch.utils.data.ConcatDataset([labeled_train_dataset_1, labeled_train_dataset_2])

    train_target_dataset = datapreprocessing.MyDataset(annotations_file=args.targetfile, img_dir=args.targetimage,
                                                       transform=train_target_transform)

    test_dataset = val_dataset = datapreprocessing.MyDataset(annotations_file=args.targetfile,
                                                             img_dir=args.targetimage,
                                                             transform=val_transform)

    train_target_loader = DataLoader(train_target_dataset, batch_size=args.unlabeled_batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # 源域平衡采样
    train_source_balance_loader = rebalance_loader(train_source_dataset, args, model=None, use_true_target=True,
                                                   device=device)
    train_source_balance_iter = ForeverDataIterator(train_source_balance_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, args.num_classes, bottleneck_dim=args.bottleneck_dim,
                                 width=args.bottleneck_dim, pool_layer=pool_layer, finetune=True).to(device)
    ema_classifier = create_model(classifier, ema=True)

    # mdd_loss
    # mdd = WorstCaseEstimationLoss(args.margin).to(device)
    from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy as MarginDisparityDiscrepancy
    mdd = MarginDisparityDiscrepancy(args.margin).to(device)

    # define optimizer and lr_scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    ema_optimizer = WeightEMA(classifier, ema_classifier, alpha=args.ema_decay)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # 测试模型
    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best_auprc'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        test_auprc(test_loader, classifier, args, device)
        return

    # initialize q_hat
    q_hat = (torch.ones(args.num_classes) / args.num_classes).to(device)

    # start training
    best_mean_acc = 0.
    best_auprc = 0.
    for epoch in range(args.epochs):
        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_source_balance_iter, train_target_iter, classifier, ema_classifier, mdd,
              optimizer, ema_optimizer, lr_scheduler, epoch, args, device, q_hat, train_source_dataset,
          test_dataset, train_target_dataset, save_path_2)
        # evaluate on validation set
        acc1, loss, mean_acc, auroc, auprc = utils.validate(val_loader, ema_classifier, args, device,
                                                            save_path=save_path_1)
        # remember best acc@1 and save checkpoint
        torch.save(ema_classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if mean_acc > best_mean_acc:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_mean_acc = max(mean_acc, best_mean_acc)
        if auprc > best_auprc:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best_auprc'))
        best_auprc = max(auprc, best_auprc)

    print("best_mean_acc = {:3.3f}".format(best_mean_acc))
    print("best_auprc = {:3.3f}".format(best_auprc))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    logger.close()


def train(train_source_balance_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          classifier: ImageClassifier, ema_classifier, mdd, optimizer: SGD, ema_optimizer: WeightEMA,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, device, q_hat, train_source_dataset,
          test_dataset, train_target_dataset, save_path_2):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    self_training_losses = AverageMeter('Self Training Loss', ':6.2f')
    pseudo_label_ratios = AverageMeter('Pseudo Label Ratio', ':3.1f')
    pseudo_label_accs = AverageMeter('Pseudo Label Acc', ':3.1f')
    kl_losses = AverageMeter('KL Loss', ':6.2f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, trans_losses, cls_accs, self_training_losses,
         kl_losses, pseudo_label_accs, pseudo_label_ratios],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    classifier.train()
    mdd.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        optimizer.zero_grad()

        x_s, labels_s = next(train_source_balance_iter)
        (x_t, x_t_strong), labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        x_t_strong = x_t_strong.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t, x_t_strong), dim=0)
        outputs, outputs_adv, features = classifier(x)
        y_s, y_t, y_t_strong = outputs.chunk(3, dim=0)
        y_s_adv, y_t_adv, _ = outputs_adv.chunk(3, dim=0)

        # compute cross entropy loss on source domain
        cls_loss = F.cross_entropy(y_s, labels_s)

        # compute margin disparity discrepancy between domains
        # for adversarial classifier, minimize negative mdd is equal to maximize mdd
        transfer_loss = - mdd(y_s, y_s_adv, y_t, y_t_adv)

        # update q_hat
        q = torch.softmax(y_t.detach(), dim=1).mean(dim=0)
        q_hat = args.moment * q_hat + (1 - args.moment) * q

        # 伪标签
        _, pseudo_labels = F.softmax((y_t.detach() - args.tau * torch.log(q_hat)), dim=1).max(dim=1)
        # MASK
        energy = - torch.logsumexp(y_t.detach(), dim=1)
        mask_raw = energy.le(args.e_cutoff)
        mask = mask_raw.float()
        # KL_loss
        kl_loss = (kl_div_with_logit(y_t.detach(), y_t_strong) * mask).mean()

        # self-training loss
        self_training_loss = (F.cross_entropy((y_t_strong + args.tau * torch.log(q_hat)), pseudo_labels) * mask).mean()

        # total loss
        loss = cls_loss + transfer_loss + kl_loss + self_training_loss
        classifier.step()

        cls_acc = accuracy(y_s, labels_s)[0]
        losses.update(loss.item(), x_s.size(0))
        cls_losses.update(cls_loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        self_training_losses.update(self_training_loss.item(), x_s.size(0))
        kl_losses.update(kl_loss.item(), x_s.size(0))

        # ratio of pseudo labels
        n_pseudo_labels = mask.sum()
        ratio = n_pseudo_labels / x_t.size(0)
        pseudo_label_ratios.update(ratio.item() * 100, x_t.size(0))

        # accuracy of pseudo labels
        if n_pseudo_labels > 0:
            pseudo_labels = pseudo_labels * mask - (1 - mask)
            n_correct = (pseudo_labels == labels_t).float().sum()
            pseudo_label_acc = n_correct / n_pseudo_labels * 100
            pseudo_label_accs.update(pseudo_label_acc.item(), n_pseudo_labels)

        # wandb.log({
        #     'train_loss': losses.avg,
        #     'cls_loss': cls_losses.avg,
        #     'train_acc': cls_accs.avg,
        #     'trans_loss': trans_losses.avg,
        #     'train_self_training_loss': self_training_losses.avg,
        #     'train_pseudo_label_acc': pseudo_label_accs.avg,
        #     'train_pseudo_label_ratio': pseudo_label_ratios.avg
        # })

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            # 重采样
            train_target_iter = rebalance_loader(test_dataset, args, model=ema_classifier, use_true_target=False,
                                                 device=device, target_dataset=train_target_dataset, save_path = save_path_2)
            train_target_iter = ForeverDataIterator(train_target_iter)
            train_source_balance_loader = rebalance_loader(train_source_dataset, args, model=None, use_true_target=True,
                                                           device=device)
            train_source_balance_iter = ForeverDataIterator(train_source_balance_loader)

        if i % args.print_freq == 0:
            progress.display(i)


# 最后的测试
def test_auprc(test_loader, model, args, device):
    model.eval()
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            probs = torch.softmax(output, dim=1)
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])

    precision, recall, _ = precision_recall_curve(all_targets, all_probs)
    # 确保 recall 是单调递增的
    decreasing_indices = np.argsort(recall)[::-1]
    recall = recall[decreasing_indices]
    precision = precision[decreasing_indices]
    auprc_value = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUPRC = {auprc_value:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (AUPRC)')
    plt.legend()
    plt.grid(True)
    log_address = os.path.join(args.log, args.domain, 'AUPRC.png')
    plt.savefig(log_address)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDD for Unsupervised Domain Adaptation')

    # dataset parameters
    parser.add_argument('--domain', type=str, default='HLH')
    parser.add_argument('--targetfile', type=str, default='../data/metadata/HLH.csv',
                        help='Path to the annotations file')
    parser.add_argument('--targetimage', type=str, default='../data/B',
                        help='Path to the image directory')

    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.2], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')

    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')

    # 网络结构不太一样
    parser.add_argument('--bottleneck-dim', default=1024, type=int)
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')

    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('-ub', '--unlabeled-batch-size', default=32, type=int,
                        help='mini-batch size of unlabeled data (target domain) (default: 32)')

    # 算法超参数
    parser.add_argument('--margin', type=float, default=4., help="margin gamma")
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--moment', default=0.999, type=float,
                        help='momentum coefficient for updating q_hat (default: 0.999)')
    parser.add_argument('--tau', default=3, type=float,
                        help='debiased strength (default: 1)')
    parser.add_argument('--e_cutoff', default=-9.5, type=float, help='INPL')

    # optimizer parameters
    parser.add_argument('--lr', '--learning-rate', default=0.004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0002, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')

    # 训练的参数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')

    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', default='True',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs/Ours',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
