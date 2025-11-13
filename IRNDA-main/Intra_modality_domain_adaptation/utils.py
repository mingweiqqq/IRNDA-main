"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os.path as osp
import time

from PIL import Image

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, \
    f1_score, balanced_accuracy_score
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

sys.path.append('../../../..')
import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset
import numpy as np
import wandb
import datapreprocessing


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + ['Digits']


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                            transform=train_target_transform)
        val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                  download=True, transform=val_transform)
        class_names = datasets.MNIST.get_classes()
        num_classes = len(class_names)
    elif dataset_name in datasets.__dict__:
        # load datasets from tllib.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        def concat_dataset(tasks, start_idx, **kwargs):
            # return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
            return MultipleDomainsDataset([dataset(task=task, **kwargs) for task in tasks], tasks,
                                          domain_ids=list(range(start_idx, start_idx + len(tasks))))

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform,
                                              start_idx=0)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform,
                                              start_idx=len(source))
        val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform,
                                     start_idx=len(source))
        if dataset_name == 'DomainNet':
            test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform,
                                          start_idx=len(source))
        else:
            test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(dataset_name)
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


import os
import pandas as pd

def save_metrics(metrics_dict, save_path="val_metrics.csv"):
    """保存指标到 CSV 文件，如果文件不存在会创建新文件，否则追加"""
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        df = pd.concat([df, pd.DataFrame([metrics_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([metrics_dict])
    df.to_csv(save_path, index=False)

def validate(val_loader, model, args, device, epoch=None, save_path="val_metrics.csv"):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test:')

    all_targets = []
    all_probs = []
    all_predictions = []

    model.eval()
    confmat = ConfusionMatrix(len(args.class_names)) if args.per_class_eval else None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images, target = images.to(device), target.to(device)

            # forward
            output = model(images)
            probs = torch.softmax(output, dim=1)
            predictions = torch.argmax(probs, dim=1)
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])
            all_predictions.extend(predictions.cpu().numpy())

            loss = F.cross_entropy(output, target)
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, predictions)

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # 计算指标
        auroc = roc_auc_score(all_targets, all_probs)
        auprc = average_precision_score(all_targets, all_probs)
        mean_acc = balanced_accuracy_score(all_targets, all_predictions)
        cm = confusion_matrix(all_targets, all_predictions)
        per_class_acc = np.diag(cm) / np.sum(cm, axis=1)
        gm = np.prod(per_class_acc) ** (1.0 / len(per_class_acc))
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')

        print(cm)
        print(f' * val_loss {losses.avg:.3f} | val_acc {top1.avg:.3f} | val_GM {gm:.3f} | val_mean_acc {mean_acc:.3f}')
        print(f' * val_auroc {auroc:.3f} | val_auprc {auprc:.3f} | val_precision {precision:.3f} | val_recall {recall:.3f} | val_f1 {f1:.3f}')
        if confmat:
            print(confmat.format(args.class_names))

        # 保存指标到 CSV
        metrics = {
            'epoch': epoch if epoch is not None else 0,
            'val_loss': losses.avg,
            'val_acc': top1.avg,
            'GM': gm,
            'mean_acc': mean_acc,
            'auroc': auroc,
            'auprc': auprc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        save_metrics(metrics, save_path)

    return top1.avg, losses.avg, mean_acc, auroc, auprc


def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        datapreprocessing.HairRemoval(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])






# Fcoal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        """
        FocalLoss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 FocalLoss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retina net中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retina net中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)

            # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
            self.alpha[0].fill_(alpha)
            self.alpha[1:].fill_(1 - alpha)

        self.gamma = gamma

        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))

    def forward(self, preds, labels):
        """
        FocalLoss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1

        ###############################
        # 一、初始操作
        ###############################

        # 按照最后一个维度重新调整矩阵形状，因为最后一个维度是分类数
        preds = preds.view(-1, preds.size(-1))

        alpha = self.alpha.to(preds.device)

        ###############################
        # 二、计算预测概率Pt
        # focalLoss(pt) = -(1-pt)^γ * log(pt)
        ###############################

        # 将 preds 张量在第 1 个维度上执行 softmax 操作，过softmax之后的，就是pt
        pt = preds_softmax = F.softmax(preds, dim=1)
        # 交叉熵损失函数 CELoss(pt) = -log(pt)，这个pt，就是预估值，多分类是softmax后的概率值，二分类是sigmoid后的值
        # 在softmax后面接着log，这样算交叉熵只用前面加个负号
        log_pt = preds_logSoftmax = torch.log(pt)

        ###############################
        # 三、选真实标签对应的数据
        ###############################

        # labels.view(-1,1) 是将 labels 张量的形状调整为 (N, 1)
        # Ensure the labels are long, not float
        labelsView = labels.view(-1, 1).long()
        # 下面操作的目的就是选出真实标签对应的pt
        pt = pt.gather(1, labelsView)
        # 下面操作的目的就是选出真实标签对应的log_pt
        log_pt = log_pt.gather(1, labelsView)

        ###############################
        # 四、不带α的focal-loss
        ###############################

        # focalLoss(pt) = -(1-pt)^γ * log(pt)
        loss = -torch.mul(torch.pow((1 - pt), self.gamma), log_pt)

        ###############################
        # 五、带上α的focal-loss
        ###############################
        # labels.view(-1) 的作用是将 labels 张量的形状调整为一个一维张量
        label_flatten = labelsView.view(-1)
        # 因为算softmax的时候，会把所有的值的softmax都算出来，然而我们需要的只要真实标签的那些而已
        # 所以要进行取值操作
        # 整句话的作用就是alpha根据label值，取到每个分类对应的数值α
        alpha = alpha.gather(0, label_flatten)
        # 损失乘上α向量操作
        loss = torch.mul(alpha, loss.t())

        # 根据需求，看损失是求平均还是求和
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# Adamatch
class DistAlignEMA():
    """
    Distribution Alignment Hook for conducting distribution alignment
    """

    def __init__(self, num_classes, momentum=0.999, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum

        # p_target
        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target)
        # print('distribution alignment p_target:', self.p_target)
        # p_model
        self.p_model = None

    @torch.no_grad()
    def dist_align(self, probs_x_ulb, probs_x_lb=None):
        # update queue
        self.update_p(probs_x_ulb, probs_x_lb)

        # dist align
        print((self.p_target + 1e-6) / (self.p_model + 1e-6))
        probs_x_ulb_aligned = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned

    @torch.no_grad()
    def update_p(self, probs_x_ulb, probs_x_lb):
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)

        # 分布式训练
        # if algorithm.distributed and algorithm.world_size > 1:
        #     if probs_x_lb is not None and self.update_p_target:
        #         probs_x_lb = concat_all_gather(probs_x_lb)
        #     probs_x_ulb = concat_all_gather(probs_x_ulb)

        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model == None:
            self.p_model = torch.mean(probs_x_ulb, dim=0)
        else:
            self.p_model = self.p_model * self.m + torch.mean(probs_x_ulb, dim=0) * (1 - self.m)

        if self.update_p_target:
            assert probs_x_lb is not None
            self.p_target = self.p_target * self.m + torch.mean(probs_x_lb, dim=0) * (1 - self.m)

    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        update_p_target = False
        if p_target_type == 'uniform':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
            update_p_target = True
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)

        return update_p_target, p_target





# Refixmatch
import torch
import torch.nn as nn
from torch.nn import functional as F
def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)

def consistency_loss(logits, targets, name='ce', mask=None):
    """
    wrapper for consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse', 'kl']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    elif name == 'kl':
        loss = F.kl_div(F.log_softmax(logits / 0.5, dim=-1), F.softmax(targets / 0.5, dim=-1), reduction='none')
        loss = torch.sum(loss * (1.0 - mask).unsqueeze(dim=-1).repeat(1, torch.softmax(logits, dim=-1).shape[1]), dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None and name != 'kl':
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()



class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """
    def forward(self, logits, targets, name='ce', mask=None):
        return consistency_loss(logits, targets, name, mask)


import math
from torch.optim.lr_scheduler import LambdaLR
def get_cosine_scheduler_with_warmup(optimizer, T_max, num_cycles=7. / 16., num_warmup_steps=0,
                                     last_epoch=-1):
    """
    Cosine learning rate scheduler from `FixMatch: Simplifying Semi-Supervised Learning with
    Consistency and Confidence (NIPS 2020) <https://arxiv.org/abs/2001.07685>`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        num_cycles (float): A scalar that controls the shape of cosine function. Default: 7/16.
        num_warmup_steps (int): Number of iterations to warm up. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    """

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, T_max - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)