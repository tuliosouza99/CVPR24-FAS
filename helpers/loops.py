import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from losses import supcon_loss
from optimizers.gacfas import GACFAS

from .performance import performances_val
from .utils import AvgrageMeter


def binary_func_sep(
    logits: torch.Tensor,
    label: torch.IntTensor,
    uuid: torch.IntTensor,
    loss_type: str,
    logit_scale: int,
    ce_loss_record_0: Optional[AvgrageMeter] = None,
    ce_loss_record_1: Optional[AvgrageMeter] = None,
    ce_loss_record_2: Optional[AvgrageMeter] = None,
    acc_record_0: Optional[AvgrageMeter] = None,
    acc_record_1: Optional[AvgrageMeter] = None,
    acc_record_2: Optional[AvgrageMeter] = None,
    device: Optional[torch.device] = None,
    return_sum: bool = True,
    n_sources: int = 3,
):
    label = label.float()
    correct_0, correct_1, correct_2 = 0, 0, 0
    total_0, total_1, total_2 = 1, 1, 1

    if loss_type == 'bce':
        logits = nn.Sigmoid()(logits * logit_scale)
        ce_loss = nn.BCELoss()

    elif loss_type == 'ce':
        assert (
            logits.size(1) >= 2
        ), f'Cannot apply CE loss with 2nd dim of {logits.size(1)}'
        label = label.long()
        ce_loss = nn.CrossEntropyLoss()

    else:
        raise ValueError('Unknown loss {}'.format(loss_type))

    logit_0 = []
    indx_0 = (uuid == 0).cpu()
    cls_loss_0 = torch.zeros(1).to(device)

    if indx_0.sum().item() > 0:
        logit_0 = logits[indx_0].squeeze()
        cls_loss_0 = ce_loss(logit_0, label[indx_0])
        predicted_0 = (
            (logit_0 > 0.5).float()
            if 'bce' in loss_type
            else (logit_0[:, 1] > 0.5).float()
        )
        total_0 += len(logit_0)
        correct_0 += predicted_0.cpu().eq(label[indx_0].cpu()).sum().item()

    logit_1 = []
    cls_loss_1 = torch.zeros(1).to(device)
    indx_1 = (uuid == 1).cpu()

    if indx_1.sum().item() > 0:
        logit_1 = logits[indx_1].squeeze()
        cls_loss_1 = ce_loss(logit_1, label[indx_1])
        predicted_1 = (
            (logit_1 > 0.5).float()
            if 'bce' in loss_type
            else (logit_1[:, 1] > 0.5).float()
        )
        total_1 += len(logit_1)
        correct_1 += predicted_1.cpu().eq(label[indx_1].cpu()).sum().item()

    logit_2 = []
    cls_loss_2 = torch.zeros(1).to(device)
    indx_2 = (uuid == 2).cpu()

    if indx_2.sum().item() > 0:
        logit_2 = logits[indx_2].squeeze()
        cls_loss_2 = ce_loss(logit_2, label[indx_2])
        predicted_2 = (
            (logit_2 > 0.5).float()
            if 'bce' in loss_type
            else (logit_2[:, 1] > 0.5).float()
        )
        total_2 += len(logit_2)
        correct_2 += predicted_2.cpu().eq(label[indx_2].cpu()).sum().item()

    if ce_loss_record_0 is not None:
        ce_loss_record_0.update(cls_loss_0.data.item(), len(logit_0))
        ce_loss_record_1.update(cls_loss_1.data.item(), len(logit_1))
        ce_loss_record_2.update(cls_loss_2.data.item(), len(logit_2))
        acc_record_0.update(correct_0 / total_0, total_0)
        acc_record_1.update(correct_1 / total_1, total_1)
        acc_record_2.update(correct_2 / total_2, total_2)

    if return_sum:
        return (cls_loss_0 + cls_loss_1 + cls_loss_2) / n_sources

    return [cls_loss_0 / n_sources, cls_loss_1 / n_sources, cls_loss_2 / n_sources][
        :n_sources
    ]


def train(
    fabric: Fabric,
    model: nn.Module,
    epoch: int,
    train_loader: DataLoader,
    optimizer: Optimizer,
    config: dict,
    minimizer: GACFAS,
    **kwargs,
):
    logger = kwargs.get('logger')

    ce_loss_record_0 = AvgrageMeter()
    ce_loss_record_1 = AvgrageMeter()
    ce_loss_record_2 = AvgrageMeter()

    acc_record_0 = AvgrageMeter()
    acc_record_1 = AvgrageMeter()
    acc_record_2 = AvgrageMeter()

    sum_bce_criterion = partial(
        binary_func_sep,
        device=fabric.device,
        loss_type=config.TRAIN.loss_func,
        logit_scale=config.TRAIN.logit_scale,
        return_sum=True,
        n_sources=len(config.train_set),
    )
    ele_bce_criterion = partial(
        binary_func_sep,
        device=fabric.device,
        loss_type=config.TRAIN.loss_func,
        logit_scale=config.TRAIN.logit_scale,
        return_sum=False,
        n_sources=len(config.train_set),
    )

    model.train()
    pbar = tqdm(enumerate(train_loader))

    for batch_idx, sample_batched in pbar:
        image_x_v1 = sample_batched["image_x_v1"]
        image_x_v2 = sample_batched["image_x_v2"]
        label = sample_batched["label"]
        uuid = sample_batched["uuid"]

        image_x2 = torch.cat([image_x_v1, image_x_v2])
        uuid2 = torch.cat([uuid, uuid])
        label2 = torch.cat([label, label])

        z, logits = model(image_x2, True)
        cls_loss = ele_bce_criterion(
            logits=logits,
            label=label2,
            uuid=uuid2,
            ce_loss_record_0=ce_loss_record_0,
            ce_loss_record_1=ce_loss_record_1,
            ce_loss_record_2=ce_loss_record_2,
            acc_record_0=acc_record_0,
            acc_record_1=acc_record_1,
            acc_record_2=acc_record_2,
        )

        feat_normed = F.normalize(z)
        f1, f2 = torch.split(feat_normed, [len(image_x_v1), len(image_x_v1)], dim=0)
        feat_loss = (
            supcon_loss(
                fabric.device,
                torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1),
                uuid * 10 + label,
                temperature=0.1,
            )
            * config.TRAIN.lambda_constrast
        )

        if (config.TRAIN.minimizer == 'gac-fas') and (
            epoch >= config.TRAIN.minimizer_warming
        ):

            model.zero_grad()
            assert isinstance(cls_loss, list) and len(cls_loss) == len(
                config.train_set
            ), f"list loss for domain is not provided"

            for idx_domain, _ in enumerate(config.train_set):
                if cls_loss[idx_domain].item() != 0:
                    fabric.backward(cls_loss[idx_domain], retain_graph=True)
                minimizer.get_perturb_norm(idx_domain)

            # ascent step
            for idx_domain, _ in enumerate(config.train_set):
                if cls_loss[idx_domain].item() != 0:
                    minimizer.ascent_step(idx_domain)

            fabric.backward(feat_loss)

            for idx_domain, _ in enumerate(config.train_set):
                if cls_loss[idx_domain].item() != 0:
                    minimizer.proxy_gradients(
                        idx_domain,
                        input=image_x2,
                        labels=label2,
                        loss_func=sum_bce_criterion,
                        uuid=uuid2,
                    )

            # Get average gradient at every top and update for main model.
            minimizer.sync_grad_step(cls_loss)
            if isinstance(cls_loss, list):
                cls_loss = sum(cls_loss)

            loss_all = cls_loss + feat_loss

        else:
            """
            Here we calculate gradient for the model
            """
            model.zero_grad()
            if isinstance(cls_loss, list):
                cls_loss = sum(cls_loss)

            loss_all = cls_loss + feat_loss
            fabric.backward(loss_all)

        if (epoch >= config.TRAIN.minimizer_warming) and (minimizer is not None):
            minimizer.descent_step()

        optimizer.step()
        optimizer.zero_grad()
        desc = 'epoch: {} [{}/{} ({:.0f}%)], loss: {:.6f} ce_loss_0={:.4f}, ce_loss_1={:.4f}, ce_loss_2={:.4f}, ACC_0={:.2f}, ACC_1={:.2f}, ACC_2={:.2f}'.format(
            epoch,
            batch_idx + 1,
            len(train_loader),
            100.0 * batch_idx / len(train_loader),
            loss_all.item(),
            ce_loss_record_0.avg,
            ce_loss_record_1.avg,
            ce_loss_record_2.avg,
            100.0 * acc_record_0.avg,
            100.0 * acc_record_1.avg,
            100.0 * acc_record_2.avg,
        )

        pbar.set_description(desc)
        if (batch_idx + 1) == len(train_loader):
            logger.log(desc, False)

    return None


def validate(
    model: nn.Module, epoch: int, test_loader: DataLoader, config: dict, **kwargs
):
    logger = kwargs.get('logger')

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        scores_list = []
        for i, sample_batched in enumerate(test_loader):
            image_x, live_label, _ = (
                sample_batched["image_x_v1"],
                sample_batched["label"],
                sample_batched["uuid"],
            )
            logits = model(image_x)
            logit = (
                logits
                if config.MODEL.num_classes == 1
                else F.softmax(logits, dim=1)[:, 1]
            )
            for i in range(len(logit)):
                scores_list.append(
                    "{} {}\n".format(logit.squeeze()[i].item(), live_label[i].item())
                )

    map_score_val_filename = os.path.join(
        config.PATH.score_path,
        "epoch_{}_{}_score.txt".format(epoch, config.test_set[0]),
    )
    logger.log("score: write test scores to {}".format(map_score_val_filename))
    with open(map_score_val_filename, 'w') as file:
        file.writelines(scores_list)

    test_ACC, fpr, FRR, HTER, auc_test, test_err, tpr = performances_val(
        map_score_val_filename
    )
    logger.log("## {} score:".format(config.test_set[0]))
    logger.log(
        "[Eval ~ {:.1f}s]\t:  HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}, TPR={:.4f}".format(
            time.time() - start_time, HTER, auc_test, test_err, test_ACC, tpr
        )
    )

    return {
        "acc": test_ACC,
        "fpr": fpr,
        "frr": FRR,
        "HTER": HTER,
        "auc": auc_test,
        "err": test_err,
        "tpr": tpr,
    }


def eval_last(eval_best, config, **kwargs):
    logger = kwargs.get('logger')
    epochs_saved = np.array(
        [
            int(
                dir.replace("epoch_", "").replace(
                    f"_{config.test_set[0]}_score.txt", ""
                )
            )
            for dir in os.listdir(config.PATH.score_path)
        ]
    )
    epochs_saved = np.sort(epochs_saved)
    last_n_epochs = epochs_saved[::-1][:10]

    HTERs, AUROCs, TPRs = [], [], []
    for epoch in last_n_epochs:
        map_score_val_filename = os.path.join(
            config.PATH.score_path,
            "epoch_{}_{}_score.txt".format(epoch, config.test_set[0]),
        )
        test_ACC, fpr, FRR, HTER, auc_test, test_err, tpr = performances_val(
            map_score_val_filename
        )
        HTERs.append(HTER)
        AUROCs.append(auc_test)
        TPRs.append(tpr)
        logger.log("## {} score:".format(config.test_set[0]))
        logger.log(
            "epoch:{:d}, test:  val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}, TPR={:.4f}".format(
                epoch, test_ACC, HTER, auc_test, test_err, test_ACC, tpr
            )
        )

    file = open(os.path.join(config.PATH.result_name, "summary.txt"), "a")
    L = [
        f"Best epoch: \t{eval_best['best_epoch']}\n",
        f"Best HTER: \t{eval_best['best_HTER'] * 100:.2f}\n",
        f"Best AUC: \t{eval_best['best_auc'] * 100:.2f}\n",
        "Average 10 last epoch:\n",
        f"HTER: \t{np.array(HTERs).mean() * 100:.2f} +- {np.array(HTERs).std() * 100:.2f}\n",
        f"AUC: \t{np.array(AUROCs).mean() * 100:.2f} +- {np.array(AUROCs).std() * 100:.2f}\n",
        f"TPR: \t{np.array(TPRs).mean() * 100:.2f} +- {np.array(TPRs).std() * 100:.2f}\n",
    ]
    file.writelines(L)
    file.close()
