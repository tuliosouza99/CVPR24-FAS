import argparse
import os
from datetime import datetime

import easydict
import torch
import yaml
from lightning.fabric import Fabric

from dataloaders.utils import get_train_test_loader
from helpers.loops import eval_last, train, validate
from helpers.utils import (FixedLengthQueue, Logger, _seed_everything,
                           check_folder, save_model, set_trainable,
                           setup_device)
from models import get_model
from optimizers.gacfas import GACFAS


def main(config):
    torch.set_float32_matmul_precision(config.SYS.matmul_precision)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.SYS.GPUs
    accelerator, n_devices, precision = setup_device(config.SYS.num_gpus, config.SYS.precision)
    fabric = Fabric(accelerator=accelerator, devices=n_devices, precision=precision)
    fabric.launch()

    now = datetime.now()
    key_names = [
        'loss_func',
        'lr',
        'fc_lr_scale',
        'weight_decay',
        'epochs',
        'logit_scale',
        'lambda_constrast',
    ]
    config.running_name = (
        config.running_name
        + '_'.join(
            [
                f"{key}({value})"
                for key, value in config.TRAIN.items()
                if key in key_names
            ]
        )
        + f"_rho({config.TRAIN.get(config.TRAIN.minimizer.upper()).get('rho')})"
        + f"_alpha({config.TRAIN.get(config.TRAIN.minimizer.upper()).get('alpha')})"
        + now.strftime('[%b %d %H.%M]')
    )

    config.PATH.result_name = os.path.join(
        config.PATH.output_folder,
        config.MODEL.model_name,
        config.protocol,
        config.running_name,
    )

    config.PATH.model_path = os.path.join(config.PATH.result_name, "model")
    check_folder(config.PATH.model_path)
    config.PATH.score_path = os.path.join(config.PATH.result_name, "score")
    check_folder(config.PATH.score_path)
    config.PATH.record_path = os.path.join(config.PATH.result_name, "log.txt")
    logger = Logger(config.PATH.record_path)

    # save config files
    with open(os.path.join(config.PATH.result_name, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    train_loader, test_loader = get_train_test_loader(config)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    model = get_model(config)
    set_trainable(model, False, ['fc'])  # If warming up
    optimizer = torch.optim.SGD(
        [
            {
                "params": model.model.parameters(),
                "lr": config.TRAIN.lr * 1,
                "weight_decay": config.TRAIN.weight_decay,
            },
            {
                "params": model.fc.parameters(),
                "lr": config.TRAIN.lr * config.TRAIN.fc_lr_scale,
                "weight_decay": config.TRAIN.weight_decay,
            },
        ],
        lr=config.TRAIN.lr,
        momentum=config.TRAIN.momentum,
        weight_decay=config.TRAIN.weight_decay,
    )
    model, optimizer = fabric.setup(model, optimizer)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.TRAIN.lr_step_size, gamma=config.TRAIN.lr_gamma
    )

    minimizer = None
    if config.TRAIN.get('minimizer') == 'gac-fas':
        print(
            "Flatter: Multi-domain gradient aligment for cross domain face anti-spoofing"
        )
        minimizer = GACFAS(
            fabric=fabric,
            model=model,
            rho=config.TRAIN.get(config.TRAIN.minimizer.upper()).rho,
            eta=config.TRAIN.get(config.TRAIN.minimizer.upper()).eta,
            alpha=config.TRAIN.get(config.TRAIN.minimizer.upper()).alpha,
            n_domains=len(config.train_set),
        )

    best_eval = easydict.EasyDict(
        {"best_epoch": -1, "best_HTER": 100, "best_auc": -100}
    )
    accum_eval = easydict.EasyDict(
        {
            'HTER': FixedLengthQueue(10),
            'auc': FixedLengthQueue(10),
            'tpr': FixedLengthQueue(10),
        }
    )
    for epoch in range(config.TRAIN.epochs):
        if epoch == config.TRAIN.warming_epochs:  # Finish warming up
            set_trainable(model, True)

        train(
            fabric,
            model,
            epoch,
            train_loader,
            optimizer,
            config,
            minimizer,
            logger=logger,
            test_loader=test_loader,
            save_func=save_model,
            scheduler=scheduler,
            best_eval=best_eval,
            accum_eval=accum_eval,
        )
        scheduler.step()

        if (epoch > 0) and (
            (epoch % config.TEST.eval_preq == 0) or (epoch >= config.TRAIN.epochs - 10)
        ):
            cur_eval = validate(model, epoch, test_loader, config, logger=logger)
            accum_eval.HTER.enqueue(cur_eval['HTER'])
            accum_eval.auc.enqueue(cur_eval['auc'])
            accum_eval.tpr.enqueue(cur_eval['tpr'])
            logger.log(
                "[Avg. result]\t:  HTER={:.4f}+-{:.4f}, AUC={:.4f}+-{:.4f}, TPR={:.4f}+-{:.4f}".format(
                    accum_eval.HTER.avg(),
                    accum_eval.HTER.std(),
                    accum_eval.auc.avg(),
                    accum_eval.auc.std(),
                    accum_eval.tpr.avg(),
                    accum_eval.tpr.std(),
                )
            )
            save_model(
                fabric,
                best_eval,
                cur_eval,
                model,
                epoch,
                optimizer,
                scheduler,
                config,
                logger=logger,
            )

    eval_last(best_eval, config, logger=logger)  # eval last 10 epochs
    logger.close()

    return None


if __name__ == '__main__':
    # args for config file
    parser = argparse.ArgumentParser(description='Training Face Anti-Spoofing')
    parser.add_argument('--config', type=str, help='configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as cf:
        config = yaml.safe_load(cf)
    config = easydict.EasyDict(config)
    _seed_everything(1223)
    main(config)
