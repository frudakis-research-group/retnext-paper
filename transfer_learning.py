import os
import json
import yaml
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, R2Score, MeanAbsoluteError
from jsonargparse import ArgumentParser
from lightning.pytorch import Trainer, seed_everything
from aidsorb.data import get_names, PCDDataset
from aidsorb.litmodels import PCDLit
from torchvision.transforms.v2 import Compose, RandomChoice
from retnext.modules import RetNeXt
from retnext.transforms import (
        AddChannelDim, BoltzmannFactor,
        RandomRotate90, RandomFlip, RandomReflect
        )


def main_cli():
    parser = ArgumentParser(prog='Finetune or train RetNeXt from scratch.')
    parser.add_class_arguments(Trainer, 'trainer')
    parser.add_argument('--voxels_path', type=str)
    parser.add_argument('--labels_path', type=str)
    parser.add_argument('--index_col', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--train_sizes', type=list)
    parser.add_argument('--n_runs', type=int)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--lineval', type=bool)
    parser.add_argument('--bias_init', type=bool, default=False)
    parser.add_argument('-c', '--config', action='config')

    return parser


def get_dataloaders(*, train_size, voxels_path, labels_path, index_col, target):
    train_names, val_names, test_names = [
            get_names(os.path.join(Path(voxels_path).parent, f'{stg}.json'))
            for stg in ['train', 'validation', 'test']
            ]

    # Use same preprocessing and augmentation as the pretrained models.
    transform_train = Compose([
        AddChannelDim(), BoltzmannFactor(),
        RandomChoice([
            torch.nn.Identity(),
            RandomRotate90(),
            RandomFlip(),
            RandomReflect()
            ])
        ])
    transform_eval = Compose([AddChannelDim(), BoltzmannFactor()])

    config_dataloaders = dict(num_workers=8, persistent_workers=True)

    train_set = PCDDataset(
            random.sample(train_names, k=train_size),  # Random subset of train_size.
            path_to_X=voxels_path,
            path_to_Y=labels_path,
            index_col=index_col,
            labels=[target],
            transform_x=transform_train,
    )

    val_set = PCDDataset(
            val_names,
            path_to_X=voxels_path,
            path_to_Y=labels_path,
            index_col=index_col,
            labels=[target],
            transform_x=transform_eval,
    )

    test_set = PCDDataset(
            test_names,
            path_to_X=voxels_path,
            path_to_Y=labels_path,
            index_col=index_col,
            labels=[target],
            transform_x=transform_eval,
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True, **config_dataloaders)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, **config_dataloaders)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, **config_dataloaders)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    parser = main_cli()
    cfg = parser.parse_args()

    results = {s: {} for s in cfg.train_sizes}

    # Add target and ckpt name to dirname.
    cfg.trainer.default_root_dir += f'from_{cfg.ckpt_path.replace("/", "_")}'
    if cfg.lineval is True:
        cfg.trainer.default_root_dir += f'-to_{cfg.target.replace("/", "_")}-lineval/'
    else:
        cfg.trainer.default_root_dir += f'-to_{cfg.target.replace("/", "_")}/'
    del cfg.config  # Not JSON serializable.

    default_root_dir = cfg.trainer.default_root_dir

    # Seeds random module as well.
    seed_everything(42, workers=True)

    print(f'\033[31;1mTraining from: {cfg.ckpt_path}\033[0m')

    for size in cfg.train_sizes:
        print(f'\033[32;1mTraining set size: {size}\033[0m')

        cfg.trainer.default_root_dir = default_root_dir + f'{size}/'  # New dir for each size.

        for run in range(cfg.n_runs):
            print(f'\033[34;1mRun number: {run}\033[0m')

            results[size][run] = {}

            # New objects for each experiment.
            init = parser.instantiate_classes(cfg)

            criterion = torch.nn.MSELoss()
            metric = MetricCollection([R2Score(), MeanAbsoluteError()])
            config_optimizer_scratch = None
            config_optimizer_finetune = None

            if cfg.ckpt_path == 'scratch':
                litmodel = PCDLit(
                        model=RetNeXt(),  # Randomly initialized model.
                        criterion=criterion,
                        metric=metric,
                        config_optimizer=config_optimizer_scratch,
                        )
            # Note: finetuning takes more time than linear evaluation and is
            # more prone to overfitting.
            else:
                litmodel = PCDLit.load_from_checkpoint(
                        cfg.ckpt_path,  # Pretrained model.
                        criterion=criterion,
                        metric=metric,
                        config_optimizer=config_optimizer_finetune,
                        )
                litmodel.freeze()

                litmodel.model.fc = torch.nn.Linear(128, 1)

                # Finetune only the last two conv layers.
                if cfg.lineval is False:
                    litmodel.model.backbone[7:9].requires_grad_(True)
                    litmodel.model.backbone[7:9].train()

            train_loader, val_loader, test_loader = get_dataloaders(
                    train_size=size,
                    voxels_path=cfg.voxels_path,
                    labels_path=cfg.labels_path,
                    index_col=cfg.index_col,
                    target=cfg.target,
                    )

            # Initialize bias of last layer with y_mean to ease optimization.
            if cfg.bias_init is True:
                train_names = list(train_loader.dataset.pcd_names)
                y_train_mean = train_loader.dataset.Y.loc[train_names].mean().item()
                torch.nn.init.constant_(litmodel.model.fc.bias, y_train_mean)

            trainer = init.trainer
            trainer.fit(litmodel, train_dataloaders=train_loader, val_dataloaders=val_loader)

            results[size][run] = trainer.test(litmodel, dataloaders=test_loader, ckpt_path='best')

            del train_loader, val_loader, test_loader

        with open(cfg.trainer.default_root_dir + 'config.yaml', 'w') as fhand:
            yaml.dump(cfg.as_dict(), fhand)

    with open(default_root_dir + 'results.json', 'w') as fhand:
        json.dump(results, fhand, indent=4)
