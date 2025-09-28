import json
import yaml

from jsonargparse import ArgumentParser
from aidsorb.litmodels import PCDLit
from aidsorb.datamodules import PCDDataModule
from lightning.pytorch import Trainer, seed_everything


def main_cli():
    parser = ArgumentParser()
    parser.add_class_arguments(PCDDataModule, 'data', fail_untyped=False)
    parser.add_class_arguments(PCDLit, 'model', fail_untyped=False)
    parser.add_class_arguments(Trainer, 'trainer')
    parser.add_argument('--train_sizes', type=list)
    parser.add_argument('--n_runs', type=int)
    parser.add_argument('-c', '--config', action='config')

    return parser


if __name__ == '__main__':
    parser = main_cli()
    cfg = parser.parse_args()

    results = {s: {} for s in cfg.train_sizes}

    # Add target name to dirname.
    cfg.trainer.default_root_dir += cfg.data.labels[0].split('/')[0] + '/'
    del cfg.config  # Not JSON serializable.

    default_root_dir = cfg.trainer.default_root_dir

    seed_everything(42, workers=True)

    for size in cfg.train_sizes:
        print(f'\033[32;1mTraining set size: {size}\033[0m')

        cfg.trainer.default_root_dir = default_root_dir + f'{size}/'  # New dir for each size.
        cfg.data.train_size = size  # Overwrite train set size.

        for run in range(cfg.n_runs):
            print(f'\033[34;1mRun number: {run}\033[0m')

            results[size][run] = {}

            init = parser.instantiate_classes(cfg)  # New objects for each experiment.
            trainer, dm, litmodel = init.trainer, init.data, init.model

            trainer.fit(litmodel, datamodule=dm)

            results[size][run] = trainer.test(litmodel, datamodule=dm, ckpt_path='best')

        with open(cfg.trainer.default_root_dir + 'results.json', 'w') as fhand:
            json.dump(results, fhand, indent=4)

        with open(cfg.trainer.default_root_dir + 'config.yaml', 'w') as fhand:
            yaml.dump(cfg.as_dict(), fhand)
