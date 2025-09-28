r"""
Extract embeddings from pretrained models and store them as CSV.
"""

from types import NoneType

import torch
from aidsorb.datamodules import PCDDataModule
from aidsorb.litmodels import PCDLit
from aidsorb.data import get_names
from retnext.modules import RetNeXt
from torchmetrics import MetricCollection, R2Score
import pandas as pd
import lightning as L
from torch.utils.data._utils.collate import default_collate_fn_map


def collate_none(batch, *, collate_fn_map):
    return None


if __name__ == '__main__':
    trainer = L.Trainer(accelerator='gpu', devices=1)

    stages = ['train', 'validation', 'test']
    names = [
        #get_names(f'/home/asarikas/databases/MOFXDB/hMOF/{stg}.json')
        #get_names(f'/home/asarikas/databases/UO/extracted_data/{stg}.json')
        get_names(f'/home/asarikas/databases/MOFXDB/Tobacco/{stg}.json')
        for stg in stages
    ]

    #targets = [
    #        'CarbonDioxide_1_298K_0.05bar_mol',
    #        'CarbonDioxide_1_298K_0.5bar_mol',
    #        'CarbonDioxide_1_298K_2.5bar_mol',
    #        'Hydrogen_1_77K_100bar_g',
    #        'Hydrogen_1_77K_2bar_g',
    #        'Methane_1_298K_0.05bar_mol',
    #        'Methane_1_298K_0.5bar_mol',
    #        'Methane_1_298K_2.5bar_mol'
    #        ]

    #checkpoints = {
    #    f'{exp}_{tgt}': f'../experiments/{exp}/{tgt}/None/lightning_logs/version_0/checkpoints/best.ckpt'
    #    for exp in ['augmentation_cubic_boltzmann']
    #    for tgt in targets
    #}
    #checkpoints['randomly_initialized'] = None
    checkpoints = {
            #'UO-augmentation_cubic_boltzmann_multitask_final_all':
            #'../ml_experiments/augmentation_cubic_boltzmann/final_all/lightning_logs/version_0/checkpoints/best.ckpt'
            'Tobacco-augmentation_cubic_boltzmann_multitask_final_all':
            '../ml_experiments/augmentation_cubic_boltzmann/final_all/lightning_logs/version_0/checkpoints/best.ckpt'
            }

    for exp_tgt, ckpt_path in checkpoints.items():
        if ckpt_path is not None:
            # Disable shuffle and drop_last to maintain correct order with train names.
            #dm = PCDDataModule.load_from_checkpoint(
            #        ckpt_path,
            #        shuffle=False,
            #        drop_last=False,
            #        train_batch_size=256,
            #        eval_batch_size=256
            #        )

            # For the UO database.
            default_collate_fn_map.update({NoneType: collate_none})
            dm = PCDDataModule.load_from_checkpoint(
                    ckpt_path,
                    #path_to_X='/home/asarikas/databases/UO/extracted_data/voxels_data_GS32_CB30',
                    path_to_X='/home/asarikas/databases/MOFXDB/Tobacco/voxels_data_GS32_CB30',
                    path_to_Y=None,
                    shuffle=False,
                    drop_last=False,
                    train_batch_size=256,
                    eval_batch_size=256
                    )
    
            # Disable augmentation.
            dm.train_transform_x = dm.eval_transform_x

            litmodel = PCDLit.load_from_checkpoint(ckpt_path)
        else:
            litmodel.model = RetNeXt()  # Randomly initialized model.

        dm.setup()
        litmodel.model.fc = torch.nn.Identity()  # So forward() returns the embeddings.
        litmodel.freeze()

        print(f'\033[34;1mExtracting embeddings from model: {ckpt_path}\033[0m')

        for dl, stg, stg_names in zip(
            [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()],
            stages,
            names,
        ):
            Z = torch.cat(trainer.predict(litmodel, dl)).numpy()
            df = pd.DataFrame(Z, index=stg_names)
            df.to_csv(f'embeddings/{exp_tgt}_{stg}.csv', index=True, index_label='name')
