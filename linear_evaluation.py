import os
import json
import random

import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error as mae
from aidsorb.data import get_names

train_names, val_names, test_names = [
    #list(get_names(f'/home/asarikas/databases/MOFXDB/hMOF/{stg}.json'))
    #list(get_names(f'/home/asarikas/databases/UO/extracted_data/{stg}.json'))
    #list(get_names(f'/home/asarikas/databases/Mercado/extracted_data/{stg}.json'))
    list(get_names(f'/home/asarikas/databases/MOFXDB/Tobacco/{stg}.json'))
    for stg in ['train', 'validation', 'test']
]

#hMOF = pd.read_csv('/home/asarikas/databases/MOFXDB/hMOF/hMOF.csv', index_col='name')
#UO = pd.read_csv('/home/asarikas/databases/UO/extracted_data/csv/logSelectivity_all_MOFs_screening_data.csv', index_col='MOFname')
Tobacco = pd.read_csv('/home/asarikas/databases/MOFXDB/Tobacco/float_Tobacco.csv', index_col='name')

n_runs = 10
train_sizes = [50, 100, 300, 500, 1000]
source_tasks = [
        #'multitask_geom_ads',
        'multitask_final_all',
        #'CarbonDioxide_1_298K_2.5bar_mol',
        #'Methane_1_298K_0.5bar_mol',
        #'Hydrogen_1_77K_100bar_g',
        ]
target_tasks = [
        #'CarbonDioxide_1_298K_0.5bar_mol/kg',
        #'Methane_1_298K_0.05bar_mol/kg',
        #'Hydrogen_1_77K_2bar_g/l',
        #'Xenon_0.2_273K_1bar_mol/kg',
        #'Krypton_0.8_273K_5bar_mol/kg',
        #'Krypton_0.8_273K_10bar_mol/kg',
        #'Xenon_0.2_273K_5bar_mol/kg',
        #'Xenon_0.2_273K_10bar_mol/kg',
        #'Krypton_0.8_273K_1bar_mol/kg',
        #'CO2_uptake_P0.15bar_T298K [mmol/g]',
        #'CO2_uptake_P0.10bar_T363K [mmol/g]',
        #'logSelectivity',
        #'working_capacity_vacuum_swing [mmol/g]',
        #'working_capacity_temperature_swing [mmol/g]',
        #'CO2_binary_uptake_P0.15bar_T298K [mmol/g]',
        #'N2_binary_uptake_P0.85bar_T298K [mmol/g]',
        #'excess_CO2_uptake_P0.15bar_T298K [mmol/g]',
        #'excess_CO2_uptake_P0.10bar_T363K [mmol/g]',
        #'excess_CO2_binary_uptake_P0.15bar_T298K [mmol/g]',
        #'absolute methane uptake low P [v STP/v]',
        #'absolute methane uptake high P [v STP/v]',
        #'deliverable capacity [v STP/v]'
        'Xenon_0.2_298K_1bar_mmol/g',
        'Krypton_0.8_298K_1bar_mmol/g',
        'Xenon_0.2_298K_5bar_mmol/g',
        'Krypton_0.8_298K_5bar_mmol/g',
        'Methane_1_298K_65bar_cm3(STP)/cm3',
        'Methane_1_298K_100bar_cm3(STP)/cm3',
        'Methane_1_298K_6bar_cm3(STP)/cm3',
        'Hydrogen_1_77K_100bar_g/l',
        'Hydrogen_1_77K_6bar_g/l',
        'Hydrogen_1_130K_100bar_g/l',
        'Hydrogen_1_200K_100bar_g/l',
        'Hydrogen_1_243K_100bar_g/l',
        'Hydrogen_1_160K_5bar_g/l',
        'Methane_1_298K_6bar_kj/mol',
        ]

random.seed(1)

for target in (pbar := tqdm(target_tasks)):
    pbar.set_description(target)

    results = {s: {} for s in train_sizes}
    #y_test =  hMOF.loc[test_names, target]
    #y_test =  UO.loc[test_names, target]
    #y_test =  COFs.loc[test_names, target]
    y_test =  Tobacco.loc[test_names, target]

    for source in source_tasks:
        #Z_train = pd.read_csv(f'playground/embeddings/augmentation_cubic_boltzmann_{source}_train.csv', index_col='name')
        #Z_test = pd.read_csv(f'playground/embeddings/augmentation_cubic_boltzmann_{source}_test.csv', index_col='name')
        #Z_train = pd.read_csv(f'playground/embeddings/UO-augmentation_cubic_boltzmann_{source}_train.csv', index_col='name')
        #Z_test = pd.read_csv(f'playground/embeddings/UO-augmentation_cubic_boltzmann_{source}_test.csv', index_col='name')
        #Z_train = pd.read_csv(f'playground/embeddings/Mercado-augmentation_cubic_boltzmann_{source}_train.csv', index_col='name')
        #Z_test = pd.read_csv(f'playground/embeddings/Mercado-augmentation_cubic_boltzmann_{source}_test.csv', index_col='name')
        Z_train = pd.read_csv(f'playground/embeddings/Tobacco-augmentation_cubic_boltzmann_{source}_train.csv', index_col='name')
        Z_test = pd.read_csv(f'playground/embeddings/Tobacco-augmentation_cubic_boltzmann_{source}_test.csv', index_col='name')

        for size in train_sizes:
            for run in range(n_runs):
                fit_names = random.sample(train_names, k=size)

                X_train = Z_train.loc[fit_names]
                #y_train = hMOF.loc[fit_names, target]
                #y_train = UO.loc[fit_names, target]
                #y_train = COFs.loc[fit_names, target]
                y_train = Tobacco.loc[fit_names, target]

                reg = Ridge(0.1, solver='cholesky')
                reg.fit(X_train, y_train)

                # Match the output returned by Lightning (list of dicts),
                # for easier visualization of the results.
                results[size][run] = [{
                        'test_R2Score': reg.score(Z_test, y_test),
                        'test_MeanAbsoluteError': mae(y_true=y_test, y_pred=reg.predict(Z_test))
                        }]

        #dirname = f'tl_experiments/from_{source.replace("/", "_")}-to_{target.replace("/", "_")}-lineval/'
        dirname = f'Tobacco_tl_experiments/from_{source.replace("/", "_")}-to_{target.replace("/", "_")}-lineval/'
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        with open(f'{dirname}/results.json', 'w') as fhand:
            json.dump(results, fhand, indent=4)
