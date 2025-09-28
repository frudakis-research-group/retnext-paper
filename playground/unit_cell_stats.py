r"""
Loop over a directory of `.cif` files, get unit cell statistics and store them.
"""

import json
import os
from argparse import ArgumentParser

from ase.io import read
from tqdm import tqdm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dirname')
    parser.add_argument('outname')
    args = parser.parse_args()

    stats = {
        'n_atoms': [],
        'volume': [],
        'element_count': {},
        'cell_lengths': [],
        'cell_angles': [],
        'mof_names': [],
    }
    fnames = os.listdir(args.dirname)

    for name in tqdm(fnames, desc='Looping over stuctures'):
        cif_path = os.path.join(args.dirname, name)
        structure = read(cif_path)

        stats['n_atoms'].append(len(structure))
        stats['volume'].append(structure.get_volume())
        stats['cell_lengths'].append(structure.cell.lengths().tolist())
        stats['cell_angles'].append(structure.cell.angles().tolist())
        stats['mof_names'].append(name)

        # Update element count.
        unique_elements = set(structure.get_chemical_symbols())
        for elm in unique_elements:
            
            if elm not in stats['element_count'].keys():
                stats['element_count'][elm] = 1
            else:
                stats['element_count'][elm] += 1

    with open(args.outname, 'w') as fhand:
        json.dump(stats, fhand, indent=4)
