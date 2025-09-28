import glob

from tqdm import tqdm
import pandas as pd
from sklearn.manifold import TSNE


def tsne_reduce(embeddings):
    reducer = TSNE(random_state=42, n_jobs=-1)
    reducer.set_output(transform='pandas')

    return reducer.fit_transform(embeddings)


if __name__ == '__main__':
    paths = glob.glob('embeddings/augmentation*test.csv')
    paths += glob.glob('embeddings/randomly*test.csv')

    for csv_path in (pbar := tqdm(paths, desc='Looping over embeddings')):
        pbar.set_description(csv_path)
        embeddings = pd.read_csv(csv_path, index_col='name')
        tsne_embeddings = tsne_reduce(embeddings)
        tsne_embeddings.to_csv(f'tsne_{csv_path}', index=True, index_label='name')
