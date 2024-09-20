import os.path
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tifffile import tifffile as tiff
from tqdm.auto import tqdm

from experiments import setup_experiments


def main(exp_name):
    exps = setup_experiments(exp_name)
    exp_path = Path(exps.path)
    data_path = exp_path / 'data' / 'pca'
    pca_path = data_path / f'train_{exps.dim_red}_{exps.n_comps}.pkl'
    model_path = exp_path / 'models' / f'{exps.dim_red}_{exps.n_comps}' / f'{exps.n_evecs}'

    if not os.path.exists(pca_path):
        raise RuntimeError('Encoding path does not exist. Please run the encoding script first.')

    if not os.path.exists(model_path):
        raise RuntimeError('Model path does not exist. Please run the train script first.')

    generate_cf(exps)


def generate_cf(exps):
    exp_path = Path(exps.path)
    pca_path = exp_path / 'data' / 'pca'
    model_path = exp_path / 'models' / f'{exps.dim_red}_{exps.n_comps}' / f'{exps.n_evecs}'
    save_path = exp_path / 'cfs' / 'no_bias'
    test_path = pca_path / f'test_{exps.dim_red}_{exps.n_comps}.pkl'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        user_input = input('Counterfactuals already exist. Existing images will be overwritten. Continue? [Yes/no]')
        if user_input.lower() == "yes":
            print("Continuing...")
        else:
            RuntimeError("Exiting...")

    with open(test_path, 'rb') as f:
        test = pickle.load(f)

    disease = test['disease']
    bias = test['bias']
    encoded = test['encoded_data']
    pca = test['pca']
    img_names = test['img_names']

    nsamples = len(img_names)
    print('Number of samples: ', nsamples)

    cf_vals = {1: 0}

    cf = np.zeros((nsamples, exps.n_comps))
    for e in range(0, exps.n_comps - 1, exps.n_evecs):
        ed = encoded[:, e:e + exps.n_evecs]
        ds = np.hstack([disease[:, np.newaxis], bias[:, np.newaxis], ed])

        macaw = torch.load(model_path / f'{e}.pt')
        cc = macaw.counterfactual(ds, cf_vals)
        cf[:, e:e + exps.n_evecs] = cc[:, exps.n_causes:]

    images = pca.inverse_transform(cf).reshape(nsamples, *exps.img_size)
    images = np.clip(images, 0, 1)

    for i, name in tqdm(zip(images, img_names)):
        tiff.imwrite(save_path / f'{name}', i)


if __name__ == '__main__':
    exp_name = sys.argv[1]
    main(exp_name)
