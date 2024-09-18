import os.path
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
from torch.utils.tensorboard import SummaryWriter

import utils.visualize as vis
from experiments import setup_experiments
from macaw import MACAW
from utils.helpers import seed_all


def main(exp_name):
    seed = 42
    exps = setup_experiments(exp_name)
    g = seed_all(seed=seed, deterministic=True)

    data_path = Path(exps.path) / 'data'
    pca_path = data_path / f'train_{exps.dim_red}_{exps.n_comps}.pkl'

    if not os.path.exists(pca_path):
        print('Encoding path does not exist. Please run the encoding script first.')

    if exps.debug:
        view_reconstruction(exps)

    train_macaw(exps, g)


def train_macaw(exps, g):
    data_path = Path(exps.path) / 'data'
    model_path = Path(exps.path) / 'models' / f'{exps.dim_red}_{exps.n_comps}' / f'{exps.n_evecs}'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    else:
        user_input = input('Model path already exists. Existing models will be overwritten. Continue? [Yes/no]')
        if user_input.lower() == "yes":
            print("Continuing...")
        else:
            print("Exiting...")

    train_path = data_path / f'train_{exps.dim_red}_{exps.n_comps}.pkl'
    val_path = data_path / f'val_{exps.dim_red}_{exps.n_comps}.pkl'

    with open(train_path, 'rb') as f:
        train = pickle.load(f)

    disease_t = train['disease']
    bias_t = train['bias']
    encoded_t = train['encoded_data']

    with open(val_path, 'rb') as f:
        val = pickle.load(f)

    disease_v = val['disease']
    bias_v = val['bias']
    encoded_v = val['encoded_data']

    # Build causal graph
    disease_to_latents = [(0, i) for i in range(exps.n_causes, exps.n_causes + exps.n_evecs)]
    bias_to_latents = [(1, i) for i in range(exps.n_causes, exps.n_causes + exps.n_evecs)]
    autoregressive_latents = [(i, j) for i in range(exps.n_causes, exps.n_evecs + exps.n_causes) for j in
                              range(i + 1, exps.n_evecs + exps.n_causes)]
    edges = disease_to_latents + bias_to_latents + autoregressive_latents

    # Prior distributions
    P_bias = np.sum(bias_t) / len(bias_t)
    print(f'P_bias: {P_bias}')

    P_dis = np.sum(disease_t) / len(disease_t)
    print(f'P_disease:{P_dis}')

    priors = [(slice(0, 1), td.Bernoulli(torch.tensor([P_dis]).to(torch.device(exps.device)))),  # disease
              (slice(1, 2), td.Bernoulli(torch.tensor([P_bias]).to(torch.device(exps.device)))),  # bias
              (slice(exps.n_causes, exps.n_evecs + exps.n_causes),  # nevecs
               td.Normal(torch.zeros(exps.n_evecs).to(exps.device), torch.ones(exps.n_evecs).to(exps.device)))
              ]

    losses = {}
    for e in range(0, exps.n_comps - 1, exps.n_evecs):
        writer = SummaryWriter(f'logs/{e}')

        save_path = model_path / f'{e}.pt'
        ed_t = encoded_t[:, e:e + exps.n_evecs]
        ed_v = encoded_v[:, e:e + exps.n_evecs]
        print(f"Starting training for model: {e} - {e + exps.n_evecs}")

        train_ds = np.hstack([disease_t[:, np.newaxis], bias_t[:, np.newaxis], ed_t])
        val_ds = np.hstack([disease_v[:, np.newaxis], bias_v[:, np.newaxis], ed_v])

        macaw = MACAW.MACAW(exps)
        loss_vals_train, loss_vals_val = macaw.fit_with_priors(train_ds, val_ds, edges, priors, writer)
        losses[e] = {"train": loss_vals_train, "val": loss_vals_val}
        torch.save(macaw, save_path)
        writer.close()

    model_path = model_path / 'hyperparameters.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump({'exps': exps}, f)


def view_reconstruction(exps):
    data_path = Path(exps.path) / 'data' / f'val_{exps.dim_red}_{exps.n_comps}.pkl'

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    imgs = data['imgs']
    pca = data['pca']
    disease = data['disease']
    bias = data['bias']

    sample_imgs = imgs[:5, :]
    t = pca.transform(sample_imgs)
    X_recon = pca.inverse_transform(t)
    crop_size = exps.crop_size

    plt.rcParams["figure.figsize"] = 10, 10
    diff = sample_imgs - X_recon
    fig = vis.img_grid(
        [d.reshape(crop_size, crop_size) for d in sample_imgs] + [d.reshape(crop_size, crop_size) for d in X_recon],
        clim=(0, 1), rows=2, cols=5, titles=[f'Disease:{d}, Bias:{b}' for d, b in zip(disease, bias)])
    fig = vis.img_grid([d.reshape(crop_size, crop_size) for d in diff], clim=(-.5, .5), cols=5, cmap='seismic')
    plt.show()


if __name__ == '__main__':
    exp_name = sys.argv[1]
    main(exp_name)
