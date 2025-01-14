import os.path
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributions as td
from torch.utils.tensorboard import SummaryWriter

from experiments import setup_experiments
from macaw import MACAW
from utils.helpers import seed_all


def main(exp_name):
    seed = 42
    exps = setup_experiments(exp_name)
    g = seed_all(seed=seed, deterministic=True)

    data_path = Path(exps.path) / 'data' / 'pca'
    pca_path = data_path / f'train_{exps.dim_red}_{exps.n_comps}.pkl'

    if not os.path.exists(pca_path):
        raise RuntimeError('Encoding does not exist. Please run the encoding script first.')

    train_macaw(exps, g)


def train_macaw(exps, g):
    data_path = Path(exps.path) / 'data' / 'pca'
    model_path = Path(exps.path) / 'models' / f'{exps.dim_red}_{exps.n_comps}' / f'{exps.n_evecs}'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

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
    # edges = disease_to_latents + bias_to_latents

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


if __name__ == '__main__':
    exp_name = sys.argv[1]
    main(exp_name)
