import os.path
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import pad_list_data_collate, DataLoader
from monai.transforms import Compose, ToTensor
from torchvision.transforms import CenterCrop

import utils.visualize as vis
from dimRed import DimRed
from experiments import setup_experiments
from utils.customTransforms import ToFloatUKBB
from utils.datasets import SimBADataset
from utils.helpers import seed_all, seed_worker


def main(exp_name):
    seed = 42
    exps = setup_experiments(exp_name)
    g = seed_all(seed=seed, deterministic=True)

    data_path = Path(exps.path) / 'data'

    if not os.path.exists(data_path):
        save_data(exps, g)
    else:
        print('Data path already exists. Delete the directory to run PCA.')

    if exps.debug:
        view_reconstruction(exps)


def save_data(exps, g):
    exp_path = Path(exps.path)

    train_csv_path = exp_path / 'csv' / 'train.csv'
    val_csv_path = exp_path / 'csv' / 'val.csv'
    test_csv_path = exp_path / 'csv' / 'test.csv'

    train_img_path = exp_path / 'images' / 'train'
    val_img_path = exp_path / 'images' / 'val'
    test_img_path = exp_path / 'images' / 'test'

    data_path = exp_path / 'data'
    os.mkdir(data_path)

    print('Loading training data...')
    imgs, disease, bias, img_names = data_as_list(train_csv_path, train_img_path, exps, g)

    print('Computing dimensionality reduction...')
    pca = DimRed(ncomps=exps.n_comps, method=exps.dim_red)
    pca.fit(imgs)
    encoded_data = pca.transform(imgs)

    print('Saving encoded training data...')
    with open(data_path / f'train_{exps.dim_red}_{exps.n_comps}.pkl', 'wb') as f:
        pickle.dump({'imgs': imgs, 'disease': disease, 'bias': bias, 'pca': pca, 'encoded_data': encoded_data,
                     'img_names': img_names}, f)

    print('Loading validation data...')
    imgs, disease, bias, img_names = data_as_list(val_csv_path, val_img_path, exps, g)
    encoded_data = pca.transform(imgs)

    print('Saving encoded validation data...')
    with open(data_path / f'val_{exps.dim_red}_{exps.n_comps}.pkl', 'wb') as f:
        pickle.dump({'imgs': imgs, 'disease': disease, 'bias': bias, 'pca': pca, 'encoded_data': encoded_data,
                     'img_names': img_names}, f)

    print('Loading test data...')
    imgs, disease, bias, img_names = data_as_list(test_csv_path, test_img_path, exps, g)
    encoded_data = pca.transform(imgs)

    print('Saving encoded test data...')
    with open(data_path / f'test_{exps.dim_red}_{exps.n_comps}.pkl', 'wb') as f:
        pickle.dump({'imgs': imgs, 'disease': disease, 'bias': bias, 'pca': pca, 'encoded_data': encoded_data,
                     'img_names': img_names}, f)


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


def data_as_list(csv_path, img_path, exps, g):
    t = Compose([ToTensor(), CenterCrop(exps.crop_size), ToFloatUKBB()])

    ds = SimBADataset(csv_path, img_path, exps.exp_name == 'no_bias', transform=t)
    dloader = DataLoader(ds, batch_size=exps.batch_size, shuffle=True, num_workers=2,
                         worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda.is_available(),
                         collate_fn=pad_list_data_collate)

    imgs_list = []
    disease_list = []
    bias_list = []
    img_names_list = []

    for d, b, i, imn in dloader:
        disease_list.append(d.numpy())
        bias_list.append(b.numpy())
        imgs_list.append(i.numpy())
        img_names_list.append(imn)

    imgs = np.concatenate(imgs_list, axis=0)
    imgs = imgs.reshape(imgs.shape[0], -1)

    img_names = np.concatenate(img_names_list, axis=0)
    disease = np.concatenate(disease_list, axis=0)
    bias = np.concatenate(bias_list, axis=0)

    return imgs, disease, bias, img_names


if __name__ == '__main__':
    exp_name = sys.argv[1]
    main(exp_name)
