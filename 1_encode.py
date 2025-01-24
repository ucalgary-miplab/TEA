import os.path
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import pad_list_data_collate, DataLoader
from monai.transforms import Compose, ToTensor

import utils.visualize as vis
from utils.dimRed import DimRed
from experiments import setup_experiments
from utils.datasets import SimBADataset
from utils.helpers import seed_all, seed_worker


def main(exp_name):
    seed = 42
    exps = setup_experiments(exp_name)
    g = seed_all(seed=seed, deterministic=True)

    encode_data(exps, g)

    if exps.debug:
        view_reconstruction(exps)


def encode_data(exps, g):
    data_path = Path(exps.path)
    pca_path = data_path / "pca"

    train_csv_path = data_path / "train.csv"
    val_csv_path = data_path / "val.csv"
    test_csv_path = data_path / "test.csv"

    train_img_path = data_path / "train"
    val_img_path = data_path / "val"
    bias_test_img_path = data_path / "test" / "bias"
    no_bias_test_img_path = data_path / "test" / "no_bias"

    if not os.path.exists(pca_path):
        os.mkdir(pca_path)

    print("Loading training data...")
    imgs, disease, bias, img_names = data_as_list(
        train_csv_path, train_img_path, exps, g
    )

    print("Computing dimensionality reduction...")
    pca = DimRed(ncomps=exps.n_comps, method=exps.dim_red)
    pca.fit(imgs)
    encoded_data = pca.transform(imgs)

    print("Saving encoded training data...")
    with open(pca_path / f"train_{exps.dim_red}_{exps.n_comps}.pkl", "wb") as f:
        pickle.dump(
            {
                "imgs": imgs,
                "disease": disease,
                "bias": bias,
                "pca": pca,
                "encoded_data": encoded_data,
                "img_names": img_names,
            },
            f,
        )

    print("Loading validation data...")
    imgs, disease, bias, img_names = data_as_list(val_csv_path, val_img_path, exps, g)
    encoded_data = pca.transform(imgs)

    print("Saving encoded validation data...")
    with open(pca_path / f"val_{exps.dim_red}_{exps.n_comps}.pkl", "wb") as f:
        pickle.dump(
            {
                "imgs": imgs,
                "disease": disease,
                "bias": bias,
                "pca": pca,
                "encoded_data": encoded_data,
                "img_names": img_names,
            },
            f,
        )

    print("Loading bias test data...")
    imgs, disease, bias, img_names = data_as_list(
        test_csv_path, bias_test_img_path, exps, g, bias_label=1
    )
    encoded_data = pca.transform(imgs)

    print("Saving encoded test data...")
    with open(pca_path / f"bias_test_{exps.dim_red}_{exps.n_comps}.pkl", "wb") as f:
        pickle.dump(
            {
                "imgs": imgs,
                "disease": disease,
                "bias": bias,
                "pca": pca,
                "encoded_data": encoded_data,
                "img_names": img_names,
            },
            f,
        )

    print("Loading no bias test data...")
    imgs, disease, bias, img_names = data_as_list(
        test_csv_path, no_bias_test_img_path, exps, g, bias_label=0
    )
    encoded_data = pca.transform(imgs)

    print("Saving encoded test data...")
    with open(pca_path / f"no_bias_test_{exps.dim_red}_{exps.n_comps}.pkl", "wb") as f:
        pickle.dump(
            {
                "imgs": imgs,
                "disease": disease,
                "bias": bias,
                "pca": pca,
                "encoded_data": encoded_data,
                "img_names": img_names,
            },
            f,
        )


def view_reconstruction(exps):
    data_path = Path(exps.path) / "pca" / f"val_{exps.dim_red}_{exps.n_comps}.pkl"

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    imgs = data["imgs"]
    pca = data["pca"]
    disease = data["disease"]
    bias = data["bias"]

    sample_imgs = imgs[:5, :]
    t = pca.transform(sample_imgs)
    X_recon = pca.inverse_transform(t)

    plt.rcParams["figure.figsize"] = 10, 10
    diff = sample_imgs - X_recon
    fig = vis.img_grid(
        [d.reshape(exps.img_size) for d in sample_imgs]
        + [d.reshape(exps.img_size) for d in X_recon],
        clim=(0, 1),
        rows=2,
        cols=5,
        titles=[f"Disease:{d}, Bias:{b}" for d, b in zip(disease, bias)],
    )
    fig = vis.img_grid(
        [d.reshape(exps.img_size) for d in diff],
        clim=(-0.5, 0.5),
        cols=5,
        cmap="seismic",
    )
    plt.show()


def data_as_list(csv_path, img_path, exps, g, bias_label=None):
    t = Compose([ToTensor()])

    ds = SimBADataset(csv_path, img_path, bias_label=bias_label, transform=t)
    dloader = DataLoader(
        ds,
        batch_size=exps.batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pad_list_data_collate,
    )

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


if __name__ == "__main__":
    exp_name = sys.argv[1]
    main(exp_name)
