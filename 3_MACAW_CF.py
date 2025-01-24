import os.path
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tifffile import tifffile as tiff
from tqdm.auto import tqdm

from experiments import setup_experiments


def main(exp_name, add_or_remove):
    exps = setup_experiments(exp_name)
    data_path = Path(exps.path)

    model_path = (
        data_path / "models" / f"{exps.dim_red}_{exps.n_comps}" / f"{exps.n_evecs}"
    )

    if add_or_remove == "add":
        cf_vals = {1: 1}
        save_path = data_path / "macaw_cfs" / "bias"
        test_path = (
            data_path / "pca" / f"no_bias_test_{exps.dim_red}_{exps.n_comps}.pkl"
        )
    elif add_or_remove == "remove":
        cf_vals = {1: 0}
        save_path = data_path / "macaw_cfs" / "no_bias"
        test_path = data_path / "pca" / f"bias_test_{exps.dim_red}_{exps.n_comps}.pkl"
    else:
        raise ValueError("second argument must be add or remove")

    validate_paths(model_path, save_path, test_path)

    with open(test_path, "rb") as f:
        test = pickle.load(f)

    disease = test["disease"]
    bias = test["bias"]
    encoded = test["encoded_data"]
    pca = test["pca"]
    img_names = test["img_names"]

    nsamples = len(img_names)
    print("Number of samples: ", nsamples)

    cf = np.zeros((nsamples, exps.n_comps))
    for e in range(0, exps.n_comps - 1, exps.n_evecs):
        ed = encoded[:, e : e + exps.n_evecs]
        ds = np.hstack([disease[:, np.newaxis], bias[:, np.newaxis], ed])

        macaw = torch.load(model_path / f"{e}.pt")
        cc = macaw.counterfactual(ds, cf_vals)
        cf[:, e : e + exps.n_evecs] = cc[:, exps.n_causes :]

    images = pca.inverse_transform(cf).reshape(nsamples, *exps.img_size)
    images = np.clip(images, 0, 1)

    for i, name in tqdm(zip(images, img_names)):
        tiff.imwrite(save_path / f"{name.replace('nii.gz', 'tiff')}", i)


def validate_paths(model_path, save_path, test_path):
    if not os.path.exists(model_path):
        raise RuntimeError(
            "Model path does not exist. Please run the train script first."
        )

    if not os.path.exists(test_path):
        raise RuntimeError(
            "Encoding test path does not exist. Please run the encoding script first."
        )

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        user_input = input(
            "Counterfactuals already exist. Existing images will be overwritten. Continue? [Yes/no]"
        )
        if user_input.lower() == "yes":
            print("Continuing...")
        else:
            raise RuntimeError("Exiting...")


if __name__ == "__main__":
    exp_name = sys.argv[1]
    add_or_remove = sys.argv[2]
    main(exp_name, add_or_remove)
