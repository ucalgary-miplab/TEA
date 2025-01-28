import sys
from pathlib import Path
import os
import numpy as np
import tifffile.tifffile as tiff
from tqdm.auto import tqdm

from experiments import setup_experiments
import os
import glob
import nibabel as nib


def main(exp_name):
    exp_name = "exp205"
    exps = setup_experiments(exp_name)
    data_path = Path(exps.path)

    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    preprocess(train_dir)
    preprocess(val_dir)
    preprocess(test_dir)


def preprocess(input_dir):
    image_names = glob.glob(os.path.join(input_dir, "*.nii.gz"))

    for name in tqdm(image_names):
        img = nib.load(name).get_fdata().astype("f4")
        img = extract_slice(img)
        img = crop_pad(img)
        img = normalize(img)
        tiff.imwrite(f"{name.replace('.nii.gz', '.tiff')}", img)


def extract_slice(img, slice_num=75):
    return img[:, :, slice_num].T


def crop_pad(img, target_height=192, target_width=192):
    height, width = img.shape
    if width == target_width and height == target_height:
        return img

    if width > target_width:
        left = (width - target_width) // 2
        img = img[:, left : left + target_width]

    if height > target_height:
        top = (height - target_height) // 2
        img = img[top : top + target_height, :]

    new_height, new_width = img.shape
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2

    padded_img = np.zeros((target_height, target_width), dtype=np.float32)
    padded_img[top : top + new_height, left : left + new_width] = img

    return padded_img


def normalize(image):
    image = image.astype("f8")
    maxv = np.max(image)
    minv = np.min(image)
    return ((image - minv) / maxv).astype("f4")


if __name__ == "__main__":
    exp_name = sys.argv[1]
    main(exp_name)
