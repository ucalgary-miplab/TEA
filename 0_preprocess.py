import sys
from pathlib import Path

import numpy as np
import tifffile.tifffile as tiff
from tqdm.auto import tqdm

from experiments import setup_experiments


def main(exp_name):
    exps = setup_experiments(exp_name)
    data_path = Path(exps.path) / 'data'

    train_dir = data_path / 'images' / 'train'
    val_dir = data_path / 'images' / 'val'
    test_dir = data_path / 'images' / 'test'

    preprocess(train_dir)
    preprocess(val_dir)
    preprocess(test_dir)


def preprocess(input_dir):
    import os
    import glob

    image_names = glob.glob(os.path.join(input_dir, '*.tiff'))

    for name in tqdm(image_names):
        im = tiff.imread(name)
        im = normalize(im)
        tiff.imwrite(f'{name}', im)


def normalize(image):
    image = image.astype('f8')
    maxv = np.max(image)
    minv = np.min(image)
    return ((image - minv) / maxv).astype('f4')


if __name__ == '__main__':
    exp_name = sys.argv[1]
    main(exp_name)
