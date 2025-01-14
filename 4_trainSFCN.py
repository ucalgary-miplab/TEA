import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader
from monai.data.utils import pad_list_data_collate
from monai.transforms import Compose, ToTensor
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from experiments import setup_experiments
from sfcn.SFCN import SFCNModel
from utils.datasets import SimBADataset
from utils.helpers import seed_all


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_sfcn(model, train_loader, device, optimizer, loss_func):
    model.train()
    losses = []
    for d, b, i, imn in train_loader:
        inputs = i.to(device)
        labels = b.to(device)
        outputs = model(inputs)

        optimizer.zero_grad()

        loss = loss_func(torch.squeeze(outputs), labels.float())
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def test_sfcn(model, val_loader, device):
    model.eval()
    predictions = []
    targets = []
    imns = []
    with torch.no_grad():
        for d, b, i, imn in val_loader:
            inputs = i.to(device)
            outputs = 1 * (model(inputs).detach().cpu().numpy().squeeze() > 0.5)
            predictions += outputs.tolist()
            targets += b.detach().cpu().numpy().squeeze().tolist()
            imns += imn

    targets = np.array(targets).squeeze()
    predictions = np.array(predictions).squeeze()
    return targets, predictions, imns, np.mean(predictions == targets)


def main(exp_name):
    seed = 42
    exps = setup_experiments(exp_name)
    g = seed_all(seed=seed, deterministic=True)

    exp_path = Path(exps.path)
    data_path = Path(exps.path) / 'data'

    train_csv = data_path / 'csv' / 'train.csv'
    train_images = data_path / 'images' / 'train'

    val_csv = data_path / 'csv' / 'val.csv'
    val_images = data_path / 'images' / 'val'

    test_csv = data_path / 'csv' / 'test.csv'
    test_images = data_path / 'images' / 'test'
    macaw_cf_images = exp_path / 'macaw_cfs' / 'no_bias'
    hvae_cf_images = exp_path / 'hvae_cfs' / 'no_bias'

    t = Compose([ToTensor()])

    train_ds = SimBADataset(train_csv, train_images, exps.exp_name == 'no_bias', transform=t)
    train_loader = DataLoader(train_ds, batch_size=exps.sfcn['batch_size'], shuffle=True,
                              num_workers=exps.sfcn['workers'],
                              worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda.is_available(),
                              collate_fn=pad_list_data_collate)

    val_ds = SimBADataset(val_csv, val_images, exps.exp_name == 'no_bias', transform=t)
    val_loader = DataLoader(val_ds, batch_size=exps.sfcn['batch_size'], shuffle=True,
                            num_workers=exps.sfcn['workers'],
                            worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda.is_available(),
                            collate_fn=pad_list_data_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SFCNModel().to(device)

    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), exps.sfcn['lr'])

    for epoch in (pbar := tqdm(range(exps.sfcn['epochs']))):
        train_loss = train_sfcn(model, train_loader, device, optimizer, loss_function)
        _, _, _,val_accuracy = test_sfcn(model, val_loader, device)
        pbar.set_description(f'Epoch {epoch + 1} - Training Loss: {train_loss:.3f}, Val accuracy: {val_accuracy:.3f}')

    test_ds = SimBADataset(test_csv, test_images, exps.exp_name == 'no_bias', transform=t)
    test_loader = DataLoader(test_ds, batch_size=exps.sfcn['batch_size'], shuffle=False,
                             num_workers=exps.sfcn['workers'],
                             worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda.is_available(),
                             collate_fn=pad_list_data_collate)

    t_t, p_t, _, test_accuracy = test_sfcn(model, test_loader, device)
    print(f'Test accuracy: {test_accuracy:.3f}')
    print(f'Test confusion matrix\n {confusion_matrix(t_t, p_t)}')

    cf_ds = SimBADataset(test_csv, macaw_cf_images, no_bias=True, transform=t)
    cf_loader = DataLoader(cf_ds, batch_size=exps.sfcn['batch_size'], shuffle=False,
                           num_workers=exps.sfcn['workers'],
                           worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda.is_available(),
                           collate_fn=pad_list_data_collate)

    t_cf, p_cf, imn_cf, cf_accuracy = test_sfcn(model, cf_loader, device)
    print(f'MACAW CF accuracy: {cf_accuracy:.3f}')
    print(f'MACAW CF confusion matrix\n {confusion_matrix(t_t, p_cf)}')

    df = pd.DataFrame(zip(imn_cf,p_cf), columns=['filename', 'predictions'])
    df.to_csv(macaw_cf_images/'predictions.csv', index=False)

    cf_ds = SimBADataset(test_csv, hvae_cf_images, no_bias=True, transform=t)
    cf_loader = DataLoader(cf_ds, batch_size=exps.sfcn['batch_size'], shuffle=False,
                           num_workers=exps.sfcn['workers'],
                           worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda.is_available(),
                           collate_fn=pad_list_data_collate)

    t_cf, p_cf, imn_cf, cf_accuracy = test_sfcn(model, cf_loader, device)
    print(f'H-VAE CF accuracy: {cf_accuracy:.3f}')
    print(f'H-VAE CF confusion matrix\n {confusion_matrix(t_t, p_cf)}')

    df = pd.DataFrame(zip(imn_cf,p_cf), columns=['filename', 'predictions'])
    df.to_csv(hvae_cf_images/'predictions.csv', index=False)


if __name__ == '__main__':
    exp_name = sys.argv[1]
    main(exp_name)
