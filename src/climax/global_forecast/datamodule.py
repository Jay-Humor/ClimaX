# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Optional

import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from dataset import (
    Forecast,
    IndividualForecastDataIter,
    NpyReader,
    ShuffleIterableDataset,
)

class GlobalForecastDataModule:
    def __init__(
        self,
        root_dir,
        variables,
        buffer_size,
        out_variables=None,
        predict_range: int = 6,
        hrs_each_step: int = 1,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        if num_workers > 1:
            raise NotImplementedError(
                "num_workers > 1 is not supported yet. Performance will likely degrage too with larger num_workers."
            )

        self.root_dir = root_dir
        self.variables = variables
        self.buffer_size = buffer_size
        self.out_variables = out_variables
        self.predict_range = predict_range
        self.hrs_each_step = hrs_each_step
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if isinstance(out_variables, str):
            self.out_variables = [out_variables]

        self.lister_train = glob.glob(os.path.join(root_dir, "train", '*'))
        self.lister_val = glob.glob(os.path.join(root_dir, "val", '*'))
        self.lister_test = glob.glob(os.path.join(root_dir, "test", '*'))

        self.transforms = self.get_normalize()
        self.output_transforms = self.get_normalize(out_variables)

        self.val_clim = self.get_climatology("val", out_variables)
        self.test_clim = self.get_climatology("test", out_variables)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None
        self.setup()

    def get_normalize(self, variables=None):
        if variables is None:
            variables = self.variables
        normalize_mean = dict(np.load(os.path.join(self.root_dir, "normalize_mean.npz"))) # type: ignore
        mean = []
        for var in variables:
            if var != "total_precipitation":
                mean.append(normalize_mean[var])
            else:
                mean.append(np.array([0.0]))
        normalize_mean = np.concatenate(mean)
        normalize_std = dict(np.load(os.path.join(self.root_dir, "normalize_std.npz"))) # type: ignore
        normalize_std = np.concatenate([normalize_std[var] for var in variables])
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.root_dir, "lon.npy"))
        return lat, lon

    def get_climatology(self, partition="val", variables=None):
        path = os.path.join(self.root_dir, partition, "climatology.npz")
        clim_dict = np.load(path)
        if variables is None:
            variables = self.variables
        clim = np.concatenate([clim_dict[var] for var in variables]) # type: ignore
        clim = torch.from_numpy(clim)
        return clim

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ShuffleIterableDataset(
                IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            file_list=self.lister_train,
                            start_idx=0,
                            end_idx=1,
                            variables=self.variables,
                            out_variables=self.out_variables,
                            shuffle=True,
                            multi_dataset_training=False,
                        ),
                        max_predict_range=self.predict_range,
                        # random_lead_time=False,
                        random_lead_time=True,
                        hrs_each_step=self.hrs_each_step,
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                ),
                buffer_size=self.buffer_size,
            )

            self.data_val = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        file_list=self.lister_val,
                        start_idx=0,
                        end_idx=1,
                        variables=self.variables,
                        out_variables=self.out_variables,
                        shuffle=False,
                        multi_dataset_training=False,
                    ),
                    max_predict_range=self.predict_range,
                    random_lead_time=False,
                    hrs_each_step=self.hrs_each_step,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
            )

            self.data_test = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        file_list=self.lister_test,
                        start_idx=0,
                        end_idx=1,
                        variables=self.variables,
                        out_variables=self.out_variables,
                        shuffle=False,
                        multi_dataset_training=False,
                    ),
                    max_predict_range=self.predict_range,
                    random_lead_time=False,
                    hrs_each_step=self.hrs_each_step,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train, # type: ignore
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    out_variables = batch[0][4]
    return (
        inp,
        out,
        lead_times,
        [v for v in variables],
        [v for v in out_variables],
    )
