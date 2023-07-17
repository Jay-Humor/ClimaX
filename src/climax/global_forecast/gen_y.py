# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import torch
from datamodule import GlobalForecastDataModule
import logging
import numpy as np

from torch.utils.tensorboard import SummaryWriter # type: ignore

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training (default: 0)')

    # Seed everything
    parser.add_argument('--seed_everything', type=int, default=42)

    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    
    # Checkpoint
    parser.add_argument('--save_interval', type=int, default=5)

    # Model
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=10000)
    parser.add_argument('--max_epochs', type=int, default=100000)
    parser.add_argument('--warmup_start_lr', type=float, default=1e-8)
    parser.add_argument('--eta_min', type=float, default=1e-8)
    parser.add_argument('--pretrained_path', type=str, default='https://climaxrelease.blob.core.windows.net/checkpoints/ClimaX-5.625deg.ckpt')
    parser.add_argument('--ckpt_path', type=str, default=None)


    # Data
    parser.add_argument('--root_dir', type=str, default="/public/home/dqren/raindata/AIR/data/climax-data")
    parser.add_argument('--log_dir', type=str, default='/public/home/qindaotest/huangshijie/ClimaX-Pytorch/log')
    parser.add_argument('--predict_range', type=int, default=12)
    parser.add_argument('--hrs_each_step', type=int, default=1)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--pin_memory', type=bool, default=False)

    # Variables
    parser.add_argument('--variables', type=list, default=[
        # "land_sea_mask",
        # "orography",
        # "lattitude",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential_1000",
        # "u_component_of_wind_1000",
        # "temperature_1000",
        "relative_humidity_1000",
        # "specific_humidity_1000",
    ])
    parser.add_argument('--net_default_vars', type=list, default=[
        "land_sea_mask",
        "orography",
        "lattitude",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential_1000",
        # "u_component_of_wind_1000",
        # "temperature_1000",
        "relative_humidity_1000",
        # "specific_humidity_1000",
    ])

    # parser.add_argument('--out_variables', type=list, default=["geopotential_1000", "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"])
    parser.add_argument('--out_variables', type=list, default=["2m_temperature"])

    args = parser.parse_args()
    return args

def test(dataloader):
    total_loss = {}
    count = 0
    with torch.no_grad():
        preds = list()
        for batch in dataloader:
            batch = [item if index < 2 else item for index, item in enumerate(batch)]
            pred = batch[1]
            count += 1
            preds.append(pred.cpu().numpy())
    preds = np.array(preds)
    return preds, total_loss


def main():
    # Parse arguments
    args = parse_arguments()

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(args.log_dir)

    # Initialize the data module
    datamodule = GlobalForecastDataModule(
        root_dir=args.root_dir,
        variables=args.variables,
        out_variables=args.out_variables,
        predict_range=args.predict_range,
        hrs_each_step=args.hrs_each_step,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # Test the model
    test_dataloader = datamodule.test_dataloader()
    preds, test_loss = test(test_dataloader)
    
    # Save the predictions to disk
    np.save(os.path.join(args.log_dir, f'y.npy'), preds)
    logging.info(f"Saved predictions to {os.path.join(args.log_dir, f'y.npy')}")
    logging.info(f"Test loss: {test_loss}")

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
