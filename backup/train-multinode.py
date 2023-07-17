# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datamodule import GlobalForecastDataModule
from module import GlobalForecastModule
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import logging

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')

    # Seed everything
    parser.add_argument('--seed_everything', type=int, default=42)

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

    # Data
    parser.add_argument('--root_dir', type=str, default='/home/humor/sugon/data-minimal')
    parser.add_argument('--predict_range', type=int, default=72)
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
        "u_component_of_wind_1000",
        "temperature_1000",
        "relative_humidity_1000",
        "specific_humidity_1000",
    ])
    parser.add_argument('--net_default_vars', type=list, default=[
        "land_sea_mask",
        "orography",
        "lattitude",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential_1000",
        "u_component_of_wind_1000",
        "temperature_1000",
        "relative_humidity_1000",
        "specific_humidity_1000",
    ])

    parser.add_argument('--out_variables', type=list, default=["geopotential_1000", "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"])

    args = parser.parse_args()
    return args



def train(model, dataloader, optimizer, lr_scheduler, device, epoch, total_epochs, local_rank):
    total_loss = 0
    count = 0
    progress_bar = tqdm(
        dataloader,
        desc=f"\033[1;34mEpoch {epoch + 1}/{total_epochs} [Train]\033[0m",
        ncols=100,
        bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]",
        ascii=True,
        disable=local_rank != 0,
    )
    for batch in progress_bar:
        optimizer.zero_grad()
        batch = [item.to(device) if index < 3 else item for index, item in enumerate(batch)]
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
        count += 1
    return total_loss / count

def test(model, dataloader, device, local_rank):
    total_loss = 0
    count = 0
    progress_bar = tqdm(
        dataloader,
        desc="\033[1;32mTesting\033[0m",
        ncols=100,
        bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]",
        ascii=True,
        disable=local_rank != 0,
    )
    with torch.no_grad():
        for batch in progress_bar:
            batch = [item.to(device) if index < 3 else item for index, item in enumerate(batch)]
            loss = model.test_step(batch)
            total_loss += loss.item()
            count += 1
    return total_loss / count

def eval(model, dataloader, device, local_rank):
    total_loss = 0
    count = 0
    progress_bar = tqdm(
        dataloader,
        desc="\033[1;32mEvaluating\033[0m",
        ncols=100,
        bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]",
        ascii=True,
        disable=local_rank != 0,
    )
    with torch.no_grad():
        for batch in progress_bar:
            batch = [item.to(device) if index < 3 else item for index, item in enumerate(batch)]
            loss = model.validation_step(batch)
            total_loss += loss.item()
            count += 1
    return total_loss / count

def main(local_rank):
    args = parse_arguments()

    # Enable ROCm backend
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    cudnn.benchmark = True
    cudnn.enabled = True

    # Initialize the distributed environment
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=4)
    device = torch.device(f'cuda:{local_rank}')
    
    # Set the PYTORCH_HIP_ALLOC_CONF environment variable
    os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb: 10240'

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize the data module and model
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

    model = GlobalForecastModule(
        default_vars=args.net_default_vars,
        lr=args.lr,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        warmup_start_lr=args.warmup_start_lr,
        eta_min=args.eta_min,
        pretrained_path=args.pretrained_path
    )
    normalization = datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    model.set_denormalization(mean_denorm, std_denorm)
    model.set_lat_lon(*datamodule.get_lat_lon())
    model.set_pred_range(datamodule.predict_range)
    model.set_val_clim(datamodule.val_clim)
    model.set_test_clim(datamodule.test_clim)

    model.to(device)

    # Set up the optimizer
    optimizer, lr_scheduler = model.configure_optimizers()

    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=0)

    # Train the model
    for epoch in range(args.max_epochs):
        train_dataloader = datamodule.train_dataloader()
        train_loss = train(model, train_dataloader, optimizer, lr_scheduler, device, epoch, args.max_epochs, local_rank)
        logging.info(f"Epoch {epoch + 1}/{args.max_epochs}, Training loss: {train_loss:.4f}")

        val_dataloader = datamodule.val_dataloader()
        train_loss = eval(model, train_dataloader, optimizer, lr_scheduler, device, epoch, args.max_epochs, local_rank)
        logging.info(f"Epoch {epoch + 1}/{args.max_epochs}, Training loss: {train_loss:.4f}")

    # Test the trained model
    test_dataloader = datamodule.test_dataloader()
    test_loss = test(model, test_dataloader, device, local_rank)
    logging.info(f"Test loss: {test_loss:.4f}")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    mp.spawn(main, args=(), nprocs=4, join=True)