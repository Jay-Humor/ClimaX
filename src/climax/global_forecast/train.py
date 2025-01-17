# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.distributed as dist

import os
import argparse
import torch
from datamodule import GlobalForecastDataModule
from module import GlobalForecastModule
from tqdm import tqdm
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
    parser.add_argument('--pretrained_path', type=str, default='/public/home/qindaotest/huangshijie/ClimaX-Pytorch/ckpts/5.625deg.ckpt')


    # Data
    parser.add_argument('--root_dir', type=str, default='/home/humor/sugon/data-minimal')
    parser.add_argument('--log_dir', type=str, default='/home/humor/sugon/ClimaX-Pytorch/logs')
    parser.add_argument('--predict_range', type=int, default=72)
    parser.add_argument('--hrs_each_step', type=int, default=1)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=2)
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

def train(model, dataloader, optimizer, lr_scheduler, device, epoch, total_epochs):
    total_loss = 0
    count = 0
    progress_bar = tqdm(
        dataloader,
        desc=f"[Epoch {epoch + 1}/{total_epochs} [Train]]",
        ncols=100,
        bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]",
        ascii=True,
        position=0,
        leave=False
    )
    for batch in dataloader:
        optimizer.zero_grad()
        batch = [item.to(device) if index < 2 else item for index, item in enumerate(batch)]
        loss = model.training_step(batch)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)
        total_loss += loss.item()
        count += 1
    return total_loss / count

def test(model, dataloader, device):
    total_loss = {}
    count = 0
    progress_bar = tqdm(
        dataloader,
        desc="[Testing]",
        ncols=100,
        bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]",
        ascii=True,
        position=0,
        leave=False
    )
    with torch.no_grad():
        preds = list()
        for batch in dataloader:
            batch = [item.to(device) if index < 2 else item for index, item in enumerate(batch)]
            pred, loss = model.test_step(batch)
            progress_bar.update(1)
            for key in loss.keys():
                if key not in total_loss:
                    total_loss[key] = 0
                total_loss[key] += loss[key].item()
            count += 1
            preds.append(pred.cpu().numpy())
    preds = np.array(preds)
    for key in total_loss.keys():
        total_loss[key] /= count
    return preds, total_loss

def eval(model, dataloader, device):
    total_loss = {}
    count = 0
    progress_bar = tqdm(
        dataloader,
        desc="[Evaluating]",
        ncols=100,
        bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]",
        ascii=True,
        position=0,
        leave=False
    )
    with torch.no_grad():
        for batch in dataloader:
            batch = [item.to(device) if index < 2 else item for index, item in enumerate(batch)]
            loss = model.validation_step(batch)
            progress_bar.update(1)
            for key in loss.keys():
                if key not in total_loss:
                    total_loss[key] = 0
                total_loss[key] += loss[key].item()
            count += 1
    for key in total_loss.keys():
        total_loss[key] /= count
    return total_loss

def main():
    # Parse arguments
    args = parse_arguments()

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(args.log_dir)

    # Initialize the distributed process group if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl")
        local_rank = args.local_rank
    torch.cuda.set_device(local_rank) # type: ignore

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

    # Initialize the model
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

    # Set up the data module and model
    normalization = datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm # type: ignore
    model.set_denormalization(mean_denorm, std_denorm)
    model.set_lat_lon(*datamodule.get_lat_lon())
    model.set_pred_range(datamodule.predict_range)
    model.set_val_clim(datamodule.val_clim)
    model.set_test_clim(datamodule.test_clim)

    # Set up the device
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda", local_rank) # type: ignore
        model.net = torch.nn.parallel.DistributedDataParallel(model.net.to(device), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) # type: ignore
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.net = torch.nn.DataParallel(model.net.to(device), device_ids=[0]) # type: ignore

    # Set up the optimizer and learning rate scheduler
    optimizer, lr_scheduler = model.configure_optimizers()

    best_val_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(args.max_epochs):
        # Train the model
        train_dataloader = datamodule.train_dataloader()
        train_loss = train(model, train_dataloader, optimizer, lr_scheduler, device, epoch, args.max_epochs)
        logging.info(f"Epoch {epoch + 1}/{args.max_epochs}, Training loss: {train_loss:.4f}")

        # Evaluate the model
        val_dataloader = datamodule.val_dataloader()
        val_loss = eval(model, val_dataloader, device)
        logging.info(f"Epoch {epoch + 1}/{args.max_epochs}, Evaluation loss: {val_loss}")

        # Early stopping logic
        if val_loss['w_mse'] < best_val_loss:
            best_val_loss = val_loss['w_mse']
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss")
                break
        
        # Save the model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_name = f"model_checkpoint_epoch{epoch}.pt"
            checkpoint_path = os.path.join(args.log_dir, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            logging.info(f"Saved model checkpoint to {checkpoint_path}")

        # Log training, validation losses, and learning rate to TensorBoard
        writer.add_scalars("Loss", {"Train": train_loss, "Validation_w_mse": val_loss['w_mse'], "Validation_w_rmse": val_loss['w_rmse'], "Validation_acc": val_loss['acc']}, global_step=epoch)
        writer.add_scalar("Learning Rate", lr_scheduler.get_last_lr()[0], global_step=epoch)


    # Test the model
    test_dataloader = datamodule.test_dataloader()
    preds, test_loss = test(model, test_dataloader, device)
    
    # Save the predictions to disk
    np.save(os.path.join(args.log_dir, f'output.npy'), preds)
    logging.info(f"Saved predictions to {os.path.join(args.log_dir, f'output.npy')}")
    logging.info(f"Test loss: {test_loss}")

    # Close the TensorBoard writer
    writer.close()

    # Clean up the distributed process group:
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
