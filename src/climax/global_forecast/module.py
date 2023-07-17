# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import Any

import torch
import torch.nn as nn
from torchvision.transforms import transforms
import logging

from arch import ClimaX
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
)
from utils.pos_embed import interpolate_pos_embed


class GlobalForecastModule(nn.Module):
    """PyTorch module for global forecasting with the ClimaX model.

    Args:
        net (ClimaX): ClimaX model.
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        default_vars: list,
        pretrained_path: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):
        super().__init__()
        self.net=ClimaX(default_vars)
        self.lr=lr
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.weight_decay=weight_decay
        self.warmup_epochs=warmup_epochs
        self.max_epochs=max_epochs
        self.warmup_start_lr=warmup_start_lr
        self.eta_min=eta_min
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith('http'):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        # logging.info(checkpoint_model.keys())
        # if self.net.parallel_patch_embed:
        #     if "net.token_embeds.proj_weights" not in checkpoint_model.keys():
        #         raise ValueError(
        #             "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
        #         )
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    def training_step(self, batch: Any):
        x, y, lead_times, variables, out_variables = batch

        loss_dict, _ = self.net.forward(x, y, lead_times, variables, out_variables, [lat_weighted_mse], lat=self.lat)
        loss_dict = loss_dict[0] # type: ignore
        loss = loss_dict["loss"]

        return loss

    def validation_step(self, batch: Any):
        x, y, lead_times, variables, out_variables = batch

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        loss, preds = self.net.forward(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            metric=None,
            lat=self.lat,
        )

        transform=self.denormalization
        clim=self.val_clim
        log_postfix=log_postfix
        metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc]
        all_loss_dicts = [m(preds, y, transform, out_variables, self.lat, clim, log_postfix) for m in metrics]

        loss_dict = {}

        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        return loss_dict

    def test_step(self, batch: Any):
        x, y, lead_times, variables, out_variables = batch

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"
        
        loss, preds = self.net.forward(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            metric=None,
            lat=self.lat,
        )

        transform=self.denormalization
        clim=self.val_clim
        log_postfix=log_postfix
        metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc]
        all_loss_dicts = [m(preds, y, transform, out_variables, self.lat, clim, log_postfix) for m in metrics]

        loss_dict = {}

        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        return preds , loss_dict

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.lr,
                    "betas": (self.beta_1, self.beta_2),
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.lr,
                    "betas": (self.beta_1, self.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.warmup_epochs,
            self.max_epochs,
            self.warmup_start_lr,
            self.eta_min,
        )

        return optimizer, lr_scheduler
