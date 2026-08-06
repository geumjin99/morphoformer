"""PyTorch Lightning training module for MorphoFormer."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict
import numpy as np
from sklearn.metrics import r2_score

from .losses.mcl_loss import MorphologyConsistentMTLLoss


class MorphoFormerLitModule(pl.LightningModule):
    """Unified training / validation / test loop for MorphoFormer."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        scheduler_T_max: int = 150,
        scheduler_eta_min: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        self.model = model
        self.loss_fn = loss_fn
        self.val_outputs = []
        self.test_outputs = []

    # ── epoch hook: forward current epoch to MCL warmup scheduler ──────────

    def on_train_epoch_start(self):
        if isinstance(self.loss_fn, MorphologyConsistentMTLLoss):
            self.loss_fn.set_epoch(self.current_epoch)

    # ── shared step ─────────────────────────────────────────────────────────

    def _compute_loss(self, batch):
        x, bh_true, bf_true = batch[:3]
        bh_true = bh_true.unsqueeze(1)
        bf_true = bf_true.unsqueeze(1)

        outputs = self.model(x)

        # MorphoFormer with uncertainty: (bh_pred, bf_pred, log_sigma_h, log_sigma_f, bh_from_bf)
        if len(outputs) == 5:
            bh_pred, bf_pred, log_sigma_h, log_sigma_f, bh_from_bf = outputs
            loss = self.loss_fn(
                bh_pred, bh_true, bf_pred, bf_true,
                log_sigma_h=log_sigma_h, log_sigma_f=log_sigma_f,
                bh_from_bf=bh_from_bf,
            )
        # MorphoFormer without uncertainty: (bh_pred, bf_pred, bh_from_bf)
        elif len(outputs) == 3:
            bh_pred, bf_pred, bh_from_bf = outputs
            loss = self.loss_fn(
                bh_pred, bh_true, bf_pred, bf_true,
                bh_from_bf=bh_from_bf,
            )
        # Legacy 4-output: (bh_pred, bf_pred, log_sigma_h, log_sigma_f)
        elif len(outputs) == 4:
            bh_pred, bf_pred, log_sigma_h, log_sigma_f = outputs
            loss = self.loss_fn(bh_pred, bh_true, bf_pred, bf_true,
                                log_sigma_h=log_sigma_h, log_sigma_f=log_sigma_f)
        else:
            bh_pred, bf_pred = outputs
            loss = self.loss_fn(bh_pred, bh_true, bf_pred, bf_true)

        return loss, bh_pred, bf_pred, bh_true, bf_true

    # ── steps ───────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        loss, bh_pred, bf_pred, bh_true, bf_true = self._compute_loss(batch)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/mae_h', torch.abs(bh_pred - bh_true).mean(), prog_bar=True)
        self.log('train/mae_f', torch.abs(bf_pred - bf_true).mean(), prog_bar=True)
        if isinstance(self.loss_fn, MorphologyConsistentMTLLoss):
            self.log('train/lambda_consist', self.loss_fn.effective_lambda)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, bh_pred, bf_pred, bh_true, bf_true = self._compute_loss(batch)
        out = {
            'bh_pred': bh_pred.detach().cpu(),
            'bf_pred': bf_pred.detach().cpu(),
            'bh_true': bh_true.detach().cpu(),
            'bf_true': bf_true.detach().cpu(),
        }
        if dataloader_idx == 0:
            self.val_outputs.append(out)
        else:
            self.test_outputs.append(out)
        return loss

    def on_validation_epoch_end(self):
        _PROG_BAR = {'mae_h', 'r2_h', 'mae_f', 'combined_mae'}
        for outputs, prefix in [(self.val_outputs, 'val'), (self.test_outputs, 'test_epoch')]:
            if not outputs:
                continue
            bh_preds = torch.cat([o['bh_pred'] for o in outputs])
            bf_preds = torch.cat([o['bf_pred'] for o in outputs])
            bh_trues = torch.cat([o['bh_true'] for o in outputs])
            bf_trues = torch.cat([o['bf_true'] for o in outputs])
            for name, val in self._compute_metrics(bh_preds, bh_trues, bf_preds, bf_trues).items():
                self.log(f'{prefix}/{name}', val,
                         prog_bar=(prefix == 'val' and name in _PROG_BAR))
            outputs.clear()

    def test_step(self, batch, batch_idx):
        loss, bh_pred, bf_pred, bh_true, bf_true = self._compute_loss(batch)
        self.test_outputs.append({
            'bh_pred': bh_pred.detach().cpu(),
            'bf_pred': bf_pred.detach().cpu(),
            'bh_true': bh_true.detach().cpu(),
            'bf_true': bf_true.detach().cpu(),
        })
        return loss

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return
        bh_preds = torch.cat([o['bh_pred'] for o in self.test_outputs])
        bf_preds = torch.cat([o['bf_pred'] for o in self.test_outputs])
        bh_trues = torch.cat([o['bh_true'] for o in self.test_outputs])
        bf_trues = torch.cat([o['bf_true'] for o in self.test_outputs])
        for name, val in self._compute_metrics(bh_preds, bh_trues, bf_preds, bf_trues).items():
            self.log(f'test/{name}', val)
        self.test_outputs.clear()

    # ── metrics ─────────────────────────────────────────────────────────────

    def _compute_metrics(self, bh_pred, bh_true, bf_pred, bf_true) -> Dict[str, float]:
        bh_pred = bh_pred.numpy().flatten()
        bh_true = bh_true.numpy().flatten()
        bf_pred = bf_pred.numpy().flatten()
        bf_true = bf_true.numpy().flatten()

        def safe_r2(t, p):
            try:
                return float(r2_score(t, p))
            except Exception:
                return 0.0

        m = {}
        m['mae_h']  = float(np.mean(np.abs(bh_pred - bh_true)))
        m['rmse_h'] = float(np.sqrt(np.mean((bh_pred - bh_true) ** 2)))
        m['me_h']   = float(np.mean(bh_pred - bh_true))
        m['r2_h']   = safe_r2(bh_true, bh_pred)
        m['mae_f']  = float(np.mean(np.abs(bf_pred - bf_true)))
        m['rmse_f'] = float(np.sqrt(np.mean((bf_pred - bf_true) ** 2)))
        m['me_f']   = float(np.mean(bf_pred - bf_true))
        m['r2_f']   = safe_r2(bf_true, bf_pred)
        m['combined_mae'] = m['mae_h'] + m['mae_f']
        return m

    # ── optimizer ───────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.scheduler_T_max,
            eta_min=self.hparams.scheduler_eta_min,
        )
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
