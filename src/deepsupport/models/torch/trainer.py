"""
DeepSupport PyTorch Training Engine
===================================
Standardisierter, robuster PyTorch-Trainingsloop mit:
- AdamW Optimizer & Weight Decay
- Learning Rate Scheduling (ReduceLROnPlateau / CosineAnnealing)
- Early Stopping mit Best-Weights-Restoration
- Gradient Clipping gegen Gradient Explosion
- History-Tracking (kompatibel mit metrics_logger.plot_learning_curve)
"""

import time
import copy
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    """
    Early Stopping Überwachung mit Best-Model-Restoration.
    """
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.early_stop: bool = False

    def __call__(self, val_metric: float, model: nn.Module, epoch: int) -> bool:
        score = -val_metric if self.mode == "min" else val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop

    def restore_best_weights(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ModelTrainer:
    """
    Generischer PyTorch-Trainer für Überwachte Regressions-, Klassifikations-
    und Survival-Modelle.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        clip_grad_norm: float = 1.0,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.clip_grad_norm = clip_grad_norm

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            self.optimizer.zero_grad()
            batch_loss = self._step(batch)
            batch_loss.backward()

            if self.clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            batch_size = next(iter(batch.values())).size(0)
            running_loss += batch_loss.item() * batch_size
            n_samples += batch_size

        return running_loss / max(1, n_samples)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0
        n_samples = 0

        for batch in val_loader:
            batch_loss = self._step(batch)
            batch_size = next(iter(batch.values())).size(0)
            running_loss += batch_loss.item() * batch_size
            n_samples += batch_size

        return running_loss / max(1, n_samples)

    def _step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Führt einen einzelnen Vorwärtsschritt und Verlustberechnung durch."""
        # Auf Device verschieben
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if hasattr(self.model, "compute_loss"):
            return self.model.compute_loss(batch)
        elif self.loss_fn is not None:
            if "x" in batch and "event" in batch:
                preds = self.model(batch["x"])
                return self.loss_fn(preds, batch["event"])
            elif "sequence" in batch and "target_note" in batch:
                preds = self.model(batch["sequence"], batch.get("mask"), batch.get("context"))
                return self.loss_fn(preds, batch["target_note"])
            else:
                raise ValueError("batch-Format passt zu keiner Standard-Verlustsignatur.")
        else:
            raise NotImplementedError("Weder model.compute_loss noch loss_fn bereitgestellt.")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
        verbose: bool = True,
        eval_fn: Optional[Callable[[nn.Module, DataLoader], float]] = None,
    ) -> Dict[str, List[float]]:
        """
        Führt vollständigen Trainingslauf mit Early Stopping durch.
        """
        early_stopping = EarlyStopping(patience=patience, mode="min")
        history = {"loss": [], "val_loss": [], "epoch_time_s": []}

        if verbose:
            print(f"[PyTorch Trainer] Training auf Gerät: {self.device} | Epochen: {epochs} | Patience: {patience}")

        start_total = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            dt = time.time() - t0

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["epoch_time_s"].append(dt)

            # Learning Rate Scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Zeit: {dt:.2f}s | LR: {current_lr:.2e}")

            if early_stopping(val_loss, self.model, epoch):
                if verbose:
                    print(f"[PyTorch Trainer] Early Stopping in Epoche {epoch} (Beste Epoche: {early_stopping.best_epoch} mit Val Loss {early_stopping.best_score * -1:.5f})")
                break

        early_stopping.restore_best_weights(self.model)
        total_time = time.time() - start_total
        history["total_time_s"] = total_time

        if verbose:
            print(f"[PyTorch Trainer] Training beendet nach {len(history['loss'])} Epochen in {total_time:.2f}s.")

        return history
