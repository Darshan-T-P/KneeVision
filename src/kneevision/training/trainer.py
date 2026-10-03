import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from kneevision.evaluation.report import compute_metrics
from kneevision.training.losses import ordinal_to_class
from kneevision.utils.helpers import get_rng_state
from kneevision.utils.logging import setup_logger

logger = setup_logger("trainer")


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p, 1 - self.decay)

    def state_dict(self):
        return self.model.state_dict()


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.stopped = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False

    def state_dict(self):
        return {"best_score": self.best_score, "counter": self.counter, "stopped": self.stopped}

    def load_state_dict(self, d):
        self.best_score = d["best_score"]
        self.counter = d["counter"]
        self.stopped = d["stopped"]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
    ema: EMA | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast("cuda", enabled=scaler is not None):
            loss = criterion(model(images), labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_kappa: bool = True,
    return_metrics: bool = False,
    num_classes: int | None = None,
    class_names: list[str] | None = None,
    return_predictions: bool = False,
) -> tuple[float, float] | tuple[float, float, dict] | tuple[float, float, dict, list, list]:
    """Evaluate on the given split.

    Pure evaluation instrumentation: it never touches gradients, optimizers,
    schedulers, EMA, early stopping, the RNG, dataloader ordering, or checkpoint
    state. ``return_predictions=True`` additionally returns the (predicted,
    label) integer lists so callers can persist the confirmation matrix of the
    selected model without re-deriving metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Validation"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        if getattr(model, "ordinal", False):
            predicted = ordinal_to_class(outputs)
        else:
            _, predicted = torch.max(outputs, 1)

        total_loss += loss.item()
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)

    # Decisions are driven only by this authoritative metric bundle (report.py);
    # the selection score is the validation QWK on this split.
    metrics = compute_metrics(all_labels, all_preds, num_classes=num_classes,
                              class_names=class_names)
    score = metrics["qwk"] if use_kappa else metrics["accuracy"]

    if return_predictions:
        return avg_loss, score, metrics, all_preds, all_labels
    if return_metrics:
        return avg_loss, score, metrics
    return avg_loss, score


def save_checkpoint(path, model, optimizer, scheduler, ema, early_stop,
                    epoch, best_kappa, history, model_name, ordinal,
                    metadata: dict | None = None, include_rng_state: bool = True):
    """Persist a full, resumable training state.

    Beyond weights and optimizer/scheduler state this stores: the experiment
    pointer, the best validation QWK, the full per-epoch history, and — when
    ``metadata`` is supplied — experiment/seed/config/environment provenance.
    ``include_rng_state`` captures the process RNG so a resumed run continues
    from the exact sampler state.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict() if ema else None,
        "early_stop_state": early_stop.state_dict() if early_stop else None,
        "epoch": epoch,
        "best_kappa": best_kappa,
        "history": history,
        "model_name": model_name,
        "ordinal": ordinal,
        "metadata": metadata,
        "rng_state": get_rng_state() if include_rng_state else None,
    }, path)
    logger.info("Checkpoint saved to %s", path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, ema=None, early_stop=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ema and ckpt.get("ema_state_dict"):
        ema.model.load_state_dict(ckpt["ema_state_dict"])
    if early_stop and ckpt.get("early_stop_state"):
        early_stop.load_state_dict(ckpt["early_stop_state"])
    return (ckpt.get("epoch", 0), ckpt.get("best_kappa", -1.0),
            ckpt.get("history", {}), ckpt.get("rng_state"), ckpt.get("metadata"))
