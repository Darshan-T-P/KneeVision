import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from kneevision.training.losses import ordinal_to_class


class EMA:
    """Exponential Moving Average of model weights."""
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
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

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
) -> tuple[float, float]:
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

    if use_kappa:
        score = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    else:
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        score = correct / len(all_labels)

    return avg_loss, score


def save_checkpoint(path, model, optimizer, scheduler, ema, early_stop,
                    epoch, best_kappa, history, model_name, ordinal):
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
    }, path)
    print(f"  Checkpoint saved to {path}")


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
    return ckpt.get("epoch", 0), ckpt.get("best_kappa", -1.0), ckpt.get("history", {})
