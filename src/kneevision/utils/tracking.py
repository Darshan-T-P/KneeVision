import importlib.abc as abc
import importlib.resources.abc as resources_abc

if not hasattr(abc, "Traversable"):
    abc.Traversable = resources_abc.Traversable

import mlflow
import torch
from typing import Any
from kneevision.config.settings import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, IMAGE_SIZE,
    WEIGHT_DECAY, MAX_GRAD_NORM, LABEL_SMOOTHING, MIXUP_ALPHA,
    SAMPLER_POWER, EARLY_STOP_PATIENCE,
)
from kneevision.utils.logging import setup_logger

logger = setup_logger("tracking")


class MLflowTracker:
    def __init__(self, experiment_name: str | None = None):
        self.experiment_name = experiment_name or MLFLOW_EXPERIMENT_NAME
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
        self.active_run = None

    def start_run(self, run_name: str | None = None, tags: dict | None = None):
        self.active_run = mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)

    def log_params(self, params: dict[str, Any]):
        mlflow.log_params(params)

    def log_params_from_settings(self, extra: dict | None = None):
        params = {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "image_size": IMAGE_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "label_smoothing": LABEL_SMOOTHING,
            "mixup_alpha": MIXUP_ALPHA,
            "sampler_power": SAMPLER_POWER,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        }
        if extra:
            params.update(extra)
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        mlflow.log_metrics(metrics, step=step)

    def log_epoch_metrics(self, epoch: int, train_loss: float, val_loss: float,
                          val_kappa: float, ema_kappa: float, lr: float):
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_kappa": val_kappa,
            "ema_kappa": ema_kappa,
            "learning_rate": lr,
        }, step=epoch)

    def log_model(self, model: torch.nn.Module, model_name: str, artifact_path: str = "model"):
        mlflow.pytorch.log_model(model, artifact_path=artifact_path, registered_model_name=model_name)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def log_artifacts(self, local_dir: str):
        mlflow.log_artifacts(local_dir)

    def set_tag(self, key: str, value: Any):
        mlflow.set_tag(key, value)

    def end_run(self):
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    @property
    def run_id(self) -> str | None:
        return self.active_run.info.run_id if self.active_run else None
