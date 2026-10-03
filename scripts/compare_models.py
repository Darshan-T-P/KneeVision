import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader
import time

from kneevision.config.settings import (
    RAW_DATA_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, IMAGE_SIZE,
    WEIGHT_DECAY, MAX_GRAD_NORM, LABEL_SMOOTHING, MIXUP_ALPHA, SAMPLER_POWER,
    EARLY_STOP_PATIENCE, CHECKPOINT_INTERVAL, MLFLOW_ENABLED, MODELS_DIR,
    NUM_KL_CLASSES, SEED, NUM_WORKERS,
)
from kneevision.models.image_model import KneeXRayClassifier
from kneevision.data.dataset import KneeXRayDataset, MixUpDataset, make_weighted_sampler
from kneevision.data.prepare import get_paths_and_labels, class_weights as make_class_weights, minority_labels
from kneevision.data.transforms import train_transform, minority_transform, val_transform
from kneevision.training.trainer import train_epoch, validate, EMA, EarlyStopping, save_checkpoint, load_checkpoint
from kneevision.training.losses import FocalLoss, OrdinalLoss
from kneevision.evaluation.report import compute_confusion_matrix
from kneevision.utils.helpers import (
    set_seed, get_device, seed_worker, get_git_commit,
    get_environment_info, set_rng_state,
)
from kneevision.utils.logging import setup_logger

logger = setup_logger("compare_models")

VAL_SPLIT_DIR = RAW_DATA_DIR / "val"

TAG_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def artifact_stem(model_name: str, ordinal: bool = False, binary: bool = False, tag: str = "") -> str:
    """Unique (model, mode, tag) stem used for every artifact filename.

    Tagged runs write to `best_{stem}.pt`, `checkpoint_{stem}.pt`, etc., so a
    `--tag` fully isolates an experiment's artifacts from untagged runs. The
    definitive (untagged) names such as `best_densenet121_ordinal.pt` are never
    produced when a tag is supplied.
    """
    # NOTE: ordinal and binary must combine into a distinct suffix. The old
    # `"_ordinal" if ordinal else ("_binary" if binary else "")` collapsed
    # `--binary --ordinal` to the same "_ordinal" suffix as a plain 5-class
    # ordinal run, so an untagged binary+ordinal run would silently overwrite
    # the frozen, audited `best_{model}_ordinal.pt` definitive checkpoint.
    if ordinal and binary:
        suffix = "_ordinal_binary"
    elif ordinal:
        suffix = "_ordinal"
    elif binary:
        suffix = "_binary"
    else:
        suffix = ""
    tag_part = f"_{tag}" if tag else ""
    return f"{model_name}{suffix}{tag_part}"


def validate_cli(args) -> str | None:
    """Return an error message for invalid CLI combinations, else None.

    Kept as a pure function so the rules are unit-testable without invoking the
    training harness.
    """
    if args.soft_ordinal_targets and not args.ordinal:
        return "--soft-ordinal-targets requires --ordinal"
    if args.mixup_alpha is not None and args.mixup_alpha < 0:
        return "--mixup-alpha must be >= 0 (0 disables MixUp)"
    if args.class_weights is not None and args.class_weights < 0:
        return "--class-weights must be >= 0 (0 disables class weighting)"
    if args.label_smoothing is not None and not (0.0 <= args.label_smoothing < 1.0):
        return "--label-smoothing must be in [0, 1)"
    if args.tag and not TAG_RE.fullmatch(args.tag):
        return "--tag may only contain [A-Za-z0-9_-]"
    return None


def build_provenance(*, model_name: str, ordinal: bool, binary: bool,
                     num_classes: int, batch_size: int, num_epochs: int,
                     patience: int, seed: int, device,
                     use_mixup_alpha: float, soft_targets: bool,
                     class_weights_enabled: bool, use_label_smoothing: float,
                     tag: str) -> dict:
    """The experiment/config/environment record written into every checkpoint."""
    return {
        "experiment": f"{model_name}_{('ordinal' if ordinal else 'binary' if binary else 'standard')}",
        "model_name": model_name,
        "ordinal": ordinal,
        "binary": binary,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "patience": patience,
        "seed": seed,
        "image_size": IMAGE_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": use_label_smoothing if not ordinal else None,
        "mixup_alpha": use_mixup_alpha,
        "sampler_power": SAMPLER_POWER,
        "class_weights": class_weights_enabled if not ordinal else None,
        "soft_targets": soft_targets,
        "tag": tag or None,
        "criterion": (
            f"OrdinalLoss(alpha=None, soft_targets={soft_targets})" if ordinal
            else f"FocalLoss(gamma=2.0, label_smoothing={use_label_smoothing}, "
                  f"class_weights={class_weights_enabled})"
        ),
        "selection_split": "val",
        "device": str(device),
        "git_commit": get_git_commit(),
        **get_environment_info(),
    }

if MLFLOW_ENABLED:
    from kneevision.utils.tracking import MLflowTracker
    tracker = MLflowTracker()


def map_binary(labels: list[int]) -> list[int]:
    """KL 0-1 -> 0 (No OA), KL 2-4 -> 1 (radiographic OA)."""
    return [0 if grade <= 1 else 1 for grade in labels]


def run(model_name: str, ordinal: bool = False, resume: bool = False,
        batch_size: int = BATCH_SIZE, num_epochs: int = NUM_EPOCHS,
        binary: bool = False, patience: int = EARLY_STOP_PATIENCE,
        seed: int = SEED, num_workers: int = NUM_WORKERS,
        soft_targets: bool = False, mixup_alpha: float | None = None,
        class_weights: float | None = None, label_smoothing: float | None = None,
        tag: str = ""):
    num_classes = 2 if binary else NUM_KL_CLASSES
    mode = "Binary" if binary else ("Ordinal" if ordinal else "Standard")
    run_name = f"{model_name}_{mode.lower()}{('_' + tag) if tag else ''}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger.info("=" * 60)
    logger.info("Training: %s  |  %s  | %d classes | Batch: %d | Seed: %d", model_name, mode, num_classes, batch_size, seed)
    logger.info("=" * 60)

    # Reseed here, inside run(), not once in __main__: with multi-backbone
    # invocations every model then starts from an identical RNG state and the
    # runs are comparable.
    set_seed(seed)
    device = get_device()

    # Effective configuration: CLI overrides fall back to the settings defaults,
    # so an invocation without the new flags behaves exactly as before.
    use_mixup_alpha = MIXUP_ALPHA if mixup_alpha is None else mixup_alpha
    use_label_smoothing = LABEL_SMOOTHING if label_smoothing is None else label_smoothing
    class_weights_enabled = class_weights is None or class_weights > 0.0

    provenance = build_provenance(
        model_name=model_name, ordinal=ordinal, binary=binary,
        num_classes=num_classes, batch_size=batch_size, num_epochs=num_epochs,
        patience=patience, seed=seed, device=device,
        use_mixup_alpha=use_mixup_alpha, soft_targets=soft_targets,
        class_weights_enabled=class_weights_enabled,
        use_label_smoothing=use_label_smoothing, tag=tag,
    )

    if MLFLOW_ENABLED:
        run_tags = {"model": model_name, "mode": mode, "project": "kneevision",
                    "seed": str(seed), "git_commit": provenance.get("git_commit")}
        if tag:
            run_tags["tag"] = tag
        tracker.start_run(run_name=run_name, tags=run_tags)
        tracker.log_params(provenance)

    IS_CUDA = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if IS_CUDA else None
    train_paths, train_labels = get_paths_and_labels(RAW_DATA_DIR / "train")
    # Model selection happens exclusively on the patient-disjoint val split.
    # The held-out test split is never loaded anywhere in this script.
    val_paths, val_labels = get_paths_and_labels(VAL_SPLIT_DIR)

    if binary:
        train_labels = map_binary(train_labels)
        val_labels = map_binary(val_labels)

    minority_set = minority_labels(train_labels, num_classes=num_classes)

    train_ds = KneeXRayDataset(
        train_paths, train_labels,
        transform=train_transform,
        minority_transform=minority_transform,
        minority_labels=minority_set,
    )

    if use_mixup_alpha > 0:
        train_ds = MixUpDataset(train_ds, alpha=use_mixup_alpha, num_classes=num_classes)

    sampler = make_weighted_sampler(train_labels, power=SAMPLER_POWER)
    val_ds = KneeXRayDataset(val_paths, val_labels, val_transform)

    # An explicit generator + worker_init_fn makes the batch stream reproducible
    # (and resumable) across identical seeds on any worker topology.
    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, worker_init_fn=seed_worker,
                              generator=loader_gen, pin_memory=IS_CUDA)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, worker_init_fn=seed_worker,
                            generator=loader_gen, pin_memory=IS_CUDA)

    model = KneeXRayClassifier(model_name, num_classes, ordinal=ordinal).to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s", f"{params:,}")

    if ordinal:
        criterion = OrdinalLoss(num_classes=num_classes, soft_targets=soft_targets)
    else:
        if class_weights_enabled:
            class_weights_t = torch.tensor(make_class_weights(train_labels, num_classes=num_classes), dtype=torch.float).to(device)
        else:
            class_weights_t = None
        criterion = FocalLoss(alpha=class_weights_t, gamma=2.0, label_smoothing=use_label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    warmup_epochs = min(5, max(1, num_epochs // 6))
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        span = max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / span * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema = EMA(model, decay=0.995)

    save_dir = MODELS_DIR
    save_dir.mkdir(exist_ok=True)
    stem = artifact_stem(model_name, ordinal=ordinal, binary=binary, tag=tag)
    ckpt_path = save_dir / f"checkpoint_{stem}.pt"
    best_path = save_dir / f"best_{stem}.pt"

    early_stop = EarlyStopping(patience=patience)
    start_epoch = 1
    best_kappa = -1.0
    history = {}

    if resume and ckpt_path.exists():
        ep, bk, hist, rng_state, ckpt_meta = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, ema, early_stop)
        start_epoch = ep + 1
        best_kappa = bk
        history = hist or {}
        model = model.to(device)
        # Restore the RNG so the resumed batches/sampling continue where the
        # interrupted run left off.
        set_rng_state(rng_state)
        logger.info("Resumed from epoch %d (best kappa=%.4f)", ep, bk)
        if MLFLOW_ENABLED:
            tracker.set_tag("resumed_from_epoch", ep)
            tracker.set_tag("resumed_best_kappa", bk)

    start = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=MAX_GRAD_NORM, ema=ema, scaler=scaler,
        )
        val_loss, val_qwk, val_metrics = validate(
            model, val_loader, criterion, device, use_kappa=True,
            return_metrics=True, num_classes=num_classes)
        ema_loss, ema_qwk, ema_metrics = validate(
            ema.model, val_loader, criterion, device, use_kappa=True,
            return_metrics=True, num_classes=num_classes)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        history[epoch] = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_qwk": float(val_qwk),
            "val_mae": val_metrics["mae"],
            "val_within_1": val_metrics["within_1_accuracy"],
            "val_within_2": val_metrics["within_2_accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "ema_qwk": float(ema_qwk),
            "lr": lr,
        }

        logger.info(
            "Epoch %2d | %ds | LR %.2e | Train Loss: %.4f | Val Loss: %.4f | QWK: %.4f | EMA QWK: %.4f | MAE: %.3f | Within-1: %.3f",
            epoch, epoch_time, lr, train_loss, val_loss, val_qwk, ema_qwk,
            val_metrics["mae"], val_metrics["within_1_accuracy"],
        )

        if MLFLOW_ENABLED:
            tracker.log_epoch_metrics(epoch, train_loss, val_loss, val_qwk, ema_qwk, lr)
            # The core ordinal bundle (QWK, MAE, within-1/2, accuracy) is logged
            # per epoch; `qwk` is already tracked above as `val_kappa`.
            tracker.log_metrics({
                f"val_{k}": v for k, v in val_metrics.items() if k != "qwk"
            }, step=epoch)

        best = max(val_qwk, ema_qwk)
        if best > best_kappa:
            best_kappa = best
            ckpt = ema.state_dict() if ema_qwk >= val_qwk else model.state_dict()
            torch.save(ckpt, best_path)
            meta = {
                "image_size": IMAGE_SIZE,
                "model_name": model_name,
                "num_classes": num_classes,
                "binary": binary,
                "ordinal": ordinal,
                "best_kappa": float(best),
                "best_validation_qwk": float(best),
                "epoch": epoch,
                "seed": seed,
                "git_commit": provenance.get("git_commit"),
                **get_environment_info(),
            }
            best_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

            # Confusion matrix of the SELECTED weights on the validation split
            # only: true KL class on rows, predicted KL class on columns,
            # KL0..KL{n-1}. Pure evaluation instrumentation of already-chosen
            # weights — it never influences loss, gradients, early stopping,
            # EMA, the RNG, or the selection rule above.
            selected = ema.model if ema_qwk >= val_qwk else model
            _, _, _, cm_preds, cm_labels = validate(
                selected, val_loader, criterion, device, use_kappa=True,
                return_metrics=True, return_predictions=True, num_classes=num_classes)
            cm = compute_confusion_matrix(cm_labels, cm_preds, num_classes=num_classes)
            cm_path = best_path.with_suffix(".cm.json")
            cm_path.write_text(json.dumps({
                "matrix": cm.astype(int).tolist(),
                "rows": f"true KL class (KL0..KL{num_classes - 1})",
                "columns": f"predicted KL class (KL0..KL{num_classes - 1})",
                "classes": [f"KL{i}" for i in range(num_classes)],
                "split": "val",
                "selected_weights": "ema" if ema_qwk >= val_qwk else "raw",
                "epoch": epoch,
                "best_validation_qwk": float(best),
                "seed": seed,
                "git_commit": provenance.get("git_commit"),
            }, indent=2))
            logger.info("  -> Saved best (best validation QWK=%.4f)", best)
            if MLFLOW_ENABLED:
                tracker.set_tag("best_kappa", best)
                tracker.set_tag("best_epoch", epoch)
                tracker.log_artifact(str(best_path))
                tracker.log_artifact(str(cm_path))

        stopped_now = early_stop.step(best)
        if epoch % CHECKPOINT_INTERVAL == 0 or stopped_now:
            save_checkpoint(ckpt_path, model, optimizer, scheduler, ema,
                           early_stop, epoch, best_kappa, history, model_name,
                           ordinal, metadata=provenance)

        if early_stop.stopped:
            logger.info("Early stopping (no improvement for %d epochs)", patience)
            break

    total_time = time.time() - start
    logger.info(
        "%s | Params: %s | Best Kappa: %.4f | Time: %ds",
        model_name, f"{params:,}", best_kappa, total_time,
    )

    if MLFLOW_ENABLED:
        tracker.log_metrics({"best_kappa": best_kappa, "total_time_sec": total_time, "total_params": params})
        tracker.end_run()

    return best_kappa, params, total_time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and compare X-ray backbones")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="max training epochs (default: %(default)s)")
    parser.add_argument("--models", nargs="+", default=["densenet121", "efficientnet-b4"],
                        metavar="BACKBONE", help="backbones to train")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--binary", action="store_true",
                        help="train a 2-class head (KL0-1 vs KL2-4) instead of 5-class KL grading")
    parser.add_argument("--ordinal", action="store_true",
                        help="use CORAL ordinal loss/decoding instead of Focal loss (KL grades are ordered)")
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE,
                        help="early-stopping patience in epochs (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="reproducibility seed (default: %(default)s)")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="DataLoader workers (default: %(default)s)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the existing model checkpoint (models/checkpoint_{name}*)")
    parser.add_argument("--soft-ordinal-targets", action="store_true",
                        help="ORDINAL ONLY: use native soft ordinal targets (expected cumulative-tail "
                             "t_k = P(Y > k)) for 2-D MixUp distributions instead of argmax-hardening")
    parser.add_argument("--mixup-alpha", type=float, default=None, metavar="FLOAT",
                        help="MixUp/CutMix alpha (default: %(default)s; 0 disables MixUp)")
    parser.add_argument("--class-weights", type=float, default=None, metavar="FLOAT",
                        help="Focal only: class-weighting magnitude; 0 disables inverse-frequency "
                             "class weighting (default: %(default)s = current behavior, enabled)")
    parser.add_argument("--label-smoothing", type=float, default=None, metavar="FLOAT",
                        help="Focal only: label smoothing in [0, 1) (default: %(default)s = current 0.1)")
    parser.add_argument("--tag", type=str, default="", metavar="NAME",
                        help="isolate all artifacts under this tag, e.g. '--tag focal_mixup' "
                             "writes best_densenet121_focal_mixup.pt")
    args = parser.parse_args()

    cli_error = validate_cli(args)
    if cli_error:
        parser.error(cli_error)

    logger.info("Epochs: %d | Models: %s | Binary: %s | Ordinal: %s | Patience: %d | Seed: %d",
                args.epochs, ", ".join(args.models), args.binary, args.ordinal, args.patience, args.seed)
    results = []
    for name in args.models:
        kappa, params, t = run(
            name, ordinal=args.ordinal, resume=args.resume,
            batch_size=args.batch_size, num_epochs=args.epochs,
            binary=args.binary, patience=args.patience,
            seed=args.seed, num_workers=args.num_workers,
            soft_targets=args.soft_ordinal_targets, mixup_alpha=args.mixup_alpha,
            class_weights=args.class_weights, label_smoothing=args.label_smoothing,
            tag=args.tag)
        results.append((name, params, kappa, t))

    logger.info("=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    for name, params, kappa, t in results:
        logger.info("%-20s | Params: %7s | Best Kappa: %.4f | Time: %ds", name, f"{params:,}", kappa, t)