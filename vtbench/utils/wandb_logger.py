"""
W&B Logger Utility for VTBench Experiments
==========================================
Lightweight wrapper around wandb that:
  - Is a no-op if wandb is not installed or WANDB_DISABLED=true
  - Provides a consistent API across all experiment scripts
  - Groups runs by experiment name and dataset

Usage:
    from vtbench.utils.wandb_logger import WandbLogger

    # At experiment start
    logger = WandbLogger(
        project="vtbench",
        experiment="6a_encodings",
        config=cfg,
    )

    # For each training run
    run_id = logger.start_run(
        name="gasf_resnet18_seed42",
        tags=["gasf", "resnet18"],
        config={"encoding": "gasf", "model": "resnet18", "seed": 42},
    )

    # During training
    logger.log({"epoch": epoch, "train_loss": loss, "val_acc": acc})

    # After training
    logger.end_run({"test_accuracy": best_acc})

    # At experiment end
    logger.finish()
"""

import os


def _wandb_available():
    """Check if wandb is importable and not explicitly disabled."""
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1", "yes"):
        return False
    try:
        import wandb
        return True
    except ImportError:
        return False


class WandbLogger:
    """
    Thin wrapper around wandb for VTBench experiments.

    If wandb is not installed or WANDB_DISABLED=true, all methods
    are silent no-ops. This lets experiment code include wandb
    logging without any conditional guards.
    """

    def __init__(self, project="vtbench", experiment=None, config=None,
                 entity=None):
        """
        Initialize the logger.

        Args:
            project: wandb project name
            experiment: experiment name (used as group)
            config: top-level experiment config dict
            entity: wandb team/entity (optional)
        """
        self.enabled = _wandb_available()
        self.project = project
        self.experiment = experiment
        self.entity = entity
        self.base_config = config or {}
        self._run = None

        if self.enabled:
            import wandb
            self._wandb = wandb
            # Set default environment
            os.environ.setdefault("WANDB_SILENT", "true")
        else:
            self._wandb = None

    @property
    def active(self):
        """Whether wandb is enabled and a run is active."""
        return self.enabled and self._run is not None

    def start_run(self, name=None, tags=None, config=None):
        """
        Start a new wandb run for one training job.

        Args:
            name: run name (e.g., "gasf_resnet18_seed42")
            tags: list of tags
            config: run-specific config dict (merged with base)

        Returns:
            run ID string, or None if disabled
        """
        if not self.enabled:
            return None

        run_config = {**self.base_config}
        if config:
            run_config.update(config)

        self._run = self._wandb.init(
            project=self.project,
            entity=self.entity,
            group=self.experiment,
            name=name,
            tags=tags or [],
            config=run_config,
            reinit=True,
        )
        return self._run.id

    def log(self, metrics, step=None):
        """Log metrics for the current run."""
        if not self.active:
            return
        if step is not None:
            self._wandb.log(metrics, step=step)
        else:
            self._wandb.log(metrics)

    def log_epoch(self, epoch, train_loss=None, val_loss=None,
                  train_acc=None, val_acc=None, lr=None, **kwargs):
        """Convenience method for logging typical training epoch metrics."""
        metrics = {"epoch": epoch}
        if train_loss is not None:
            metrics["train/loss"] = train_loss
        if val_loss is not None:
            metrics["val/loss"] = val_loss
        if train_acc is not None:
            metrics["train/accuracy"] = train_acc
        if val_acc is not None:
            metrics["val/accuracy"] = val_acc
        if lr is not None:
            metrics["lr"] = lr
        metrics.update(kwargs)
        self.log(metrics, step=epoch)

    def end_run(self, summary=None):
        """
        End the current run with optional summary metrics.

        Args:
            summary: dict of final metrics (e.g., {"test_accuracy": 0.95})
        """
        if not self.active:
            return
        if summary:
            for k, v in summary.items():
                self._run.summary[k] = v
        self._wandb.finish()
        self._run = None

    def log_run_result(self, name, config, accuracy, **extra):
        """
        Convenience: start run, log result, end run. For experiments
        that don't need per-epoch logging.

        Args:
            name: run name
            config: run config dict
            accuracy: final test accuracy
            **extra: additional summary metrics
        """
        self.start_run(
            name=name,
            tags=[t for t in [config.get("encoding", ""), config.get("model", "")] if t],
            config=config,
        )
        summary = {"test_accuracy": accuracy, **extra}
        self.end_run(summary)

    def finish(self):
        """Clean up. Call at the very end of the experiment."""
        if self._run is not None:
            self.end_run()


# ====================================================================
# Convenience: module-level singleton for simple usage
# ====================================================================

_global_logger = None


def init_logger(project="vtbench", experiment=None, config=None):
    """Initialize global logger singleton."""
    global _global_logger
    _global_logger = WandbLogger(project, experiment, config)
    return _global_logger


def get_logger():
    """Get global logger (returns a no-op logger if not initialized)."""
    global _global_logger
    if _global_logger is None:
        _global_logger = WandbLogger()
    return _global_logger
