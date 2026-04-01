"""
Layer 1: Training Process Heartbeat
====================================
Each training script writes status every ~10-30 seconds to a JSON file.
If the machine hard-crashes, the last heartbeat shows exactly where it died.

Usage in experiment scripts:
    from vtbench.utils.heartbeat import Heartbeat

    hb = Heartbeat("6a")  # experiment name
    for run in runs:
        hb.pulse(dataset=ds, encoding=enc, model=model, run=f"{i}/{total}")
        train_and_evaluate(...)
        hb.pulse(dataset=ds, encoding=enc, model=model, run=f"{i}/{total}",
                 accuracy=acc, status="done")
    hb.close()
"""

import json
import os
import time
from datetime import datetime

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


HEARTBEAT_DIR = os.path.join("results", "heartbeats")


class Heartbeat:
    """Writes periodic heartbeat files for crash diagnosis."""

    def __init__(self, experiment_name, heartbeat_dir=None):
        self.experiment_name = experiment_name
        self.hb_dir = heartbeat_dir or HEARTBEAT_DIR
        os.makedirs(self.hb_dir, exist_ok=True)
        self.hb_file = os.path.join(self.hb_dir, f"{experiment_name}.json")
        self.start_time = time.time()
        self.pulse_count = 0

        # Write initial heartbeat
        self._write({
            "experiment": experiment_name,
            "status": "started",
            "start_time": self._ts(),
            "pid": os.getpid(),
        })

    def pulse(self, **kwargs):
        """Write a heartbeat with current GPU/training state."""
        self.pulse_count += 1
        data = {
            "experiment": self.experiment_name,
            "status": kwargs.get("status", "running"),
            "timestamp": self._ts(),
            "uptime_sec": round(time.time() - self.start_time, 1),
            "pulse_count": self.pulse_count,
            "pid": os.getpid(),
        }

        # GPU info (best-effort, never crash on this)
        if HAS_TORCH and torch.cuda.is_available():
            try:
                data["gpu_mem_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
                data["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
                data["gpu_mem_max_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            except Exception:
                data["gpu_mem_mb"] = -1

        # Training context from kwargs
        for key in ("dataset", "encoding", "model", "run", "epoch",
                    "loss", "accuracy", "batch", "seed", "method"):
            if key in kwargs:
                data[key] = kwargs[key]

        self._write(data)

    def close(self, final_status="completed"):
        """Write final heartbeat."""
        self._write({
            "experiment": self.experiment_name,
            "status": final_status,
            "timestamp": self._ts(),
            "total_time_sec": round(time.time() - self.start_time, 1),
            "total_pulses": self.pulse_count,
            "pid": os.getpid(),
        })

    def _write(self, data):
        """Atomic write to heartbeat file."""
        tmp = self.hb_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            # On Linux, os.replace() is atomic and overwrites existing files.
            # On Windows, os.replace() also works (Python 3.3+).
            os.replace(tmp, self.hb_file)
        except Exception:
            # Never crash the training because of heartbeat I/O
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    @staticmethod
    def _ts():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
