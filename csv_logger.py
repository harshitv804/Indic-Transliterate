"""CSV logging for the transliteration training loop."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path


def perplexity(loss: float) -> float:
    """exp(loss); guards the OverflowError math.exp raises above ~88."""
    return math.exp(loss) if loss < 88.0 else float("inf")


class CSVLogger:
    COLUMNS = [
        "run_id",
        "epoch",
        "step",
        "train_lm_loss",
        "train_ppl",
        "val_lm_loss",
        "val_ppl",
        "lr",
    ]

    def __init__(
        self,
        log_dir: str | Path = "logs",
        run_id: str | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id if run_id is not None else str(int(time.time()))
        self.path = self.log_dir / f"run_{self.run_id}.csv"
        exists = self.path.exists() and self.path.stat().st_size > 0

        self.file = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.COLUMNS)

        if not exists:
            self.writer.writeheader()
            self.file.flush()

        print(f"[csv] Logging to {self.path}")

    def log(
        self,
        epoch: int,
        step: int,
        logs: dict[str, float],
        val_lm_loss: float | None = None,
    ) -> None:
        row = {
            "run_id": self.run_id,
            "epoch": epoch,
            "step": step,
            "train_lm_loss": logs["lm_loss"],
            "train_ppl": perplexity(logs["lm_loss"]),
            "val_lm_loss": (val_lm_loss if val_lm_loss is not None else ""),
            "val_ppl": (perplexity(val_lm_loss) if val_lm_loss is not None else ""),
            "lr": logs["lr"],
        }
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()
