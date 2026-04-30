from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from .diagnostics import DiagnosticsHistory
from .optimizer_utils import ImprovementTracker, safe_float


class PhysicalOptimizerBase:
    def _initialize_physical_optimizer(self, optimizer_name: str) -> None:
        self.optimizer_name = optimizer_name
        self._history = DiagnosticsHistory(optimizer_name=optimizer_name)
        self._tracker = ImprovementTracker()
        self._global_step = 0
        self._external_metrics: dict[str, Any] = {}

    def set_current_loss(self, loss_value: float | int | torch.Tensor | None) -> float | None:
        loss_float = None if loss_value is None else safe_float(loss_value, default=float("nan"))
        if loss_float is not None and loss_float == loss_float:
            self._tracker.update(loss_float)
            return loss_float
        self._tracker.update(None)
        return None

    def _prepare_closure(self, closure) -> tuple[Any, float | None]:
        loss_tensor = None
        current_loss = self._tracker.last_loss
        if closure is not None:
            with torch.enable_grad():
                loss_tensor = closure()
            current_loss = self.set_current_loss(loss_tensor)
        return loss_tensor, current_loss

    def _record_step(self, metrics: dict[str, Any]) -> dict[str, Any]:
        self._global_step += 1
        metrics = dict(metrics)
        metrics.setdefault("step", self._global_step)
        metrics.setdefault("loss_improvement", self._tracker.loss_improvement)
        metrics.setdefault("stagnation_counter", self._tracker.stagnation_counter)
        return self._history.append(metrics)

    def set_external_metrics(self, **metrics: Any) -> None:
        self._external_metrics.update(metrics)

    def clear_external_metrics(self) -> None:
        self._external_metrics.clear()

    @property
    def external_metrics(self) -> dict[str, Any]:
        return dict(self._external_metrics)

    def latest_diagnostics(self) -> dict[str, Any]:
        return self._history.latest()

    def diagnostics_dataframe(self) -> pd.DataFrame:
        return self._history.to_frame()

    def reset_diagnostics(self) -> None:
        self._history.clear()
        self._global_step = 0
        self.clear_external_metrics()

    @property
    def stagnation_counter(self) -> int:
        return self._tracker.stagnation_counter

    @property
    def current_loss(self) -> float | None:
        return self._tracker.last_loss

    @property
    def best_loss(self) -> float | None:
        return self._tracker.best_loss
