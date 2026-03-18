from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
from enum import Enum
import json

from hdt.parameters import Components, Models, Horizon, Segmentation
import numpy as np



@dataclass(frozen=True, slots=True)
class ForecastConfig:
    exp: str
    model_type: Any
    st_lt: Any
    sensor_prefix: str
    mode_prefix: str
    diff_prefix: str
    mode_component: str

    def to_dict(self):
        def _serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        return {k: _serialize(v) for k, v in asdict(self).items()}

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @staticmethod
    def from_json(json_str: str) -> ForecastConfig:
        data = json.loads(json_str)

        if "model_type" in data and isinstance(data["model_type"], str):
            try:
                data["model_type"] = Models(data["model_type"])
            except ValueError:
                pass
        elif "st_lt" in data and isinstance(data["st_lt"], str):
            try:
                data["st_lt"] = Horizon(data["st_lt"])
            except ValueError:
                pass

        return ForecastConfig(**data)


@dataclass(frozen=True, slots=True)
class TrainPredictConfig:
    component: Any
    fold: int
    bundle_file: str

    prediction_horizon: int
    prediction_stride: int
    input_size: int


    gs: bool = False
    sim: bool = False
    train_boundaries: bool = False  # If False, the true control modes with true boundaries are used
    lower_vals: list[float] = field(default_factory=list)
    upper_vals: list[float] = field(default_factory=list)
    lower: float = None
    upper: float = None
    actual_boundaries: Any = None
    segmentation: Any = Segmentation.OFF
    multi: bool = False
    # For the CPEE

    def to_dict(self):
        def _serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        return {k: _serialize(v) for k, v in asdict(self).items()}

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @staticmethod
    def from_json(json_str: str) -> TrainPredictConfig:
        data = json.loads(json_str)

        if "component" in data and isinstance(data["component"], str):
            try:
                data["component"] = Components(data["component"])
            except ValueError:
                pass

        return TrainPredictConfig(**data)


@dataclass(frozen=True, slots=True)
class SeriesBundle:
    # All with modes
    train_scaled: Any
    val_scaled: Any
    test_scaled: Any


@dataclass(frozen=True, slots=True)
class WorkerBundle:
    chunks: Any

    past_covariates: Any = None
    val_past_covariates: Any = None
    all_past_covariates: Any = None


@dataclass(frozen=True, slots=True)
class EvalConfig:
    scale_min: list[float] | None = None
    scale_max: list[float] | None = None

    def to_dict(self):
        def _serialize(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, list):
                return [_serialize(i) for i in obj]
            return obj

        return {k: _serialize(v) for k, v in asdict(self).items()}

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @staticmethod
    def from_json(json_str: str) -> EvalConfig:
        data = json.loads(json_str)
        return EvalConfig(**data)

