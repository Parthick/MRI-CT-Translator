from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    model_path: Path
    dataset: Path
    img_shape: tuple
    epochs: int
    lr: float
    l1: str
    l2: str
