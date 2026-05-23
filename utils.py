import inspect
import json
import logging
import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback
from pathlib import Path
from torchvision.transforms import v2 as transforms


class TransformAdapter:
    """Expose both callable and .transform APIs for transform compatibility."""

    def __init__(self, fn):
        self.fn = fn

    def transform(self, x, *args, **kwargs):
        return self.fn(x)

    def __call__(self, x):
        return self.fn(x)


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.WrapTorchTransform(
        TransformAdapter(transforms.Resize(img_size)),
        source=source,
        target=target,
    )
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)


def get_swm_cache_subdir(name: str) -> Path:
    """Return a stable_worldmodel cache subdirectory across minor API versions."""
    import stable_worldmodel as swm

    candidates = [
        getattr(getattr(swm, "data", None), "get_cache_dir", None),
        getattr(getattr(getattr(swm, "data", None), "utils", None), "get_cache_dir", None),
    ]
    fn = next((candidate for candidate in candidates if candidate is not None), None)
    if fn is None:
        raise AttributeError("Could not find stable_worldmodel get_cache_dir")

    try:
        sig = inspect.signature(fn)
        if "sub_folder" in sig.parameters:
            return Path(fn(sub_folder=name))
        if "subfolder" in sig.parameters:
            return Path(fn(subfolder=name))
    except Exception:
        pass

    return Path(fn()) / name


def save_pretrained_compatible(model, run_name, config, filename):
    """Save LeWM weights across stable_worldmodel checkpoint API versions."""
    save_pretrained = None
    try:
        from stable_worldmodel.wm.utils import save_pretrained
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("stable_worldmodel.wm"):
            raise
    if save_pretrained is not None:
        try:
            return save_pretrained(
                model,
                run_name=run_name,
                config=config,
                filename=filename,
            )
        except TypeError as exc:
            if "sub_folder" not in str(exc) and "subfolder" not in str(exc):
                raise

    from omegaconf import OmegaConf

    ckpt_dir = get_swm_cache_subdir("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = ckpt_dir / filename
    torch.save(model.state_dict(), checkpoint_path)

    if config is not None:
        config_path = ckpt_dir / "config.json"
        config_data = OmegaConf.to_container(config, resolve=True)
        with config_path.open("w") as f:
            json.dump(config_data, f, indent=2)

    logging.info("Model saved to %s", checkpoint_path)


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained_compatible(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )
