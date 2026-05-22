import numpy as np
import torch
import torch.nn.functional as F
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    return ImagePreprocessor(source=source, target=target, img_size=img_size)


class ImagePreprocessor(torch.nn.Module):
    """Picklable image preprocessor for HDF5 sequence samples."""

    def __init__(self, source: str, target: str, img_size: int = 224):
        super().__init__()
        self.source = source
        self.target = target
        self.img_size = img_size
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        )

    def forward(self, sample):
        x = torch.as_tensor(sample[self.source])
        x = x.float() if x.is_floating_point() else x.float().div(255.0)
        x = self._to_channel_first(x)

        leading_shape = x.shape[:-3]
        channels, height, width = x.shape[-3:]
        if channels == 1:
            x = x.expand(*leading_shape, 3, height, width)
        elif channels > 3:
            x = x[..., :3, :, :]

        if x.shape[-2:] != (self.img_size, self.img_size):
            flat = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])
            flat = F.interpolate(
                flat,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            x = flat.reshape(*leading_shape, x.shape[-3], self.img_size, self.img_size)

        view_shape = [1] * x.ndim
        view_shape[-3] = 3
        mean = self.mean.to(device=x.device, dtype=x.dtype).view(*view_shape)
        std = self.std.to(device=x.device, dtype=x.dtype).view(*view_shape)

        out = dict(sample)
        out[self.target] = (x - mean) / std
        return out

    @staticmethod
    def _to_channel_first(x):
        if x.ndim < 3:
            raise ValueError(f"Expected image tensor with at least 3 dims, got {tuple(x.shape)}")

        if x.shape[-1] in (1, 3, 4):
            return x.movedim(-1, -3)

        if x.shape[-3] in (1, 3, 4):
            return x

        raise ValueError(
            "Could not infer image channel dimension from shape "
            f"{tuple(x.shape)}; expected channel count 1, 3, or 4."
        )


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
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )
