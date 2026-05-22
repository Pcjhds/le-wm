"""Build-time compatibility check for the NERC OpenShift training image."""

from pathlib import Path
import os
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import stable_pretraining as spt
import torch

from train import get_swm_cache_dir, load_swm_dataset
from utils import get_column_normalizer, get_img_preprocessor


def create_tiny_hdf5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_steps = 32
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=np.array([num_steps], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))
        handle.create_dataset(
            "pixels",
            data=np.zeros((num_steps, 8, 8, 3), dtype=np.uint8),
        )
        handle.create_dataset(
            "action",
            data=np.linspace(0, 1, num_steps * 2, dtype=np.float32).reshape(
                num_steps, 2
            ),
        )
        handle.create_dataset(
            "proprio",
            data=np.linspace(1, 2, num_steps * 3, dtype=np.float32).reshape(
                num_steps, 3
            ),
        )
        handle.create_dataset(
            "state",
            data=np.linspace(2, 3, num_steps * 4, dtype=np.float32).reshape(
                num_steps, 4
            ),
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir)
        dataset_path = cache_dir / "datasets" / "pusht_expert_train.h5"
        create_tiny_hdf5(dataset_path)

        os.environ["STABLEWM_HOME"] = str(cache_dir)
        os.environ["LOCAL_DATASET_DIR"] = str(cache_dir)
        checkpoints_dir = get_swm_cache_dir("checkpoints")
        assert checkpoints_dir.name == "checkpoints"

        dataset_cfg = {
            "num_steps": 4,
            "frameskip": 5,
            "keys_to_load": ["pixels", "action", "proprio", "state"],
            "keys_to_cache": ["action", "proprio", "state"],
        }
        dataset = load_swm_dataset("pusht_expert_train", str(cache_dir), dataset_cfg)

        assert len(dataset) > 0
        assert dataset.get_dim("action") == 2
        transforms = [
            get_img_preprocessor(source="pixels", target="pixels", img_size=16)
        ]
        for key in ["action", "proprio", "state"]:
            transforms.append(get_column_normalizer(dataset, key, key))

        dataset.transform = spt.data.transforms.Compose(*transforms)
        sample = dataset[0]
        for key in dataset_cfg["keys_to_load"]:
            assert key in sample, f"missing key in sample: {key}"
        assert torch.is_floating_point(sample["pixels"])
        assert sample["pixels"].shape[-3:] == (3, 16, 16)

    print("openshift training smoke check ok")


if __name__ == "__main__":
    main()
