from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  # h5 读取插件
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from stable_pretraining import data as dt
from stable_pretraining.backbone.utils import vit_hf
from torch import nn


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


# ============================================================
# Motion 分支辅助：flow transform、flow stats、2 通道 ViT
# ============================================================


def get_flow_preprocessor(
    source: str = "pixels",
    target: str = "flow_pixels",
    img_size: int = 224,
    mean=None,
    std=None,
    motion_steps: int | None = None,
):
    """raw UV flow 的 preprocessor。

    输入 ``[T, H, W, 2]`` (NHWC, int16/float)，T = flow HDF5Dataset 的 num_steps。
    输出 ``[motion_steps, 2, img_size, img_size]`` float32，按 channel 归一化。
    """
    # mean/std 预计算成 [1, 2, 1, 1] 形状，方便在 NCHW 上 broadcast
    mean_t = torch.as_tensor(mean, dtype=torch.float32).view(1, 2, 1, 1) if mean is not None else None
    std_t = torch.as_tensor(std, dtype=torch.float32).view(1, 2, 1, 1) if std is not None else None

    def flow_fn(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), 0.0)
        x = x[:motion_steps]
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[-1] != img_size or x.shape[-2] != img_size:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        if mean_t is not None and std_t is not None:
            x = (x - mean_t.to(x.device)) / std_t.to(x.device)
        return x

    return dt.transforms.WrapTorchTransform(flow_fn, source=source, target=target)


def compute_flow_channel_stats(
    flow_h5_path: str | Path,
    flow_key: str = "pixels",
    max_samples: int = 8192,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机抽样估计 raw flow 的 channel-wise mean/std，避免全量扫描 2.3M 张 flow。"""
    rng = np.random.default_rng(seed)
    with h5py.File(flow_h5_path, "r") as f:
        ds = f[flow_key]
        n_take = min(max_samples, ds.shape[0])
        rows = np.sort(rng.choice(ds.shape[0], size=n_take, replace=False))
        data = np.asarray(ds[rows.tolist()], dtype=np.float32)  # (N, H, W, 2)
    data = np.nan_to_num(data, nan=0.0)
    # 在 N/H/W 上聚合，保留 channel
    mean = torch.from_numpy(data.mean(axis=(0, 1, 2)))
    std = torch.from_numpy(data.std(axis=(0, 1, 2))).clamp_min(1e-6)
    return mean, std


def build_vit_encoder(
    encoder_scale: str,
    patch_size: int,
    image_size: int,
    in_channels: int = 3,
):
    """统一构造 state encoder (3 通道) 与 motion encoder (2 通道) 的 ViT。

    in_channels != 3 时，直接替换 HuggingFace ViT 的 patch projection 为目标通道数；
    raw UV flow 因此原生作为 2 通道输入，不做 2->3 适配。
    """
    model = vit_hf(
        encoder_scale,
        patch_size=patch_size,
        image_size=image_size,
        pretrained=False,
        use_mask_token=False,
    )
    if in_channels != 3:
        pe = model.embeddings.patch_embeddings
        old = pe.projection
        pe.projection = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        pe.num_channels = in_channels
        model.config.num_channels = in_channels
    return model


class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")
