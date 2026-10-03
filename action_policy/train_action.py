"""Action policy 独立训练入口。

假设 LeWM 已经训好并 dump 为 ``torch.save(model_object)`` 的整 nn.Module
（来自 ``utils.ModelObjectCallBack``）。本脚本：

1. 从 ``cfg.backbone.ckpt_path`` 加载 JEPA；
2. 包装成冻结的 ``JEPABackbone``；
3. 构造 ``FlowPolicy`` 并训练。

只优化 policy 自身的参数，backbone 不更新。
"""

import sys
from pathlib import Path

import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from action_policy.backbone import JEPABackbone
from action_policy.policy import FlowPolicy
from utils import get_column_normalizer, get_img_preprocessor

try:
    import wandb
except ImportError:
    wandb = None


def prepare_batch(batch, history_size, chunk_size):
    """从原始 batch 切出 (pixels, goal, action_chunk)。"""
    pixels = batch["pixels"]
    actions = batch["action"]
    action_start = history_size - 1
    return {
        "pixels": pixels[:, :history_size],
        "goal": pixels[:, -1:],
        "action_chunk": actions[:, action_start : action_start + chunk_size], 
    }


def build_dataset(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


def run_epoch(policy, loader, optim, device, history_size, chunk_size, train: bool,
              use_wandb: bool = False, log_every: int = 0, global_step: int = 0):
    """跑一个 epoch；train 时按 log_every 间隔写 step-level wandb 日志。"""
    policy.train(train)
    total, n = 0.0, 0
    step = global_step
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        inputs = prepare_batch(batch, history_size, chunk_size)
        if train:
            loss, logs = policy.training_step(inputs)
            optim.zero_grad()
            loss.backward()
            optim.step()
        else:
            with torch.no_grad():
                loss, logs = policy.training_step(inputs)
        total += loss.item() * inputs["pixels"].size(0)
        n += inputs["pixels"].size(0)
        # step-level wandb：只在 train 时记，val 用 epoch 平均
        if train and use_wandb and log_every and step % log_every == 0:
            wandb.log({
                "train/step_loss": loss.item(),
                "train/sigma_mean": logs["sigma_mean"].item(),
                "step": step,
            })
        step += 1
    return total / max(n, 1), step


@hydra.main(version_base=None, config_path="../config/train", config_name="action_flow")
def run(cfg):
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    dataset = build_dataset(cfg)
    raw_action_dim = dataset.get_dim("action")
    effective_action_dim = cfg.data.dataset.frameskip * raw_action_dim
    with open_dict(cfg):
        cfg.policy.action_dim = effective_action_dim

    rnd = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd
    )
    train_loader = DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd
    )
    val_loader = DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    jepa = torch.load(cfg.backbone.ckpt_path, map_location=device, weights_only=False)
    backbone = JEPABackbone(jepa).to(device)

    policy = FlowPolicy(
        backbone,
        history_size=cfg.history_size,
        chunk_size=cfg.chunk_size,
        action_dim=effective_action_dim,
        d_act=cfg.policy.d_act,
        num_layers=cfg.policy.num_layers,
        num_heads=cfg.policy.num_heads,
        dim_head=cfg.policy.dim_head,
        mlp_dim=cfg.policy.mlp_dim,
        dropout=cfg.policy.dropout,
        freq_dim=cfg.policy.freq_dim,
        num_train_timesteps=cfg.flow_match.num_train_timesteps,
        shift=cfg.flow_match.shift,
    ).to(device)

    optim = torch.optim.AdamW(
        policy.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    use_wandb = cfg.wandb.enabled and wandb is not None
    if use_wandb:
        wandb.init(**cfg.wandb.config, config=OmegaConf.to_container(cfg))

    log_every = cfg.get("log_every", 50)

    start_epoch = 0
    global_step = 0
    resume_from = cfg.get("resume_from", None)
    if resume_from:
        state = torch.load(resume_from, map_location=device, weights_only=False)
        policy.load_state_dict(state["policy"], strict=False)
        optim.load_state_dict(state["optim"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    for epoch in range(start_epoch, cfg.max_epochs):
        train_loss, global_step = run_epoch(
            policy, train_loader, optim, device, cfg.history_size, cfg.chunk_size, train=True,
            use_wandb=use_wandb, log_every=log_every, global_step=global_step,
        )
        val_loss, _ = run_epoch(
            policy, val_loader, optim, device, cfg.history_size, cfg.chunk_size, train=False,
        )
        print(f"epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")
        if use_wandb:
            wandb.log({
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss,
                "epoch": epoch,
                "step": global_step,
            })

        trainable = {
            k: v for k, v in policy.state_dict().items() if not k.startswith("backbone.")
        }
        torch.save(trainable, out_dir / f"policy_epoch{epoch}.ckpt")
        torch.save(
            {"policy": trainable, "optim": optim.state_dict(),
             "epoch": epoch, "global_step": global_step},
            out_dir / "training_state.ckpt",
        )


if __name__ == "__main__":
    run()
