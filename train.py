import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import (
    ModelObjectCallBack,
    build_vit_encoder,
    compute_flow_channel_stats,
    get_column_normalizer,
    get_flow_preprocessor,
    get_img_preprocessor,
)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds

    state_weight = cfg.loss.state.weight
    motion_weight = cfg.loss.motion.weight
    state_sigreg_weight = cfg.loss.state_sigreg.weight
    motion_sigreg_weight = cfg.loss.motion_sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM state loss
    state_loss = (pred_emb - tgt_emb).pow(2).mean()
    state_sigreg_loss = self.sigreg(emb.transpose(0, 1))

    # motion 分支：仅训练期产生 motion_loss，rollout/eval 不会走这条路径
    if cfg.motion.enabled:
        output = self.model.encode_motion(output, source_key=cfg.motion.flow_output_key)
        tgt_vel_emb = output["vel_emb"]
        pred_vel_emb = self.model.predict_motion(ctx_emb, ctx_act)
        motion_loss = (pred_vel_emb - tgt_vel_emb).pow(2).mean()
        motion_sigreg_loss = self.sigreg(output["vel_emb"].transpose(0, 1))
    else:
        motion_loss = torch.zeros((), device=state_loss.device, dtype=state_loss.dtype)
        motion_sigreg_loss = torch.zeros((), device=state_loss.device, dtype=state_loss.dtype)

    output["state_loss"] = state_loss
    output["motion_loss"] = motion_loss
    output["state_sigreg_loss"] = state_sigreg_loss
    output["motion_sigreg_loss"] = motion_sigreg_loss
    output["loss"] = (
        state_weight * state_loss
        + motion_weight * motion_loss
        + state_sigreg_weight * state_sigreg_loss
        + motion_sigreg_weight * motion_sigreg_loss
    )

    log_stage = {"fit": "train", "validate": "val"}.get(stage, stage)
    loss_keys = [
        "loss",
        "state_loss",
        "motion_loss",
        "state_sigreg_loss",
        "motion_sigreg_loss",
    ]
    metrics_dict = {f"{log_stage}/{k}": output[k].detach() for k in loss_keys}
    self.log_dict(metrics_dict, on_step=True, on_epoch=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    # motion 分支：flow h5 每条 record 是 1 帧位移 (i -> i+1)。
    # 取每个 history 步末端的 1 帧 flow 作为 v_i 监督：z_i = pixels[s + i*frameskip]
    # 对应 v_i = flow[s + (i+1)*frameskip - 1]，即末尾帧到其下一帧的位移。
    if cfg.motion.enabled:
        # 若 config 未提供 flow stats，则随机抽样估计 channel-wise mean/std
        if cfg.motion.stats.mean is None or cfg.motion.stats.std is None:
            mean_t, std_t = compute_flow_channel_stats(
                cfg.motion.flow_path,
                flow_key=cfg.motion.flow_key,
                max_samples=cfg.motion.stats.max_samples,
                seed=cfg.seed,
            )
            with open_dict(cfg):
                cfg.motion.stats.mean = mean_t.tolist()
                cfg.motion.stats.std = std_t.tolist()

        flow_path = Path(cfg.motion.flow_path)
        flow_dataset = swm.data.HDF5Dataset(
            name=flow_path.stem,
            cache_dir=str(flow_path.parent),
            num_steps=cfg.data.dataset.num_steps,
            frameskip=cfg.data.dataset.frameskip,
            keys_to_load=[cfg.motion.flow_key],
            transform=get_flow_preprocessor(
                source=cfg.motion.flow_key,
                target=cfg.motion.flow_output_key,
                img_size=cfg.img_size,
                mean=cfg.motion.stats.mean,
                std=cfg.motion.stats.std,
                motion_steps=cfg.wm.history_size,
            ),
        )
        # flow start = rgb start + (frameskip - 1)，每条 v_i 落在对应 history 步末端；
        # shift 后右端比 rgb 多 (frameskip-1) 帧，再过滤一次保证 flow 访问合法
        offset = cfg.data.dataset.frameskip - 1
        valid = [
            (ep, s) for ep, s in flow_dataset.clip_indices
            if s + offset + flow_dataset.span <= flow_dataset.lengths[ep]
        ]
        dataset.clip_indices = valid
        flow_dataset.clip_indices = [(ep, s + offset) for ep, s in valid]

        # small-data subsample: deterministically cap clip_indices in lockstep
        max_clips = cfg.get("max_clips")
        if max_clips is not None and 0 < max_clips < len(dataset.clip_indices):
            sub_gen = torch.Generator().manual_seed(cfg.seed)
            perm = torch.randperm(len(dataset.clip_indices), generator=sub_gen)[:max_clips].tolist()
            dataset.clip_indices = [dataset.clip_indices[i] for i in perm]
            flow_dataset.clip_indices = [flow_dataset.clip_indices[i] for i in perm]

        dataset = swm.data.MergeDataset(
            [dataset, flow_dataset],
            keys_from_dataset=[dataset.column_names, [cfg.motion.flow_output_key]],
        )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = build_vit_encoder(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        in_channels=3,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    # motion 分支模块：与 state 分支不共享权重，仅在 cfg.motion.enabled 时构造
    motion_modules = {}
    if cfg.motion.enabled:
        motion_modules = dict(
            encoder_vel=build_vit_encoder(
                cfg.encoder_scale,
                patch_size=cfg.patch_size,
                image_size=cfg.img_size,
                in_channels=cfg.motion.input_channels,
            ),
            predictor_vel=ARPredictor(
                num_frames=cfg.wm.history_size,
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                **cfg.predictor,
            ),
            projector_vel=MLP(
                input_dim=hidden_dim,
                output_dim=embed_dim,
                hidden_dim=2048,
                norm_fn=torch.nn.BatchNorm1d,
            ),
            pred_vel_proj=MLP(
                input_dim=hidden_dim,
                output_dim=embed_dim,
                hidden_dim=2048,
                norm_fn=torch.nn.BatchNorm1d,
            ),
        )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        **motion_modules, #自动解包 motion_modules 字典到 JEPA 构造函数参数
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or "" 
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    ) #epoch_interval=1 表示每个 epoch 保存一次模型对象

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
