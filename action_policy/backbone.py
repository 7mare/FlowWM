"""Backbone 接口与 JEPA 实现。

policy 只依赖 ``Backbone`` 抽象（``encode_obs`` / ``encode_goal``），
backbone 内部的具体网络（JEPA / DINO / VLM）对 policy 不可见。
"""

import torch
from torch import nn


class Backbone(nn.Module):
    """policy 看到的最小 backbone 接口。

    子类需要：
      - 设置 ``self.embed_dim``
      - 实现 ``encode_obs(pixels) -> [B, H, D]``
      - 实现 ``encode_goal(pixels) -> [B, 1, D]``
    """

    embed_dim: int

    def encode_obs(self, pixels):
        raise NotImplementedError

    def encode_goal(self, pixels):
        raise NotImplementedError


class JEPABackbone(Backbone):
    """包装一个冻结的 LeWM ``JEPA`` 实例。

    构造时把整个 JEPA（含 encoder/projector/action_encoder/predictor/pred_proj
    /encoder_vel/projector_vel/predictor_vel/pred_vel_proj）一次性
    ``requires_grad_(False) + .eval()``，即使当前 path 不调用的子模块
    也保持冻结，避免 optimizer 误吞或 dropout/BN 行为漂移。
    """

    def __init__(self, jepa: nn.Module):
        super().__init__()
        self.jepa = jepa
        self.jepa.eval()
        self.jepa.requires_grad_(False)
        self.embed_dim = jepa.projector.net[-1].out_features

    def train(self, mode: bool = True):
        super().train(mode)
        self.jepa.eval()
        return self

    @torch.no_grad()
    def encode_obs(self, pixels):
        return self.jepa.encode({"pixels": pixels})["emb"]

    @torch.no_grad()
    def encode_goal(self, pixels):
        return self.jepa.encode({"pixels": pixels})["emb"]
