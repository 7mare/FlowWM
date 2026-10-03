"""LeWM action policy：把 scheduler / tokens / expert / policy 全部装在这一个文件里。

模块布局（按依赖顺序）：

1. ``FlowMatchScheduler``          —— flow-matching 噪声调度
2. ``sinusoidal_embedding_1d``     —— TimeEmbedder 用的频率编码
3. ``build_sincos_1d``             —— 固定 sin-cos 1D 位置编码
4. ``TimeEmbedder``                —— t → (time_emb, adaln_params)
5. ``ContextTokenBuilder``         —— z_ctx + z_goal → cross-attn 的 K/V
6. ``ActionTokenEncoder``          —— noisy action chunk → action expert 的 query
7. ``CrossAttention``              —— Q 来自 action 序列，K/V 来自 context
8. ``ActionExpertBlock``           —— AdaLN + cross-attn + self-attn + FFN
9. ``ActionExpert``                —— N 层 block + 末端 LayerNorm
10. ``ActionDecoder``              —— AdaLN(time_emb) + LayerNorm + MLP head
11. ``FlowPolicy``                 —— 顶层组装；含 training_step / generate / get_action
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from action_policy.backbone import Backbone
from module import MLP, Attention, Embedder, FeedForward, modulate


# ============================================================
# 1. Flow-matching scheduler
# ============================================================

class FlowMatchScheduler:
    """Flow-matching σ-shift 噪声调度（不含可训练参数）。

    训练：``a_noisy = (1 - σ) * a_clean + σ * ε``，``target = ε - a_clean``；
    推理：Euler 积分 ``t : 1 → 0``。``shift=1`` 等价 vanilla flow matching。
    """

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0):
        # 预计算训练用 σ 网格：σ_i = shift·t_i / (1 + (shift-1)·t_i)，t_i ∈ (0, 1]
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        ts = torch.linspace(1.0 / num_train_timesteps, 1.0, num_train_timesteps)
        self.sigmas = shift * ts / (1.0 + (shift - 1.0) * ts)
        # 把 σ 放大到 [0, num_train_timesteps] 作为 TimeEmbedder 输入的 scalar t
        self.timesteps = self.sigmas * num_train_timesteps

        self.inference_sigmas = None
        self.inference_timesteps = None
        self.dt = None

    def sample_train(self, batch_size: int, device):
        """训练时给每条样本随机采一个 timestep；返回 broadcast-friendly σ 与 TimeEmbedder 用的 t_scalar。"""
        t_id = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
        # σ shape [B,1,1] 用于和 action_chunk [B,K,A] 广播
        sigma = self.sigmas.to(device)[t_id].view(-1, 1, 1)
        t_scalar = self.timesteps.to(device)[t_id]
        return sigma, t_scalar, t_id

    def add_noise(self, a_clean, noise, sigma):
        """Flow-matching 混合：a_noisy = (1-σ)·a_clean + σ·ε。"""
        return (1.0 - sigma) * a_clean + sigma * noise

    def velocity_target(self, a_clean, noise):
        """Flow-matching velocity 目标：v* = ε - a_clean。"""
        return noise - a_clean

    def set_inference_steps(self, num_inference_steps: int):
        """推理前调用一次，预计算 Euler 网格 + 每步 dt（负值，σ 从 1 走到 0）。"""
        ts = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        sigmas = self.shift * ts / (1.0 + (self.shift - 1.0) * ts)
        self.inference_sigmas = sigmas
        self.inference_timesteps = sigmas * self.num_train_timesteps
        # dt[i] = σ_{i+1} - σ_i，最后一步落到 0
        next_sigmas = torch.cat([sigmas[1:], torch.zeros(1)])
        self.dt = next_sigmas - sigmas

    def scale_for_model_input(self, t_idx: int, device):
        """推理第 t_idx 步喂给 TimeEmbedder 的 scalar t，shape=[1]。"""
        return self.inference_timesteps.to(device)[t_idx : t_idx + 1]


# ============================================================
# 2-3. sin-cos 工具
# ============================================================

def sinusoidal_embedding_1d(freq_dim: int, t: torch.Tensor) -> torch.Tensor:
    """把标量 t [B] 编成 sin/cos 频率特征，给 TimeEmbedder 用。"""
    half = freq_dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.cos(), args.sin()], dim=-1)


def build_sincos_1d(length: int, dim: int) -> torch.Tensor:
    """生成不可训练的 1D sin-cos 位置编码 buffer。"""
    pos = torch.arange(length, dtype=torch.float32)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
    )
    args = pos.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ============================================================
# 4-6. 输入侧 token 模块
# ============================================================

class TimeEmbedder(nn.Module):
    """t -> (time_emb [B, D_act], adaln_params [B, 6, D_act])。

    6 路 AdaLN 参数：3 路给 cross-attn，3 路给 FFN（与 ConditionalBlock 同款分配）。
    """

    def __init__(self, d_act: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        # base time embedding，给 ActionDecoder head 用
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_act),
            nn.SiLU(),
            nn.Linear(d_act, d_act),
        )
        # 再投到 6×d_act，每个 ActionExpertBlock 把它 chunk(6) 拆成 6 路调制参数
        self.adaln_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_act, 6 * d_act),
        )

    def forward(self, t: torch.Tensor):
        emb = sinusoidal_embedding_1d(self.freq_dim, t)
        time_emb = self.mlp(emb)
        adaln = self.adaln_head(time_emb).reshape(t.size(0), 6, -1)
        return time_emb, adaln


class ContextTokenBuilder(nn.Module):
    """cat([z_ctx, z_goal]) -> Linear -> +type -> +pos -> C ∈ [B, H+1, D_act]。

    输出是 cross-attention 的 K/V；type embedding 让模型分辨 obs 与 goal。
    """

    def __init__(self, d_backbone: int, d_act: int, history_size: int):
        super().__init__()
        self.history_size = history_size
        # 从 backbone embed_dim 投到 ActionExpert 工作维度，做容量解耦
        self.context_proj = nn.Linear(d_backbone, d_act)
        self.type_emb = nn.Embedding(2, d_act)  # 0 = obs, 1 = goal
        self.register_buffer("pos_embed", build_sincos_1d(history_size + 1, d_act))

    def forward(self, z_ctx: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_ctx, z_goal], dim=1)
        x = self.context_proj(x)
        # 前 H 个 token 加 obs type，最后 1 个加 goal type
        type_ids = torch.cat(
            [
                torch.zeros(self.history_size, dtype=torch.long, device=x.device),
                torch.ones(1, dtype=torch.long, device=x.device),
            ]
        )
        x = x + self.type_emb(type_ids).unsqueeze(0)
        x = x + self.pos_embed.unsqueeze(0)
        return x


class ActionTokenEncoder(nn.Module):
    """noisy_actions [B, K, A] -> [B, K, D_act]。query 序列就是这 K 个 token，没有 state/register。"""

    def __init__(self, d_act: int, action_dim: int, chunk_size: int):
        super().__init__()
        # 复用 module.Embedder（Conv1d(k=1)+Linear+SiLU+Linear）
        self.action_mlp = Embedder(
            input_dim=action_dim,
            smoothed_dim=action_dim,
            emb_dim=d_act,
            mlp_scale=4,
        )
        self.register_buffer("pos_embed", build_sincos_1d(chunk_size, d_act))

    def forward(self, noisy_actions: torch.Tensor) -> torch.Tensor:
        x = self.action_mlp(noisy_actions)
        x = x + self.pos_embed.unsqueeze(0)
        return x


# ============================================================
# 7-10. 处理侧 expert 模块
# ============================================================

class CrossAttention(nn.Module):
    """Q 来自 action 序列，K/V 来自 backbone context tokens（与 module.Attention 的差别仅在 to_q / to_kv 分离）。"""

    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.0):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.dropout = dropout

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.to_q(x_q), "b n (h d) -> b h n d", h=self.heads)
        k, v = self.to_kv(x_kv).chunk(2, dim=-1)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=False)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ActionExpertBlock(nn.Module):
    """单层 expert block：AdaLN -> cross-attn -> self-attn -> AdaLN -> FFN。

    AdaLN 6 参与 ConditionalBlock 同款（3 给 cross-attn，3 给 FFN）；self-attn 不带 AdaLN 是本设计的选择。
    """

    def __init__(
        self,
        d_act: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # elementwise_affine=False：让 AdaLN 的 shift/scale 独占 norm 的 γ/β 角色
        self.norm1 = nn.LayerNorm(d_act, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_act, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(d_act, heads, dim_head, dropout)
        # 复用 module.Attention，forward 时 causal=False 关掉因果掩码
        self.self_attn = Attention(d_act, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ffn = FeedForward(d_act, mlp_dim, dropout=dropout)
        # block-owned AdaLN bias：与外部 adaln_params 相加，给本层一个可学习偏置
        self.modulation = nn.Parameter(torch.zeros(6, d_act))

    def forward(
        self, a: torch.Tensor, c: torch.Tensor, adaln_params: torch.Tensor
    ) -> torch.Tensor:
        # 把 [B,6,D] 拆成 6 个 [B,1,D]，可与 action seq [B,K,D] broadcast
        params = adaln_params + self.modulation.unsqueeze(0)
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            p.unsqueeze(1) for p in params.unbind(dim=1)
        )
        # 1) cross-attn：吸收 backbone context（主要信息通路）
        a = a + gate1 * self.cross_attn(modulate(self.norm1(a), shift1, scale1), c)
        # 2) self-attn：建模 chunk 内时序关系
        a = a + self.self_attn(a, causal=False)
        # 3) FFN
        a = a + gate2 * self.ffn(modulate(self.norm2(a), shift2, scale2))
        return a


class ActionExpert(nn.Module):
    """堆 N 层 ActionExpertBlock + 末端 LayerNorm（沿用 module.Transformer 末端 norm 风格）。"""

    def __init__(
        self,
        num_layers: int,
        d_act: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(d_act, heads, dim_head, mlp_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_act)

    def forward(
        self, a: torch.Tensor, c: torch.Tensor, adaln_params: torch.Tensor
    ) -> torch.Tensor:
        for blk in self.blocks:
            a = blk(a, c, adaln_params)
        return self.norm(a)


class ActionDecoder(nn.Module):
    """AdaLN(time_emb) -> LayerNorm -> MLP head -> velocity field。head 最后一层零初始化，保证训练初值稳定。"""

    def __init__(self, d_act: int, action_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_act, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Parameter(torch.zeros(2, d_act))
        # 复用 module.MLP（norm_fn=None, act_fn=SiLU）：Linear→Identity→SiLU→Linear
        self.head = MLP(d_act, d_act, action_dim, norm_fn=None, act_fn=nn.SiLU)
        # zero-init 输出层：初始 velocity = 0，flow matching 起步稳定
        nn.init.zeros_(self.head.net[-1].weight)
        nn.init.zeros_(self.head.net[-1].bias)

    def forward(self, a_final: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        params = self.modulation.unsqueeze(0) + time_emb.unsqueeze(1).expand(-1, 2, -1)
        shift, scale = (p.unsqueeze(1) for p in params.unbind(dim=1))
        z = modulate(self.norm(a_final), shift, scale)
        return self.head(z)


# ============================================================
# 11. 顶层 policy
# ============================================================

class FlowPolicy(nn.Module):
    """顶层组装：backbone（冻结）+ ContextTokens + ActionExpert + Decoder + FlowMatchScheduler。

    对外暴露三个接口：
      - ``training_step``：训练时算 flow-matching MSE 损失；
      - ``generate``：推理时 Euler 去噪出 action chunk；
      - ``get_action``：eval 时被 ``swm.policy.FeedForwardPolicy`` 调用的 Actionable 接口。
    """

    def __init__(
        self,
        backbone: Backbone,
        *,
        history_size: int,
        chunk_size: int,
        action_dim: int,
        d_act: int = 384,
        num_layers: int = 8,
        num_heads: int = 8,
        dim_head: int = 48,
        mlp_dim: int = 1536,
        dropout: float = 0.0,
        freq_dim: int = 256,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        action_block: int = 1,
        num_inference_steps: int = 20,
    ):
        super().__init__()
        self.backbone = backbone
        # 保险冻结：JEPABackbone 构造时已冻结，这里再确认一次防止外部漏冻
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        self.history_size = history_size
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        # 训练时一个 token 对应 action_block 个 raw 动作；eval 时需要按 action_block 拆回 raw
        self.action_block = action_block
        # eval 默认 Euler 步数，可通过 generate(num_inference_steps=...) override
        self.num_inference_steps = num_inference_steps
        self._action_plan = None
        self._action_step = 0

        self.context_builder = ContextTokenBuilder(backbone.embed_dim, d_act, history_size)
        self.action_tokens = ActionTokenEncoder(d_act, action_dim, chunk_size)
        self.time_embedder = TimeEmbedder(d_act, freq_dim=freq_dim)
        self.expert = ActionExpert(num_layers, d_act, num_heads, dim_head, mlp_dim, dropout)
        self.decoder = ActionDecoder(d_act, action_dim)
        self.scheduler = FlowMatchScheduler(num_train_timesteps, shift)

    def train(self, mode: bool = True):
        """覆写：policy.train() 时不解冻 backbone，保持 eval 模式避免 dropout/BN 漂移。"""
        super().train(mode)
        self.backbone.eval()
        return self

    def trainable_parameters(self):
        """只返回 requires_grad=True 的参数（即 backbone 之外的全部模块），喂给 optimizer。"""
        return [p for p in self.parameters() if p.requires_grad]

    def _expert_forward(self, a_noisy, c, t_scalar):
        """ActionExpert 一次完整前向：训练与推理共用，保证两条路径行为一致。"""
        time_emb, adaln_params = self.time_embedder(t_scalar)
        a = self.action_tokens(a_noisy)
        a = self.expert(a, c, adaln_params)
        return self.decoder(a, time_emb)

    def training_step(self, batch):
        """flow-matching 训练一步：返回 (loss, logs)。"""
        pixels = batch["pixels"]
        goal = batch["goal"]
        action_chunk = batch["action_chunk"]
        bsz = pixels.size(0)
        device = pixels.device

        # backbone 冻结 + no_grad，仅作为只读特征提取
        z_ctx = self.backbone.encode_obs(pixels)
        z_goal = self.backbone.encode_goal(goal)
        c = self.context_builder(z_ctx, z_goal)

        # flow-matching 噪声混合 + velocity target
        sigma, t_scalar, _ = self.scheduler.sample_train(bsz, device)
        noise = torch.randn_like(action_chunk)
        a_noisy = self.scheduler.add_noise(action_chunk, noise, sigma)
        v_target = self.scheduler.velocity_target(action_chunk, noise)

        v_hat = self._expert_forward(a_noisy, c, t_scalar)
        loss = F.mse_loss(v_hat, v_target)
        logs = {"action_loss": loss.detach(), "sigma_mean": sigma.mean().detach()}
        return loss, logs

    @torch.no_grad()
    def generate(self, obs_pixels, goal_pixels, num_inference_steps: int = 20):
        """推理：从纯噪声 a_T 开始 Euler 积分 t:1→0，输出 action chunk [B, K, A]。"""
        bsz = obs_pixels.size(0)
        device = obs_pixels.device

        # context 只跟 obs/goal 有关，循环外算一次复用
        z_ctx = self.backbone.encode_obs(obs_pixels)
        z_goal = self.backbone.encode_goal(goal_pixels)
        c = self.context_builder(z_ctx, z_goal)

        self.scheduler.set_inference_steps(num_inference_steps)
        a_t = torch.randn(bsz, self.chunk_size, self.action_dim, device=device)
        dt = self.scheduler.dt.to(device)

        for i in range(num_inference_steps):
            t_scalar = self.scheduler.scale_for_model_input(i, device).expand(bsz)
            v_hat = self._expert_forward(a_t, c, t_scalar)
            # Euler 一步：a_t ← a_t + v̂·dt（dt 为负，σ 从 1 减到 0）
            a_t = a_t + v_hat * dt[i]

        return a_t

    @torch.no_grad()
    def get_action(self, info_dict):
        """eval 时被 ``swm.policy.FeedForwardPolicy`` 调用，返回当前一步动作 [E, raw_dim]。

        约定：FeedForwardPolicy 外面会 .cpu().numpy() + process['action'].inverse_transform，
        所以这里只用返回正确 shape 的 tensor 即可。
        """
        pixels = info_dict["pixels"]            # [E, H, 3, H_img, W_img]
        goal = info_dict["goal"]
        # env 可能给单帧 goal [E, 3, H, W]，补一个时间维
        if goal.ndim == 4:
            goal = goal.unsqueeze(1)
        # 多帧 goal 时只取最后一帧（与训练保持一致）
        elif goal.size(1) > 1:
            goal = goal[:, -1:]

        if self._action_plan is None or self._action_step == self._action_plan.size(1):
            chunk = self.generate(pixels, goal, num_inference_steps=self.num_inference_steps)
            raw_dim = chunk.size(-1) // self.action_block
            # 每段完整执行 15 个原始动作，执行完后再根据新观测重规划。
            self._action_plan = chunk.reshape(chunk.size(0), -1, raw_dim)
            self._action_step = 0

        action = self._action_plan[:, self._action_step]
        self._action_step += 1
        return action
