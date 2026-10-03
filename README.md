# FlowWM
### Optical-Flow-Supervised LeWorldModel

FlowWM builds on [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm), a stable end-to-end JEPA world model trained from raw pixels. We add a **training-only optical-flow (motion) branch**: alongside the usual next-embedding prediction, the world model also has to predict a latent of the optical flow that follows each context step. This extra signal pushes the shared state representation to encode *how things move*, not only *what the scene looks like*.

The motion branch is used **only during training**. Encoding, rollout and the planning cost (`JEPA.encode` / `rollout` / `criterion` / `get_cost`) are the same as in LeWM, so planning cost and speed are unchanged. The improvements come purely from a better-trained latent space.

## Results

Planning success rate (%) with the same CEM planner as LeWM:

<div align="center">

| Method | PushT | Cube |
|:---|:---:|:---:|
| LeWM | 96 | 74 |
| **FlowWM (ours)** | **100** | **86** |

</div>

## Method

LeWM trains an encoder, an action-conditioned autoregressive predictor and projector heads with two losses: next-embedding prediction and SIGReg, a regularizer that pushes latent embeddings toward a Gaussian distribution. FlowWM keeps that state branch unchanged and adds a parallel **motion branch** that shares no weights with it:

| | State branch (LeWM) | Motion branch (new, train-only) |
|:---|:---|:---|
| Input | RGB frame, 3 channels | raw UV optical flow, 2 channels |
| Encoder | ViT (`encoder`) | ViT with a 2-channel patch embedding (`encoder_vel`) |
| Predictor | `predictor`: (z, a) → next z | `predictor_vel`: (z, a) → motion latent |
| Loss | MSE + SIGReg | MSE + SIGReg |

The motion predictor reads the **state** latents and actions and predicts the flow latent for each context step. Because the motion loss backpropagates into the state encoder, the state latent itself has to carry motion information. The total objective is:

```
L = w_state · L_state + w_motion · L_motion + λ_state · SIGReg(z) + λ_motion · SIGReg(v)
```

The defaults in `config/train/lewm.yaml` are `w_state=1.0`, `w_motion=0.5`, `λ_state=0.08`, `λ_motion=0.2`. Set `motion.enabled=False` to recover plain LeWM training.

Optical flow is normalized per channel; the mean/std are estimated from a random sample of the flow file when they are not given in the config. Flow maps are resized to `img_size` and fed natively as 2-channel images, with no 2→3 channel adaptation.

The repository also includes an optional **flow-matching action policy** (`action_policy/`). It is a diffusion-style action head on top of a frozen FlowWM/LeWM backbone and is selected with `policy_type=flow` at evaluation time. The results above use CEM planning, not this policy.

## Installation

This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning and evaluation, and on [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training.

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
uv pip install h5py hdf5plugin
```

## Data

### RGB datasets
Use the same HDF5 datasets as LeWM. Download them from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the `.h5` files under `$STABLEWM_HOME`. The configs in this repo resolve paths through this variable, so **export it explicitly**:

```bash
export STABLEWM_HOME=/path/to/your/storage
```

### Optical-flow datasets
The motion branch needs one optical-flow HDF5 file per dataset, frame-aligned with the RGB file:

| Dataset | RGB file | Flow file |
|:---|:---|:---|
| PushT | `pusht_expert_train.h5` | `pusht_expert_train_flow_uv.h5` |
| Cube | `ogbench/cube_single_expert.h5` | `cube_single_expert_flow_uv.h5` |

Each flow file has the same episode layout as its RGB file. Its `pixels` key holds the raw UV flow from frame `t` to frame `t+1`, with shape `(N, H, W, 2)` and type int16 or float. During training, the flow at the end of each context step is used as the motion target.

## Training

`jepa.py` contains the model; the motion-branch modules are `encoder_vel`, `predictor_vel`, `projector_vel` and `pred_vel_proj`. Training is configured with [Hydra](https://hydra.cc/) under `config/train/`. Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`.

**Cube** (default config):
```bash
python train.py
```

**PushT**:
```bash
python train.py data=pusht \
  motion.flow_path=$STABLEWM_HOME/pusht_expert_train_flow_uv.h5 \
  output_model_name=lewm_pusht subdir=pusht
```

The model object is saved after every epoch to `$STABLEWM_HOME/<subdir>/<output_model_name>_epoch_<N>_object.ckpt`. For quick experiments, `max_clips=<N>` deterministically subsamples the training clips.

## Planning

Evaluation configs live under `config/eval/`. Set `policy` to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# PushT
python eval.py --config-name=pusht.yaml policy=pusht/lewm_pusht_epoch_<N>

# Cube
python eval.py --config-name=cube.yaml policy=cube_full/lewm_cube_full_epoch_<N>
```

### Optional: flow-matching action policy
Train a policy head on a frozen FlowWM checkpoint and evaluate it instead of CEM:

```bash
python action_policy/train_action.py \
  backbone.ckpt_path=$STABLEWM_HOME/cube_full/lewm_cube_full_epoch_<N>_object.ckpt

python eval.py --config-name=cube.yaml policy_type=flow \
  flow.backbone_ckpt=$STABLEWM_HOME/cube_full/lewm_cube_full_epoch_<N>_object.ckpt \
  flow.policy_ckpt=outputs/action_flow/policy_epoch<M>.ckpt
```

## Acknowledgements & Citation

This repository is a fork of [LeWorldModel](https://github.com/lucas-maes/le-wm) by Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun and Randall Balestriero. All credit for the base architecture, training recipe and planning pipeline goes to them. If you use this code, please cite LeWM:

```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT; see [LICENSE](LICENSE). The original copyright notice of LeWorldModel is retained.
