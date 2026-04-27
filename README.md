# Diff-Muscle

**Diff-Muscle: Efficient Learning for Musculoskeletal Robotic Table Tennis. [[Paper](https://arxiv.org/abs/2603.08617)]**

<p align="center">
  <img width="1020" height="467" alt="image" src="images\diff-muscle.png" />
  <p align="center"><i>Overview of Diff-Muscle</i></p>
</p>

---

## Overview

This repository is the official implementation of our paper [Diff-muscle](https://arxiv.org/abs/2603.08617) and the First Place solution for 2025 NeurIPS - MyoChallenge: Towards Human Athletic Intelligence Table Tennis Track - Team ActingAI. We use [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) for GPU-accelerated training, supporting thousands of parallel environments.



---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and requires an **NVIDIA GPU** with CUDA 12.4+.

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone <your-repo-url>
cd Diff-Muscle

# 3. Install all dependencies (includes mjlab and mujoco-warp)
uv sync
source ./.venv/bin/activate
```

This project bundles [mjlab](https://github.com/mujocolab/mjlab) in src/mjlab/. Running uv sync installs mjlab from this local source along with all its dependencies, including [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) (pinned to a tested revision). No separate mjlab installation is required.

---

## Training

### Single GPU

```bash
python train_multigpu.py
```

### Multi-GPU

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" python train_multigpu.py --gpu-ids all
```

Training configuration is in `default_config.yaml`. Key parameters:

**Environment**


| Parameter            | Default    | Description                            |
| -------------------- | ---------- | -------------------------------------- |
| `num_envs`           | 1024       | Number of parallel environments        |
| `action_type`        | `joint_pd` | Action space: `joint_pd`, `muscle_act` |
| `max_episode_length` | 300        | Max steps per episode                  |
| `frame_skip`         | 5          | Physics steps per control step         |


**Runner**


| Parameter                 | Default | Description                                     |
| ------------------------- | ------- | ----------------------------------------------- |
| `num_steps_per_env`       | 20      | Rollout length per environment per update       |
| `max_iterations`          | 3000    | Total number of policy update steps             |
| `empirical_normalization` | `true`  | Normalize observations using running statistics |
| `eval_interval`           | 500     | Run evaluation every N iterations               |
| `save_interval`           | 300     | Save checkpoint every N iterations              |
| `eval_episodes`           | 20      | number of episodes to evaluate                  |


**PPO Algorithm**


| Parameter             | Default        | Description                                                                |
| --------------------- | -------------- | -------------------------------------------------------------------------- |
| `learning_rate`       | 0.0005         | Initial learning rate (decays linearly by default)                         |
| `schedule`            | `linear_decay` | LR schedule: `linear_decay` or `adaptive`                                  |
| `num_learning_epochs` | 5              | Gradient update epochs per rollout                                         |
| `num_mini_batches`    | 4              | Mini-batches per epoch (`batch = num_envs × num_steps / num_mini_batches`) |


---

## Project Structure

```
Diff-Muscle/
├── tabletennis_env.py        # Main RL environment (TableTennisWarpEnv)
├── planner.py                # Physics-based ball trajectory planner
├── muscle_utils.py           # Muscle activation utilities (PD + FLV inverse dynamics)
├── on_policy_runner.py       # PPO on-policy training runner
├── train_multigpu.py  # Training entry point
├── default_config.yaml# PPO hyperparameters and logging config
├── tabletennis.xml           # MuJoCo scene definition
├── assets/                   # 3D mesh assets (paddle, ball, table)
├── myo_sim/                  # MyoSim musculoskeletal models (Apache-2.0)
├── src/mjlab/                # mjlab framework source
└── tests/                    # Unit tests
```

## Third-Party Code

- `**src/mjlab/**` — [mjlab](https://github.com/mujocolab/mjlab) framework (Apache-2.0)
- `**src/mjlab/utils/lab_api/**` — Utilities forked from [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause)

---

## Citation

If you find this open source release useful, please reference in your paper:

```
@article{zhao2026diff,
  title={Diff-Muscle: Efficient Learning for Musculoskeletal Robotic Table Tennis},
  author={Zhao, Wentao and Guo, Jun and Huang, Kangyao and Liu, Xin and Liu, Huaping},
  journal={arXiv preprint arXiv:2603.08617},
  year={2026}
}
```
