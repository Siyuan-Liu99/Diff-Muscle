# Table Tennis with Musculoskeletal Control

A MuJoCo-based reinforcement learning environment for training a musculoskeletal arm model to play table tennis, built on top of [mjlab](https://github.com/mujocolab/mjlab) and the [MyoSim](https://github.com/MyoHub/myo_sim) musculoskeletal model library.

---

## Overview

This project simulates a table tennis task where a **MyoSim musculoskeletal arm** (27 DoF, 63 muscles) must hit incoming balls over the net and land them on the opponent's side of the table.

Key features:
- **Musculoskeletal control**: Supports multiple action types — `joint_pd`, `muscle_pd`, `muscle_act`, and `muscle_vae`
- **Hierarchical planning**: A physics-based trajectory planner (`planner.py`) computes high-level target paddle poses and velocities from ball kinematics
- **Parallel simulation**: GPU-accelerated via [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), supports thousands of parallel environments
- **Multi-GPU training**: Uses `torchrunx` for distributed PPO training across multiple GPUs
- **Domain randomization**: Ball position, velocity, paddle mass, and ball friction are randomized at reset

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and requires an **NVIDIA GPU**.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <your-repo-url>
cd tabletennis_double_opensource

# Install dependencies
uv sync
```

---

## Training

### Single GPU

```bash
uv run python train_single_multigpu.py
```

### Multi-GPU

```bash
uv run torchrun --nproc_per_node=<NUM_GPUS> train_single_multigpu.py
```

Training configuration is in `default_config_single.yaml`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `num_envs` | 1024 | Number of parallel environments |
| `action_type` | `joint_pd` | Action space: `joint_pd`, `muscle_pd`, `muscle_act`, `muscle_vae` |
| `max_episode_length` | 300 | Max steps per episode |
| `frame_skip` | 5 | Physics steps per control step |

---

## Project Structure

```
tabletennis_double_opensource/
├── tabletennis_env.py        # Main RL environment (TableTennisWarpEnv)
├── planner.py                # Physics-based ball trajectory planner
├── muscle_utils.py           # Muscle activation utilities (PD + FLV inverse dynamics)
├── on_policy_runner.py       # PPO on-policy training runner
├── train_single_multigpu.py  # Training entry point
├── default_config_single.yaml# PPO hyperparameters and logging config
├── tabletennis.xml           # MuJoCo scene definition
├── assets/                   # 3D mesh assets (paddle, ball, table)
├── myo_sim/                  # MyoSim musculoskeletal models (Apache-2.0)
├── src/mjlab/                # mjlab framework source
└── tests/                    # Unit tests
```

---

## Environment Details

### Observation Space

The actor observation includes: pelvis position, body joint positions/velocities, ball position/velocity, paddle position/velocity/orientation, reach error, contact information, current activations, and planner targets.

### Action Space

Depends on `action_type`:
- **`joint_pd`**: Target joint positions → muscle activations via FK + PD control
- **`muscle_act`**: Direct muscle activations
- **`muscle_vae`**: Target muscle lengths → activations via FLV inverse dynamics

### Reward

| Component | Weight | Description |
|---|---|---|
| `paddle_pos_err` | 20 | Paddle reaches target hit position |
| `paddle_ori_err` | 10 | Paddle orientation aligned with desired hit direction |
| `hit_with_paddle` | 100 | Ball contacts paddle |
| `fall_opponent` | 100 | Ball lands on opponent's side |
| `fall_hit_plane` | 100 | Returned ball can reach opponent's body plane |
| `net_penalty` | -20 | Ball hits the net |

### Termination

Episode ends on: timeout (5s), ball out of range, or ball leaves the paddle without a valid hit.

---

## Third-Party Code

- **`src/mjlab/`** — [mjlab](https://github.com/mujocolab/mjlab) framework (Apache-2.0)
- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause)
- **`myo_sim/`** — [MyoSim](https://github.com/MyoHub/myo_sim) musculoskeletal models (Apache-2.0)

---

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).
