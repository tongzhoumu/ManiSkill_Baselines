<h1>ManiSkill Baselines</span></h1>

This repository contains **unofficial** baselines for [ManiSkill](https://maniskill2.github.io/) (more specifically, version `0.5.3`). These baselines are **heavily tuned** so they generally give you better sample efficiency and performance. 


----

## Installation

1. Install all dependencies via `mamba` or `conda` by running the following command:

```bash
mamba env create -f environment.yml
mamba activate ms
```

Note: `mamba` is a drop-in replacement for `conda`. Feel free to use `conda` if you prefer it.


2. Download and link the necessary assets for ManiSkill

```bash
python -m mani_skill2.utils.download_asset all # if you need the assets for all tasks
python -m mani_skill2.utils.download_asset ${ENV_ID} # if you only need the assets for one task
```

which downloads assets to `./data`. You may move these assets to any location. Then, add the following line to your `~/.bashrc` or `~/.zshrc`:

```bash
export MS2_ASSET_DIR=<path>/<to>/<data>
```

and restart your terminal. 


----

## TODOs
- [x] SAC state
- [x] PPO state
- [ ] SAC rgbd (a few examples)
- [ ] PPO rgbd (a few examples)
- [x] diffusion policy state (a few examples)
- [x] diffusion policy rgbd (a few examples)
- [ ] BeT state (a few examples)
- [ ] MPC

----

## Benchmark Overview

|       **Task**      | **SAC (state)** | **PPO (state)** | **Diffusion Policy (state)** | **Diffusion Policy (RGBD)** |
|---------------------|:---------------:|:---------------:|:----------------------------:|:---------------------------:|
| PickCube            |        ✅        |        ✅        |               ✅              |              ✅              |
| StackCube           |        ✅        |        ❌        |               ✅              |              ✅              |
| PickSingleYCB       |        ✅        |        ✅        |                              |                             |
| PickSingleEGAD      |        ✅        |        ✅        |                              |                             |
| PickClutterYCB      |        ✅        |        ⚠️        |                              |                             |
| PegInsertionSide    |        ✅        |                 |               ✅              |              ❌              |
| TurnFaucet          |        ✅        |                 |               ⚠️              |              ⚠️              |
| PlugCharger         |        ⚠️        |                 |                              |                             |
| PandaAvoidObstacles |        ❌        |                 |                              |                             |
| OpenCabinetDrawer   |        ✅        |        ⚠️        |                              |                             |
| OpenCabinetDoor     |        ✅        |        ⚠️        |                              |                             |
| MoveBucket          |        ✅        |        ❌        |                              |                             |
| PushChair           |        ⚠️        |                 |               ⚠️              |              ⚠️              |

- ✅ = works well
- ⚠️ = works, but there is still room for improvement
- ❌ = doesn't work at all
- blank = not tested yet

----


## Run Experiments

The following commands should be run under the repo root dir.

### SAC

```bash
python rl/sac_state.py --env-id PickCube-v1 --total-timesteps 500_000
python rl/sac_state.py --env-id StackCube-v1 --total-timesteps 5_000_000
python rl/sac_state.py --env-id PickSingleYCB-v1 --total-timesteps 5_000_000
python rl/sac_state.py --env-id PickSingleEGAD-v1 --total-timesteps 2_000_000
python rl/sac_state.py --env-id PickClutterYCB-v1 --total-timesteps 15_000_000
python rl/sac_state.py --env-id PegInsertionSide-v1 --total-timesteps 10_000_000 --gamma 0.9 --control-mode pd_ee_delta_pose
python rl/sac_state.py --env-id TurnFaucet-v0 --total-timesteps 20_000_000 --gamma 0.95 --control-mode pd_ee_delta_pose
python rl/sac_state.py --env-id PlugCharger-v0 --total-timesteps 15_000_000 --control-mode pd_ee_delta_pose
python rl/sac_state.py --env-id OpenCabinetDrawer_unified-v1 --total-timesteps 3_000_000 --gamma 0.95 --bootstrap-at-done truncated --control-mode base_pd_joint_vel_arm_pd_joint_vel
python rl/sac_state.py --env-id OpenCabinetDoor_unified-v1 --total-timesteps 5_000_000 --gamma 0.95 --bootstrap-at-done truncated --control-mode base_pd_joint_vel_arm_pd_joint_vel
python rl/sac_state.py --env-id MoveBucket_unified-v1 --total-timesteps 80_000_000 --gamma 0.9 --bootstrap-at-done truncated --control-mode base_pd_joint_vel_arm_pd_joint_vel --eval-freq 500_000 --log-freq 20_000
python rl/sac_state.py --env-id PushChair_unified-v1 --total-timesteps 20_000_000 --gamma 0.9 --bootstrap-at-done truncated --control-mode base_pd_joint_vel_arm_pd_joint_vel --eval-freq 500_000 --log-freq 20_000
```

Notes:
- If you want to use [Weights and Biases](https://wandb.ai) (`wandb`) to track learning progress, please add `--track` to your commands.
- You can tune `--num-envs` to get better speed.

### PPO

```bash
python rl/ppo_state.py --env-id PickCube-v1 --total-timesteps 3_000_000
python rl/ppo_state.py --env-id PickSingleYCB-v1 --total-timesteps 50_000_000 --gamma 0.9 --utd 0.025
python rl/ppo_state.py --env-id PickSingleEGAD-v1 --total-timesteps 5_000_000 --utd 0.025
python rl/ppo_state.py --env-id PickClutterYCB-v1 --total-timesteps 50_000_000
python rl/ppo_state.py --env-id OpenCabinetDrawer_unified-v1 --total-timesteps 30_000_000 --gamma 0.95 --utd 0.025 --bootstrap-at-done truncated --control-mode base_pd_joint_vel_arm_pd_joint_vel --eval-freq 500_000 --log-freq 20_000
python rl/ppo_state.py --env-id OpenCabinetDoor_unified-v1 --total-timesteps 50_000_000 --gamma 0.95 --utd 0.025 --bootstrap-at-done truncated --control-mode base_pd_joint_vel_arm_pd_joint_vel --eval-freq 500_000 --log-freq 20_000
```

Notes:
- PPO usually yields worse sample effiency when comapred to SAC.


### Diffusion Policy

State observation:
```bash
python bc/diffusion_unet.py --env-id PegInsertionSide-v0 --demo-path PATH_TO_MS2_OFFICIAL_DEMO
```

RGBD observation:
```bash
python bc/diffusion_unet_rgbd.py --env-id StackCube-v0 --demo-path PATH_TO_MS2_OFFICIAL_DEMO
```

## Acknowledgments

This codebase is built upon [CleanRL](https://github.com/vwxyzjn/cleanrl) repository.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.