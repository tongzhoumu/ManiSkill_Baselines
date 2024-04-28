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

Please check [ManiSkill's documentation](https://github.com/haosulab/ManiSkill?tab=readme-ov-file#installation) for more details.

----

# TODOs
- [x] SAC state
- [ ] PPO state
- [ ] SAC rgbd (a few examples)
- [ ] PPO rgbd (a few examples)
- [x] diffusion policy state (a few examples)
- [x] diffusion policy rgbd (a few examples)
- [ ] BeT state (a few examples)

----

## Benchmark Overview

|       **Task**      | **SAC(state)** | **Diffusion Policy (state)** | **Diffusion Policy (RGBD)** |
|:-------------------:|:--------------:|:----------------------------:|:---------------------------:|
| PickCube            | ✅              | ✅                            |              ✅              |
| StackCube           | ✅              | ✅                            |              ✅              |
| PickSingleYCB       | ✅              |                              |                             |
| PickSingleEGAD      | ✅              |                              |                             |
| PickClutterYCB      | ✅              |                              |                             |
| PegInsertionSide    | ✅              | ✅                            |              ❌              |
| TurnFaucet          | ✅              | ⚠️                            |              ⚠️              |
| PlugCharger         | ⚠️              |                              |                             |
| PandaAvoidObstacles | ❌              |                              |                             |
| OpenCabinetDrawer   | ✅              |                              |                             |
| OpenCabinetDoor     | ✅              |                              |                             |
| MoveBucket          | ✅              |                              |                             |
| PushChair           | ⚠️              | ⚠️                            |              ⚠️              |

- ✅ = works well
- ⚠️ = doesn't work well
- ❌ = doesn't work at all
- blank = not tested yet

----


## Run Experiments

The following commands should be run under the repo root dir.

### SAC

```bash
python rl/sac_state.py --env-id PickCube-v1 --total-timesteps 500000
python rl/sac_state.py --env-id StackCube-v1 --total-timesteps 5000000
python rl/sac_state.py --env-id PickSingleYCB-v1 --total-timesteps 5000000
python rl/sac_state.py --env-id PickSingleEGAD-v1 --total-timesteps 2000000
python rl/sac_state.py --env-id PickClutterYCB-v1 --total-timesteps 15000000
python rl/sac_state.py --env-id PegInsertionSide-v1 --total-timesteps 10000000 --gamma 0.9 --control-mode pd_ee_delta_pose
python rl/sac_state.py --env-id TurnFaucet-v0 --total-timesteps 20000000 --gamma 0.95 --control-mode pd_ee_delta_pose
python rl/sac_state.py --env-id PlugCharger-v0 --total-timesteps 15000000 --control-mode pd_ee_delta_pose
python rl/sac_state.py --env-id OpenCabinetDrawer_unified-v1 --total-timesteps 3000000 --gamma 0.95 --bootstrap-at-done truncated --control-mode base_pd_joint_vel_arm_pd_joint_vel
python rl/sac_state.py --env-id OpenCabinetDoor_unified-v1 --total-timesteps 5000000 --gamma 0.95 --bootstrap-at-done truncated --control-mode base_pd_joint_vel_arm_pd_joint_vel
python rl/sac_state.py --env-id MoveBucket_unified-v1 --total-timesteps 80000000 --gamma 0.9 --bootstrap-at-done truncated --control-mode base_pd_joint_vel_arm_pd_joint_vel --eval-freq 500000 --log-freq 20000
python rl/sac_state.py --env-id PushChair_unified-v1 --total-timesteps 20000000 --gamma 0.9 --bootstrap-at-done truncated --control-mode base_pd_joint_vel_arm_pd_joint_vel --eval-freq 500000 --log-freq 20000
```

Notes:
- If you want to use [Weights and Biases](https://wandb.ai) (`wandb`) to track learning progress, please add `--track` to your commands.
- You can tune `--num-envs` to get better speed.

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