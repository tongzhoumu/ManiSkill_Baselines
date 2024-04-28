ALGO_NAME = 'Diffusion_UNet_RGBD'

import argparse
import os
import random
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import datetime
from collections import defaultdict, deque
from utils.profiling import NonOverlappingTimeProfiler

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from utils.sampler import IterationBasedBatchSampler
from utils.torch_utils import worker_init_fn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from nets.diffusion_policy.conditional_unet1d_colab import ConditionalUnet1D
from nets.cnn.plain_conv import PlainConv, PlainConv_MS1


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default='test',
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Rookie-dev",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="StackCube-v0",
        help="the id of the environment")
    parser.add_argument("--demo-path", type=str, default='data/ms2_official_demos/rigid_body/StackCube-v0/trajectory.rgbd.pd_ee_delta_pos_64.h5',
        help="the path of demo dataset (pkl or h5)")
    parser.add_argument("--num-demo-traj", type=int, default=None)
    parser.add_argument("--total-iters", type=int, default=1_000_000, # for easier task, we can train shorter
        help="total timesteps of the experiments")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the replay memory")
    
    # Diffusion Policy specific arguments
    parser.add_argument("--lr", type=float, default=1e-4) # 1e-4 is a safe choice
    parser.add_argument("--obs-horizon", type=int, default=2)
    # Seems not very important in ManiSkill, 1, 2, 4 work well
    parser.add_argument("--act-horizon", type=int, default=8)
    # Seems not very important in ManiSkill, 4, 8, 15 work well
    parser.add_argument("--pred-horizon", type=int, default=16)
    # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=64) # not very important
    parser.add_argument("--unet-dims", metavar='N', type=int, nargs='+', default=[64, 128, 256]) # ~4.5M params
    parser.add_argument("--n-groups", type=int, default=8)
    # Jiayuan told me it is better to let each group has at least 8 channels; it seems 4 and 8 are similar

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    parser.add_argument("--num-eval-envs", type=int, default=10) # NOTE: should not be too large, otherwise bias to short episodes
    parser.add_argument("--num-dataload-workers", type=int, default=0) # TODO: to tune
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pos')
    parser.add_argument("--obj-ids", metavar='N', type=str, nargs='+', default=[])
    parser.add_argument("--image-size", type=str, default='64x64',
        help="the size of observation image, HxW")
    parser.add_argument("--random-shift", type=int, default=0)

    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    if args.num_eval_envs == 1:
        args.sync_venv = True
    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, 'Control mode mismatched'
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1
    if args.image_size:
        args.image_size = tuple(map(int, args.image_size.split('x')))
    # fmt: on
    return args

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode

class SeqActionWrapper(gym.Wrapper):
    def step(self, action_seq):
        rew_sum = 0
        for action in action_seq:
            obs, rew, terminated, truncated, info = self.env.step(action)
            rew_sum += rew
            if terminated or truncated:
                break
        return obs, rew_sum, terminated, truncated, info

from gymnasium import spaces
class MS2_RGBDObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.get_wrapper_attr('obs_mode') == 'rgbd'
        self.state_obs_extractor = self.build_state_obs_extractor(env.unwrapped.spec.id)
        self.observation_space = self.build_obs_space(env, np.float32, self.state_obs_extractor)
        self.concat_fn = partial(np.concatenate, axis=-1)
        self.transpose_fn = partial(np.transpose, axes=(2, 0, 1)) # channel first, (H, W, C) -> (C, H, W)

    def observation(self, obs):
        return self.convert_obs(obs, self.concat_fn, self.transpose_fn, self.state_obs_extractor)
    
    @staticmethod
    def build_state_obs_extractor(env_id):
        env_name = env_id.split('-')[0]
        if env_name in ['TurnFaucet', 'StackCube']:
            return lambda obs: list(obs['extra'].values())
        elif env_name == 'PushChair':
            return lambda obs: list(obs['agent'].values()) + list(obs['extra'].values())
        else:
            raise NotImplementedError(f'Please tune state obs by yourself')
    
    @staticmethod
    def build_obs_space(env, depth_dtype, state_obs_extractor):
        # NOTE: We have to use float32 for gym AsyncVecEnv since it does not support float16, but we can use float16 for MS2 vec env
        obs_space = env.observation_space

        # Unified version
        state_dim = sum([v.shape[0] for v in state_obs_extractor(obs_space)])

        single_img_space = next(iter(env.observation_space['image'].values()))
        h, w, _ = single_img_space['rgb'].shape
        n_images = len(env.observation_space['image'])

        return spaces.Dict({
            'state': spaces.Box(-float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32),
            'rgb': spaces.Box(0, 255, shape=(n_images*3, h,w), dtype=np.uint8),
            'depth': spaces.Box(-float("inf"), float("inf"), shape=(n_images,h,w), dtype=depth_dtype),
        })
    
    @staticmethod
    def convert_obs(obs, concat_fn, transpose_fn, state_obs_extractor):
        img_dict = obs['image']
        new_img_dict = {
            key: transpose_fn(
                concat_fn([v[key] for v in img_dict.values()])
            ) # (C, H, W) or (B, C, H, W)
            for key in ['rgb', 'depth']
        }
        # if isinstance(new_img_dict['depth'], torch.Tensor): # MS2 vec env uses float16, but gym AsyncVecEnv uses float32
        #     new_img_dict['depth'] = new_img_dict['depth'].to(torch.float16)

        # Unified version
        states_to_stack = state_obs_extractor(obs)
        for j in range(len(states_to_stack)):
            if states_to_stack[j].dtype == np.float64:
                states_to_stack[j] = states_to_stack[j].astype(np.float32)
        try:
            state = np.hstack(states_to_stack)
        except: # dirty fix for concat trajectory of states
            state = np.column_stack(states_to_stack)
        if state.dtype == np.float64:
            for x in states_to_stack:
                print(x.shape, x.dtype)
            import pdb; pdb.set_trace()

        out_dict = {
            'state': state,
            'rgb': new_img_dict['rgb'], 
            'depth': new_img_dict['depth'],
        }
        return out_dict
    
from gymnasium.wrappers.frame_stack import FrameStack, LazyFrames, Box
class DictFrameStack(FrameStack):
    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
        lz4_compress: bool = False,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, num_stack=num_stack, lz4_compress=lz4_compress
        )
        gym.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        new_observation_space = gym.spaces.Dict()
        for k, v in self.observation_space.items():
            low = np.repeat(v.low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat(v.high[np.newaxis, ...], num_stack, axis=0)
            new_observation_space[k] = Box(
                low=low, high=high, dtype=v.dtype
            )
        self.observation_space = new_observation_space

    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return {
            k: LazyFrames([x[k] for x in self.frames], self.lz4_compress)
            for k in self.observation_space.keys()
        }


from functools import partial
def make_vec_env(env_id, num_envs, seed, control_mode=None, image_size=None, video_dir=None, other_kwargs={}, gym_vec_env=True):
    assert gym_vec_env or video_dir is None, 'Saving video is only supported for gym vec env'
    cam_cfg = {'width': image_size[0], 'height': image_size[1]} if image_size else None
    wrappers = [
        gym.wrappers.RecordEpisodeStatistics,
        gym.wrappers.ClipAction,
        partial(DictFrameStack, num_stack=other_kwargs['obs_horizon']),
        SeqActionWrapper,
    ]
    env_kwargs = {'model_ids': other_kwargs['obj_ids']} if len(other_kwargs['obj_ids']) > 0 else {}
    if gym_vec_env:
        wrappers = [MS2_RGBDObsWrapper] + wrappers
        if video_dir:
            wrappers = [partial(RecordEpisode, output_dir=video_dir, save_trajectory=False, info_on_video=True)] + wrappers
        def make_single_env(_seed):
            def thunk():
                env = gym.make(env_id, reward_mode='sparse', obs_mode='rgbd', control_mode=control_mode, 
                               render_mode='cameras' if video_dir else None, camera_cfgs=cam_cfg, **env_kwargs)
                for wrapper in wrappers: env = wrapper(env)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env
            return thunk
        # must use AsyncVectorEnv, so that the renderers will be in different processes
        # but in BC, we can use SyncVectorEnv if we only create one renderer, since we do not have training envs
        VecEnv = gym.vector.SyncVectorEnv if num_envs == 1 \
            else lambda x: gym.vector.AsyncVectorEnv(x, context='forkserver')
        envs = VecEnv([make_single_env(seed + i) for i in range(num_envs)])
    else:
        raise NotImplementedError()

    return envs

def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out

class SmallDemoDataset_DiffusionPolicy(Dataset): # Load everything into memory
    def __init__(self, data_path, obs_process_fn, obs_space, num_traj):
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            from utils.ms2_data import load_demo_dataset
            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['observations'] is a list of dict, each dict is a traj, with keys in obs_space, values with length L+1
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        print('Raw trajectory loaded, start to pre-process the observations...')

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories['observations']:
            _obs_traj_dict = reorder_keys(obs_traj_dict, obs_space) # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            _obs_traj_dict['depth'] = torch.Tensor(_obs_traj_dict['depth'].astype(np.float32) / 1024).to(torch.float16)
            _obs_traj_dict['rgb'] = torch.from_numpy(_obs_traj_dict['rgb']) # still uint8
            _obs_traj_dict['state'] = torch.from_numpy(_obs_traj_dict['state'])
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories['observations'] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())
        # Pre-process the actions
        for i in range(len(trajectories['actions'])):
            trajectories['actions'][i] = torch.Tensor(trajectories['actions'][i])
        print('Obs/action pre-processing is done, start to pre-compute the slice indices...')

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,))
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            raise NotImplementedError(f'Control Mode {args.control_mode} not supported')
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = args.obs_horizon, args.pred_horizon
        self.slices = []
        num_traj = len(trajectories['actions'])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories['actions'][traj_idx].shape[0]
            assert trajectories['observations'][traj_idx]['state'].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon) for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)
        
        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories['actions'][traj_idx].shape

        obs_traj = self.trajectories['observations'][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start):start+self.obs_horizon] # start+self.obs_horizon is at least 1
            if start < 0: # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]]*abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories['actions'][traj_idx][max(0, start):end]
        if start < 0: # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L: # pad after the trajectory
            gripper_action = act_seq[-1, -1] # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end-L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert obs_seq['state'].shape[0] == self.obs_horizon and act_seq.shape[0] == self.pred_horizon
        return {
            'observations': obs_seq,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.slices)

class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space['state'].shape) == 2 # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1 # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space['state'].shape[1]
        _, C, H, W = envs.single_observation_space['rgb'].shape

        visual_feature_dim = 256
        CNN_class = PlainConv if C == 6 else PlainConv_MS1
        self.visual_encoder = CNN_class(in_channels=int(C/3*4), out_dim=visual_feature_dim)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim, # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2', # has big impact on performance, try not to change
            clip_sample=True, # clip output to [-1,1] to improve stability
            prediction_type='epsilon' # predict noise (instead of denoised action)
        )

        if args.random_shift > 0:
            from utils.torch_utils import RandomShiftsAug
            self.aug = RandomShiftsAug(args.random_shift)

    def encode_obs(self, obs_seq, eval_mode):
        rgb = obs_seq['rgb'].float() / 255.0 # (B, obs_horizon, 3*k, H, W)
        depth = obs_seq['depth'].float() # (B, obs_horizon, 1*k, H, W)
        img_seq = torch.cat([rgb, depth], dim=2) # (B, obs_horizon, C, H, W), C=4*k
        img_seq = img_seq.flatten(end_dim=1) # (B*obs_horizon, C, H, W)
        if hasattr(self, 'aug') and not eval_mode:
            img_seq = self.aug(img_seq) # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq) # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(rgb.shape[0], self.obs_horizon, visual_feature.shape[1]) # (B, obs_horizon, D)
        feature = torch.cat((visual_feature, obs_seq['state']), dim=-1) # (B, obs_horizon, D+obs_state_dim)
        return feature.flatten(start_dim=1) # (B, obs_horizon * (D+obs_state_dim))
    
    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq['state'].shape[0]

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(obs_seq, eval_mode=False) # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(
            action_seq, noise, timesteps)
        
        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)
    
    def get_eval_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq['state']: (B, obs_horizon, obs_state_dim)
        B = obs_seq['state'].shape[0]
        with torch.no_grad():
            obs_cond = self.encode_obs(obs_seq, eval_mode=True) # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq['state'].device)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end] # (B, act_horizon, act_dim)

def collect_episode_info(infos, result=None):
    if result is None:
        result = defaultdict(list)
    if "final_info" in infos: # infos is a dict
        indices = np.where(infos["_final_info"])[0] # not all envs are done at the same time
        for i in indices:
            info = infos["final_info"][i] # info is also a dict
            ep = info['episode']
            print(f"global_step={cur_iter}, ep_return={ep['r'][0]:.2f}, ep_len={ep['l'][0]}, success={info['success']}")
            result['return'].append(ep['r'][0])
            result['len'].append(ep["l"][0])
            result['success'].append(info['success'])
    return result

def to_tensor(x, device):
    if isinstance(x, dict):
        return {k: to_tensor(v, device) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)

def evaluate(n, agent, eval_envs, device):
    agent.eval()
    print('======= Evaluation Starts =========')
    result = defaultdict(list)
    obs, info = eval_envs.reset() # don't seed here
    while len(result['return']) < n:
        with torch.no_grad():
            action = agent.get_eval_action(to_tensor(obs, device))
        obs, rew, terminated, truncated, info = eval_envs.step(action.cpu().numpy())
        collect_episode_info(info, result)
    print('======= Evaluation Ends =========')
    agent.train()
    return result

def save_ckpt(tag):
    os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, f'{log_path}/checkpoints/{tag}.pt')

if __name__ == "__main__":
    args = parse_args()

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tag = '{:s}_{:d}'.format(now, args.seed)
    if args.exp_name: tag += '_' + args.exp_name
    log_name = os.path.join(args.env_id, ALGO_NAME, tag)
    log_path = os.path.join(args.output_dir, log_name)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=log_name.replace(os.path.sep, "__"),
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    import json
    with open(f'{log_path}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    eval_envs = make_vec_env(args.env_id, args.num_eval_envs, args.seed+1000, args.control_mode, args.image_size,
                             video_dir=f'{log_path}/videos' if args.capture_video else None, other_kwargs=args.__dict__,
                            )
                            #  gym_vec_env=args.capture_video)
    eval_envs.reset(seed=args.seed+1000) # seed eval_envs here, and no more seeding during evaluation
    envs = eval_envs
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # dataloader setup
    obs_process_fn = partial(
        MS2_RGBDObsWrapper.convert_obs, 
        concat_fn=partial(np.concatenate, axis=-1), 
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)), # (B, H, W, C) -> (B, C, H, W)
        state_obs_extractor=MS2_RGBDObsWrapper.build_state_obs_extractor(args.env_id),
    )
    tmp_env = gym.make(args.env_id, obs_mode='rgbd'); orignal_obs_space = tmp_env.observation_space; tmp_env.close()
    dataset = SmallDemoDataset_DiffusionPolicy(args.demo_path, obs_process_fn, orignal_obs_space, args.num_demo_traj)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        pin_memory=True,
        persistent_workers=(args.num_dataload_workers > 0),
    )

    # agent setup
    agent = Agent(envs, args).to(device)
    optimizer = optim.AdamW(params=agent.parameters(), 
        lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)


    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()
    best_success_rate = -1

    timer = NonOverlappingTimeProfiler()

    for iteration, data_batch in enumerate(train_dataloader):
        cur_iter = iteration + 1
        timer.end('data')

        # copy data from cpu to gpu
        obs_batch_dict = data_batch['observations']
        obs_batch_dict = {k: v.cuda(non_blocking=True) for k, v in obs_batch_dict.items()}
        act_batch = data_batch['actions'].cuda(non_blocking=True)

        # forward and compute loss
        total_loss = agent.compute_loss(
            obs_seq=obs_batch_dict, # obs_batch_dict['state'] is (B, L, obs_dim)
            action_seq=act_batch, # (B, L, act_dim)
        )
        timer.end('forward')

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step() # step lr scheduler every batch, this is different from standard pytorch behavior
        timer.end('backward')

        # update Exponential Moving Average of the model weights
        ema.step(agent.parameters())
        timer.end('EMA')

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if cur_iter % args.log_freq == 0: 
            print(cur_iter, total_loss.item())
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter)
            writer.add_scalar("losses/total_loss", total_loss.item(), cur_iter)
            timer.dump_to_writer(writer, cur_iter)

        # Evaluation
        if cur_iter % args.eval_freq == 0:
            ema.copy_to(ema_agent.parameters())
            result = evaluate(args.num_eval_episodes, ema_agent, eval_envs, device)
            for k, v in result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), cur_iter)
            
            timer.end('eval')
            sr = np.mean(result['success'])
            if sr > best_success_rate:
                best_success_rate = sr
                save_ckpt('best_eval_success_rate')
                print(f'### Update best success rate: {sr:.4f}')

        # Checkpoint
        if args.save_freq and cur_iter % args.save_freq == 0:
            save_ckpt(str(cur_iter))

    envs.close()
    writer.close()
