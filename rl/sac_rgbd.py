ALGO_NAME = 'SAC-RGBD'

import os
import argparse
import random
from distutils.util import strtobool

os.environ["OMP_NUM_THREADS"] = "1"

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import DictReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import datetime
from collections import defaultdict
from functools import partial
from utils.profiling import NonOverlappingTimeProfiler
from nets.cnn.plain_conv import PlainConv, make_mlp

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
    parser.add_argument("--wandb-project-name", type=str, default="ManiSkill-baselines",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="PickCube-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=300_000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.8,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.01,
        help="target smoothing coefficient (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=512, # to be tuned
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=4000,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=1,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1,
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--correct-alpha", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--utd", type=float, default=0.25,
        help="Update-to-Data ratio (number of gradient updates / number of env steps)")
    parser.add_argument("--training-freq", type=int, default=64)

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--sync-venv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--log-freq", type=int, default=2000)
    parser.add_argument("--save-freq", type=int, default=1_000_000)
    parser.add_argument("--bootstrap-at-done", type=str, choices=['always', 'never', 'truncated'], default='always',
        help="in ManiSkill variable episode length and dense reward setting, set to always if positive reawrd, truncated if negative reward.")
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pos')
    parser.add_argument("--from-ckpt", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=64,
        help="the size of observation image, e.g. 64 means 64x64")

    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    if args.buffer_size is None:
        args.buffer_size = args.total_timesteps
    args.buffer_size = min(args.total_timesteps, args.buffer_size)
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    assert args.training_freq % args.num_envs == 0
    assert (args.training_freq * args.utd).is_integer()
    # fmt: on
    return args

import mani_skill2.envs
import env_wrappers.better_rewards # use the rewards designed by Tongzhou Mu
import env_wrappers.better_obs # use the observations designed by Tongzhou Mu
from mani_skill2.utils.common import flatten_state_dict, flatten_dict_space_keys
from mani_skill2.utils.wrappers import RecordEpisode
from gymnasium.core import ObservationWrapper
from gymnasium import spaces

class MS2_RGBDObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.obs_mode == 'rgbd'
        self.observation_space = self.build_obs_space(env, depth_dtype=np.float32)

    def observation(self, obs):
        img_dict = obs['image']
        new_img_dict = {
            key: np.concatenate([v[key] for v in img_dict.values()], axis=-1)
            for key in ['rgb', 'depth']
        }

        states = [flatten_state_dict(obs["agent"])]
        if len(obs["extra"]) > 0:
            states.append(flatten_state_dict(obs["extra"]))
        new_img_dict['state'] = np.hstack(states)

        return new_img_dict

    @staticmethod
    def build_obs_space(env, depth_dtype=np.float16):
        obs_space = getattr(env, 'single_observation_space', env.observation_space)
        state_dim = 0
        for k in ['agent', 'extra']:
            state_dim += sum([v.shape[0] for v in flatten_dict_space_keys(obs_space[k]).spaces.values()])

        single_img_space = list(obs_space['image'].values())[0]
        h, w, _ = single_img_space['rgb'].shape
        k = len(obs_space['image']) # number of cameras

        return spaces.Dict({
            'state': spaces.Box(-float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32),
            'rgb': spaces.Box(0, 255, shape=(h,w,k*3), dtype=np.uint8),
            'depth': spaces.Box(-float("inf"), float("inf"), shape=(h,w,k), dtype=depth_dtype),
        })
    # NOTE: We have to use float32 for gym AsyncVecEnv since it does not support float16, but we can use float16 for MS2 vec env

def make_vec_env(env_id, num_envs, seed, control_mode=None, image_size=None, video_dir=None):
    cam_cfg = {'width': image_size, 'height': image_size} if image_size else None
    wrappers = [
        gym.wrappers.RecordEpisodeStatistics,
        gym.wrappers.ClipAction,
    ]
    if video_dir:
        wrappers.append(partial(RecordEpisode, output_dir=video_dir, save_trajectory=False, info_on_video=True))
    wrappers.append(MS2_RGBDObsWrapper)
    def make_single_env(_seed):
        def thunk():
            env = gym.make(env_id, reward_mode='dense', obs_mode='rgbd', control_mode=control_mode, camera_cfgs=cam_cfg)
            for wrapper in wrappers: env = wrapper(env)
            env.action_space.seed(_seed)
            env.observation_space.seed(_seed)
            return env
        return thunk
    # must use AsyncVectorEnv, so that the renderers will be in different processes
    envs = gym.vector.AsyncVectorEnv([make_single_env(seed + i) for i in range(num_envs)], context='forkserver')

    return envs

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, envs, encoder):
        super().__init__()
        self.encoder = encoder
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space['state'].shape[0]
        self.mlp = make_mlp(encoder.encoder.out_dim+action_dim+state_dim, [512, 256, 1], last_act=False)

    def forward(self, obs, action, visual_feature=None, detach_encoder=False):
        if visual_feature is None:
            visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = torch.cat([visual_feature, obs["state"], action], dim=1)
        return self.mlp(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5

class EncoderObsWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, obs):
        rgb = obs['rgb'].float() / 255.0 # (B, H, W, 3*k)
        depth = obs['depth'].float() # (B, H, W, 1*k)
        img = torch.cat([rgb, depth], dim=3) # (B, H, W, C)
        img = img.permute(0, 3, 1, 2) # (B, C, H, W)
        return self.encoder(img)

class Actor(nn.Module):
    def __init__(self, envs, visual_feature_dim=256):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space['state'].shape[0]
        self.encoder = EncoderObsWrapper(
            PlainConv(in_channels=8, out_dim=visual_feature_dim) # assume image is 64x64
        )
        self.mlp = make_mlp(visual_feature_dim+state_dim, [512, 256], last_act=True)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling
        self.action_scale = torch.FloatTensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0)

    def get_feature(self, obs, detach_encoder=False):
        visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = torch.cat([visual_feature, obs['state']], dim=1)
        return self.mlp(x), visual_feature

    def forward(self, obs, detach_encoder=False):
        x, visual_feature = self.get_feature(obs, detach_encoder)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, visual_feature

    def get_eval_action(self, obs):
        mean, log_std, _ = self(obs)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, obs, detach_encoder=False):
        mean, log_std, visual_feature = self(obs, detach_encoder)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, visual_feature

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

def to_tensor(x, device):
    if isinstance(x, dict):
        return {k: to_tensor(v, device) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)

def collect_episode_info(infos, result=None):
    if result is None:
        result = defaultdict(list)
    if "final_info" in infos: # infos is a dict
        indices = np.where(infos["_final_info"])[0] # not all envs are done at the same time
        for i in indices:
            info = infos["final_info"][i] # info is also a dict
            ep = info['episode']
            print(f"global_step={global_step}, ep_return={ep['r'][0]:.2f}, ep_len={ep['l'][0]}, success={info['success']}")
            result['return'].append(ep['r'][0])
            result['len'].append(ep["l"][0])
            result['success'].append(info['success'])
    return result

def evaluate(n, agent, eval_envs, device):
    print('======= Evaluation Starts =========')
    agent.eval()
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
                             video_dir=f'{log_path}/videos' if args.capture_video else None)
    eval_envs.reset(seed=args.seed+1000) # seed eval_envs here, and no more seeding during evaluation
    envs = make_vec_env(args.env_id, args.num_envs, args.seed, args.control_mode, args.image_size)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs, actor.encoder).to(device)
    qf2 = SoftQNetwork(envs, actor.encoder).to(device)
    qf1_target = SoftQNetwork(envs, actor.encoder).to(device)
    qf2_target = SoftQNetwork(envs, actor.encoder).to(device)
    if args.from_ckpt is not None:
        ckpt = torch.load(args.from_ckpt)
        actor.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.mlp.parameters()) + 
        list(qf2.mlp.parameters()) + 
        list(qf1.encoder.parameters()), 
        lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = DictReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False, # stable-baselines3 has not fully supported Gymnasium's termination signal
    )

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.seed) # in Gymnasium, seed is given to reset() instead of seed()
    global_step = 0
    global_update = 0
    learning_has_started = False
    num_updates_per_training = int(args.training_freq * args.utd)
    result = defaultdict(list)
    timer = NonOverlappingTimeProfiler()

    while global_step < args.total_timesteps:

        # Collect samples from environemnts
        for local_step in range(args.training_freq // args.num_envs):
            global_step += 1 * args.num_envs

            # ALGO LOGIC: put action logic here
            if not learning_has_started:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _, _, _ = actor.get_action(to_tensor(obs, device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            result = collect_episode_info(infos, result)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = {k:v.copy() if isinstance(v, np.ndarray) else v.clone() for k,v in next_obs.items()}
            if args.bootstrap_at_done == 'never':
                stop_bootstrap = truncations | terminations # always stop bootstrap when episode ends
            else:
                if args.bootstrap_at_done == 'always':
                    need_final_obs = truncations | terminations # always need final obs when episode ends
                    stop_bootstrap = np.zeros_like(terminations, dtype=bool) # never stop bootstrap
                else: # bootstrap at truncated
                    need_final_obs = truncations & (~terminations) # only need final obs when truncated and not terminated
                    stop_bootstrap = terminations # only stop bootstrap when terminated, don't stop when truncated
                for idx, _need_final_obs in enumerate(need_final_obs):
                    if _need_final_obs:
                        t_obs = infos["final_observation"][idx]
                        for key in real_next_obs:
                            real_next_obs[key][idx] = t_obs[key]
            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        
        timer.end('collect')

        # ALGO LOGIC: training.
        if global_step < args.learning_starts:
            continue

        learning_has_started = True
        for local_update in range(num_updates_per_training):
            global_update += 1
            data = rb.sample(args.batch_size)

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _, visual_feature = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions, visual_feature)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions, visual_feature)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

            visual_feature = actor.encoder(data.observations)
            qf1_a_values = qf1(data.observations, data.actions, visual_feature).view(-1)
            qf2_a_values = qf2(data.observations, data.actions, visual_feature).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _, visual_feature = actor.get_action(data.observations, detach_encoder=True)
                qf1_pi = qf1(data.observations, pi, visual_feature, detach_encoder=True)
                qf2_pi = qf2(data.observations, pi, visual_feature, detach_encoder=True)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _, _ = actor.get_action(data.observations)
                    if args.correct_alpha:
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    else:
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()
                    # log_alpha has a legacy reason: https://github.com/rail-berkeley/softlearning/issues/136#issuecomment-619535356

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        
        timer.end('train')

        # Log training-related data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            if len(result['return']) > 0:
                for k, v in result.items():
                    writer.add_scalar(f"train/{k}", np.mean(v), global_step)
                result = defaultdict(list)
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            timer.dump_to_writer(writer, global_step)
            if args.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # Evaluation
        if (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            result = evaluate(args.num_eval_episodes, actor, eval_envs, device)
            for k, v in result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), global_step)
            timer.end('eval')
        
        # Checkpoint
        if args.save_freq and ( global_step >= args.total_timesteps or \
                (global_step - args.training_freq) // args.save_freq < global_step // args.save_freq):
            os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
            torch.save({
                'actor': actor.state_dict(),
                'qf1': qf1_target.state_dict(),
                'qf2': qf2_target.state_dict(),
                'log_alpha': log_alpha,
            }, f'{log_path}/checkpoints/{global_step}.pt')

    envs.close()
    writer.close()
