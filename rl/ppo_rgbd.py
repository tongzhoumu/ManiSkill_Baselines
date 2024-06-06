ALGO_NAME = 'PPO-RGBD'

import os
import argparse
import random
from distutils.util import strtobool

os.environ["OMP_NUM_THREADS"] = "1"

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
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
    parser.add_argument("--total-timesteps", type=int, default=5_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-steps-per-collect", type=int, default=8000, # this hp is pretty important
        help="the number of steps to run in all environment in total per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.8,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.9,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--minibatch-size", type=int, default=400,
        help="the size of mini-batches")
    parser.add_argument("--utd", type=float, default=0.05,
        help="Update-to-Data ratio (number of gradient updates / number of env steps)")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.1,
        help="the target KL divergence threshold")
    parser.add_argument("--critic-warmup-epochs", type=int, default=4)
    parser.add_argument("--finite-horizon-gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--log-freq", type=int, default=10000)
    parser.add_argument("--sync-venv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--rew-norm", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--save-freq", type=int, default=1_000_000)
    parser.add_argument("--bootstrap-at-done", type=str, choices=['always', 'never', 'truncated'], default='always',
        help="in ManiSkill variable episode length and dense reward setting, set to always if positive reawrd, truncated if negative reward.")
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pos')
    parser.add_argument("--image-size", type=int, default=64,
        help="the size of observation image, e.g. 64 means 64x64")

    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    assert args.num_steps_per_collect % args.num_envs == 0
    args.num_steps = int(args.num_steps_per_collect // args.num_envs)
    assert args.num_steps_per_collect % args.minibatch_size == 0
    args.num_minibatches = int(args.num_steps_per_collect // args.minibatch_size)
    assert (args.num_steps_per_collect * args.utd).is_integer()
    args.num_updates_per_collect = int(args.num_steps_per_collect * args.utd)
    assert args.num_updates_per_collect % args.num_minibatches == 0
    args.update_epochs = int(args.num_updates_per_collect // args.num_minibatches)
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    args.critic_warmup_updates = args.critic_warmup_epochs * args.num_minibatches
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

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space['state'].shape[0]
        self.encoder = PlainConv(in_channels=8, out_dim=256)
        self.critic = make_mlp(256+state_dim, [512, 256, 1], last_act=False)
        self.actor_mean = make_mlp(256+state_dim, [512, 256, action_dim], last_act=False)
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_feature(self, obs):
        # Preprocess the obs before passing to the real network, similar to the Dataset class in supervised learning
        rgb = obs['rgb'].float() / 255.0 # (B, H, W, 3*k)
        depth = obs['depth'].float() # (B, H, W, 1*k)
        img = torch.cat([rgb, depth], dim=3) # (B, H, W, C)
        img = img.permute(0, 3, 1, 2) # (B, C, H, W)
        feature = self.encoder(img)
        return torch.cat([feature, obs['state']], dim=1)

    def get_value(self, obs):
        x = self.get_feature(obs)
        return self.critic(x)

    def get_eval_action(self, obs):
        x = self.get_feature(obs)
        return self.actor_mean(x)

    def get_action_and_value(self, obs, action=None):
        x = self.get_feature(obs)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def to_tensor(x):
    if isinstance(x, dict):
        return {k: to_tensor(v) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)

def unsqueeze(x, dim):
    if isinstance(x, dict):
        return {k: unsqueeze(v, dim) for k, v in x.items()}
    return x.unsqueeze(dim)

class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v)
                else:
                    self.data[k] = torch.zeros(buffer_shape + v.shape).to(device)

    def keys(self):
        return self.data.keys()
    
    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v
    
    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


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
            action = agent.get_eval_action(to_tensor(obs))
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
    tmp_env = gym.make(args.env_id)
    if tmp_env.spec.max_episode_steps > args.num_steps:
        print("\033[93mWARN: num_steps is less than max episode length. "
            "Consider raise num_steps_per_collect or lower num_envs. Continue?\033[0m")
        aaa = input()
    del tmp_env
    eval_envs = make_vec_env(args.env_id, args.num_eval_envs, args.seed+1000, args.control_mode, args.image_size,
                             video_dir=f'{log_path}/videos' if args.capture_video else None)
    eval_envs.reset(seed=args.seed+1000) # seed eval_envs here, and no more seeding during evaluation
    envs = make_vec_env(args.env_id, args.num_envs, args.seed, args.control_mode, args.image_size)
    if args.rew_norm:
        envs = gym.wrappers.NormalizeReward(envs, args.gamma)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # agent setup
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # ALGO Logic: Storage setup
    # each obs is like {'image': {'rgb': (B,H,W,6), 'depth': (B,H,W,2)}, 'state': (B,D)}
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = to_tensor(next_obs)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = int(np.ceil(args.total_timesteps / args.num_steps_per_collect))
    result = defaultdict(list)
    timer = NonOverlappingTimeProfiler()

    for update in range(1, num_updates + 1):
        print('== Epoch:', update)

        # Needed to account for value bootstraping on timeout (which is truncated)
        # This fix is from https://github.com/vwxyzjn/cleanrl/pull/204
        # for detailed explanation see:
        # https://github.com/DLR-RM/stable-baselines3/issues/633
        # https://github.com/DLR-RM/stable-baselines3/pull/658
        # Here we will create separate buffer to not change original rewards or values, as they can be used in other places
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        agent.eval()
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, truncations, terminations, infos = envs.step(action.cpu().numpy())
            done = truncations | terminations
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            # Handle `final_observation` when we need to boostrap
            if args.bootstrap_at_done != 'never':
                if args.bootstrap_at_done == 'always':
                    need_final_obs = truncations | terminations # always need final obs when episode ends
                else: # bootstrap at truncated
                    need_final_obs = truncations & (~terminations) # only need final obs when truncated and not terminated
                for idx, _need_final_obs in enumerate(need_final_obs):
                    if _need_final_obs:
                        final_obs = unsqueeze(to_tensor(infos["final_observation"][idx]), dim=0)
                        with torch.no_grad():
                            final_value = agent.get_value(final_obs)
                        final_values[step, idx] = final_value.item()

            next_obs= to_tensor(next_obs)
            next_done = torch.Tensor(done).to(device)

            result = collect_episode_info(infos, result)
        
        timer.end('collect')

        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.num_steps_per_collect)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.num_steps_per_collect, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                
                if args.target_kl is not None and approx_kl > args.target_kl: # break asap
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                if args.critic_warmup_updates > 0:
                    loss = v_loss * args.vf_coef
                    args.critic_warmup_updates -= 1
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl: # break the outer loop
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        timer.end('train')

        # Log training-related data
        if (global_step - args.num_steps_per_collect) // args.log_freq < global_step // args.log_freq:
            if len(result['return']) > 0:
                for k, v in result.items():
                    writer.add_scalar(f"train/{k}", np.mean(v), global_step)
                result = defaultdict(list)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("losses/update_epochs", epoch + 1, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            timer.dump_to_writer(writer, global_step)

        # Evaluation
        if (global_step - args.num_steps_per_collect) // args.eval_freq < global_step // args.eval_freq:
            result = evaluate(args.num_eval_episodes, agent, eval_envs, device)
            for k, v in result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), global_step)
            timer.end('eval')
        
        # Checkpoint
        if args.save_freq and ( update == num_updates or \
                (global_step - args.num_steps_per_collect) // args.save_freq < global_step // args.save_freq):
            os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
            torch.save({
                'agent': agent.state_dict(),
            }, f'{log_path}/checkpoints/{global_step}.pt')

    envs.close()
    writer.close()
