# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from env.quad.quad_rotor_still import QuadStillEnv
from lyapunov_scaling import LyapunovNet, ScalingMethod, certification_loss, lie_margin, lyapunov_loss, run_tag

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Pendulum-v1"
    """the environment id of the task"""
    total_timesteps: int = 100_000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # Lyapunov scaling arguments (thesis Chapter 3.1)
    scaling_method: ScalingMethod = "min_lie"
    """Lyapunov scaling method: none | min_lie | regularization | structural"""
    scaling_mu: float = 0.1
    """mu hyperparameter: minimum Lie derivative (min_lie) or regularization weight (regularization); unused otherwise"""
    eq_obs: Optional[Tuple[float, ...]] = None
    """equilibrium observation (defaults to all zeros)"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
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
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    tag = run_tag(args.scaling_method, args.scaling_mu)
    run_name = f"{args.env_id}__{args.exp_name}_{tag}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "data", args.env_id, "clean_lsac", tag, "seed_" + str(args.seed))
    os.makedirs(data_path, exist_ok=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    lyapunov = LyapunovNet(
        envs.single_observation_space.shape[0],
        envs.single_action_space.shape[0],
        structural=args.scaling_method == "structural",
    ).to(device)
    margin = lie_margin(args.scaling_method, args.scaling_mu)
    eq_state = torch.zeros((1, envs.single_observation_space.shape[0]), dtype=torch.float, device=device)
    if args.eq_obs is not None:
        eq_state = torch.tensor([args.eq_obs], dtype=torch.float, device=device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    dt = envs.envs[0].unwrapped.dt

    # Initialize log_beta as a leaf tensor
    log_beta = nn.Parameter(torch.log(torch.tensor([10.0], dtype=torch.float, device=device)))
    beta_optimizer = optim.Adam([log_beta], lr=3e-4)
    beta = torch.exp(log_beta.detach())

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    episode_rewards = []
    episode_steps = []
    episode_count = 0
    max_reward = -np.inf
    beta_arr = [beta.item()]

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        next_done = np.logical_or(terminations, truncations)

        for i, done in enumerate(next_done):
            if done:
                episode_rewards.append(infos["episode"]["r"][i])
                episode_steps.append((global_step + 1) * args.num_envs)
                episode_count += 1

                if episode_count % 50 == 0:
                    avg_reward = np.mean(episode_rewards[-50:])
                    print(f"Episodes Completed={episode_count}, Average Reward={avg_reward}, Global Step={(global_step + 1) * args.num_envs}, Time={time.time() - start_time:.0f}")
                    writer.add_scalar("charts/episode_return", avg_reward, episode_count)

                    if avg_reward > max_reward:
                        max_reward = avg_reward
                        if args.save_model:
                            torch.save(actor.state_dict(), os.path.join(data_path, f"{args.exp_name}.cleanrl_model.pth"))
                            torch.save(qf1.state_dict(), os.path.join(data_path, "qf1.pth"))
                            torch.save(qf2.state_dict(), os.path.join(data_path, "qf2.pth"))
                            torch.save(lyapunov.state_dict(), os.path.join(data_path, "lyapunov.pth"))
                            print("Saving model...")


        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         if info is not None:
        #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #             writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #             writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #             break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = obs[idx]
        rb.add(obs, real_next_obs, actions, rewards, next_done, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions.detach())
                qf2_next_target = qf2_target(data.next_observations, next_state_actions.detach())
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi.detach()
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            temp_time = time.time()
            # Update Lyapunov function
            with torch.no_grad():
                eq_mean, _ = actor.forward(eq_state)
                eq_action = torch.tanh(eq_mean) * actor.action_scale + actor.action_bias

            ly_eq = lyapunov(eq_state, eq_action, eq_state, eq_action)
            ly_val = lyapunov(data.observations, data.actions, eq_state, eq_action)
            ly_lie = (lyapunov(data.next_observations, next_state_actions, eq_state, eq_action) - ly_val) / dt

            ly_loss = lyapunov_loss(args.scaling_method, args.scaling_mu, ly_val, ly_lie, ly_eq, data.observations, eq_state)
            ly_cert_loss = certification_loss(ly_val, ly_lie, ly_eq).detach()

            lyapunov.optimizer.zero_grad()
            ly_loss.backward()
            lyapunov.optimizer.step()
            # print(time.time() - temp_time)
            # temp_time = time.time()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    
                    next_state_actions, _, _ = actor.get_action(data.next_observations)
                    ly_lie = (
                        lyapunov(data.next_observations, next_state_actions, eq_state, eq_action)
                        - lyapunov(data.observations, data.actions, eq_state, eq_action).detach()
                    ) / dt

                    actor_loss = ((alpha * log_pi) - min_qf_pi + beta.detach() * F.relu(ly_lie + margin)).mean()
                    # actor_loss = ((alpha * log_pi) - min_qf_pi).mean()


                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # print(time.time() - temp_time)
            # temp_time = time.time()

            log_beta_loss = -log_beta * (ly_lie.detach().mean())
            beta_optimizer.zero_grad()
            log_beta_loss.backward()
            beta_optimizer.step()

            beta = torch.exp(log_beta.detach())
            # print(time.time() - temp_time)
            # temp_time = time.time()
            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/lyapunov_loss", ly_loss.item(), global_step)
                writer.add_scalar("losses/lyapunov_cert_loss", ly_cert_loss.item(), global_step)
                writer.add_scalar("losses/beta", beta.item(), global_step)
                beta_arr.append(beta.item())
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if args.save_model:
        np.save(os.path.join(data_path, "returns.npy"), np.array(episode_rewards))
        np.save(os.path.join(data_path, "steps.npy"), np.array(episode_steps))
        np.save(os.path.join(data_path, "beta.npy"), np.array(beta_arr))
        print(f"Saved model to {data_path}")
        print(f"Best episodic return: {max_reward}")

    envs.close()
    writer.close()