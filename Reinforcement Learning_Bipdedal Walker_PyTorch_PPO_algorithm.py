import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal
import time
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import pandas as pd

# ============ GROUP 2 PPO ALGORITHM ============


""" When running this code, the model will load our trained agent and continue the training loop.
    Average episode scores are printed to the terminal which will give a good indication as to the performance of
    the current model. Our agent consistently scores around 290, which suggests that it is stuck in a local minima.
    To train from scratch, comment out the line (355) which loads the pretrained model. """


# torch.autograd.set_detect_anomaly(True) # check for anomalies in the gradient update

# setup
version = 'GROUP_2_PPO_V8_FINAL'
env_name = "BipedalWalker-v3"

# Hyperparameters
gamma = 0.99  # Discount factor
gae_lambda = 0.95  # GAE lambda parameter
lr = 1e-4 # Learning rate for both actor and critic
eps_clip = 0.2  # Clipping epsilon for PPO
K_epochs = 10  # Number of epochs to update policy
update_timestep = 4000  # Timesteps to update the policy, maybe change to 6000?
mini_batch_size = 64  # Mini-batch size for PPO update
max_timesteps = 1e6  # Max timesteps to run the environment
log_std = -1  # Initial value for log_std (-0.5 = 0.6065, -1 = 0.3679)

render = False  # Render environment  # todo

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment Setup
env = gym.make(env_name, render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, log_std):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.__create_network()

        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std)

    def __create_network(self):
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        action_mean = self.network(state)
        return action_mean

    def act(self, state):
        action_mean = self.forward(state)
        cov_matrix = torch.diag_embed(self.log_std.exp())
        dist = MultivariateNormal(action_mean, cov_matrix)
        action = dist.sample()

        # Calculate log probability before clipping
        action_logprob = dist.log_prob(action)

        # Clip the actions to be within the range [-1, 1]
        # action = torch.clamp(action, -1, 1)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.forward(state)
        cov_matrix = torch.diag_embed(self.log_std.exp())
        dist = MultivariateNormal(action_mean, cov_matrix)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim

        self.__create_network()

    def __create_network(self):
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.network(state)


class PPO:

    def __init__(self, state_dim, action_dim, log_std):

        self.actor = Actor(state_dim, action_dim, log_std).to(device)
        self.critic = Critic(state_dim).to(device)

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=lr)

        self.actor_old = Actor(state_dim, action_dim, log_std).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):

        # Convert lists to tensors and move to device
        old_states = torch.stack(memory.states).to(device).detach()
        old_next_states = torch.stack(memory.next_states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Get state values from the critic
        state_values = self.critic(old_states).squeeze()
        next_state_values = self.critic(old_next_states).squeeze()

        # Handle terminal states (create a copy of next_state_values)
        adjusted_next_state_values = next_state_values.clone()
        for t in range(len(memory.is_terminals)):
            if memory.is_terminals[t]:
                adjusted_next_state_values[t] = 0  # If it's a terminal state, no next state value

        # Compute GAE for each episode separately
        advantages = []
        returns = []
        last_advantage = 0

        episode_start = 0
        for t in range(len(memory.is_terminals)):
            if memory.is_terminals[t] or t == len(memory.is_terminals) - 1:
                episode_end = t + 1
                episode_advantages = torch.zeros(episode_end - episode_start).to(device)
                for i in reversed(range(episode_start, episode_end)):
                    delta = memory.rewards[i] + gamma * adjusted_next_state_values[i] - state_values[i]
                    last_advantage = delta + gamma * gae_lambda * last_advantage
                    episode_advantages[i - episode_start] = last_advantage
                advantages.append(episode_advantages)
                returns.append(episode_advantages + state_values[episode_start:episode_end])
                episode_start = episode_end
                last_advantage = 0

        # Flatten the advantages and returns
        advantages = torch.cat(advantages)
        returns = torch.cat(returns)

        # Normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = advantages.detach()  # Important to detach after normalisation

        # PPO update with mini-batch
        for _ in range(K_epochs):
            # Generate a random permutation of indices
            shuffled_indices = torch.randperm(len(old_states))

            # Use the shuffled indices to shuffle the data
            old_states = old_states[shuffled_indices]
            old_actions = old_actions[shuffled_indices]
            old_logprobs = old_logprobs[shuffled_indices]
            returns = returns[shuffled_indices]
            advantages = advantages[shuffled_indices]

            # Then proceed with your mini-batch sampling
            for index in range(0, len(old_states), mini_batch_size):
                sampled_indices = slice(index, index + mini_batch_size)

                sampled_states = old_states[sampled_indices]
                sampled_actions = old_actions[sampled_indices]
                sampled_logprobs = old_logprobs[sampled_indices]
                sampled_returns = returns[sampled_indices]
                sampled_advantages = advantages[sampled_indices]

                # Recompute the logprobs and state_values for the mini-batch
                logprobs, dist_entropy = self.actor.evaluate(sampled_states, sampled_actions)
                state_values = self.critic(sampled_states).squeeze()

                # Find the ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(logprobs - sampled_logprobs.detach())

                # PPO loss
                surr1 = ratios * sampled_advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * sampled_advantages
                actor_loss = -torch.min(surr1, surr2)

                critic_loss = 0.5 * self.MseLoss(state_values,
                                                 sampled_returns.detach())  # Detach returns to avoid multiple backward passes

                # Total loss
                loss = actor_loss.mean() + critic_loss.mean() - 0.01 * dist_entropy.mean()
                # loss = actor_loss.mean() + critic_loss.mean()

                # Take optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Copy new weights into old actor policy
        self.actor_old.load_state_dict(self.actor.state_dict())

    def select_action(self, state, memory):

        state = torch.FloatTensor(state).to(device)
        action, action_logprob = self.actor_old.act(state)

        # Convert action to array and ensure it is the correct shape
        action = action.cpu().numpy().flatten()

        memory.states.append(state)
        memory.actions.append(torch.tensor(action).to(device))
        memory.logprobs.append(action_logprob)

        return action

    def save(self, checkpoint_path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, checkpoint_path)

    def load(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_old.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


class Memory:

    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

        self.rollout_reward = []
        self.rolllout_separator = []
        self.discounted_rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.rollout_reward[:]
        del self.rolllout_separator[:]
        del self.discounted_rewards[:]

    def discount_rewards(self):

        self.discounted_rewards = self.rewards.copy()

        for start, stop in self.rolllout_separator:

            G = 0  # Initialise the cumulative discounted reward

            for t in reversed(range(start, stop)):
                G = self.rewards[t] + gamma * G
                self.discounted_rewards[t] = G

        return self.discounted_rewards

    def __repr__(self):

        return (f'Memory('
                f'actions={len(self.actions)}, '
                f'states={len(self.states)}, '
                f'next_states={len(self.next_states)}, '
                f'logprobs={len(self.logprobs)}, '
                f'rewards={len(self.rewards)}, '
                f'is_terminals={len(self.is_terminals)})')


def test_model(env, agent, runs=100, record=True, video_folder='videos'):
    if record:
        env = RecordVideo(env, video_folder=video_folder,
                          name_prefix="bipedalwalker_episode")

    trajectory_rewards = []

    for i_episode in range(1, int(runs) + 1):

        state = np.array(env.reset()[0])
        state = torch.from_numpy(state)
        episode_reward = 0

        while True:

            action, _ = agent.act(state)

            next_state, reward, done, truncated, _ = env.step(action)

            memory.rewards.append(reward)
            memory.next_states.append(torch.FloatTensor(next_state).to(device))

            memory.is_terminals.append(done)

            episode_reward += reward

            if render:
                env.render()

            if done or truncated:
                trajectory_rewards.append(episode_reward)
                break
            else:
                state = np.array(next_state)
                state = torch.from_numpy(state)

        episode_reward += reward

    trajectory_rewards = np.array(trajectory_rewards)

    return trajectory_rewards.mean(), trajectory_rewards.std()


if __name__ == "__main__":

    memory = Memory()
    ppo = PPO(state_dim, action_dim, log_std)

    ppo.load(r'ppo_bipedalwalker_group_2.pth')

    # ===== UNCOMMENT THIS CODE TO TEST THE MODEL, TO RUN TRAINING LOOP ENSURE IT IS COMMENTED OUT AGAIN =====
    #
    # agent = ppo.actor
    # mean, std = test_model(env, agent, runs=100, record=False)
    # print(f'mean score: {mean}, std +/-: {std}')
    # exit()

    # ===== MAIN TRAINING LOOP =====

    print_running_reward = 0
    timestep = 0
    env = RecordEpisodeStatistics(env)
    master_list = []

    for i_episode in range(1, int(max_timesteps) + 1):

        state, info = env.reset()
        episode_reward = 0
        start = timestep

        rewards_list = []

        done, truncated = False, False

        for t in range(1, update_timestep + 1):

            timestep += 1

            action = ppo.select_action(state, memory)

            next_state, reward, done, truncated, _ = env.step(action)

            memory.rewards.append(reward)
            memory.next_states.append(torch.FloatTensor(next_state).to(device))

            memory.is_terminals.append(done)

            episode_reward += reward

            if render:
                env.render()

            if done or truncated:
                stop = timestep
                memory.rolllout_separator.append((start, stop))
                memory.rollout_reward.append(episode_reward)
                break
            else:
                state = next_state

        print_running_reward += episode_reward

        # Update PPO agent

        if timestep >= update_timestep:
            mem = memory.rollout_reward.copy()
            master_list.append(mem)
            ppo.update(memory)

            memory.clear_memory()
            timestep = 0
            er = np.array(mem)

            print(f'Episode {i_episode}, Avg reward: {er.mean():.2f}, STD: {er.std():.2f}.')

        # Save every 500 episodes
        if i_episode % 500 == 0:
            ppo.save(f"ppo_bipedalwalker_{version}_{i_episode}.pth")

            # temp_master = np.array(master_list)
            # print(temp_master)
            #
            # averages = np.mean(temp_master, axis=1)
            # std = np.std(temp_master, axis=1)
            #
            # print(averages, std, averages.shape, std.shape)
            #
            # df = pd.DataFrame({'mean scores': averages, 'std': std})
            # df.to_excel(f'model_scores_v{version}.xlsx', index=False)
            # del df

    env.close()
