import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datetime import datetime
from collections import deque


class Agent:
    """The agent that will interact with and learn from the environment"""

    def __init__(self, config):
        self.config = config  # Collect everything needed for the Agent class from the config object
        self.device = config.device

        print("DDPG Agent using: ", self.device)

        self.local_actor = config.actor_fn().to(self.device)
        if config.local_actor_saved_model is not None:
            self.local_actor.load_state_dict(torch.load(config.local_actor_saved_model))

        self.target_actor = config.actor_fn().to(self.device)
        self.actor_opt = config.actor_opt_fn(self.local_actor.parameters())

        self.local_critic = config.critic_fn().to(self.device)
        if config.local_critic_saved_model is not None:
            self.local_critic.load_state_dict(torch.load(config.local_critic_saved_model))

        self.target_critic = config.critic_fn().to(self.device)
        self.critic_opt = config.critic_opt_fn(self.local_critic.parameters())

        self.noise = config.noise_fn()
        self.buffer = config.replay_fn()
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.learn_every = config.learn_every
        self.learn_number = config.learn_number

    def step(self, state, action, reward, next_state, done_signal, step):
        self.buffer.add(state, action, reward, next_state, done_signal)

        if len(self.buffer) > self.buffer.batch_size and step % self.learn_every == 0:
            for _ in range(self.learn_number):
                self.learn()

    def act(self, state, add_noise=True):
        """
        Returns actions for given state as per current policy.
        :param state: (array_like) current state
        :param add_noise: (bool) Whether add noise to the action to be taken (set in the config file)
        :return: (float) The action to be taken by the agent, from a range of [-1, 1] (inclusive)
        """
        # Register the current state to PyTorch
        state = torch.from_numpy(state).float().to(self.device)
        self.local_actor.eval()

        # Freeze gradient calculation on the action to be taken, given the current state
        with torch.no_grad():
            action = self.local_actor(state).cpu().data.numpy()

        self.local_actor.train()

        if add_noise:
            action += self.noise.sample() * self.epsilon

        # Make sure that the action stays within the desired range of [-1, 1] by clipping it out
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self):
        states, actions, rewards, next_states, done_signals = self.buffer.sample()

        # ----------------------
        # Critic model updates -
        # ----------------------

        # Predict actions for the next states with the target actor model
        target_next_actions = self.target_actor(next_states)
        # Compute Q values for the next states and actions with the target critic model
        target_next_qs = self.target_critic(next_states, target_next_actions)
        # Compute target Q values for the current states using the Bellman equation
        """
        Q_targets = r + (gamma * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        target_qs = rewards + (self.config.discount * target_next_qs * (1 - done_signals))
        # Compute Q values for the current states and actions with the critic model
        local_qs = self.local_critic(states, actions)
        # Compute and minimize the critic loss
        critic_loss = F.mse_loss(local_qs, target_qs)  # TODO: Try out other loss metrics.
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), 1)
        self.critic_opt.step()

        # ---------------------
        # Actor model updates -
        # ---------------------

        # Predict actions for current states from the actor model
        online_actions = self.local_actor(states)
        # Compute and minimize the actor loss
        actor_loss = -self.local_critic(states, online_actions).mean()
        self.actor_opt.zero_grad()
        # Perform back-propagation
        actor_loss.backward()
        # Optimise network weights towards minimising the loss
        self.actor_opt.step()

        # Update target critic and actor models
        self.soft_update(self.local_critic, self.target_critic)
        self.soft_update(self.local_actor, self.target_actor)

        # Update epsilon and noise
        self.epsilon *= self.epsilon_decay
        self.noise.reset()

    def soft_update(self, local_model, target_model):
        """
        θ_target = τ*θ_local + (1 - τ)*θ_target
            where τ is the interpolation parameter (self.config.target_mix)
        :param local_model: (pytorch.model) The model that is being trained
        :param target_model: (pytorch.model) The network with frozen weights
        :return: None
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # Mix in local's network weight's into the frozen target network's weight by a small percentage
            target_param.data.copy_(self.config.target_mix * local_param.data + (1. - self.config.target_mix) *
                                    target_param.data)


def run(agent, print_every=10):
    config = agent.config

    scores_deque = deque(maxlen=100)
    scores = []
    steps_across_episodes = 0
    cur_ts = str(int(datetime.now().timestamp() * 1000))

    for episode in range(1, config.max_episodes + 1, 1):
        agent.reset()
        score = np.zeros(config.num_agents)

        env_info = config.env.reset(train_mode=True)[config.brain_name]
        states = env_info.vector_observations

        for step in range(config.max_steps):
            actions = agent.act(states)
            env_info = config.env.step(actions)[config.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            done_signals = env_info.local_done

            for state, action, reward, next_state, done_signal in zip(states, actions, rewards, next_states,
                                                                      done_signals):
                agent.step(state, action, reward, next_state, done_signal, steps_across_episodes)

            score += rewards
            steps_across_episodes += 1
            states = next_states

            if np.any(done_signals):
                # Terminal state reached. Stop running this episode and go to the next.
                break

        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        mean_score = np.mean(scores_deque)

        # Save score at every step, in case of failures..
        with open('scores_' + cur_ts + '.txt', 'w') as f:  # Replace content with new scores..
            for score in scores:
                f.write(str(score) + '\n')

        f.close()

        if steps_across_episodes % print_every == 0:
            print('\rEpisode No. {}\tAverage reward: '
                  '{:.2f}\tAverage reward over 100 episodes: {:.2f}'.format(episode, np.mean(score), mean_score))
            torch.save(agent.local_actor.state_dict(), 'local_actor_' + cur_ts + '.pth')
            torch.save(agent.local_critic.state_dict(), 'local_critic_' + cur_ts + '.pth')
            torch.save(agent.target_actor.state_dict(), 'target_actor_' + cur_ts + '.pth')
            torch.save(agent.target_critic.state_dict(), 'target_critic_' + cur_ts + '.pth')

        if config.goal_score <= mean_score:
            print(
                '\nEnvironment solved in {:d} '
                'episodes!\tAverage score over 100 episodes: {:.2f}'.format(episode - 100, mean_score))

            torch.save(agent.local_actor.state_dict(), config.local_actor_path)
            torch.save(agent.local_critic.state_dict(), config.local_critic_path)
            torch.save(agent.target_actor.state_dict(), config.target_actor_path)
            torch.save(agent.target_critic.state_dict(), config.target_critic_path)

            break

    return scores
