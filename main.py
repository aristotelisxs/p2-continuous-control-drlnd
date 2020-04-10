import argparse

from datetime import datetime
from torch import optim
from config import Config
from memory import ReplayBuffer
from unityagents import UnityEnvironment
from utils import OrnsteinUhlenbeck
from model.ddpg_agent import Agent as ddpg_agent
from model.ddpg_agent import run as ddpg_agent_run
from model.ddpg_actor import Actor as ddpg_actor
from model.ddpg_critic import Critic as ddpg_critic

# TODO: Implement changes as seen in the Continuous Control python notebook..

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', dest='epsilon',
                        help="A value between 0 and 1 to indicate the probability with which a random (exploratory) "
                             "or the best known action will be taken in any given state during agent training.",
                        type=float, required=False, default=1)
    parser.add_argument('--discount', dest='discount',
                        help="Discount rewards by a factor between 0 and 1.",
                        type=float, required=False, default=.99)
    parser.add_argument('--epsilon_decay', dest='epsilon_decay',
                        help="The rate at which epsilon will be reduced at each optimisation step",
                        type=float, required=False, default=.99)
    parser.add_argument('--target_mix', dest='target_mix',
                        help="How much of the local network's weights to 'mix in' with the target network's at each "
                             "time step",
                        type=float, required=False, default=1e-3)
    parser.add_argument('--max_episodes', dest='max_episodes',
                        help="Maximum episodes to run the agent training procedure for.",
                        type=int, required=False, default=int(2000))
    parser.add_argument('--max_steps', dest='max_steps',
                        help="Maximum steps that an agent can take within an episode before reaching the terminal "
                             "state",
                        type=int, required=False, default=int(1e6))
    parser.add_argument('--buffer_size', dest='buffer_size',
                        help="Maximum steps that an agent can take within an episode before reaching the terminal "
                             "state",
                        type=int, required=False, default=int(1e6))

    parser.add_argument('--seed', dest='seed',
                        help="Seed to reproduce results",
                        type=int, required=True)
    parser.add_argument('--reacher_fp', dest='reacher_fp',
                        help="The relative to the script's location or full file path of the folder containing the "
                             "Unity environment for the robotic arm control.",
                        type=str, required=False, default='Reacher_Windows_x86_64_20_agents/Reacher.exe')
    parser.add_argument('--fc1_units', dest='fc1_units',
                        help="The number of units for the first hidden layer of the fully connected inference network",
                        type=int, required=False, default=256)
    parser.add_argument('--fc2_units', dest='fc2_units',
                        help="The number of units for the second hidden layer of the fully connected inference network",
                        type=int, required=False, default=256)
    parser.add_argument('--lr_actor', dest='lr_actor',
                        help="The learning rate used for the network weights' update step",
                        type=int, required=False, default=1e-4)
    parser.add_argument('--lr_critic', dest='lr_critic',
                        help="The learning rate used for the network weights' update step",
                        type=int, required=False, default=1e-3)
    parser.add_argument('--batch_size', dest='batch_size',
                        help="Batch size to use for the neural network updates",
                        type=int, required=False, default=128)
    parser.add_argument('--learn_every', dest='learn_every',
                        help="After how many times should the network be allowed to update its network weights and "
                             "learn",
                        type=int, required=False, default=20)
    parser.add_argument('--learn_number', dest='learn_number',
                        help="How many times should the network be updated after `learn_every` steps",
                        type=int, required=False, default=10)

    args = parser.parse_args()

    config = Config(seed=args.seed)
    config.env = UnityEnvironment(file_name=args.reacher_fp)

    config.brain_name = config.env.brain_names[0]
    env_info = config.env.reset(train_mode=True)[config.brain_name]
    config.num_agents = len(env_info.agents)

    config.state_size = env_info.vector_observations.shape[1]
    config.action_size = config.env.brains[config.brain_name].vector_action_space_size

    config.actor_fn = lambda: ddpg_actor(config.state_size, config.action_size,
                                         fc1_units=args.fc1_units, fc2_units=args.fc2_units)
    config.actor_opt_fn = lambda params: optim.Adam(params, lr=args.lr_actor)

    config.critic_fn = lambda: ddpg_critic(config.state_size, config.action_size,
                                           fc1_units=args.fc1_units, fc2_units=args.fc2_units)
    config.critic_opt_fn = lambda params: optim.Adam(params, lr=args.lr_critic)

    config.replay_fn = lambda: ReplayBuffer(config.action_size, buffer_size=args.buffer_size, batch_size=args.batch_size,
                                            seed=args.seed, device=config.device
                                            )
    config.noise_fn = lambda: OrnsteinUhlenbeck(config.action_size, mu=0., theta=0.15, sigma=0.05)

    # Time specific outputs..
    cur_ts = str(int(datetime.now().timestamp() * 1000))
    config.actor_path = 'actor_' + cur_ts + '.pth'
    config.critic_path = 'critic_' + cur_ts + '.pth'
    config.scores_path = 'scores_' + cur_ts + '.png'

    # Update from command line arguments
    config.discount = args.discount
    config.target_mix = args.target_mix
    config.epsilon = args.epsilon
    config.epsilon_decay = args.epsilon_decay

    config.max_episodes = args.max_episodes
    config.max_steps = args.max_steps

    config.learn_every = args.learn_every
    config.learn_number = args.learn_number

    agent = ddpg_agent(config)
    ddpg_agent_run(agent)
