# Imports for Malmo
from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import os
import sys
import time
import gym

# Imports for minecraft gym
sys.path.append("..")
import MalmoPython
path = os.path.abspath(MalmoPython.__file__)
import mcenv

# Imports for pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
print("PyTorch:\t{}".format(torch.__version__))

# Other imports
import csv
from collections import namedtuple
from itertools import count
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import random


# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.03
GAMMA = 0.99
REWARD_THRESH = 85
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
NUM_EPISODES = 3000

# Results write file 
FILE_NAME = "results2.csv"
f = open(FILE_NAME, mode="w")
writer = csv.writer(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

# Write hyperparamters info to file
writer.writerow([learning_rate])
writer.writerow([GAMMA])
writer.writerow([REWARD_THRESH])

# Define ReplayMemory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 
						'reward'))


class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


n_actions = 0
class DQN(nn.Module):

	def __init__(self):
		super(DQN, self).__init__()
		# state_space = env.observation_space.shape[0]
		obs_shape = env.observation_space.shape
		# TODO: HERE
#		state_space = (obs_shape[0] * obs_shape[1] * obs_shape[2])
		state_space = (obs_shape[0] * obs_shape[1])
		action_space = env.action_space.n

		global n_actions
		n_actions = action_space

		# ----- DQN ----- 
		self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)

		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		def conv2d_size_out(size, kernel_size = 5, stride = 2):
			return (size - (kernel_size - 1) - 1) // stride  + 1

		w = obs_shape[1]
		h = obs_shape[0]
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		linear_input_size = convw * convh * 32
		self.head = nn.Linear(linear_input_size, action_space)

		"""
		self.sm1 = nn.Softmax(dim=-1)
		# Overall reward and loss history
		self.reward_history = []
		self.loss_history = []
		self.reset()

	def reset(self):
		# Episode policy and reward history
		self.episode_actions = torch.Tensor([])
		self.episode_rewards = []
		"""

	def forward(self, x):
		# ----- DQN ----- 
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))

		x = self.head(x.view(x.size(0), -1))
		# return self.sm1(x) 
		return x

"""
prev_action = -1
def predict(state):
	# Select an action (0 or 1) by running policy model
	# and choosing based on the probabilities in state
#	state = torch.from_numpy(state.flatten()).type(torch.FloatTensor)
	state = torch.from_numpy(state).type(torch.FloatTensor)
	action_probs = policy(state)
	distribution = Categorical(action_probs)

	global prev_action 
	if prev_action == 4: 
		action = torch.tensor(0)
	else:
		action = distribution.sample()
	prev_action = action

	# Add log probability of our chosen action to our history
	policy.episode_actions = torch.cat([
		policy.episode_actions,
		distribution.log_prob(action).reshape(1)
	])

	return action


def update_policy():
	R = 0
	rewards = []

	# Discount future rewards back to the present using gamma
	for r in policy.episode_rewards[::-1]:
		R = r + gamma * R
		rewards.insert(0, R)

	# Scale rewards
	rewards = torch.FloatTensor(rewards)
	rewards = (rewards - rewards.mean()) / \
		(rewards.std() + np.finfo(np.float32).eps)

	# Calculate loss
	loss = (torch.sum(torch.mul(policy.episode_actions, rewards).mul(-1), -1))

	# Update network weights
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	# Save and intialize episode history counters
	policy.loss_history.append(loss.item())
	policy.reward_history.append(np.sum(policy.episode_rewards))
	policy.reset()


def train(episodes):
	scores = []
	for episode in range(episodes):
		# Reset environment and record the starting state
		state = env.reset()
		cum_reward = 0

		for time in range(1000):
			# Transform state shape from (H, W, C) to (B, C, W, H)
			state = np.moveaxis(state, 2, 0)
			state = state[np.newaxis, :]

			action = predict(state)

			# Uncomment to render the visual state in a window
			# env.render()

			# Step through environment using chosen action
			state, reward, done, _ = env.step(action.item())
			cum_reward += reward

			# Save reward
			policy.episode_rewards.append(reward)

			if done:
				break

		update_policy()

		if episode % 50 == 0:
			print('Episode {}\tAverage length (last 100 episodes): {:.2f}' \
				  .format(episode, cum_reward))

		if cum_reward > REWARD_THRESH:
			print("Average cumulative reward is greater than {}".format(
				  REWARD_THRESH))
			break
"""

# Training functions
steps_done = 0
prev_action = -1

def select_action(state):
	global prev_action 
	global steps_done

	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
		math.exp(-1. * steps_done / EPS_DECAY)
	steps_done += 1

	if prev_action == 4: 
		action = torch.tensor(0).view(1, 1)
	elif sample > eps_threshold:
		with torch.no_grad():
			# t.max(1) will return largest column value of each row.
			# second column on max result is index of where max element was
			# found, so we pick action with the larger expected reward.
			action = policy_net(state).max(1)[1].view(1, 1)
	else:
		action = torch.tensor([[random.randrange(n_actions)]], device=device, 
							  dtype=torch.long)

	prev_action = action
	return action


def optimize_model():
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	# detailed explanation). This converts batch-array of Transitions
	# to Transition of batch-arrays.
	batch = Transition(*zip(*transitions))

	# Compute a mask of non-final states and concatenate the batch elements
	# (a final state would've been the one after which simulation ended)
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state
												if s is not None])
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken. These are the actions which would've been taken
	# for each batch state according to policy_net
	state_action_values = policy_net(state_batch).gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	# Expected values of actions for non_final_next_states are computed based
	# on the "older" target_net; selecting their best reward with max(1)[0].
	# This is merged based on the mask, such that we'll have either the expected
	# state value or 0 in case the state was final.
	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Compute Huber loss
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()


def train(num_episodes): 
	for i_episode in range(num_episodes):
		# Initialize the environment and state
		state = env.reset()
		state = screenToTensor(state)

		cum_reward = 0

		for t in count():
			# Select and perform an action
			action = select_action(state)
			next_state, reward, done, _ = env.step(action.item())
			reward = torch.tensor([reward], device=device).type(
				torch.FloatTensor)

			next_state = screenToTensor(next_state)
			cum_reward += reward

			# Store the transition in memory
			memory.push(state, action, next_state, reward)

			# Move to the next state
			state = next_state

			# Perform one step of the optimization (on the target network)
			optimize_model()
			if done:
				episode_durations.append(t + 1)
				# plot_durations()
				break

		print("Cumulative reward = {}".format(cum_reward))

		# Update the target network, copying all weights and biases in DQN
		if i_episode % TARGET_UPDATE == 0:
			target_net.load_state_dict(policy_net.state_dict())


# Utilities graphing function
episode_durations = []

def plot_durations():
	plt.figure(2)
	plt.clf()
	durations_t = torch.tensor(episode_durations, dtype=torch.float)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(durations_t.numpy())
	# Take 100 episode averages and plot them too
	if len(durations_t) >= 100:
		means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated


def screenToTensor(screen): 
	# Transform state shape from (H, W, C) to (C, W, H)
	result = np.moveaxis(screen, 2, 0)

	# Convert to float, rescale, convert to torch tensor
	result = np.ascontiguousarray(result, dtype=np.float32) / 255
	result = torch.from_numpy(result)

	# Resize, and add a batch dimension (B, C, H, W)
	return result.unsqueeze(0).to(device)


# Get minecraft gym environment
env = gym.make('MinecraftEnv-v0')
env.init(client_pool=[("localhost", 10000)], start_minecraft=False, 
		 allowDiscreteMovement=["move", "turn", "jump"], 
		 videoResolution=(360, 240), videoWithDepth=True)

# Set up training & utilities
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

# Start training
train(NUM_EPISODES)

"""
# Start training
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
train(episodes=5000)
"""

# Close write file
f.close()

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
