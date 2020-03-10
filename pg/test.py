from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import os
import sys
import time
import gym

sys.path.append("..")
import MalmoPython
path = os.path.abspath(MalmoPython.__file__)
import mcenv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
print("PyTorch:\t{}".format(torch.__version__))

import csv

env = gym.make('MinecraftEnv-v0')
env.init(client_pool=[("localhost", 10000)], start_minecraft=False, 
		 allowDiscreteMovement=["move", "turn", "jump"], videoResolution=False)

# Hyperparameters
learning_rate = 0.7
gamma = 0.99

# Results write file 
f = open("results4.csv", mode="w")
writer = csv.writer(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

writer.writerow([learning_rate])
writer.writerow([gamma])

class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
#		state_space = env.observation_space.shape[0]
		obs_shape = env.observation_space.shape
		state_space = (obs_shape[0] * obs_shape[1] * obs_shape[2])
		action_space = env.action_space.n
		num_hidden = 128

		self.l1 = nn.Linear(state_space, num_hidden, bias=False)
		self.l2 = nn.Linear(num_hidden, action_space, bias=False)

		# Overall reward and loss history
		self.reward_history = []
		self.loss_history = []
		self.reset()

	def reset(self):
		# Episode policy and reward history
		self.episode_actions = torch.Tensor([])
		self.episode_rewards = []

	def forward(self, x):
		model = torch.nn.Sequential(
			self.l1,
			nn.Dropout(p=0.5),
			nn.ReLU(),
			self.l2,
			nn.Softmax(dim=-1)
		)
		return model(x)


prev_action = -1
def predict(state):
	# Select an action (0 or 1) by running policy model
	# and choosing based on the probabilities in state
	state = torch.from_numpy(state.flatten()).type(torch.FloatTensor)
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

		# Calculate score to determine when the environment has been solved
		scores.append(time)
		mean_score = np.mean(scores[-100:])

		print("Episode {} Cumulative Reward = {}".format(episode, cum_reward))
		writer.writerow([cum_reward])

		if episode % 50 == 0:
			print('Episode {}\tAverage length (last 100 episodes): {:.2f}'.format(
				episode, mean_score))

#		if mean_score > env.spec.reward_threshold:
		if mean_score > 599:
			print("Solved after {} episodes! Running average is now {}. Last episode ran to {} time steps."
				  .format(episode, mean_score, time))
			break


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
train(episodes=10000)

# Close write file
f.close()

# number of episodes for rolling average
window = 50

fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
std = pd.Series(policy.reward_history).rolling(window).std()
ax1.plot(rolling_mean)
ax1.fill_between(range(len(policy.reward_history)), rolling_mean -
				 std, rolling_mean+std, color='orange', alpha=0.2)
ax1.set_title(
	'Episode Length Moving Average ({}-episode window)'.format(window))
ax1.set_xlabel('Episode')
ax1.set_ylabel('Episode Length')

ax2.plot(policy.reward_history)
ax2.set_title('Episode Length')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Episode Length')

fig.tight_layout(pad=2)
plt.show()



"""
class policy_estimator():
	def __init__(self, env):
		self.n_inputs = env.observation_space.shape[0]
		self.n_outputs = env.action_space.n
		
		# Define network
		self.network = nn.Sequential(
			nn.Linear(self.n_inputs, 16), 
			nn.ReLU(), 
			nn.Linear(16, self.n_outputs),
			nn.Softmax(dim=-1))
	
	def predict(self, state):
		action_probs = self.network(torch.FloatTensor(state))
		return action_probs

def discount_rewards(rewards, gamma=0.99):
	r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
	# Reverse the array direction for cumsum and then
	# revert back to the original order
	r = r[::-1].cumsum()[::-1]
	return r - r.mean()

def reinforce(env, policy_estimator, num_episodes=2000,
			  batch_size=10, gamma=0.99):
	# Set up lists to hold results
	total_rewards = []
	batch_rewards = []
	batch_actions = []
	batch_states = []
	batch_counter = 1
			
	# Define optimizer
	optimizer = optim.Adam(policy_estimator.network.parameters(), lr=0.01)
			
	action_space = np.arange(env.action_space.n)
	ep = 0
	while ep < num_episodes:
		s_0 = env.reset()
		states = []
		rewards = []
		actions = []
		done = False
		while done == False:
			# Get actions and convert to numpy array
			action_probs = policy_estimator.predict(s_0).detach().numpy()
			action = np.random.choice(action_space, p=action_probs)
			s_1, r, done, _ = env.step(action)
				
			states.append(s_0)
			rewards.append(r)
			actions.append(action)
			s_0 = s_1
					
			# If done, batch data
			if done:
				batch_rewards.extend(discount_rewards(rewards, gamma))
				batch_states.extend(states)
				batch_actions.extend(actions)
				batch_counter += 1
				total_rewards.append(sum(rewards))
					
				# If batch is complete, update network
				if batch_counter == batch_size:
					optimizer.zero_grad()
					state_tensor = torch.FloatTensor(batch_states)
					reward_tensor = torch.FloatTensor(batch_rewards)
					# Actions are used as indices, must be 
					# LongTensor
					action_tensor = torch.LongTensor(batch_actions)

					# Calculate loss
					logprob = torch.log(
						policy_estimator.predict(state_tensor))

					selected_logprobs = reward_tensor * \
						torch.index_select(logprob, 1, action_tensor).squeeze()
						# torch.gather(logprob, 1, 
						# action_tensor).squeeze()
					loss = -selected_logprobs.mean()
							
					# Calculate gradients
					loss.backward()
					# Apply gradients
					optimizer.step()
					
					batch_rewards = []
					batch_actions = []
					batch_states = []
					batch_counter = 1
						
				avg_rewards = np.mean(total_rewards[-100:])
				# Print running average
				print("\rEp: {} Average of last 100: {:.2f}\n".format(
					 ep + 1, avg_rewards), end="")
				ep += 1
						
	return total_rewards


env = gym.make('CartPole-v0')
policy_est = policy_estimator(env)
rewards = reinforce(env, policy_est)

plt.plot(rewards)
plt.show()

env = gym.make('MinecraftEnv-v0')
env.init(client_pool=[("localhost", 10000)], start_minecraft=False, allowDiscreteMovement=["move", "turn"], videoResolution=False)
policy_est = policy_estimator(env)
rewards = reinforce(env, policy_est)
"""

"""
env = gym.make('MinecraftEnv-v0')
env.init(client_pool=[("localhost", 10000)], start_minecraft=False, allowDiscreteMovement=["move", "turn"], videoResolution=False)
# env.configure(allowDiscreteMovement=["move", "turn"], log_level="INFO")

for _ in range(10):
	t = time.time()
	env.reset()
	t2 = time.time()
	print("Startup time:", t2 - t)
	done = False
	s = 0
	while not done:
		obs, reward, done, info = env.step(env.action_space.sample())
		env.render()
		#print "obs:", obs.shape
		#print "reward:", reward
		#print "done:", done
		#print "info", info
		s += 1
	t3 = time.time()
	print((t3 - t2), "seconds total,", s, "steps total,", s / (t3 - t2), "steps/second")

env.close()
"""

"""
if sys.version_info[0] == 2:
	sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
	import functools
	print = functools.partial(print, flush=True)


missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
			<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
			
			  <About>
				<Summary>Hello world!</Summary>
			  </About>
			  
			<ServerSection>
			  <ServerInitialConditions>
				<Time>
					<StartTime>1000</StartTime>
					<AllowPassageOfTime>false</AllowPassageOfTime>
				</Time>
				<Weather>clear</Weather>
			  </ServerInitialConditions>
			  <ServerHandlers>
				  <FileWorldGenerator src="/Users/richardhsu/Library/MalmoPlatform/Minecraft/run/saves/maze"/>
				  <ServerQuitFromTimeUp timeLimitMs="30000"/>
				  <ServerQuitWhenAnyAgentFinishes/>
				</ServerHandlers>
			  </ServerSection>
			  
			  <AgentSection mode="Survival">
				<Name>MalmoTutorialBot</Name>
				<AgentStart>
					<Placement x="205.300" y="70.00000" z="201.700" yaw="-90"/>
					<Inventory>
						<InventoryItem slot="0" type="diamond_pickaxe"/>
					</Inventory>
				</AgentStart>
				<AgentHandlers>
				  <ObservationFromFullStats/>
				  <ContinuousMovementCommands turnSpeedDegs="180"/>
				  <InventoryCommands/>
				  <AgentQuitFromReachingPosition>
					<Marker x="213.700" y="70.00000" z="197.300" tolerance="0.5" description="Goal_found"/>
				  </AgentQuitFromReachingPosition>
				</AgentHandlers>
			  </AgentSection>
			</Mission>'''

# Create default Malmo objects:

agent_host = MalmoPython.AgentHost()
try:
	agent_host.parse( sys.argv )
except RuntimeError as e:
	print('ERROR:',e)
	print(agent_host.getUsage())
	exit(1)
if agent_host.receivedArgument("help"):
	print(agent_host.getUsage())
	exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
	try:
		agent_host.startMission( my_mission, my_mission_record )
		break
	except RuntimeError as e:
		if retry == max_retries - 1:
			print("Error starting mission:",e)
			exit(1)
		else:
			time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
	print(".", end="")
	time.sleep(0.1)
	world_state = agent_host.getWorldState()
	for error in world_state.errors:
		print("Error:",error.text)

print()
print("Mission running ", end=' ')

# ADD YOUR CODE HERE
# TO GET YOUR AGENT TO THE DIAMOND BLOCK
agent_host.sendCommand("move 1")
agent_host.sendCommand("strafe -0.25")

# Loop until mission ends:
while world_state.is_mission_running:
	print(".", end="")
	time.sleep(0.1)
	world_state = agent_host.getWorldState()
	for error in world_state.errors:
		print("Error:",error.text)

print()
print("Mission ended")
# Mission has ended.
"""
