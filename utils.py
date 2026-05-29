import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import random



class BetaActor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(BetaActor, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.alpha_head = nn.Linear(net_width, action_dim)
		self.beta_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))

		alpha = F.softplus(self.alpha_head(a)) + 1.0
		beta = F.softplus(self.beta_head(a)) + 1.0

		return alpha,beta

	def get_dist(self,state):
		alpha,beta = self.forward(state)
		dist = Beta(alpha, beta)
		return dist

	def deterministic_act(self, state):
		alpha, beta = self.forward(state)
		mode = (alpha) / (alpha + beta)
		return mode

class RNNBetaActor(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_size,num_layers=2):
		super(RNNBetaActor, self).__init__()

        # GRU layer
		self.gru = nn.GRU(input_size=state_dim+action_dim,
		                  hidden_size=hidden_size,
		                  num_layers=num_layers,
		                  batch_first=False)
        
        # Output layer
		self.alpha_head = nn.Linear(hidden_size, action_dim)
		self.beta_head = nn.Linear(hidden_size, action_dim)

	def forward(self, state_past_action,last_hidden):
		gru_out, h_n = self.gru(input=state_past_action,hx=last_hidden)
		# Take output from last time step
		last_output = gru_out[:, :, :]

		alpha = F.softplus(self.alpha_head(last_output)) + 1.0
		beta = F.softplus(self.beta_head(last_output)) + 1.0

		return alpha,beta,h_n

	def get_dist(self,state_past_action,last_hidden):
		alpha,beta,hidden_state = self.forward(state_past_action, last_hidden)
		dist = Beta(alpha, beta)
		return dist, hidden_state

	def deterministic_act(self, state_past_action, last_hidden):
		alpha, beta, hidden_state = self.forward(state_past_action, last_hidden)
		mode = (alpha) / (alpha + beta)
		return mode, hidden_state
	

class GaussianActor_musigma(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(GaussianActor_musigma, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.sigma_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		sigma = F.softplus( self.sigma_head(a) ) + 1e-6
		return mu,sigma

	def get_dist(self, state):
		mu,sigma = self.forward(state)
		dist = Normal(mu,sigma)
		return dist

	def deterministic_act(self, state):
		mu, sigma = self.forward(state)
		return mu

class RNNGaussianActor_musigma(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_size,num_layers=2):
		super(RNNGaussianActor_musigma, self).__init__()

        # GRU layer
		self.gru = nn.GRU(input_size=state_dim+action_dim,
		                  hidden_size=hidden_size,
		                  num_layers=num_layers,
		                  batch_first=False)


		self.mu_head = nn.Linear(hidden_size, action_dim)
		self.sigma_head = nn.Linear(hidden_size, action_dim)

	def forward(self, state_past_action,last_hidden):
		gru_out, h_n = self.gru(input=state_past_action,hx=last_hidden)
		# Take output from last time step
		last_output = gru_out[:, :, :]

		mu = torch.sigmoid(self.mu_head(last_output))
		sigma = F.softplus( self.sigma_head(last_output) ) + 1e-6
		return mu,sigma,h_n

	def get_dist(self, state_past_action,last_hidden):
		mu,sigma,hidden_state = self.forward(state_past_action, last_hidden)
		dist = Normal(mu,sigma)
		return dist, hidden_state

	def deterministic_act(self, state_past_action, last_hidden):
		mu, sigma, hidden_state = self.forward(state_past_action, last_hidden)
		return mu, hidden_state

class GaussianActor_mu(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, log_std=0):
		super(GaussianActor_mu, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.mu_head.weight.data.mul_(0.1)
		self.mu_head.bias.data.mul_(0.0)

		self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

	def forward(self, state):
		a = torch.relu(self.l1(state))
		a = torch.relu(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		return mu

	def get_dist(self,state):
		mu = self.forward(state)
		action_log_std = self.action_log_std.expand_as(mu)
		action_std = torch.exp(action_log_std)

		dist = Normal(mu, action_std)
		return dist

	def deterministic_act(self, state):
		return self.forward(state)

class RNNGaussianActor_mu(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_size,num_layers=2, log_std=0):
		super(RNNGaussianActor_mu, self).__init__()

        # GRU layer
		self.gru = nn.GRU(input_size=state_dim+action_dim,
		                  hidden_size=hidden_size,
		                  num_layers=num_layers,
		                  batch_first=False)


		self.mu_head = nn.Linear(hidden_size, action_dim)
		self.mu_head.weight.data.mul_(0.1)
		self.mu_head.bias.data.mul_(0.0)

		self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

	def forward(self, state_past_action,last_hidden):
		gru_out, h_n = self.gru(input=state_past_action,hx=last_hidden)
		# Take output from last time step
		last_output = gru_out[:, :, :]

		mu = torch.sigmoid(self.mu_head(last_output))
		return mu, h_n

	def get_dist(self, state_past_action,last_hidden):
		mu, hidden_state = self.forward(state_past_action,last_hidden)
		action_log_std = self.action_log_std.expand_as(mu)
		action_std = torch.exp(action_log_std)

		dist = Normal(mu, action_std)
		return dist, hidden_state

	def deterministic_act(self, state_past_action,last_hidden):
		return self.forward(state_past_action,last_hidden)


class Critic(nn.Module):
	def __init__(self, state_dim,net_width):
		super(Critic, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, net_width)
		self.C3 = nn.Linear(net_width, 1)

	def forward(self, state):
		v = torch.tanh(self.C1(state))
		v = torch.tanh(self.C2(v))
		v = self.C3(v)
		return v

def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		print('Wrong Input.')
		raise


def Action_adapter(a,min_action,amplitude_action):
	#from [0,1] to [-max,max]
	#return  (2*(a-0.5))*amplitude_action
	return  a*amplitude_action+min_action

def Reward_adapter(r, EnvIdex):
	# For BipedalWalker
	if EnvIdex == 0 or EnvIdex == 1:
		if r <= -100: r = -1
	# For Pendulum-v0
	elif EnvIdex == 3:
		r = (r + 8) / 8
	return r

def evaluate_policy(env, agent,device, min_action,amplitude_action, episodes_num,seed_number=None,e_seed=None):
	total_scores = 0
	for j in range(episodes_num):
		
		obs, info = env.reset() if (seed_number==None) else (env.reset()if (e_seed==None) else env.reset(seed=random.randint(e_seed,e_seed+seed_number-1)))
		done = False
		while not done:
			action, logprob_a = agent.select_action(obs, deterministic=True) # Take deterministic actions when evaluation
			act = Action_adapter(action, min_action,amplitude_action)  # [0,1] to [-max,max]
			next_obs, reward, termination, truncation, info = env.step(act)
			done = (termination or truncation)

			total_scores += reward
			obs = next_obs

	return total_scores/episodes_num

import numpy as np

def evaluate_policy_rnn(env, agent,device, min_action,amplitude_action, episodes_num,first_p_act_dim,h_0_dim,seed_number=None,e_seed=None):
	total_scores = 0
	for j in range(episodes_num):
		
		last_hidden_state=np.zeros(shape=(h_0_dim[0],1,h_0_dim[2]),dtype=np.float32)
		past_action=np.full(fill_value=0.5,shape=(1,first_p_act_dim[1]),dtype=np.float32)

		obs, info = env.reset() if (seed_number==None) else (env.reset()if (e_seed==None) else env.reset(seed=random.randint(e_seed,e_seed+seed_number-1)))
		obs = obs[None, :]
		done = False
		while not done:
			state_past_action = np.concatenate([obs,past_action], axis=-1)
			action_v, logprob_a, hidden_state = agent.select_action(state_past_action,last_hidden_state, deterministic=True) # Take deterministic actions when evaluation
			action = action_v[-1,:,:]
			act = Action_adapter(action, min_action,amplitude_action)  # [0,1] to [-max,max]
			next_obs, reward, termination, truncation, info = env.step(act[0,:])
			done = (termination or truncation)

			total_scores += reward
			obs = next_obs[None, :]
			past_action = action
			last_hidden_state = hidden_state

	return total_scores/episodes_num