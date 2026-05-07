from utils import BetaActor, GaussianActor_musigma, GaussianActor_mu, Critic
import numpy as np
import copy
import torch
import math


class PPO_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)

		# Choose distribution for the actor
		if self.Distribution == 'Beta':
			self.actor = BetaActor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_ms':
			self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_m':
			self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		else: print('Dist Error')
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		# Build Critic
		self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

		# Build Trajectory holder

		# Build Trajectory holder
		#self.obs_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		#self.action_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		#self.reward_hoder = np.zeros((self.T_horizon, 1),dtype=np.float32)
		#self.next_obs_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		#self.logprob_a_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		#self.done_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
		#self.termination_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
		
		self.obs_hoder = np.zeros((self.T_horizon, self.num_envs,self.state_dim),dtype=np.float32)
		self.action_hoder = np.zeros((self.T_horizon, self.num_envs,self.action_dim),dtype=np.float32)
		self.reward_hoder = np.zeros((self.T_horizon, self.num_envs, 1),dtype=np.float32)
		self.next_obs_hoder = np.zeros((self.T_horizon, self.num_envs, self.state_dim),dtype=np.float32)
		self.logprob_a_hoder = np.zeros((self.T_horizon, self.num_envs, self.action_dim),dtype=np.float32)
		self.done_hoder = np.zeros((self.T_horizon, self.num_envs, 1),dtype=np.bool_)
		self.termination_hoder = np.zeros((self.T_horizon, self.num_envs, 1),dtype=np.bool_)


	def select_action(self, state, deterministic):
		with torch.no_grad():
			#state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			state = torch.FloatTensor(state).to(self.dvc)
			if deterministic:
				# only used when evaluate the policy.Making the performance more stable
				action = self.actor.deterministic_act(state)
				#return action.cpu().numpy()[0], None  # action is in shape (adim, 0)
				return action.cpu().numpy(), None # (num_envs, action_dim)

			else:
				# only used when interact with the env
				dist = self.actor.get_dist(state)
				action = dist.sample()
				action = torch.clamp(action, 0, 1) #remove if needed
				#logprob_a = dist.log_prob(action).cpu().numpy().flatten()
				logprob_a = dist.log_prob(action).cpu().numpy()
				#return action.cpu().numpy()[0], logprob_a # both are in shape (adim, 0)
				return action.cpu().numpy(), logprob_a # both are in shape (num_envs, adim)


	def train(self, ):
		self.entropy_coef*=self.entropy_coef_decay

		'''Prepare PyTorch data from Numpy data'''
		obs = torch.from_numpy(self.obs_hoder).to(self.dvc)
		action = torch.from_numpy(self.action_hoder).to(self.dvc)
		reward = torch.from_numpy(self.reward_hoder).to(self.dvc)
		next_obs = torch.from_numpy(self.next_obs_hoder).to(self.dvc)
		logprob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
		done = torch.from_numpy(self.done_hoder).to(self.dvc)
		termination = torch.from_numpy(self.termination_hoder).to(self.dvc)

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(obs)
			vs_next = self.critic(next_obs)


			not_term = (~termination).float()
			not_done = (~done).float()

			'''dw for TD_target and Adv'''
			deltas = reward + self.gamma * vs_next * (not_term) - vs
			#deltas = deltas.cpu().flatten().numpy()

			T, N = deltas.shape[0], deltas.shape[1]

			#adv = [0]
			adv = torch.zeros_like(deltas)

			advantage = torch.zeros((N, 1), device=self.dvc)

			'''done for GAE'''
			#for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
			#	advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
			#	adv.append(advantage)
			#adv.reverse()
			#adv = copy.deepcopy(adv[0:-1])
			#adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)

			for t in reversed(range(T)):
				advantage = deltas[t] + self.gamma * self.lambd * not_done[t] * advantage
				adv[t] = advantage
			td_target = adv + vs
			adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps #all samples
			#or adv = (adv - adv.mean(dim=0, keepdim=True)) / (adv.std(dim=0, keepdim=True) + 1e-4) # per env normalization

		# from (T, N, dim) → (T*N, dim)
		obs = obs.reshape(-1, obs.shape[-1])
		action = action.reshape(-1, action.shape[-1])
		td_target = td_target.reshape(-1, 1)
		adv = adv.reshape(-1, 1)
		logprob_a = logprob_a.reshape(-1, logprob_a.shape[-1])

		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(obs.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(obs.shape[0] / self.c_optim_batch_size))

		for i in range(self.K_epochs):

			#Shuffle the trajectory, Good for training
			perm = np.arange(obs.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.dvc)
			#obs, action, td_target, adv, logprob_a = \
			#	obs[perm].clone(), action[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			obs = obs[perm]
			action = action[perm]
			td_target = td_target[perm]
			adv = adv[perm]
			logprob_a = logprob_a[perm]

			'''update the actor'''
			for i in range(a_optim_iter_num):
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, obs.shape[0]))
				distribution = self.actor.get_dist(obs[index])
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				logprob_a_now = distribution.log_prob(action[index])
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_gradient_norm)
				self.actor_optimizer.step()

			'''update the critic'''
			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, obs.shape[0]))
				c_loss = (self.critic(obs[index]) - td_target[index]).pow(2).mean()
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()

	def put_data(self, obs, action, reward, next_obs, logprob_a, done, termination, idx):

		self.obs_hoder[idx] = obs
		self.action_hoder[idx] = action

		reward = reward.reshape(-1, 1)
		self.reward_hoder[idx] = reward

		self.next_obs_hoder[idx] = next_obs
		self.logprob_a_hoder[idx] = logprob_a

		done = done.reshape(-1,1)
		termination = termination.reshape(-1,1)
		self.done_hoder[idx] = done
		self.termination_hoder[idx] = termination

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep=None):
		if timestep != None:
			self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
			self.critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))
		else:
			self.actor.load_state_dict(torch.load("./model/{}_actor.pth".format(EnvName), map_location=self.dvc))
			self.critic.load_state_dict(torch.load("./model/{}_q_critic.pth".format(EnvName), map_location=self.dvc))


class BC_PPO_agent(PPO_agent):

	def train(self,obs_ex,action_ex):
		self.entropy_coef*=self.entropy_coef_decay

		'''Prepare PyTorch data from Numpy data'''
		obs = torch.from_numpy(self.obs_hoder).to(self.dvc)
		action = torch.from_numpy(self.action_hoder).to(self.dvc)
		reward = torch.from_numpy(self.reward_hoder).to(self.dvc)
		next_obs = torch.from_numpy(self.next_obs_hoder).to(self.dvc)
		logprob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
		done = torch.from_numpy(self.done_hoder).to(self.dvc)
		termination = torch.from_numpy(self.termination_hoder).to(self.dvc)

		obs_expt = torch.from_numpy(obs_ex).to(self.dvc)
		action_expt = torch.from_numpy(action_ex).to(self.dvc)

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(obs)
			vs_next = self.critic(next_obs)


			not_term = (~termination).float()
			not_done = (~done).float()

			'''dw for TD_target and Adv'''
			deltas = reward + self.gamma * vs_next * (not_term) - vs
			#deltas = deltas.cpu().flatten().numpy()

			T, N = deltas.shape[0], deltas.shape[1]

			#adv = [0]
			adv = torch.zeros_like(deltas)

			advantage = torch.zeros((N, 1), device=self.dvc)

			'''done for GAE'''
			#for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
			#	advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
			#	adv.append(advantage)
			#adv.reverse()
			#adv = copy.deepcopy(adv[0:-1])
			#adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)

			for t in reversed(range(T)):
				advantage = deltas[t] + self.gamma * self.lambd * not_done[t] * advantage
				adv[t] = advantage
			td_target = adv + vs
			adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps #all samples
			#or adv = (adv - adv.mean(dim=0, keepdim=True)) / (adv.std(dim=0, keepdim=True) + 1e-4) # per env normalization

		# from (T, N, dim) → (T*N, dim)
		obs = obs.reshape(-1, obs.shape[-1])
		action = action.reshape(-1, action.shape[-1])
		td_target = td_target.reshape(-1, 1)
		adv = adv.reshape(-1, 1)
		logprob_a = logprob_a.reshape(-1, logprob_a.shape[-1])

		#expert
		obs_expt = obs_expt.reshape(-1, obs_expt.shape[-1])
		action_expt = action_expt.reshape(-1, action_expt.shape[-1])

		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(obs.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(obs.shape[0] / self.c_optim_batch_size))

		for i in range(self.K_epochs):

			#Shuffle the trajectory, Good for training
			perm = np.arange(obs.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.dvc)
			#obs, action, td_target, adv, logprob_a = \
			#	obs[perm].clone(), action[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			obs = obs[perm]
			action = action[perm]
			td_target = td_target[perm]
			adv = adv[perm]
			logprob_a = logprob_a[perm]

			#expert
			perm = np.arange(obs_expt.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.dvc)
			if obs_expt.shape[0] > self.T_horizon:
				obs_expt = obs_expt[perm[0:self.T_horizon]]
				action_expt = action_expt[perm[0:self.T_horizon]]
			else:
				obs_expt = obs_expt[perm]
				action_expt = action_expt[perm]

			'''update the actor'''
			for i in range(a_optim_iter_num):
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, obs.shape[0]))
				distribution = self.actor.get_dist(obs[index])
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				logprob_a_now = distribution.log_prob(action[index])
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				### BC predict actions ###
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, obs_expt.shape[0]))
				# MSE #
				pred_action = self.actor.deterministic_act(obs_expt[index])

				bc_loss = (action_expt[index] - pred_action).pow(2).mean()

				# Log Prob #

				#dist_e = self.actor.get_dist(obs_expt[index])
				#logprob_e = dist_e.log_prob(action_expt[index]).sum(dim=1, keepdim=True)
				#bc_loss = -self.bc_alpha * logprob_e.mean()
				######
				a_loss = (1.0-self.bc_alpha)*a_loss.mean() + bc_loss
				self.actor_optimizer.zero_grad()
				a_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_gradient_norm)
				self.actor_optimizer.step()

			'''update the critic'''
			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, obs.shape[0]))
				c_loss = (self.critic(obs[index]) - td_target[index]).pow(2).mean()
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()

		self.bc_alpha*=self.bc_alpha

class PPO_RNN_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)

		# Choose distribution for the actor
		if self.Distribution == 'Beta':
			self.actor = BetaActor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_ms':
			self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_m':
			self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		else: print('Dist Error')
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		# Build Critic
		self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

		# Build Trajectory holder

		# Build Trajectory holder
		#self.obs_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		#self.action_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		#self.reward_hoder = np.zeros((self.T_horizon, 1),dtype=np.float32)
		#self.next_obs_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		#self.logprob_a_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		#self.done_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
		#self.termination_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
		
		self.obs_hoder = np.zeros((self.T_horizon, self.num_envs,self.state_dim),dtype=np.float32)
		self.past_action_hoder = np.zeros((self.T_horizon, self.num_envs,self.action_dim),dtype=np.float32)
		self.action_hoder = np.zeros((self.T_horizon, self.num_envs,self.action_dim),dtype=np.float32)
		self.reward_hoder = np.zeros((self.T_horizon, self.num_envs, 1),dtype=np.float32)
		self.next_obs_hoder = np.zeros((self.T_horizon, self.num_envs, self.state_dim),dtype=np.float32)
		self.logprob_a_hoder = np.zeros((self.T_horizon, self.num_envs, self.action_dim),dtype=np.float32)
		self.done_hoder = np.zeros((self.T_horizon, self.num_envs, 1),dtype=np.bool_)
		self.termination_hoder = np.zeros((self.T_horizon, self.num_envs, 1),dtype=np.bool_)


	def select_action(self, state, deterministic):
		with torch.no_grad():
			#state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			state = torch.FloatTensor(state).to(self.dvc)
			if deterministic:
				# only used when evaluate the policy.Making the performance more stable
				action = self.actor.deterministic_act(state)
				#return action.cpu().numpy()[0], None  # action is in shape (adim, 0)
				return action.cpu().numpy(), None # (num_envs, action_dim)

			else:
				# only used when interact with the env
				dist = self.actor.get_dist(state)
				action = dist.sample()
				action = torch.clamp(action, 0, 1) #remove if needed
				#logprob_a = dist.log_prob(action).cpu().numpy().flatten()
				logprob_a = dist.log_prob(action).cpu().numpy()
				#return action.cpu().numpy()[0], logprob_a # both are in shape (adim, 0)
				return action.cpu().numpy(), logprob_a # both are in shape (num_envs, adim)


	def train(self, ):
		self.entropy_coef*=self.entropy_coef_decay

		'''Prepare PyTorch data from Numpy data'''
		obs = torch.from_numpy(self.obs_hoder).to(self.dvc)
		past_action = torch.from_numpy(self.past_action_hoder).to(self.dvc)
		action = torch.from_numpy(self.action_hoder).to(self.dvc)
		reward = torch.from_numpy(self.reward_hoder).to(self.dvc)
		next_obs = torch.from_numpy(self.next_obs_hoder).to(self.dvc)
		logprob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
		done = torch.from_numpy(self.done_hoder).to(self.dvc)
		termination = torch.from_numpy(self.termination_hoder).to(self.dvc)

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(obs)
			vs_next = self.critic(next_obs)


			not_term = (~termination).float()
			not_done = (~done).float()

			'''dw for TD_target and Adv'''
			deltas = reward + self.gamma * vs_next * (not_term) - vs
			#deltas = deltas.cpu().flatten().numpy()

			T, N = deltas.shape[0], deltas.shape[1]

			#adv = [0]
			adv = torch.zeros_like(deltas)

			advantage = torch.zeros((N, 1), device=self.dvc)

			'''done for GAE'''
			#for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
			#	advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
			#	adv.append(advantage)
			#adv.reverse()
			#adv = copy.deepcopy(adv[0:-1])
			#adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)

			for t in reversed(range(T)):
				advantage = deltas[t] + self.gamma * self.lambd * not_done[t] * advantage
				adv[t] = advantage
			td_target = adv + vs
			adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps #all samples
			#or adv = (adv - adv.mean(dim=0, keepdim=True)) / (adv.std(dim=0, keepdim=True) + 1e-4) # per env normalization

		# from (T, N, dim) → (T*N, dim)
		obs = obs.reshape(-1, obs.shape[-1])
		action = action.reshape(-1, action.shape[-1])
		td_target = td_target.reshape(-1, 1)
		adv = adv.reshape(-1, 1)
		logprob_a = logprob_a.reshape(-1, logprob_a.shape[-1])

		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(obs.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(obs.shape[0] / self.c_optim_batch_size))

		for i in range(self.K_epochs):

			#Shuffle the trajectory, Good for training
			perm = np.arange(obs.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.dvc)
			#obs, action, td_target, adv, logprob_a = \
			#	obs[perm].clone(), action[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			obs = obs[perm]
			action = action[perm]
			td_target = td_target[perm]
			adv = adv[perm]
			logprob_a = logprob_a[perm]

			'''update the actor'''
			for i in range(a_optim_iter_num):
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, obs.shape[0]))
				distribution = self.actor.get_dist(obs[index])
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				logprob_a_now = distribution.log_prob(action[index])
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_gradient_norm)
				self.actor_optimizer.step()

			'''update the critic'''
			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, obs.shape[0]))
				c_loss = (self.critic(obs[index]) - td_target[index]).pow(2).mean()
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()

	def put_data(self, obs, past_action, action, reward, next_obs, logprob_a, done, termination, idx):

		self.obs_hoder[idx] = obs
		self.past_action_hoder[idx] = past_action
		self.action_hoder[idx] = action

		reward = reward.reshape(-1, 1)
		self.reward_hoder[idx] = reward

		self.next_obs_hoder[idx] = next_obs
		self.logprob_a_hoder[idx] = logprob_a

		done = done.reshape(-1,1)
		termination = termination.reshape(-1,1)
		self.done_hoder[idx] = done
		self.termination_hoder[idx] = termination

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep=None):
		if timestep != None:
			self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
			self.critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))
		else:
			self.actor.load_state_dict(torch.load("./model/{}_actor.pth".format(EnvName), map_location=self.dvc))
			self.critic.load_state_dict(torch.load("./model/{}_q_critic.pth".format(EnvName), map_location=self.dvc))

class PPO_expert_agent(object):

	def __init__(self, **kwargs):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)

		# Choose distribution for the actor
		if self.Distribution == 'Beta':
			self.actor = BetaActor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_ms':
			self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_m':
			self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		else: print('Dist Error')
		#self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		# Build Critic
		self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
		#self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

		# Build Trajectory holder
		self.obs_hoder = np.zeros((self.T_horizon, self.num_envs,self.state_dim),dtype=np.float32)
		self.action_hoder = np.zeros((self.T_horizon, self.num_envs,self.action_dim),dtype=np.float32)
		self.next_obs_hoder = np.zeros((self.T_horizon, self.num_envs, self.state_dim),dtype=np.float32)
		self.done_hoder = np.zeros((self.T_horizon, self.num_envs, 1),dtype=np.bool_)
		self.termination_hoder = np.zeros((self.T_horizon, self.num_envs, 1),dtype=np.bool_)

	def select_action(self, state, deterministic):
		with torch.no_grad():
			#state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			state = torch.FloatTensor(state).to(self.dvc)
			if deterministic:
				# only used when evaluate the policy.Making the performance more stable
				action = self.actor.deterministic_act(state)
				#return action.cpu().numpy()[0], None  # action is in shape (adim, 0)
				return action.cpu().numpy(), None # (num_envs, action_dim)

			else:
				# only used when interact with the env
				dist = self.actor.get_dist(state)
				action = dist.sample()
				action = torch.clamp(action, 0, 1) #remove if needed
				#logprob_a = dist.log_prob(action).cpu().numpy().flatten()
				logprob_a = dist.log_prob(action).cpu().numpy()
				#return action.cpu().numpy()[0], logprob_a # both are in shape (adim, 0)
				return action.cpu().numpy(), logprob_a # both are in shape (num_envs, adim)
			
	def put_data(self, obs, action, next_obs, done, termination, idx):

		self.obs_hoder[idx] = obs
		self.action_hoder[idx] = action
		self.next_obs_hoder[idx] = next_obs

		done = done.reshape(-1,1)
		termination = termination.reshape(-1,1)
		self.done_hoder[idx] = done
		self.termination_hoder[idx] = termination


	def load(self,EnvName, timestep=None):
		if timestep != None:
			self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
			self.critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))
		else:
			self.actor.load_state_dict(torch.load("./model/{}_actor.pth".format(EnvName), map_location=self.dvc))
			self.critic.load_state_dict(torch.load("./model/{}_q_critic.pth".format(EnvName), map_location=self.dvc))