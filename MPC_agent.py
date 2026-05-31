import numpy as np
import copy
import torch
import math
from scipy.optimize import minimize
from tqdm import tqdm
import gymnasium as gym
import random

#dx = 10            # Distance step (m)
#dxc = 2                 # Distance step (m) for simulation of continuous system
Np = 20                              # Prediction horizon
Nc = 20                               # Control horizon
#sim_steps = int(xfinal/dx)  # Total simulation steps (20 km / 10 m) = 2000 steps

'''
def lander_dynamics(state, action):
    """
    Computes the continuous-time derivatives of the lander state.
    state: [x, y, vx, vy, theta, omega]
    action: [main_engine, side_engine] (Expected to be continuous [-1, 1])
    """
    # Unpack state
    x, y, vx, vy, theta, omega = state
    
    # Unpack actions
    a_main = action[0]
    a_side = action[1]
    
    # --- Physical Parameters (Tune these to perfectly match Box2D) ---
    MASS = 1.0           
    INERTIA = 0.5        
    GRAVITY = -10.0      
    MAIN_FORCE_MAX = 20.0 
    SIDE_FORCE_MAX = 5.0  
    SIDE_TORQUE_ARM = 0.5 # Distance from center of mass to side thrusters
    
    # --- Action Mapping ---
    # Gym main engine usually only fires if action > 0
    main_thrust = a_main * MAIN_FORCE_MAX if a_main > 0 else 0.0
    
    # Gym side engines: left fires if < -0.5, right fires if > 0.5 (approx mapping)
    # We will treat it continuously: negative = left thruster, positive = right thruster
    side_thrust = a_side * SIDE_FORCE_MAX 
    
    # --- Forces in Global Frame ---
    # Main engine pushes UP relative to lander
    # Side engines push LATERAL relative to lander
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    force_x = -main_thrust * sin_t + side_thrust * cos_t
    force_y = main_thrust * cos_t + side_thrust * sin_t + (MASS * GRAVITY)
    
    # --- Torque ---
    # Side engines create a moment around the center of mass
    torque = -side_thrust * SIDE_TORQUE_ARM 
    
    # --- Derivatives ---
    dx = vx
    dy = vy
    dvx = force_x / MASS
    dvy = force_y / MASS
    dtheta = omega
    domega = torque / INERTIA
    
    return np.array([dx, dy, dvx, dvy, dtheta, domega])

def rk4_step(state, action, dt=0.02):
    """
    Steps the physics model forward by dt using Runge-Kutta 4 integration.
    Gymnasium's default FPS is 50, so dt = 1/50 = 0.02 seconds.
    """
    # Calculate the 4 Runge-Kutta slopes
    k1 = lander_dynamics(state, action)
    k2 = lander_dynamics(state + 0.5 * dt * k1, action)
    k3 = lander_dynamics(state + 0.5 * dt * k2, action)
    k4 = lander_dynamics(state + dt * k3, action)
    
    # Compute the weighted average of the slopes
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return next_state

def predict_lunar_lander(gym_obs, action, dt=0.02):
    """
    Wrapper to make our RK4 math perfectly compatible with Gym's 8D state vector.
    gym_obs: [x, y, vx, vy, angle, v_angle, left_leg_contact, right_leg_contact]
    """
    # 1. Extract the 6 kinematic variables (ignore leg contacts for free-flight)
    kinematic_state = gym_obs[0:6]
    
    # 2. Run the math predictor
    next_kinematic = rk4_step(kinematic_state, action, dt)
    
    # 3. Rebuild the 8D Gym Observation
    # We assume leg contacts remain the same as the current step during free-flight
    next_gym_obs = np.zeros(8)
    next_gym_obs[0:6] = next_kinematic
    next_gym_obs[6] = gym_obs[6] 
    next_gym_obs[7] = gym_obs[7] 
    
    # 4. (Optional) Ground collision clamp
    # If the lander hits the ground (y <= 0), stop its downward velocity
    if next_gym_obs[1] <= 0:
        next_gym_obs[1] = 0.0
        next_gym_obs[3] = 0.0 # Zero out vy
        next_gym_obs[6] = 1.0 # Simulate leg contact
        next_gym_obs[7] = 1.0 
        
    return next_gym_obs
'''
def set_lunar_lander_state(predictor_env, start_state, env_seed):
    # 1. Standard reset to initialize the objects and seed
    # Make sure env_seed is a standard Python int
    predictor_env.reset(seed=int(env_seed))
    
    # 2. Access the completely unwrapped environment instance
    raw_env = predictor_env.unwrapped
    
    VIEWPORT_W = 600
    VIEWPORT_H = 400
    SCALE = 30.0
    FPS = 50
    
    # 3. Parse and scale values
    x_val = start_state[0] * (VIEWPORT_W / SCALE / 2)
    y_val = start_state[1] * (VIEWPORT_H / SCALE / 2) + (134 / SCALE)
    vx_val = start_state[2] * (FPS / SCALE)
    vy_val = start_state[3] * (FPS / SCALE)
    angle_val = -start_state[4]            
    v_angle_val = -start_state[5] * FPS    
    
    # 4. CRITICAL FIX: Cast every value explicitly to a native Python float()
    raw_env.lander.position = (float(x_val), float(y_val))
    raw_env.lander.linearVelocity = (float(vx_val), float(vy_val))
    raw_env.lander.angle = float(angle_val)
    raw_env.lander.angularVelocity = float(v_angle_val)
    
    # 5. Handle leg contacts safely (convert numpy booleans to pure Python bools)
    raw_env.legs[0].ground_contact = bool(start_state[6])
    raw_env.legs[1].ground_contact = bool(start_state[7])

# Cost function for MPC
def cost_function_LLdV3(predictor_env,env_seed,start_state, a,Qx,Qu,penalized_states,set_point,Nc, Np):
	num_inputs = len(a)//Nc
	a = a.reshape((Nc, num_inputs))
	set_lunar_lander_state(predictor_env, start_state,env_seed)
	state = start_state.copy()
	#print(state)
	#print("yep")
	J = 0
	for i in range(Np):
		a_c = a[i].copy() if(i<Nc) else a[Nc-1].copy()
		state_p = state[penalized_states] - set_point
		#print(state_p)
		state_p = state_p.reshape(-1,1)
		a_c = a_c.reshape(-1,1)
		#print(state_p)
		#print(penalized_states)
		J +=  (state_p.T @ Qx @ state_p + a_c.T @ Qu @ a_c).item()              # add penalty for control signal
		a_c = a_c.reshape(-1)
		state, reward, terminated, truncated, info = predictor_env.step(a_c)
		#state = predict_lunar_lander(state, a_c, dt=0.02)
	return J

# Nonlinear constraints for optimization
def nlcon_LLdV3(predictor_env,env_seed,start_state, a,states_constrained,state_constrains,Nc, Np):
	c = []

	num_inputs = len(a)//Nc
	a = a.reshape((Nc, num_inputs))
	set_lunar_lander_state(predictor_env, start_state,env_seed)
	state = start_state.copy()

	# Np is the prediction horizon
	for i in range(Np):
		a_c = a[i] if(i<Nc) else a[Nc-1]
		violations = []
		for j in range(states_constrained.shape[0]):
			state_val = state[states_constrained[j]]
			min_violation = state_constrains[j,0] - state_val
			max_violation = state_val - state_constrains[j,1]
			violations.append(min_violation)
			violations.append(max_violation)
		c.extend(violations.copy())  # add constraints
		state, reward, terminated, truncated, info = predictor_env.step(a_c) # simulate the system and get the next state
		#state = predict_lunar_lander(state, a_c, dt=0.02)
	return np.array(c)



class MPC_expert_agent(object):

	def __init__(self, EnvName, cost_function, nlcon, **kwargs):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)

		# Choose distribution for the actor
		self.Np = Np
		self.Nc = Nc

		self.predictor_env = gym.make(EnvName)
		self.env_seed = [i + self.seed for i in range(self.num_envs)]
		self.cost_function = cost_function
		self.nlcon = nlcon
		
		#defining Costs
		self.Qx = np.eye(N=self.state_dim,M=None,k=0,dtype=np.float32) if(self.EnvIdex != 1) else np.eye(N=self.state_dim-2,M=None,k=0,dtype=np.float32)
		self.Qu = np.eye(N=self.action_dim,M=None,k=0,dtype=np.float32)
		
		self.u_lb = np.array(self.min_action)
		self.u_ub = np.array(self.max_action)

		self.penalized_states = np.array([0,1,2,3,4,5],dtype=np.int32)
		self.set_point = np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0],dtype=np.float32)
		self.states_constrained = np.array([2,3,5],dtype=np.int32)
		self.state_constrains = np.array([[-1.5,1.5],[-0.5,0.5],[-np.pi,np.pi]],dtype=np.float32)

		# Build Trajectory holder
		self.obs_hoder = np.zeros((self.T_horizon, self.num_envs,self.state_dim),dtype=np.float32)
		self.action_hoder = np.zeros((self.T_horizon, self.num_envs,self.action_dim),dtype=np.float32)
		self.next_obs_hoder = np.zeros((self.T_horizon, self.num_envs, self.state_dim),dtype=np.float32)
		self.done_hoder = np.zeros((self.T_horizon, self.num_envs, 1),dtype=np.bool_)
		self.termination_hoder = np.zeros((self.T_horizon, self.num_envs, 1),dtype=np.bool_)

	def set_env_seed(self,nenvs,seed):
		self.env_seed = [i + seed for i in range(nenvs)]
		self.num_envs = nenvs


	def select_action(self, state, deterministic):
		#with torch.no_grad():
			
		assert (deterministic), "Actions can only be deterministic!"
		
		if deterministic:
			# only used when evaluate the policy.Making the performance more stable
			action = self.deterministic_act(state)
			#return action.cpu().numpy()[0], None  # action is in shape (adim, 0)
			return action, None # (num_envs, action_dim)

	def deterministic_act(self, states):

		# Solve MPC optimization problem
		
		actions_to_apply = np.zeros(shape=(self.num_envs, self.action_dim),dtype=np.float32)

		a_init = np.zeros(self.Nc*self.action_dim)                           # initial guess for control signal
		bounds = list(zip(self.u_lb, self.u_ub)) * self.Nc
		solver_options = {'maxiter': 15}
		#print(self.num_envs)
		for i in range(self.num_envs):
			#print(states)
			obs = states[i]
			#print(obs)
			seed = self.env_seed[i]
			cons = {'type': 'ineq', 'fun': lambda a: -self.nlcon(self.predictor_env,seed,obs, a,self.states_constrained,self.state_constrains,self.Nc, self.Np)}  # nonlinear constraints
			result = minimize(lambda a: self.cost_function(self.predictor_env,seed,obs, a,self.Qx,self.Qu,self.penalized_states,self.set_point,self.Nc, self.Np), a_init, method='SLSQP', bounds=bounds, constraints=cons, options=solver_options) # optimization
			a_matrix = result.x.reshape((Nc, self.action_dim))
			actions_to_apply[i] = a_matrix[0]
			#a_array.append(a)       # only store the first control signal

		return actions_to_apply

	def put_data(self, obs, action, next_obs, done, termination, idx):

		self.obs_hoder[idx] = obs
		self.action_hoder[idx] = action
		self.next_obs_hoder[idx] = next_obs

		done = done.reshape(-1,1)
		termination = termination.reshape(-1,1)
		self.done_hoder[idx] = done
		self.termination_hoder[idx] = termination
	

	#def load(self,EnvName, timestep=None):
	#	if timestep != None:
	#		self.actor.load_state_dict(torch.load("./model/{}_MPC_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
	#	else:
	#		self.actor.load_state_dict(torch.load("./model/{}_MPC_actor.pth".format(EnvName), map_location=self.dvc))
			
'''
class MPC_expert(object):

	def __init__(self, rk4_update, cost_function, nlcon, **kwargs):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)

		# Choose distribution for the actor
		self.rk4_update = rk4_update
		self.cost_function = cost_function
		self.nlcon = nlcon
		#self.actor = 

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
			self.actor.load_state_dict(torch.load("./model/{}_MPC_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
		else:
			self.actor.load_state_dict(torch.load("./model/{}_MPC_actor.pth".format(EnvName), map_location=self.dvc))
'''
def evaluate_mpc(env, agent, episodes_num,seed_number=None,e_seed=None):
	total_scores = 0
	for j in range(episodes_num):
		
		obs, info = env.reset() if (seed_number==None) else (env.reset()if (e_seed==None) else env.reset(seed=random.randint(e_seed,e_seed+seed_number-1)))
		selected_seed = env.np_random_seed
		agent.set_env_seed(1,selected_seed)
		done = False
		while not done:
			action, logprob_a = agent.select_action(obs[None,:], deterministic=True) # Take deterministic actions when evaluation
			next_obs, reward, termination, truncation, info = env.step(action[0])
			done = (termination or truncation)
			#print("done")
			total_scores += reward
			obs = next_obs

	return total_scores/episodes_num