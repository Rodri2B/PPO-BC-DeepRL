from datetime import datetime
import os, shutil
import argparse
import torch
import gymnasium as gym

from utils import str2bool, Action_adapter, Reward_adapter, evaluate_policy
from PPO import PPO_agent, PPO_expert_agent,BC_PPO_agent

import numpy as np
import h5py
import random

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')

parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')

parser.add_argument('--num_envs', type=int, default=1, help='number of environments')
parser.add_argument('--evaluation_rollouts_num', type=int, default=1, help='number of evaluation rollouts')

parser.add_argument('--seed_number', type=int, default=None, help='number of random seeds')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--Distribution', type=str, default='Beta', help='Should be one of Beta ; GS_ms  ;  GS_m')
parser.add_argument('--eval_interval', type=int, default=int(5), help='Model evaluating interval, in steps.')

parser.add_argument('--bc_expert_model', type=str, default=None, help='Behaviour cloning expert model name')

parser.add_argument('--expert_traj', type=str, default=None, help='Expert trajectories main name')
parser.add_argument('--expert_traj_number', type=float, default=5, help='Number of expert trajectories')

opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def append_chunk(exp_traj,filename):
    # exp_traj shape: (batch_size, dim)
    with h5py.File(filename, "a") as f:
        if "data" not in f:
            # create dataset with unlimited rows
            maxshape = (None, exp_traj.shape[1])
            dset = f.create_dataset(
                "data",
                data=exp_traj,
                maxshape=maxshape,
                chunks=True,   # important for resizing
                dtype='float32'
            )
        else:
            dset = f["data"]
            old_size = dset.shape[0]
            new_size = old_size + exp_traj.shape[0]

            # resize and append
            dset.resize((new_size, exp_traj.shape[1]))
            dset[old_size:new_size] = exp_traj


def main():
    EnvName = ['Pendulum-v1','LunarLanderContinuous-v3','Humanoid-v5','HalfCheetah-v5','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']

    # Build Env
    envs = gym.make_vec(EnvName[opt.EnvIdex],num_envs=opt.num_envs, vectorization_mode="sync")
    env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
    opt.state_dim = envs.single_observation_space.shape[0]
    opt.action_dim = envs.single_action_space.shape[0]
    opt.max_action = envs.single_action_space.high#float(envs.action_space.high[0])
    opt.min_action = envs.single_action_space.low
    opt.amplitude_action = opt.max_action-opt.min_action
    opt.max_steps = opt.T_horizon#envs.spec.max_episode_steps if (envs.spec.max_episode_steps <= opt.T_horizon) else opt.T_horizon

    print('Env:',EnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,
          '  max_a:',opt.max_action,'  min_a:',opt.min_action, 'max_steps', opt.max_steps)

    device = opt.dvc
    env_min_action = np.array(opt.min_action, dtype=np.float32)
    env_amplitude_action = np.array(opt.amplitude_action, dtype=np.float32)
    #print(env_min_action.shape)
    # Seed Everything
    env_seed = opt.seed

    seed_number = opt.seed_number

    random.seed(opt.seed)
    #np.random.seed(seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))


    if not os.path.exists('model'): os.mkdir('model')
    expert = PPO_expert_agent(**vars(opt)) # transfer opt to dictionary, and use it to init PPO_agent
    assert opt.bc_expert_model != None, "Expert model path not specified!"
    expert.load(opt.bc_expert_model)

    assert opt.expert_traj != None, "Expert trajectories file name not specified!"
    if not os.path.exists('expert_traj'): os.mkdir('expert_traj')
    expt_traj_filename = f"expert_traj/{opt.expert_traj}_exp_traj.h5"

    if os.path.exists(expt_traj_filename):
        os.remove(expt_traj_filename)

    done_hist = np.zeros(shape=(1,opt.num_envs),dtype=bool,device='cpu')
    #termination_hist = np.zeros(shape=(1,opt.num_envs),dtype=bool,device='cpu')

    local_counter=0
    total_traj_length,traj_lenth, total_expert_traj = 0, 0, 0
    while total_expert_traj < opt.expert_traj_number: 
        obs_e, info = envs.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
        
        done = False

        '''Interact & trian'''
        while not done:
            '''Interact with Env'''
            with torch.no_grad():
                action_e, logprob_a_e = expert.select_action(obs_e, deterministic=True)
                act_e = Action_adapter(action_e,env_min_action,env_amplitude_action) #[0,1] to [-max,max]

            next_obs_e, reward_e, terminations_e, truncations_e, infos = envs.step(act_e) # dw: dead&win; tr: truncated               
            #reward = Reward_adapter(reward, opt.EnvIdex)
            #done = (terminations or truncations)
            #print(truncations.shape)     

            dones_e = np.logical_or(terminations_e, truncations_e)
            done_hist = np.logical_or(done_hist,dones_e)

            #termination_hist = np.logical_or(termination_hist, terminations_e)

            '''Store the current transition'''
            expert.put_data(obs_e, action_e, next_obs_e, dones_e, terminations_e, idx = traj_lenth)
            #expert.put_data(obs_e, action_e, next_obs_e, done_hist, termination_hist, idx = traj_lenth)

            if np.all(done_hist):
                done = True
                done_hist[0][:] = False
                #termination_hist[0][:] = False
                #done_hist.fill(0)
            #done = np.all(dones_e)

            obs_e = next_obs_e
                  
                
            traj_lenth += 1
            total_traj_length += 1

            '''Write if its time'''
            if traj_lenth % opt.T_horizon == 0:
                exp_traj = np.concatenate([expert.obs_hoder[0:opt.T_horizon], expert.action_hoder[0:opt.T_horizon]], axis=-1)
                exp_traj = exp_traj.reshape(-1, exp_traj.shape[-1])
                append_chunk(exp_traj,expt_traj_filename)
                traj_lenth = 0

        if traj_lenth > 0:
            exp_traj = np.concatenate([expert.obs_hoder[0:traj_lenth], expert.action_hoder[0:traj_lenth]], axis=-1)
            #print(exp_traj.shape)
            exp_traj = exp_traj.reshape(-1, exp_traj.shape[-1])
            append_chunk(exp_traj,expt_traj_filename)
            traj_lenth = 0
        
        total_expert_traj += 1

        print("Trajectory number {} saved! Shape: T:{}, N:{}, D:{}".format(total_expert_traj,total_traj_length,opt.num_envs,exp_traj.shape[1]))

        total_traj_length = 0

        '''Record & log'''
        if total_expert_traj % opt.eval_interval == 0:
            avg_ep_reward = evaluate_policy(env, expert, device, env_min_action,env_amplitude_action, opt.evaluation_rollouts_num,seed_number=(seed_number+opt.num_envs-1),e_seed=env_seed)
            print(' EnvName:',EnvName[opt.EnvIdex],'seed:',opt.seed,'trajs: {}'.format(total_expert_traj),'avg episode rw:', avg_ep_reward)


        if(seed_number != None):
            local_counter += 1
            local_counter %= seed_number
            env_seed = opt.seed + local_counter 
        else:
            env_seed += 1 

        
    
    envs.close()
    env.close()

if __name__ == '__main__':
    main()




