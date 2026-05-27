from datetime import datetime
import os, shutil
import argparse
import torch
import gymnasium as gym

from utils import str2bool, Action_adapter, Reward_adapter, evaluate_policy_rnn
from PPO import PPO_agent, PPO_expert_agent,BC_PPO_agent,PPO_RNN_agent

import numpy as np
import random
import h5py

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--num_envs', type=int, default=1, help='number of environments')
parser.add_argument('--evaluation_rollouts_num', type=int, default=1, help='number of evaluation rollouts')

parser.add_argument('--clip_gradient_norm', type=float, default=40.0, help='max gradient norm')
parser.add_argument('--seed_number', type=int, default=None, help='number of random seeds')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--Distribution', type=str, default='Beta', help='Should be one of Beta ; GS_ms  ;  GS_m')
parser.add_argument('--Max_train_steps', type=int, default=int(5e7), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(5e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')

parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')

parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')

parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')

parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')

parser.add_argument('--bc_expert_model', type=str, default=None, help='Behaviour cloning expert model name')
parser.add_argument('--bc_half_lf', type=float, default=5, help='Behaviour cloning factor half life')

parser.add_argument('--rnn_sequence_length', type=int, default=10, help='Define RNN sequence length during training')
parser.add_argument('--hidden_state_dim', type=int, default=10, help='Define RNN hidden state dimension')
parser.add_argument('--rnn_layers_num', type=int, default=10, help='Define RNN number of layers')

parser.add_argument('--load_train_data', type=str2bool, default=False, help='Load expert trajectories')
parser.add_argument('--expert_traj', type=str, default=None, help='Expert trajectories main name')
#parser.add_argument('--expert_traj_number', type=float, default=5, help='Number of expert trajectories')

opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def main():
    EnvName = ['Pendulum-v1','LunarLanderContinuous-v3','Humanoid-v5','HalfCheetah-v5','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']

    # Build Env
    envs = gym.make_vec(EnvName[opt.EnvIdex],num_envs=opt.num_envs, vectorization_mode="sync")
    env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = envs.single_observation_space.shape[0]
    opt.action_dim = envs.single_action_space.shape[0]
    opt.max_action = envs.single_action_space.high#float(envs.action_space.high[0])
    opt.min_action = envs.single_action_space.low
    opt.amplitude_action = opt.max_action-opt.min_action
    opt.max_steps = opt.T_horizon#envs.spec.max_episode_steps if (envs.spec.max_episode_steps <= opt.T_horizon) else opt.T_horizon

    '''BC Setting'''
    opt.bc_alpha_o = 0.5**(1/opt.bc_half_lf)
    opt.bc_alpha = opt.bc_alpha_o

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

    # Use tensorboard to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # Beta dist maybe need larger learning rate, Sometimes helps
    # if Dist[distnum] == 'Beta' :
    #     kwargs["a_lr"] *= 2
    #     kwargs["c_lr"] *= 4

    if not os.path.exists('model'): os.mkdir('model')
    agent = None if (opt.bc_expert_model != None or opt.load_train_data) else PPO_RNN_agent(**vars(opt))
    #agent = BC_PPO_RNN_agent(**vars(opt)) if (opt.bc_expert_model != None or opt.load_train_data) else PPO_RNN_agent(**vars(opt)) # transfer opt to dictionary, and use it to init PPO_agent
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)


    '''BC Model Setting'''

    if opt.bc_expert_model != None and ~opt.load_train_data:
        expert = PPO_expert_agent(**vars(opt))
        expert.load(opt.bc_expert_model)

        expert_envs = gym.make_vec(EnvName[opt.EnvIdex],num_envs=opt.num_envs, vectorization_mode="sync")
    elif opt.load_train_data: 
        assert opt.expert_traj != None, "Trajectory data not specified!"
        expt_traj_filename = "expert_traj/"+opt.expert_traj

        #with h5py.File(expt_traj_filename, "r") as f:
            #expert_data = f["data"]
        h5file = h5py.File(expt_traj_filename, "r")
        expert_data = h5file["data"]
        expert_trajs_lims = h5file["data_sizes"]

        entire_dataset_loaded = False
        if expert_data.shape[0] < 500*opt.T_horizon:
            expt_obs_dataset = expert_data[:,:, :opt.state_dim]
            expt_past_action_dataset = expert_data[:,:, opt.state_dim:(opt.state_dim+opt.action_dim)]
            expt_action_dataset = expert_data[:,:, (opt.state_dim+opt.action_dim):]
            entire_dataset_loaded = True

            trajectory_data_hist = []
            #bached_traj_start = 0
            for traj_e in expert_trajs_lims:

                start = traj_e[0]
                #end = traj_e[1]+1

                #traj_size = end-start

                #trajectory_data_hist.append(bached_traj_start)
                trajectory_data_hist.append(start)

                #bached_traj_start += traj_size

        else:
            #expert_batch_size = min(20,int(expert_data.shape[0]/opt.T_horizon))*opt.T_horizon
            #e_rand_idx = np.random.permutation(expert_data.shape[0])
            #e_batch_max_steps = int(expert_data.shape[0]/expert_batch_size)
            #e_batch_step = 0
            traj_mean_length = 0
            for traj_legth in expert_trajs_lims:
                length = traj_legth[1]-traj_legth[0]+1
                traj_mean_length+=length
            traj_mean_length = int(traj_mean_length / expert_trajs_lims.shape[0])

            expert_trajbatch_size = min(20,int(expert_data.shape[0]/opt.T_horizon))*opt.T_horizon
            expert_trajbatch_size = int(expert_trajbatch_size/traj_mean_length)
            if expert_trajbatch_size == 0: expert_trajbatch_size = 1

            while(expert_trajs_lims.shape[0]%expert_trajbatch_size != 0):
                expert_trajbatch_size -= 1

            e_trajrand_idx = np.random.permutation(expert_trajs_lims.shape[0])
            e_trajbatch_max_steps = int(expert_trajs_lims.shape[0]/expert_trajbatch_size)
            e_trajbatch_step = 0


            



    
    if opt.render:
        while True:
            last_hidden_state_eval=torch.zeros(shape=(opt.rnn_layers_num,opt.num_envs,opt.hidden_dim),dtype=torch.float32,device='cpu')
            past_action_eval=np.full(fill_value=0.5,shape=(opt.num_envs,opt.action_dim),dtype=np.float32)
            ep_r = evaluate_policy_rnn(env, agent, device, env_min_action,env_amplitude_action, 1,past_action_eval,last_hidden_state_eval,seed_number=seed_number,e_seed=env_seed)
            print(f'Env:{EnvName[opt.EnvIdex]}, Episode Reward:{ep_r}')
#####################################
    else:

        done_hist = np.zeros(shape=(1,opt.num_envs),dtype=bool)

        trajectory_hist = []
        #termination_hist = np.zeros(shape=(1,opt.num_envs),dtype=bool,device='cpu')

        #if opt.bc_expert_model != None: 
        #    done_hist_e = np.zeros(shape=(1,opt.num_envs),dtype=bool,device='cpu')
        #    termination_hist_e = np.zeros(shape=(1,opt.num_envs),dtype=bool,device='cpu')

        if opt.bc_expert_model != None and ~opt.load_train_data: 
            past_action_expt_hoder = np.zeros((opt.T_horizon, opt.num_envs,opt.action_dim),dtype=np.float32)

        local_counter=0
        traj_lenth, total_steps = 0, 0

        while total_steps < opt.Max_train_steps:
            obs, info = envs.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed

            last_hidden_state=torch.zeros(shape=(opt.rnn_layers_num,opt.num_envs,opt.hidden_dim),dtype=torch.float32,device='cpu')
            past_action=np.full(fill_value=0.5,shape=(opt.num_envs,opt.action_dim),dtype=np.float32)

            if (traj_lenth > 0):trajectory_hist.append(traj_lenth)

            if opt.bc_expert_model != None and ~opt.load_train_data: 
                train_past_action_T = past_action.copy()
                obs_e, info = expert_envs.reset(seed=env_seed)

            elif opt.load_train_data and (not entire_dataset_loaded):

                expert_traj_index = slice(e_trajbatch_step * expert_trajbatch_size, min((e_trajbatch_step + 1) * expert_trajbatch_size, expert_trajs_lims.shape[0]))
                bached_traj_start = 0
                trajectory_data_hist = []

                #calculate toltal size
                trajs_total_size = 0
                for traj_e in e_trajrand_idx[expert_traj_index]:
                    start = traj_e[0]
                    end = traj_e[1]+1
                    traj_size = end-start

                    trajs_total_size += traj_size

                expt_obs_dataset=np.zeros(shape=(trajs_total_size,expert_data.shape[1],opt.state_dim),dtype=np.float32)
                expt_past_action_dataset=np.zeros(shape=(trajs_total_size,expert_data.shape[1],opt.action_dim),dtype=np.float32)
                expt_action_dataset=np.zeros(shape=(trajs_total_size,expert_data.shape[1],opt.action_dim),dtype=np.float32)

                for traj_e in e_trajrand_idx[expert_traj_index]:

                    start = traj_e[0]
                    end = traj_e[1]+1

                    traj_size = end-start
                    bached_traj_end = bached_traj_start+traj_size


                    expt_obs_dataset[bached_traj_start:bached_traj_end, :, :opt.state_dim] = expert_data[start:end, :, :opt.state_dim]
                    expt_past_action_dataset[bached_traj_start:bached_traj_end, :, :opt.action_dim] = expert_data[start:end, :, opt.state_dim:(opt.state_dim+opt.action_dim)]
                    expt_action_dataset[bached_traj_start:bached_traj_end, :, :opt.action_dim] = expert_data[start:end, :, (opt.state_dim+opt.action_dim):]

                    trajectory_data_hist.append(bached_traj_start)

                    bached_traj_start += traj_size

                e_trajbatch_step += 1
                e_trajbatch_step %= e_trajbatch_max_steps

            done = False

            '''Interact & trian'''
            while not done:
                '''Interact with Env'''
                state_past_action = np.concatenate([obs,past_action], axis=-1)
                action_v, logprob_a ,hidden_st= agent.select_action(state_past_action,last_hidden_state, deterministic=False) # use stochastic when training
                action = action_v[-1,:,:]
                act = Action_adapter(action,env_min_action,env_amplitude_action) #[0,1] to [-max,max]
                next_obs, reward, terminations, truncations, infos = envs.step(act) # dw: dead&win; tr: truncated                
                #reward = Reward_adapter(reward, opt.EnvIdex)
                #done = (terminations or truncations)
                #print(truncations.shape)     

                dones = np.logical_or(terminations, truncations)
                done_hist = np.logical_or(done_hist,dones)

                #termination_hist = np.logical_or(termination_hist, terminations)

                '''Store the current transition'''
                agent.put_data(obs,last_hidden_state,past_action,action, reward, next_obs, logprob_a, dones, terminations, idx = traj_lenth)
                #agent.put_data(obs, action, reward, next_obs, logprob_a, done_hist, termination_hist, idx = traj_lenth)

                last_hidden_state = hidden_st
                past_action = action
                obs = next_obs


                if opt.bc_expert_model != None and ~opt.load_train_data: 
                    action_e, logprob_a_e = expert.select_action(obs_e, deterministic=True)
                    act_e = Action_adapter(action_e,env_min_action,env_amplitude_action) #[0,1] to [-max,max]
                    next_obs_e, reward_e, terminations_e, truncations_e, infos = expert_envs.step(act_e) # dw: dead&win; tr: truncated  
                    dones_e = np.logical_or(terminations_e, truncations_e)
                    #done_hist_e = np.logical_or(done_hist_e,dones_e)
                    #termination_hist_e = np.logical_or(termination_hist_e, terminations_e)

                    past_action_expt_hoder[traj_lenth] = train_past_action_T

                    expert.put_data(obs_e, action_e, next_obs_e, dones_e, terminations_e, idx = traj_lenth)
                    #expert.put_data(obs_e, action_e, next_obs_e, done_hist_e, termination_hist_e, idx = traj_lenth)

                    train_past_action_T = action_e
                    obs_e = next_obs_e

                if np.all(done_hist):
                    done = True
                    done_hist[0][:] = False
                    #termination_hist[0][:] = False
                    #done_hist_e[0][:] = False
                    #termination_hist_e[0][:] = False
                    #done_hist.fill(0)
                #done = np.all(dones)

                traj_lenth += 1
                total_steps += 1

                '''Update if its time'''
                if traj_lenth % opt.T_horizon == 0:
                    if opt.bc_expert_model != None and ~opt.load_train_data:
                        agent.train(expert.obs_hoder,expert.action_hoder,past_action_expt_hoder,trajectory_hist,trajectory_hist)
                    elif opt.load_train_data:
                        agent.train(expt_obs_dataset,expt_action_dataset,expt_past_action_dataset,trajectory_hist,trajectory_data_hist)
                    else:
                        agent.train(trajectory_hist)
                    traj_lenth = 0
                    trajectory_hist.clear()
                    trajectory_hist.append(traj_lenth)
##############################################
                '''Record & log'''
                if total_steps % opt.eval_interval == 0:
                    last_hidden_state_eval=torch.zeros(shape=(opt.rnn_layers_num,opt.num_envs,opt.hidden_dim),dtype=torch.float32,device='cpu')
                    past_action_eval=np.full(fill_value=0.5,shape=(opt.num_envs,opt.action_dim),dtype=np.float32)
                    avg_ep_reward = evaluate_policy_rnn(eval_env, agent,device, env_min_action,env_amplitude_action, episodes_num=opt.evaluation_rollouts_num,first_p_act=past_action_eval ,h_0=last_hidden_state_eval , seed_number=seed_number,e_seed=env_seed)  # evaluate the policy for 3 times, and get averaged result
                    if opt.write: writer.add_scalar('ep_r', avg_ep_reward, global_step=total_steps)
                    print(' EnvName:',EnvName[opt.EnvIdex],'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'avg episode rw:', avg_ep_reward)

                '''Save model'''
                if total_steps % opt.save_interval==0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))

            if(seed_number != None):
                local_counter += 1
                local_counter %= seed_number
                env_seed = opt.seed + local_counter 
            else:
                env_seed += 1 

        envs.close()
        if opt.bc_expert_model != None and ~opt.load_train_data: expert_envs.close()
        env.close()
        eval_env.close()

if __name__ == '__main__':
    main()
