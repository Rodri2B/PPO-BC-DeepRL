# PPO+BC for Continuous Environments
This is a simple implementation of an algorithm that fuses PPO and Behaviour Cloning for Deep RL. The project was mainly based on [PPO-Continuous-Pytorch](https://github.com/XinJingHao/PPO-Continuous-Pytorch) which was originally implemented only for PPO in continuous environments.

## Dependencies
```python
gymnasium==1.2.3
numpy==2.2.6
pytorch==2.6.0
h5py==3.16.0

python==3.10.20
```

### Train from scratch

#### For PPO only:
```bash
python main.py --dvc cpu --EnvIdex 1 --render False --Loadmodel False  --save_interval 100000 --Max_train_steps 1000000 --net_width 150 --num_envs 16 --seed_number 100 --load_train_data False
``` 

#### For BC+PPO using saved data:
```bash
python main.py --dvc cpu --EnvIdex 1 --render False --Loadmodel False  --save_interval 100000 --Max_train_steps 500000 --net_width 150 --num_envs 16 --seed_number 100 --load_train_data True --expert_traj lunar2ex_exp_traj.h5 --bc_half_lf 10
```
 
#### For BC+PPO using expert model:
```bash
python main.py --dvc cpu --EnvIdex 1 --render False --Loadmodel False  --save_interval 100000 --Max_train_steps 500000 --net_width 150 --num_envs 16 --seed_number 100 --load_train_data False --bc_expert_model LLdV2ex --bc_half_lf 10
```
Change the  ```--num_envs``` hyperparameter to better suit your implementation.

###  Run trained model
```bash
python main.py --dvc cpu --EnvIdex 1 --render True --Loadmodel True --net_width 150 --num_envs 16 --seed_number 100 --ModelIdex 400 --load_train_data False
``` 

### Hyperparameter Setting
For more details of Hyperparameter Setting, verify 'main.py'

### References

PPO:
<br>
* [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)<br>
* [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286)<br>

Imitation Learning:
<br>
* [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)<br>
* [Augmenting GAIL with BC for sample efficient imitation learning](https://arxiv.org/abs/2001.07798)<br>

Differentiable Predictive Control:
<br>
* [Neural Lyapunov Differentiable Predictive Control](https://arxiv.org/abs/2205.10728)<br>

Main Reference:
<br>
* [PPO-Continuous-Pytorch](https://github.com/XinJingHao/PPO-Continuous-Pytorch)<br>
