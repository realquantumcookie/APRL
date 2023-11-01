# APRL

Code to replicate [Efficient Real-World RL for Legged Locomotion via Adaptive Policy Regularization](https://sites.google.com/berkeley.edu/aprl/).
This repo contains code for training a simulated or real Go1 quadrupedal robot to walk from scratch. This code has been tested on Ubuntu 18.04 LTS with Python 3.10.

## Installation

> The tested Python version is 3.10, create a new environment with `conda create -n aprl python=3.10` and activate it with `conda activate aprl`.

First copy the `lib` directory of [Unitree SDK](https://github.com/unitreerobotics/unitree_legged_sdk/tree/go1) to the `unitree_go1_wrapper/lib`.

Then clone down this [DMCGym](https://github.com/ikostrikov/dmcgym) repo and install it with `pip install -e .`. You may need to remove the first line of `requirements.txt` in that repo and change it as follows:

```
# Before the change
gym[mujoco] >= 0.21.0, < 0.24.1

# After the change
gym >= 0.21.0, < 0.24.1
```

Then run the following in this repo to install the dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

> Special note with Windows users: Jax does not offer official cuda support for Windows binary wheels. However, you can use instructions in [this experimental repo](https://github.com/cloudhan/jax-windows-builder) to install a cuda-enabled version of Jax. Our code uses `jax[cuda111]==0.3.25`. Make sure that the `jax` and `jaxlib` packages are both installed with the correct version. Also make sure that your nvidia driver shows support for cuda 11 when you run `nvidia-smi`.

## Setting up the real robot

[Real Robot Setup](RealSetup.md)

## Training

Example command to run real training

```bash
cd training/

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py \
  --env_name=Go1SanityReal-Empty-SepRew-v0 \
  --save_buffer=True \
  --load_buffer \
  --utd_ratio=20 \
  --start_training=1000 \
  --config=configs/droq_config.py \
  --config.critic_layer_norm=True \
  --config.exterior_linear_c=12.0 \
  --config.target_entropy=-12 \
  --save_eval_videos=False \
  --eval_interval=-1 \
  --save_training_videos=False \
  --training_video_interval=5000 \
  --eval_episodes=1 \
  --max_steps=40000 \
  --log_interval=1000 \
  --save_interval=10000 \
  --seed=0 \
  --project_name=APRL_real_reproduce \
  --tqdm=True \
  --save_dir=saved_real_exp \
  --task_config.action_interpolation=True \
  --task_config.enable_reset_policy=True \
  --task_config.Kp=20 \
  --task_config.Kd=1.0 \
  --task_config.limit_episode_length=0 \
  --task_config.action_range=1.0 \
  --task_config.frame_stack=0 \
  --task_config.action_history=1 \
  --task_config.rew_target_velocity=1.5 \
  --task_config.rew_energy_penalty_weight=0.0 \
  --task_config.rew_qpos_penalty_weight=2.0 \
  --task_config.rew_smooth_torque_penalty_weight=0.005 \
  --task_config.rew_pitch_rate_penalty_factor=0.4 \
  --task_config.rew_roll_rate_penalty_factor=0.2 \
  --task_config.rew_joint_diagonal_penalty_weight=0.00 \
  --task_config.rew_joint_shoulder_penalty_weight=0.00 \
  --task_config.rew_joint_acc_penalty_weight=0.0 \
  --task_config.rew_joint_vel_penalty_weight=0.0 \
  --task_config.center_init_action=True \
  --task_config.rew_contact_reward_weight=0.0 \
  --action_curriculum_steps=30000 \
  --action_curriculum_start=0.35 \
  --action_curriculum_end=0.6 \
  --action_curriculum_linear=True \
  --action_curriculum_exploration_eps=0.15 \
  --task_config.filter_actions=8 \
  --reset_curriculum=True \
  --reset_criterion=dynamics_error \
  --task_config.rew_smooth_change_in_tdy_steps=1 \
  --threshold=1.5
```

Example command to run simulated training

```bash
cd training/

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py \
  --env_name=Go1SanityMujoco-Empty-SepRew-v0 \
  --save_buffer=True \
  --load_buffer \
  --utd_ratio=20 \
  --start_training=1000 \
  --config=configs/droq_config.py \
  --config.critic_layer_norm=True \
  --config.exterior_linear_c=12.0 \
  --config.target_entropy=-12 \
  --save_eval_videos=False \
  --eval_interval=-1 \
  --save_training_videos=False \
  --training_video_interval=5000 \
  --eval_episodes=1 \
  --max_steps=40000 \
  --log_interval=1000 \
  --save_interval=10000 \
  --seed=0 \
  --project_name=APRL_sim_reproduce \
  --tqdm=True \
  --save_dir=saved_sim_exp \
  --task_config.action_interpolation=True \
  --task_config.enable_reset_policy=False \
  --task_config.Kp=20 \
  --task_config.Kd=1.0 \
  --task_config.limit_episode_length=0 \
  --task_config.action_range=1.0 \
  --task_config.frame_stack=0 \
  --task_config.action_history=1 \
  --task_config.rew_target_velocity=1.5 \
  --task_config.rew_energy_penalty_weight=0.0 \
  --task_config.rew_qpos_penalty_weight=2.0 \
  --task_config.rew_smooth_torque_penalty_weight=0.005 \
  --task_config.rew_pitch_rate_penalty_factor=0.4 \
  --task_config.rew_roll_rate_penalty_factor=0.2 \
  --task_config.rew_joint_diagonal_penalty_weight=0.00 \
  --task_config.rew_joint_shoulder_penalty_weight=0.00 \
  --task_config.rew_joint_acc_penalty_weight=0.0 \
  --task_config.rew_joint_vel_penalty_weight=0.0 \
  --task_config.center_init_action=True \
  --task_config.rew_contact_reward_weight=0.0 \
  --action_curriculum_steps=30000 \
  --action_curriculum_start=0.35 \
  --action_curriculum_end=0.6 \
  --action_curriculum_linear=True \
  --action_curriculum_exploration_eps=0.15 \
  --task_config.filter_actions=8 \
  --reset_curriculum=True \
  --reset_criterion=dynamics_error \
  --task_config.rew_smooth_change_in_tdy_steps=1 \
  --threshold=1.5
```