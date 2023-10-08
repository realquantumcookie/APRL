# rail_unified_walker training

To train a model, run the following command

```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python train_online.py --env_name="..." --config="./configs/droq_config.py" --seed 10 --project_name="..." --save_training_videos --save_training_rollouts --utd_ratio=20 --save_eval_videos --save_eval_rollouts

bash MUJOCO_EGL_DEVICE_ID=1 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python train_online.py --env_name="Go1ResetMujoco-v0" --config="./configs/droq_config.py" --seed 10 --project_name="Yunhao_debugreset" --utd_ratio=20 --save_eval_videos
```

Eval-real

```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python evaluate.py --env_name="Go1EasyResetReal-v0" --config="./configs/droq_config.py" --seed 0 --project_name="Yunhao_debugreset" --save_eval_rollouts
```

Reset Experiments => 20230514

```bash
python train_with_reset_hardcoded.py --save_training_videos --task_config.Kp=40 --task_config.Kd=4 --task_config.action_interpolation=False --config "configs/droq_config.py" --resetter_agent_config "configs/droq_config.py" --resetter_agent_checkpoint "reset_ckpt/Go1GatedResetMujoco-v0_s4096_apw0.00_PD100.00,10.00_ai_ar0.60_nep" --project_name="Yunhao_debugreset"

MUJOCO_GL=egl python evaluate.py --env_name="Go1GatedResetMujoco-v0" --config="./configs/droq_config.py" --seed 0 --project_name="Yunhao_debugreset" --save_eval_videos --task_config.Kp=60 --task_config.Kd=5 --task_config.action_range=1.0 --task_config.action_interpolation=False --eval_episodes=5 --save_dir="./"
```