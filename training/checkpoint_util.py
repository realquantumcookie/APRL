import os
import typing
from jaxrl5.agents.agent import Agent as JaxRLAgent
from jaxrl5.data import ReplayBuffer
from flax.training import checkpoints
from natsort import natsorted
import pickle
import shutil
from rail_walker_gym.envs.wrappers.rollout_collect import Rollout

def initialize_project_log(project_dir) -> None:
    os.makedirs(project_dir, exist_ok=True)
   
    chkpt_dir = os.path.join(project_dir, 'checkpoints')
    buffer_dir = os.path.join(project_dir, "buffers")
    rollout_dir = os.path.join(project_dir, "rollouts")
    eval_rollout_dir = os.path.join(project_dir, "eval_rollouts")
    
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(buffer_dir, exist_ok=True)
    os.makedirs(rollout_dir, exist_ok=True)
    os.makedirs(eval_rollout_dir, exist_ok=True)

def list_checkpoint_steps(project_dir) -> typing.List[int]:
    chkpt_dir = os.path.join(project_dir,'checkpoints')
    chkpts = natsorted(os.listdir(chkpt_dir))
    return [int(chkpt.split('_')[-1]) for chkpt in chkpts]

def list_replay_buffer_steps(project_dir) -> typing.List[int]:
    buffer_dir = os.path.join(project_dir, "buffers")
    buffers = natsorted(os.listdir(buffer_dir))
    return [int(buffer.split('_')[-1].split('.')[0]) for buffer in buffers]

def load_checkpoint_at_step(project_dir, step : int, agent : JaxRLAgent, if_failed_return_step : int = 0) -> typing.Tuple[int, JaxRLAgent]:
    chkpt_dir = os.path.join(project_dir,'checkpoints')
    chkpts = os.listdir(chkpt_dir)
    for chkpt in chkpts:
        chkpt_int = int(chkpt.split('_')[-1])
        if chkpt_int == step:
            agent = checkpoints.restore_checkpoint(os.path.join(chkpt_dir, chkpt), agent)
            return chkpt_int, agent
    
    return if_failed_return_step, agent

def load_checkpoint_file(
    filename : str,
    agent : JaxRLAgent
):
    agent = checkpoints.restore_checkpoint(filename, agent)
    return agent

def load_latest_checkpoint(project_dir, agent : JaxRLAgent, if_failled_return_step: int = 0) -> typing.Tuple[int, JaxRLAgent]:
    chkpt_dir = os.path.join(project_dir,'checkpoints')
    chkpts = natsorted(os.listdir(chkpt_dir))
    if len(chkpts) == 0:
        return if_failled_return_step, agent
    else:
        chkpt_int = int(chkpts[-1].split('_')[-1])
        agent = checkpoints.restore_checkpoint(os.path.join(chkpt_dir, chkpts[-1]), agent)
        return chkpt_int, agent

def load_latest_replay_buffer(project_dir) -> typing.Optional[typing.Tuple[int, ReplayBuffer]]:
    buffer_dir = os.path.join(project_dir, "buffers")
    buffers = natsorted(os.listdir(buffer_dir))
    if len(buffers) == 0:
        return None
    else:
        buffer_int = int(buffers[-1].split('_')[-1].split('.')[0])
        with open(os.path.join(buffer_dir, buffers[-1]),'rb') as f:
            replay_buffer = pickle.load(f)
        return buffer_int, replay_buffer

def load_replay_buffer_at_step(project_dir, step: int) -> typing.Optional[ReplayBuffer]:
    buffer_dir = os.path.join(project_dir, "buffers")
    try:
        with open(os.path.join(buffer_dir, f'buffer_{step}.pkl'),'rb') as f:
            replay_buffer = pickle.load(f)
        return replay_buffer
    except:
        return None

def load_latest_additional_replay_buffer(project_dir) -> typing.Optional[ReplayBuffer]:
    buffer_dir = os.path.join(project_dir, "additional_buffers")
    if not os.path.exists(buffer_dir):
        return None
    buffers = natsorted(os.listdir(buffer_dir))
    if len(buffers) <= 1:
        return None
    else:
        with open(os.path.join(buffer_dir, buffers[-2]),'rb') as f:
            replay_buffer = pickle.load(f)
        return replay_buffer

def load_replay_buffer_file(filename : str) -> ReplayBuffer:
    with open(filename,'rb') as f:
        replay_buffer = pickle.load(f)
    return replay_buffer

def save_checkpoint(project_dir, step : int, agent : JaxRLAgent) -> None:
    chkpt_dir = os.path.join(project_dir,'checkpoints')
    try:
        checkpoints.save_checkpoint(
            chkpt_dir,
            agent,
            step=step,
            keep=10,
            overwrite=True
        )
    except:
        pass

def save_replay_buffer(project_dir, step: int, replay_buffer: ReplayBuffer, delete_old_buffers : bool = True) -> None:
    buffer_dir = os.path.join(project_dir, "buffers")
    if delete_old_buffers:
        try:
            shutil.rmtree(buffer_dir)
        except:
            pass
    try:
        os.makedirs(buffer_dir, exist_ok=True)
        with open(os.path.join(buffer_dir, f'buffer_{step}.pkl'),
                    'wb') as f:
            pickle.dump(replay_buffer, f)
    except:
        pass

def save_rollout(project_dir, step: int, is_training : bool, rollout : Rollout):
    rollout_dir = os.path.join(project_dir, "rollouts" if is_training else "eval_rollouts")
    try:
        os.makedirs(rollout_dir, exist_ok=True)
        rollout.export_npz(os.path.join(rollout_dir, f'rollout_{step}.npz'))
    except:
        pass

def save_episode_rollout(project_dir, step: int, is_training : bool, rollout : Rollout):
    rollout_dir = os.path.join(project_dir, "rollouts" if is_training else "eval_rollouts")
    try:
        os.makedirs(rollout_dir, exist_ok=True)
        save_file = os.path.join(rollout_dir, f'rollout_{step}.npz')
        if rollout.is_current_episode_empty():
            rollout.export_last_episode_npz(save_file)
        else:
            rollout.export_current_episode_npz(save_file)
    except:
        pass

def list_rollout_steps(project_dir, is_training : bool) -> typing.List[int]:
    rollout_dir = os.path.join(project_dir, "rollouts" if is_training else "eval_rollouts")
    rollouts = natsorted(os.listdir(rollout_dir))
    return [int(rollout.split('_')[-1].split('.')[0]) for rollout in rollouts]

def load_rollout_at_step(project_dir, step: int, is_training : bool) -> typing.Optional[Rollout]:
    rollout_dir = os.path.join(project_dir, "rollouts" if is_training else "eval_rollouts")
    try:
        rollout = Rollout.import_npz(os.path.join(rollout_dir, f'rollout_{step}.npz'))
        return rollout
    except:
        return None