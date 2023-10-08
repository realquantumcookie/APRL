import gym
import os
from typing import Tuple, Optional
from jaxrl5.agents.agent import Agent as JaxRLAgent
import numpy as np
from rail_walker_gym.envs.wrappers.rollout_collect import Rollout, RolloutCollector
import wandb
from tqdm import trange
import matplotlib.pyplot as plt
import imageio
import time
from skimage.transform import rescale, resize, downscale_local_mean

IMG_PATH = "topdownmap.png"

def log_visitation(
    env: gym.Env,
    locations : np.ndarray,
    train_step: int = 0,
    project_dir: str = None,
    log_tag: str = "training/visitation_map",
    locations_compare : np.ndarray = None,
) :
    traversible = env.joystick_policy.target_yaw_provider.traversible
    img = plt.imread(IMG_PATH)
    # import ipdb; ipdb.set_trace()
    def plot_heatmap(locations_list, start_idx, end_idx, locations_compare_list=None):
        if locations_compare_list is not None:
            assert len(locations_list) == len(locations_compare_list)

        if end_idx > len(locations_list):
            end_idx = len(locations_list)    
        if end_idx - start_idx < 100:
            return None 
        locations = locations_list[start_idx:end_idx]
        locations = np.array(locations).T / traversible.scale
        
        if locations_compare_list is not None:
            locations_compare = locations_compare_list[start_idx:end_idx]
            locations_compare = np.array(locations_compare).T / traversible.scale
        else:
            locations_compare = None
        
        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(10, 10))  # Set the figure size to create a square plot
        ax.imshow(img, extent=[-1, 1, -1, 1])
        ax.scatter(locations[0], locations[1], marker='.', color='turquoise', s=100, alpha=np.linspace(0, 1, end_idx-start_idx)**5)
        if locations_compare is not None:
            ax.scatter(locations_compare[0], locations_compare[1], marker='.', color='hotpink', s=100, alpha=np.linspace(0, 1, end_idx-start_idx)**5)
        # Set the aspect ratio to 'equal'
        ax.set_aspect('equal')

        # Set the axis limits to -1 and 1
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        # # Add a colorbar
        # cbar = fig.colorbar(im)

        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Visitation Density')

        if hasattr(traversible, "goals"):
            # Add stars to specific locations
            star_x = [g[0][0] for g in traversible.goals]
            star_y = [g[0][1] for g in traversible.goals]
            ax.scatter(star_x, star_y, marker='*', color='yellow', s=200)
        elif hasattr(traversible, "route_points"):
            star_x = [g[0][0] for g in traversible.route_points]
            star_y = [g[0][1] for g in traversible.route_points]
            ax.scatter(star_x, star_y, marker='*', color='yellow', s=200)
        else:
            pass

        # Save the plot to a file (temporary file or in-memory buffer)
        unique_id = str(int(np.round(time.time() * 1000)))
        image_path = os.path.join(project_dir, 'heatmap'+unique_id+'.png')
        plt.savefig(image_path)
        heatmap_array = imageio.v2.imread(image_path)

        # image_resized = resize(heatmap_array, (heatmap_array.shape[0] // 4, heatmap_array.shape[1] // 4),
        #                     anti_aliasing=True)
        os.remove(image_path)
        plt.close()
        return heatmap_array
        # return image_resized

    # create a gif of the heatmap over time
    heatmap_images = []
    timesteps = 1000
    for i in range(0, len(locations), timesteps//5):
    # for i in range(0, len(locations), timesteps):
        heatmap_image = plot_heatmap(locations, i, i+timesteps, locations_compare)
        if not (heatmap_image is None):
            heatmap_images.append(heatmap_image)

    if heatmap_images:
        # prepend the first image to the list
        heatmap_images = [255+np.zeros_like(heatmap_images[0])] + heatmap_images

        wandb.log({
            log_tag: wandb.Video(
                np.array(heatmap_images).transpose(0, 3, 1, 2), fps=4, format="gif"
                ),
            }, step=train_step)


def evaluate(
    agent : JaxRLAgent, 
    env: gym.Env, 
    num_episodes: int,
    train_step: int = 0,
    log_wandb: bool = False,
    log_video: bool = False,
    record_rollout : bool = False,
    enable_tqdm: bool = False
) -> Tuple[float, float, Optional[np.ndarray], Optional[Rollout]]:
    if hasattr(env, 'enableWandbVideo'):
        prev_enable_wandb_video = env.enableWandbVideo
        env.enableWandbVideo = False
    
    print("===================== Evaluating =====================")
    if record_rollout:
        env = RolloutCollector(env)
    if log_video:
        videos = []
    else:
        videos = None
    
    for episode_i in trange(num_episodes, disable=not enable_tqdm):
        episode_infos = {}
        observation = env.reset()
        done = False
        while not done:
            action = agent.eval_actions(observation)
            observation, rew, done, info = env.step(action)

            if 'TimeLimit.joystick_target_change' in info and info['TimeLimit.joystick_target_change']:
                observation, info = env.reset(return_info=True)
                done = False
            
            for key, value in info.items():
                if key in ['TimeLimit.truncated', 'TimeLimit.joystick_target_change', 'episode']:
                    continue
                value = info[key]
                if key not in episode_infos:
                    episode_infos[key] = [value]
                else:
                    episode_infos[key].append(value)
            
            if log_video:
                videos.append(env.render())
        
        if log_wandb and episode_i == 0:
            for k, v in episode_infos.items():
                if v is None or len(v) <= 0:
                    continue
                if k in ['fall_count','traversible_finished_lap_count']:
                    to_log = v[-1]
                else:
                    to_log = np.mean(v)
                wandb.log({'evaluation/' + str(k): to_log}, step=train_step)

    eval_return = np.mean(env.return_queue)
    eval_length = np.mean(env.length_queue)
    eval_rollouts = env.collected_rollouts if record_rollout else None
    
    if log_wandb:
        wandb.log({
            'evaluation/return': eval_return,
            'evaluation/length': eval_length
        }, step=train_step)
    
    if hasattr(env, 'enableWandbVideo'):
        env.enableWandbVideo = prev_enable_wandb_video
    
    print("===================== Evaluation Done =====================")

    return eval_return, eval_length, videos, eval_rollouts


def combine(one_dict, other_dict):
    combined = {}
    indices = np.random.permutation(len(one_dict['observations']) + len(other_dict['observations']))
    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty((v.shape[0] + other_dict[k].shape[0], *v.shape[1:]),
                           dtype=v.dtype)
            tmp[:v.shape[0]] = v
            tmp[v.shape[0]:] = other_dict[k]
            combined[k] = tmp[indices]

    return combined

def update_with_mixed_buffers(prior_replay_buffer, online_replay_buffer, batch_size, utd_ratio, agent, prior_buffer_ratio=0.5):
    assert 0 < prior_buffer_ratio < 1
    prior_amount = int(batch_size * utd_ratio * prior_buffer_ratio)
    online_amount = batch_size * utd_ratio - prior_amount
    online_batch = online_replay_buffer.sample(online_amount)
    prior_batch = prior_replay_buffer.sample(prior_amount)
    batch = combine(online_batch, prior_batch)

    return agent.update(batch, utd_ratio)

def update_with_delay_with_mixed_buffers(prior_replay_buffer, online_replay_buffer, batch_size : int, utd_ratio : int, agent : JaxRLAgent, prior_buffer_ratio : float = 0.5, actor_delay : int = 20, output_range : Optional[Tuple[float, float]] = None):
    assert 0 < prior_buffer_ratio < 1
    prior_amount = int(batch_size * utd_ratio * prior_buffer_ratio)
    online_amount = batch_size * utd_ratio - prior_amount
    online_batch = online_replay_buffer.sample(online_amount)
    prior_batch = prior_replay_buffer.sample(prior_amount)
    batch = combine(online_batch, prior_batch)
    return agent.update_with_delay(batch, utd_ratio, output_range=output_range, actor_delay=actor_delay)

def compute_priority_weights(x, const, scale):
    indices = np.arange(x)
    time = indices - x
    weights = np.exp(time / scale) + const
    weights = weights / np.sum(weights)
    return weights

def evaluate_route_following(
    agent : JaxRLAgent, 
    env: gym.Env, 
    max_timesteps: int,
    train_step: int = 0,
    log_video: bool = False,
    prior_replay_buffer = None,
    prior_buffer_ratio = 0.5,
    online_replay_buffer = None,
    batch_size=256,
    utd_ratio=20,
    enable_tqdm=False,
    frame_skip=2,
    seed=0,
    use_recency_bias=False,
) -> Tuple[float, float, Optional[np.ndarray], Optional[Rollout]]:

    eval_info = {
        'return': 0.0,
        'falls' : 0,
        'video' : None,
        'tracked_locations' : [],
        'time_to_lap_completion' : [],
    }

    if log_video:
        videos = []
    else:
        videos = None

    # LS HACK: this is a hack to reset the traversible to the very start of the route
    traversible = env.joystick_policy.target_yaw_provider.traversible
    # and to set the random state of the goal generator
    if hasattr(traversible, "goals"):
        traversible.goal_generator_random_state = np.random.RandomState(seed)

    if prior_replay_buffer:
        print(f"===================== Evaluating online training policy at {train_step} =====================")
    else:
        print(f"===================== Evaluating fixed policy at {train_step} =====================")
    # import ipdb; ipdb.set_trace()
    env.joystick_policy.resetters[1]._inited = False
    env.joystick_policy.resetters[0].last_position = None
    observation = env.reset()
    # print the position of the robot at reset
    # import ipdb; ipdb.set_trace()
    threed_loc = env.joystick_policy.robot.get_3d_location()
    print("Location of robot at reset:", np.around(threed_loc, 2))
    done = False  
    video = []

    time_to_lap_completion = []

    for t in trange(max_timesteps, disable=not enable_tqdm):
        action, agent = agent.sample_actions(observation)
        next_observation, rew, done, info = env.step(action)

        if 'TimeLimit.joystick_target_change' in info and info['TimeLimit.joystick_target_change']:
            observation, info = env.reset(return_info=True)
            done = False
        
        eval_info['tracked_locations'].append(info['2d_location'])
        eval_info['return'] += rew

        truncated = "TimeLimit.truncated" in info and info['TimeLimit.truncated']
        if (not done) or truncated:
            mask = 1.0
        else:
            mask = 0.0
            eval_info['falls'] += 1
        
        if traversible.is_complete:
            time_to_lap_completion.append(t)
            eval_info['time_to_lap_completion'].append(t - time_to_lap_completion[-1] if len(time_to_lap_completion) > 0 else t)


        update_info = None
        if prior_replay_buffer:
            if prior_buffer_ratio < 1.0:
                online_replay_buffer.insert(dict(
                    observations=observation,
                    actions=action,
                    rewards=rew,
                    masks=mask,
                    dones=done,
                    next_observations=next_observation
                ))
                if len(online_replay_buffer) >= 1000:
                    agent, update_info = update_with_mixed_buffers(prior_replay_buffer,
                                                               online_replay_buffer,
                                                               batch_size,
                                                               utd_ratio,
                                                               agent,
                                                               prior_buffer_ratio=prior_buffer_ratio)
            else:
                prior_replay_buffer.insert(dict(
                    observations=observation,
                    actions=action,
                    rewards=rew,
                    masks=mask,
                    dones=done,
                    next_observations=next_observation
                ))
                if use_recency_bias:
                    prioritized_indx = prior_replay_buffer.np_random.choice(
                        len(prior_replay_buffer), 
                        size=batch_size * utd_ratio, 
                        p = compute_priority_weights(len(prior_replay_buffer), 0.01, 50)
                    )

                    batch = prior_replay_buffer.sample(batch_size * utd_ratio, indx=prioritized_indx)
                else:
                    batch = prior_replay_buffer.sample(batch_size * utd_ratio)
                agent, update_info = agent.update(batch, utd_ratio)

        # if update_info:
        #     if t % 100 == 0:
        #         for k, v in update_info.items():
        #             wandb.log({f'testtime_training' + f'/{k}': v.item()}, step=t)

        observation = next_observation

        if log_video and t % frame_skip == 0:
            video.append(env.render())
            if len(video) > 60*20:
                videos.append(video)
                video = []

        if done:
            # import ipdb; ipdb.set_trace()
            assert env.joystick_policy.resetters[1]._inited
            observation, info = env.reset(return_info=True)
            done = False
    
    if log_video:
        if len(video) > 0:
            videos.append(video)

    eval_info['videos'] = videos

    return eval_info