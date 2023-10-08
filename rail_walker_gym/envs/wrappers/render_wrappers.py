import gym
from typing import Optional
import wandb
import numpy as np
import cv2 as cv
from typing import Any
import pygame
from PIL import Image

class WanDBVideoWrapper(gym.Wrapper):
    def __init__(
        self, 
        env : gym.Env, 
        video_format : str = "mp4", # "mp4" or "gif" or "webm" or "ogg"
        log_name = "training/video",
        record_every_n_steps = -1,
        frame_rate : Optional[int] = None,
        video_length_limit : int = 0, # Change to > 0 to limit video length
    ):
        super().__init__(env)
        if frame_rate is None:
            if hasattr(env,"joystick_policy"):
                control_timestep = env.joystick_policy.robot.control_timestep
            elif hasattr(env.unwrapped._env, "control_timestep"):
                control_timestep = getattr(env.unwrapped._env, "control_timestep")()
            else:
                raise AttributeError(
                "Environment must have a control_timestep() method."
                )

            frame_rate = int(1.0 / control_timestep)
            print("WanDBVideoWrapper: control_timestep = {}, framerate = {}".format(control_timestep, frame_rate))
                
        
        self._frame_rate = frame_rate
        self._frames = []
        self.log_name = log_name
        self.wandb_step = 0
        self._last_recorded_iter = 0
        self._current_recording_iter = 0
        self.video_format = video_format
        self.enableWandbVideo = True
        self.record_every_n_steps = record_every_n_steps
        self.video_length_limit = video_length_limit
        self._last_joystick_truncation = False
    
    @property
    def is_recording(self):
        return self.enableWandbVideo and (
            self._last_recorded_iter < self._current_recording_iter
        )

    @property
    def should_start_recording(self):
        return self.enableWandbVideo and not(self.is_recording) and (
            self.wandb_step - self._last_recorded_iter >= self.record_every_n_steps or
            self.record_every_n_steps < 0
        )

    def set_wandb_step(self, step : int) -> None:
        self.wandb_step = step

    def step(self, action):
        obs, rew, done, info = self.env.step(action) #ret = (obs, reward, done, info)
        self._try_append_record()
        if "TimeLimit.joystick_target_change" in info and info["TimeLimit.joystick_target_change"]:
            self._last_joystick_truncation = True
        elif done:
            self._terminate_record()
            self._last_joystick_truncation = False
        self.wandb_step += 1
        return obs, rew, done, info

    def reset(self, *args, seed: int | None = None, options: dict[str, Any] | None = None, **kwargs):
        if not self._last_joystick_truncation:
            self._terminate_record()
            if self.should_start_recording:
                setattr(self.env.unwrapped, "_is_video_rec", True)
            else:
                setattr(self.env.unwrapped, "_is_video_rec", False)
        ret = self.env.reset(*args, seed=seed, options=options, **kwargs)
        if not self._last_joystick_truncation:
            self._try_start_record()
        self._last_joystick_truncation = False
        return ret

    # Helper methods.

    def _try_append_record(self):
        if self.is_recording:
            self._frames.append(self.env.render())
        
    def _try_start_record(self):
        if self.should_start_recording:
            if hasattr(self.env,"prepended_frames") and self.env.prepended_frames is not None:
                self._frames = self.env.prepended_frames
            else:
                self._frames = []
            
            self._frames.append(self.env.render())
            self._current_recording_iter = self.wandb_step

    def _terminate_record(self) -> None:
        self._last_recorded_iter = self._current_recording_iter
        if len(self._frames) <= 0:
            return

        if self.video_length_limit > 0:
            self._frames = self._frames[-self.video_length_limit:]
        
        print("Writing {} to wandb (Step {})".format(self.log_name, self.wandb_step))
        wandb.log(
            {
                # RGBA => BGRA
                self.log_name: wandb.Video(
                    np.stack(self._frames).transpose(0, 3, 1, 2), 
                    fps=self._frame_rate, 
                    format=self.video_format
                )
            }, 
            step=self.wandb_step #self._current_recording_iter
        )
        self._frames = []

class _RenderWrapperViewer:
    def __init__(
        self
    ):
        self.screen = None
        self.clock = None
        self.fps = 60
    
    def init_pygame(self, x, y, window_name : str = "Enviroment Viewer") -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((x, y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption(window_name)
        self.clock = pygame.time.Clock()

    @property
    def has_inited(self) -> bool:
        return self.screen is not None

    def render(self, image : Image) -> bool:
        if self.screen is None:
            self.init_pygame(image.width, image.height)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        image_surface = pygame.image.fromstring(image.tobytes(), image.size, image.mode).convert()
        self.screen.fill((0,0,0))
        self.screen.blit(image_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True


class RenderViewerWrapper(gym.Wrapper):
    def __init__(self, env, window_name = "Environment Render", control_timestep = 0.05):
        super().__init__(env)
        self.window_name = window_name
        self._viewer = _RenderWrapperViewer()
        self._viewer.fps = int(1.0 / control_timestep)

    def step(self, action):
        ret = self.env.step(action) #ret = (obs, reward, done, info)

        if ret[2]:
            self._terminate_display()
        else:
            render = self.env.render()
            if render is not None:
                self._display(render)
            else:
                self._terminate_display()
        
        return ret

    def reset(self, *args, seed: int | None = None, options: dict[str, Any] | None = None, **kwargs):
        self._terminate_display()
        ret = self.env.reset(*args,seed=seed, options=options, **kwargs)
        return ret

    def _display(self, frame):
        frame = Image.fromarray(frame,mode="RGB")
        if not self._viewer.has_inited:
            self._viewer.init_pygame(frame.width, frame.height, window_name=self.window_name)
        self._viewer.render(frame)
    
    def _terminate_display(self):
        pass
