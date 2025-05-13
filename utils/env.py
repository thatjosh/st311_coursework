from __future__ import annotations
import time
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym
import minigrid

def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env

class CustomRedBlueDoorEnv(MiniGridEnv):
    def __init__(self, size=12, max_steps: int | None = None, **kwargs):
        self.size = size
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = size**2

        super().__init__(
            mission_space=mission_space,
            width=2 * size,
            height=size,
            max_steps = 20 * size**2,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "open the red door then the blue door"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the grid walls
        self.grid.wall_rect(0, 0, 2 * self.size, self.size)
        self.grid.wall_rect(self.size // 2, 0, self.size, self.size)

        # Place the agent in the top-left corner
        self.place_agent(top=(self.size // 2, 0), size=(self.size, self.size))

        # Add a red door at a random position in the left wall
        pos = self._rand_int(1, self.size - 1)
        self.red_door = Door("red")
        self.grid.set(self.size // 2, pos, self.red_door)

        # Add a blue door at a random position in the right wall
        pos = self._rand_int(1, self.size - 1)
        self.blue_door = Door("blue")
        self.grid.set(self.size // 2 + self.size - 1, pos, self.blue_door)

        # Generate the mission string
        self.mission = "open the red door then the blue door"

    def step(self, action):
        red_door_opened_before = self.red_door.is_open
        blue_door_opened_before = self.blue_door.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        red_door_opened_after = self.red_door.is_open
        blue_door_opened_after = self.blue_door.is_open

        if blue_door_opened_after:
            if red_door_opened_before:
                reward = self._reward()
                terminated = True
            else:
                reward = 0
                terminated = True

        elif red_door_opened_after:
            if blue_door_opened_before:
                reward = 0
                terminated = True

        return obs, reward, terminated, truncated, info