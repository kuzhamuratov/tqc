import robel
import gym
import argparse
import pandas as pd
import numpy as np
from mujoco_py.generated import const


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='DKittyWalkFixed-v0')
    parser.add_argument('--data', default='experiments_data/real_stand.csv')
    args = parser.parse_args()

    env = gym.make(args.env)
    print('observation_space', env.observation_space)
    print('action_space', env.action_space)

    env.reset()
    env.render(mode='human')
    env.sim_scene.renderer._onscreen_renderer._run_speed = 0.015

    full_df = pd.read_csv(args.data)
    for episode in set(full_df['episode'].array):
        episode_df = full_df[full_df['episode'] == episode]
        for step in set(episode_df['step'].array):
            df = episode_df[episode_df['step'] == step]

            def get(begin, size):
                return np.array([float(df[f'obs_{i}']) for i in range(begin, begin + size)])

            def get_state():
                return np.array([float(df[f'obs_{i}']) for i in range(env.observation_space.shape[0])])

            env.set_state({
                'root_pos': get(0, 3), # Cartesian position
                'root_euler': get(3, 3), # Euler orientation
                'root_vel': get(18, 3), # velocity
                'root_angular_vel': get(9, 3), # angular velocity
                'kitty_qpos': get(6, 12), # joint positions
                'kitty_qvel': get(24, 12), # joint velocities
            })

            env.sim_scene.renderer._onscreen_renderer.add_overlay(
                const.GRID_TOPLEFT,
                "success",
                str(float(df['success']))
            )

            env.sim_scene.renderer._onscreen_renderer.add_overlay(
                const.GRID_TOPLEFT,
                "episode",
                str(episode)
            )

            env.render(mode='human')
