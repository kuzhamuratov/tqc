import robel
import gym
import argparse
import numpy as np
import pandas as pd
import torch
import time
from tqc import DEVICE
from tqc.structures import Actor, RescaleAction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/no_field_10_degree_DKittyWalkRandomDynamics-v0_0_actor',
                        help='Path to the model file. check default in file')
    parser.add_argument('--env', default='DKittyWalkRandomDynamics-v0',
                        help='Environment name. (default: DKittyWalkRandomDynamics-v0)')
    parser.add_argument('--dont_save', default=False, action='store_true',
                        help='Do not save collected data from observations. (default: False)')
    parser.add_argument('--save_actuators', default=False, action='store_true',
                        help='Save collected data from each actuator independently (default: False)')
    parser.add_argument('--name', default='unknown',
                        help='Name for csv file containing collected data (default: unknown)')
    args = parser.parse_args()
    sim_env = RescaleAction(gym.make(args.env, angle=-np.pi/180.*5.), -1., 1.)
    real_env = RescaleAction(gym.make(args.env, angle=-np.pi/180.*5.), -1., 1.) #, device_path='/dev/ttyUSB0', torso_tracker_id=1, reset_type='scripted')

    print("Created 2 env: ", args.env)

    if args.model == 'custom':
        def policy(obs):
            # Custom policy to control movements
            return np.zeros(12)
    else:
        # download tqc agent
        state_dim = real_env.observation_space.shape[0]
        action_dim = real_env.action_space.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = Actor(state_dim, action_dim).to(device)
        policy.load_state_dict(torch.load(args.model, map_location=device))
        policy.eval()

    num_iterations = 30
    real_obs = []  # List of observations
    sim_obs = []
    successes = []
    actions = []
    episodes = []
    steps = []
    total_success = 0

    for current_iteration in range(num_iterations):
        print('CURRENT ITERATION: {}'.format(current_iteration))

        sim_env.reset()
        observation = real_env.reset()
        sim_env.set_state(real_env.get_state())

        done = False
        curr_success = 0
        action = np.zeros(12)
        cnt = 0

        while not done:
            action = policy.select_action(observation)
            actions.append(action)
            real_env.render()
            observation, reward, done, info = real_env.step(action)
            real_obs.append(observation.tolist())

            sim_env.set_state(real_env.get_state())  # Set the simulation state the same as real

            sim_observation, sim_reward, sim_done, sim_info = sim_env.step(action)
            sim_obs.append(sim_observation.tolist())
            episodes.append(current_iteration)
            steps.append(cnt)
            cnt += 1

            if info['score/success'] and curr_success == 0:
                curr_success = 1
                total_success += 1
                print('Success')

        if curr_success:
            successes += [1] * cnt
        else:
            successes += [0] * cnt

        if args.save_actuators:
            pd.DataFrame(real_obs[-160:]).to_csv('experiments_data/actuator_{}_real'.format(current_iteration) + '.csv', index=False)
            pd.DataFrame(sim_obs[-160:]).to_csv('experiments_data/actuator_{}_sim'.format(current_iteration) + '.csv', index=False)
            print('SAVED FOR ACTUATOR {}'.format(current_iteration))

        #input()

    if not args.dont_save:
        real_obs = pd.DataFrame(real_obs)
        sim_obs = pd.DataFrame(sim_obs)
        successes = pd.DataFrame(successes)
        episodes = pd.DataFrame(episodes)
        steps = pd.DataFrame(steps)
        actions = pd.DataFrame(actions)

        print(len(steps), len(real_obs), len(sim_obs))

        real = pd.concat([successes, episodes, steps, actions, real_obs], axis=1, sort=False)
        sim = pd.concat([successes, episodes, steps, actions, sim_obs], axis=1, sort=False)
        real.columns = ['success', 'episode', 'step'] +\
                       list(map(lambda x: 'action_{}'.format(x), list(range(12)))) +\
                       list(map(lambda x: 'obs_{}'.format(x), list(range(real_obs.shape[1]))))
        sim.columns = real.columns

        real.to_csv('experiments_data/real_{}.csv'.format(args.name), index=False)
        sim.to_csv('experiments_data/sim_{}.csv'.format(args.name), index=False)

        print('Saved data with postfix: {}'.format(args.name))

    print("ACTION FINISHED")
    time.sleep(3)

    sim_env.close()
    real_env.close()

    print("TOTAL SUCCESSES: ", total_success)
    print("Environment closed")
