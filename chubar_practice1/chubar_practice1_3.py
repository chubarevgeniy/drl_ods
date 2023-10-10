import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')

state_n = 500
action_n = 6

class CrossEntropyAgent():
    def __init__(self, state_n, action_n, model = None):
        self.state_n = state_n
        self.action_n = action_n
        if(model):
            self.model = model.copy
        else:
            self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state, _model = None):
        model = self.model
        if(isinstance(_model,np.ndarray)):
            model = self.model
        action = np.random.choice(np.arange(self.action_n), p=model[state])
        return int(action)
    
    def sample_determ_model(self, model):
        res = np.zeros(model.shape)
        for state in range(model.shape[0]):
            action = np.random.choice(np.arange(model.shape[1]), p=model[state])
            res[state,action] = 1
        return res

    def fit(self, elite_trajectories, update_type = 'rewrite', update_coef = 0):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                if(update_type == 'Laplace'):
                    new_model[state] += update_coef
                    new_model[state] /= (np.sum(new_model[state]) + update_coef*action_n)
                else:
                    new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()
        if(update_type == 'rewrite' or update_type == 'Laplace'):
            self.model = new_model
        if(update_type == 'Policy'):
            self.model = self.model*update_coef + new_model*(1 - update_coef)
            
        return None


def get_state(obs):
    return obs


def get_trajectory(env, agent, model = None, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()
    state = get_state(obs)

    for _ in range(max_len):
        trajectory['states'].append(state)
        
        action = agent.get_action(state, model)
        trajectory['actions'].append(action)
        
        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
        
        state = get_state(obs)

        if visualize:
            time.sleep(0.5)
            env.render()

        if done:
            break
    
    return trajectory


agent = CrossEntropyAgent(state_n, action_n)
q_param = 0.6
iteration_n = 100
trajectory_n_per_realisation = 20
samples_number = 400

hist = []

for iteration in range(iteration_n):

    #policy evaluation
    trajectories = []
    for realisation in range(samples_number):
        model = agent.sample_determ_model(agent.model)
        trajectories_r = [get_trajectory(env, agent, model) for _ in range(trajectory_n_per_realisation)]
        trajectories.append(trajectories_r)
    total_rewards = []
    for realisation in trajectories:
        total_rewards.append(np.mean([np.sum(trajectory['rewards']) for trajectory in realisation]))
    print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))
    hist.append(np.mean(total_rewards))
    #policy improvement
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = []
    for realisation in trajectories:
        total_reward = np.mean([np.sum(trajectory['rewards']) for trajectory in realisation])
        if total_reward > quantile:
            elite_trajectories.extend(realisation)

    agent.fit(elite_trajectories)

trajectory = get_trajectory(env, agent, max_len=100, visualize=True)
print('total reward:', sum(trajectory['rewards']))
np.savetxt('stoc_pol_q_'+str(q_param)+'__sample_n_'+str(samples_number)+'__traj_per_samp_'+str(trajectory_n_per_realisation)+'.txt',np.array(hist))
#print('model:')
#print(agent.model)