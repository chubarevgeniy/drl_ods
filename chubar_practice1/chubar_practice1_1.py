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

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)

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


def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()
    state = get_state(obs)

    for _ in range(max_len):
        trajectory['states'].append(state)
        
        action = agent.get_action(state)
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
iteration_n = 50
trajectory_n = 600

hist = []

for iteration in range(iteration_n):

    #policy evaluation
    trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards), 'std total reward: ',np.std(total_rewards))
    hist.append(np.mean(total_rewards))
    #policy improvement
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards'])
        if total_reward > quantile:
            elite_trajectories.append(trajectory)

    agent.fit(elite_trajectories)

trajectory = get_trajectory(env, agent, max_len=100, visualize=True)
print('total reward:', sum(trajectory['rewards']))
#np.savetxt('rewrite_pol_q_'+str(q_param)+'__tr_n_'+str(trajectory_n)+'.txt',np.array(hist))
#print('model:')
#print(agent.model)