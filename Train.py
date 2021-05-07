import os
import gym
from copy import deepcopy
from DDPG import *
import pandas as pd
import seaborn as sns
import matplotlib as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

warmup = 100
train_iter = 500000
max_episode_length = 500

def plot_rewards(episode_rewards):
    sns.set()
    fig = plt.figure(figsize=(20,7))
    rewards_smoothed = pd.Series(episode_rewards).rolling(15, min_periods=15).mean()
    plt.plot(episode_rewards,color="#f4c17c",label="episode rewards")
    plt.plot(rewards_smoothed,color="#ec5528",label="avg rewards")
    plt.ylim(-400, 300)
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.show(fig)

def train(num_iterations,
          agent,
          env,
          max_episode_length=None):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    episode_reward_list = []

    observation = None
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action
        if step <= warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > warmup:
            agent.update_alpha()
            agent.update_policy()

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:
            print('#{}: current episode_reward:{} | value_loss:{} | policy_loss:{}'
                  .format(episode+1,episode_reward,agent.value_loss,agent.policy_loss))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )
            episode_reward_list.append(episode_reward)

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

            if(episode % 30 == 0):
                agent.save_model(episode)


    return episode_reward_list

def run():
    env = gym.make("Pendulum-v0")
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    agent = DDPG(nb_states, nb_actions)
    rewards = train(train_iter, agent, env,max_episode_length=max_episode_length)
    rewards_save = np.array(rewards)
    np.save('reward_result/result.npy', rewards_save)
    plot_rewards(rewards)
    return rewards

run()

