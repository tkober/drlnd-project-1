import numpy as np
import torch
from collections import deque
from unityagents import UnityEnvironment
from agent import Agent
from learning_parameters import DeepQLearningParameters


def save_agent(agent: Agent, path: str):
    torch.save(agent.qnetwork_local.state_dict(), path)


def train_agent(agent: Agent, env: UnityEnvironment, parameters: DeepQLearningParameters, log=True, log_progress=100):
    episode_scores = []
    moving_avg = []
    avg_score_window = deque(maxlen=parameters.scores_window)
    epsilon = parameters.epsilon_start

    brain_name = env.brain_names[0]

    for i_episode in range(1, parameters.max_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]           # Reset environment
        state = env_info.vector_observations[0]                     # Get current state
        score = 0                                                   # Initialize score of this episode

        for t in range(parameters.max_t):
            action = agent.act(state, epsilon)                      # Select an action
            env_info = env.step(action)[brain_name]                 # Apply action in environment
            next_state = env_info.vector_observations[0]            # Get resulting state
            reward = env_info.rewards[0]                            # Get reward of action
            done = env_info.local_done[0]                           # Check if episode is done
            agent.step(state, action, reward, next_state, done)     # Let the agent learn

            state = next_state
            score += reward

            if done:
                break

        # Calculate current average score
        episode_scores.append(score)
        avg_score_window.append(score)
        avg_score = np.mean(avg_score_window)
        moving_avg.append(avg_score)

        if log and i_episode % log_progress == 0:
            progress = f'\rEpisode {i_episode:>6}\tAvg Score:{avg_score:.2f}'
            print(progress)

        # Check if goal score has been reached
        if log and parameters.stop_on_goal_reached and avg_score >= parameters.goal_score:
            print(f'\n\nGoal reached at episode {i_episode}\tAvg Score:{avg_score:.2f}')
            break

        # Update epsilon
        epsilon = max(parameters.epsilon_min, epsilon*parameters.epsilon_decay)

    return episode_scores, moving_avg
