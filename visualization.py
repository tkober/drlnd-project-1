import matplotlib.pyplot as plt
import numpy as np


def plot_agent_learning(name, episode_scores, running_avg_score, goal):
    fig = plt.figure(figsize=(15, 6), dpi=80)
    ax = fig.add_subplot(111)
    plt.title(name)
    plt.plot(np.arange(len(episode_scores)), episode_scores, c='lightgrey', label='Episode Score')
    plt.axhline(y=goal, linestyle='solid', c='deepskyblue', label='Goal')

    episodes_reaching_goal = np.where(np.array(running_avg_score) >= goal)
    if len(episodes_reaching_goal[0]) > 0:
        episode_reached = episodes_reaching_goal[0][0]
        plt.axvline(x=episode_reached, linestyle='dashed', c='yellow', label='Goal reached')
        plt.text(episode_reached+10, 0, episode_reached, c='yellow')

    plt.plot(np.arange(len(running_avg_score)), running_avg_score, c='r', label='Running avg. Score')

    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.legend(loc='upper left')
    plt.show()