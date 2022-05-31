[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[sarsamax]: ./assets/sarsa_max.png "Q-Learning (Sarsamax) Algorithm"
[dqn]: ./assets/dqn.png "Deep Q-Learning Algorithm"
[initial_model]: ./assets/initial_model.png "Initial Model"
[improvements]: ./assets/improvements.png "Improvements"
[demo]: ./assets/demo.png "Demo Video"

**Udacity Deep Reinforcement Learning Nanodegree**

# Project 1: Navigation

## Environment

For this project, we will train an agent to navigate (and collect bananas!) in a large, square world.  

![Expected Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Deep Q-Learning

In this project we will use **Deep Q-Networks (DQN)** to train an agent in our environment. 
The approach is combining the **temporal difference** method of Q-Learning with a **value-based** function approximation.
To be more concrete we will use a deep neural network to represent the **value function**.

#### Differences to Q-Learning

In traditional Q-Learning (also called Sarsamax) we represented our policy by a Q-table.
During training, we used to update the Q-tables value for the current state and action by considering the immediate reward and the discounted maximum reward of the next state and corresponding action.

![Q-Learning][sarsamax]

For the DQN method the goal is to get a good approximation of the **value function** for the optimal policy.
Here this function is represented by a deep neural network and its weights.

#### Experience Replay

An important aspect to consider in DQNs is that rapidly tend to overfit to (temporal) correlated training data.
A simple solution to that problem is to not let the agent learn immediately but first letting it generate episode data using the current value function.
These episode data are stored in a buffer and later drawn in random batches from it for training.
These two phases of **sampling** and **learning** are than alternated during training.

#### Algorithm

![Deep Q-Learning][dqn]


## Training

#### Architecture

As an initial architecture for our neural network we will go with a simple feed forward network featuring two hidden layers.
The base hyperparameters for the training are described in the following code listing:

```python
goal_score = 13.0

# Define Network
network_params = NetworkParameters(
    n_states=state_size,
    n_actions=action_size,
    seed=0,
    fc1_size=64,
    fc2_size=64,
    duelling_network=False,
)

# Define Parameters for Q Learning
learning_params = DeepQLearningParameters(
    max_episodes=1000,
    scores_window=100,
    goal_score=goal_score,
    stop_on_goal_reached=False,
    max_t=1000,
    epsilon_start=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    replay_buffer_size=int(1e5),
    batch_size=64,
    discount_factor=0.99,
    tau=1e-3,
    learning_rate=5e-4,
    update_interval=4,
    double_dqn=False
)
```

### Parameter Tuning
In order to boost the learning performance and consistence we will vary the following hyperparameters and train 1000 episodes with each combination:
- Hidden layer size: **(64 | 128)**
- Batch size: **(64 | 128 | 256)**
- Update interval: **(4 | 8)**

The results look like this:

| Layer Size | Batch Size | Update Interval | Max Avg. Score | Episodes to Goal |
|------------|------------|-----------------|----------------|------------------|
| 64         | 64         | 4               | 16.38          | 474              |
| 64         | 64         | 8               | 16.36          | 569              |
| 64         | 128        | 4               | 16.15          | 534              |
| 64         | 128        | 8               | 16.57          | 520              |
| 64         | 256        | 4               | 15.82          | 523              |
| 64         | 256        | 8               | 16.83          | 623              |
| 128        | 64         | 4               | 15.68          | 551              |
| 128        | 64         | 8               | 16.14          | 585              |
| 128        | 128        | 4               | 14.79          | 564              |
| 28         | 128        | 8               | 16.44          | 653              |
| 128        | 256        | 4               | 13.75          | 885              |
| 128        | 256        | 8               | 15.39          | 598              |


#### Baseline

Using the table above we pick the most simple model with a layer and batch size of 64 and an update interval of 4.
More complex models did not achieve higher average scores but also took longer to reach the goal score.

![Initial Model][initial_model]

#### Improvements

As an additional step we tried to improve our picked baseline model by applying the concepts of **duelling** networks and **Double DQNs**.
The results did not show a significant improvement, neither in speed to reach goal score nor in overall average score.
At least not for the first 1000 trained episodes.

![Improvemented Model][improvements]

## Trained Agent

[![demo]](https://www.youtube.com/watch?v=H337QB8ObBc)

## Future Improvements

Looking at the agent behaviour from the video above its biggest weakness seems to be tight corners of blue bananas piles.
For smaller piles it succeeds to back up a little and drive around them.

One possible solution might be to provide more temporal information to the agent so that it can learn to back up to an earlier position.
Another promising improvement could be the concept of **Prioritized Experience Replay**.