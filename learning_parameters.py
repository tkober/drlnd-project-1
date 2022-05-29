from dataclasses import dataclass


@dataclass()
class DeepQLearningParameters:
    max_episodes: int           # Maximum number of episodes to train
    scores_window: int          # Number of episodes scores that shall be considered for avg calculation
    goal_score: float           # Goal of the average of scores in the scores window
    stop_on_goal_reached: bool  # Set if you want training to be stopped if the goal score is reached
    max_t: int                  # Maximum number of steps per episode
    epsilon_start: float        # Start value of epsilon
    epsilon_decay: float        # Decay of epsilon per episode
    epsilon_min: float          # Minimum value of epsilon
    replay_buffer_size: int     # Capacity of replay buffer
    batch_size: int             # Size of the batch drawn from the replay buffer
    discount_factor: float      # Discount factor for rewards
    tau: float                  # For soft updating target parameters
    learning_rate: float        # Learning rate for the optimizer
    update_interval: int        # How often the network shall be updated
