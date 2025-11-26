import numpy as np
from finite_mdp import DiscreteSpace
from replay_buffer import ReplayBuffer
from utils import Transition

class QLearningAgent:
    def __init__(
        self, state_space: DiscreteSpace, action_space: DiscreteSpace, lr: float = 0.1,
        discount: float = 0.99, explore_rate: float = 0.1, buffer_capacity: int = 1,
        batch_size: int = 32) -> None:

        self.lr = lr
        self.discount = discount
        self.explore_rate = explore_rate
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = None
        self.reset()

    def reset(self) -> None:
        self.q_table = np.random.normal(size=(self.state_space.n, self.action_space.n))
        self.replay_buffer.clear()

    def act(self, state: int, exploit: bool = False) -> int:
        if exploit:action = np.argmax(self.q_table[state])
        else:
            if np.random.rand() < self.explore_rate:action = np.random.choice(self.action_space.n)
            else:action = np.argmax(self.q_table[state])
        return action

    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        transition = Transition(state, action, reward, next_state)
        self.replay_buffer.push(transition)

    def learn(self) -> dict:
        batch = self.replay_buffer.sample(self.batch_size)
        states,actions,rewards,next_states = batch.states,batch.actions,batch.rewards,batch.next_states
        td_target = rewards + self.discount * np.max(self.q_table[next_states], axis=1)
        td_error = td_target - self.q_table[states, actions]
        self.q_table[states, actions] += self.lr * td_error
        self.q_table = np.clip(self.q_table, 0, 1)
        return {'td_error': td_error,'q_table': self.q_table,'value_arr': self.get_value_arr(),'policy_arr': self.get_policy_arr()}

    def get_policy_arr(self) -> np.ndarray:
        return np.argmax(self.q_table, axis=1)

    def get_value_arr(self) -> np.ndarray:
        return np.max(self.q_table, axis=1)
