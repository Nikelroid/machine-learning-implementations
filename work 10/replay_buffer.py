import random
from collections import deque
from utils import BatchTransition, Transition

class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer = deque(maxlen=capacity) # DO NOT MODIFY

    @property
    def size(self) -> int:
        return len(self.buffer) # DO NOT MODIFY
    
    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)


    def sample(self, batch_size: int) -> BatchTransition:
        if batch_size <= 0:raise ValueError("Batch size must be greater than 0.")
        if batch_size >= len(self.buffer):samples = self.buffer
        else:samples = list(random.sample(self.buffer, batch_size))
        return BatchTransition.from_list(samples)
           
    def clear(self) -> None:
        self.buffer.clear() # DO NOT MODIFY

