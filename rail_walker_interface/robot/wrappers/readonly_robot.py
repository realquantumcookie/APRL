import numpy as np
from ..robot_wrapper import WalkerWrapper

class ReadOnlyWalkerWrapper(WalkerWrapper):
    def receive_observation(self) -> bool:
        return True
    
    def apply_action(self, action: np.ndarray) -> bool:
        return True