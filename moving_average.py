from collections import deque
import numpy as np

class MovingAverage:
    def __init__(self, window_size: int = 20):
        self.window_size   = window_size
        self.window        = deque(maxlen=window_size)  # rolling buffer
        self.running_sum   = 0.0
        self.last_value    = None   # most recent *accepted* input
        self.last_average  = None   # most recent average

    # -------------------------------------------------------------- #
    def update(self, value: float) -> float:
        # Skip duplicates
        if self.last_value is not None and value == self.last_value and abs(value - self.last_value) > np.deg2rad(20):
            return self.last_average

        # Evict oldest if the window is full
        if len(self.window) == self.window.maxlen:
            self.running_sum -= self.window[0]

        # Insert the new sample
        self.window.append(value)
        self.running_sum += value
        self.last_value = value

        # Compute the mean over the current window length
        self.last_average = self.running_sum / len(self.window)
        return self.last_average
    
    def reset(self):
        """Reset the moving average state."""
        self.window.clear()
        self.running_sum = 0.0
        self.last_value = None
        self.last_average = None