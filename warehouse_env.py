import numpy as np

class WarehouseEnv:
    def __init__(self):
        # 0: empty, 1: wall, 2: item, 3: destination
        self.grid = np.array([
            [0, 0, 0, 1, 0],
            [0, 1, 0, 1, 3],
            [0, 1, 0, 0, 0],
            [2, 0, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
        self.start_pos = (0, 0)
        self.agent_pos = list(self.start_pos)
        self.has_item = False
        self.max_steps = 50
        self.current_step = 0
        
    def reset(self):
        self.agent_pos = list(self.start_pos)
        self.has_item = False
        self.current_step = 0
        self.grid[3, 0] = 2 # Ensure item is reset
        return self._get_obs()
        
    def _get_obs(self):
        return f"Grid:\n{self.grid}\nAgent Position: {self.agent_pos}\nHas Item: {self.has_item}"

    def step(self, action):
        # actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.current_step += 1
        
        row, col = self.agent_pos
        next_row, next_col = row, col
        
        if action == 0: next_row -= 1
        elif action == 1: next_row += 1
        elif action == 2: next_col -= 1
        elif action == 3: next_col += 1
        
        reward = -1.0 # Standard step penalty
        done = False
        error_msg = "null" # Required format for 'no error' by the hackathon guidelines
        
        # Check boundaries
        if next_row < 0 or next_row >= self.grid.shape[0] or next_col < 0 or next_col >= self.grid.shape[1]:
            reward = -15.0
            error_msg = "Hit boundary"
        elif self.grid[next_row, next_col] == 1:
            reward = -15.0
            error_msg = "Hit wall"
        else:
            self.agent_pos = [next_row, next_col]
            
            # Pick up item
            if self.grid[next_row, next_col] == 2 and not self.has_item:
                self.has_item = True
                self.grid[next_row, next_col] = 0 # Clear item from grid
                
            # Drop off at destination
            if self.grid[next_row, next_col] == 3 and self.has_item:
                reward = 100.0
                done = True
                
        if self.current_step >= self.max_steps:
            done = True
            
        return self._get_obs(), float(reward), done, error_msg